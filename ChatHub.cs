using System.Collections.Concurrent;
using Microsoft.AspNetCore.SignalR;
using System.IO;
using System.Linq;

namespace Qwen3VL;

public class ChatHub : Hub
{
    private readonly Qwen3VLPipeline _pipeline;
    private readonly SessionStore _store;

    // 全局活跃会话存储：SessionId -> (ChatSession, Semaphore)
    private static readonly ConcurrentDictionary<string, (Qwen3VLPipeline.ChatSession Session, SemaphoreSlim Lock)> ActiveSessions = new();
    
    // 连接 -> 活跃会话映射
    private static readonly ConcurrentDictionary<string, string> ConnectionSessions = new();
    
    // 连接 -> 取消令牌
    private static readonly ConcurrentDictionary<string, CancellationTokenSource> ActiveCts = new();

    public ChatHub(Qwen3VLPipeline pipeline, SessionStore store)
    {
        _pipeline = pipeline;
        _store = store;
    }

    /// <summary>创建新会话，返回 SessionId</summary>
    public string CreateSession()
    {
        // 查找内存中是否已有空会话（且尚未存盘到DB的，或者DB里也没有历史的）
        foreach (var kvp in ActiveSessions)
        {
            if (kvp.Value.Session.GetHistory().Count == 0 && _store.GetTurnCount(kvp.Key) == 0)
            {
                ConnectionSessions[Context.ConnectionId] = kvp.Key;
                return kvp.Key;
            }
        }

        var sessionId = Guid.NewGuid().ToString("N")[..8];
        var session = _pipeline.StartSession();

        ActiveSessions[sessionId] = (session, new SemaphoreSlim(1, 1));
        ConnectionSessions[Context.ConnectionId] = sessionId;
        return sessionId;
    }

    /// <summary>获取会话列表（SQLite）</summary>
    public List<SessionStore.SessionInfo> ListSessions()
    {
        return _store.ListSessions();
    }

    /// <summary>重命名会话</summary>
    public bool RenameSession(string sessionId, string newTitle)
    {
        _store.UpdateTitle(sessionId, newTitle);
        if (ActiveSessions.TryGetValue(sessionId, out var entry))
        {
            entry.Session.Title = newTitle;
        }
        return true;
    }

    /// <summary>切换会话（冷加载，不重放 KV Cache）</summary>
    public object? SwitchSession(string sessionId)
    {
        if (string.IsNullOrEmpty(sessionId)) return null;
        ConnectionSessions[Context.ConnectionId] = sessionId;
        return _store.GetHistory(sessionId);
    }

    /// <summary>物理删除会话</summary>
    public bool DeleteSession(string sessionId)
    {
        _store.DeleteSession(sessionId);
        if (ActiveSessions.TryRemove(sessionId, out var entry))
        {
            entry.Session.Dispose();
            entry.Lock.Dispose();
        }
        return true;
    }

    /// <summary>发送消息（包含升温逻辑）</summary>
    public async Task SendMessage(string sessionId, string prompt, string? imageName, string? videoName, int maxFrames, GenerationConfig? config, int? turnIndex = null)
    {
        if (string.IsNullOrEmpty(sessionId)) return;

        // 如果会话不在 ActiveSessions，初始化它
        if (!ActiveSessions.TryGetValue(sessionId, out var entry))
        {
            var session = _pipeline.StartSession();
            entry = (session, new SemaphoreSlim(1, 1));
            ActiveSessions[sessionId] = entry;
        }

        await entry.Lock.WaitAsync();
        try
        {
            var connectionId = Context.ConnectionId;

            // 升温逻辑
            if (entry.Session.GetHistory().Count == 0 && _store.GetTurnCount(sessionId) > 0)
            {
                await Clients.Caller.SendAsync("ReceiveStatus", "正在恢复对话上下文...");
                var turns = _store.GetTurnsForReplay(sessionId);
                var systemPrompt = _store.GetSystemPrompt(sessionId);
                var title = _store.ListSessions().FirstOrDefault(s => s.Id == sessionId)?.Title;
                
                var legacyTurns = turns.Select(t => new Qwen3VLPipeline.ChatSession.TurnData
                {
                    Prompt = t.Prompt,
                    ImagePath = t.ImagePath,
                    VideoPath = t.VideoPath,
                    MaxVideoFrames = t.MaxVideoFrames,
                    GeneratedIds = t.GeneratedIds
                }).ToList();

                await Task.Run(() => entry.Session.Load(systemPrompt, title, legacyTurns));
                await Clients.Caller.SendAsync("ReceiveStatus", "");
            }

            // 初始化 DB 条目
            if (!_store.SessionExists(sessionId))
            {
                _store.SaveSession(sessionId, "新会话", null);
            }

            // 设置标题
            if (string.IsNullOrEmpty(entry.Session.Title) || entry.Session.Title == "新会话" || entry.Session.Title == sessionId)
            {
                var newTitle = prompt.Length > 15 ? prompt[..15] + "..." : prompt;
                entry.Session.Title = newTitle;
                _store.UpdateTitle(sessionId, newTitle);
            }

            // 注入 System Prompt
            if (entry.Session.GetHistory().Count == 0)
            {
                string? sp = config?.Mode switch
                {
                    "computer_use" => SystemPrompts.ComputerUse,
                    "2d_grounding" => SystemPrompts.Grounding,
                    "ocr" => SystemPrompts.Ocr,
                    "doc_parsing" => SystemPrompts.DocumentParsing,
                    "spatial" => SystemPrompts.Spatial,
                    _ => null
                };
                if (sp != null)
                {
                    entry.Session.InitSystemPrompt(sp);
                    _store.SaveSession(sessionId, entry.Session.Title, sp);
                }
            }

            // 执行推理
            string? imagePath = ResolveFilePath(imageName, "uploads");
            string? videoPath = ResolveFilePath(videoName, "uploads");
            
            var cts = new CancellationTokenSource();
            ActiveCts[connectionId] = cts;
            
            await Clients.Caller.SendAsync("GenerationStarted");

            List<int> resultIds = new();
            object? lastStats = null;
            await Task.Run(() =>
            {
                resultIds = entry.Session.Chat(
                    prompt, imagePath, videoPath, maxFrames, config,
                    silent: true, ct: cts.Token,
                    onToken: token => _ = Clients.Client(connectionId).SendAsync("ReceiveToken", token),
                    onStats: stats => { lastStats = stats; _ = Clients.Client(connectionId).SendAsync("ReceiveStats", stats); }
                );
            }, cts.Token);

            int effectiveTurnIndex = turnIndex ?? entry.Session.GetHistory().Count - 1;
            int effectiveVersionIndex = 0;
            if (turnIndex.HasValue)
            {
                effectiveVersionIndex = _store.GetMaxVersion(sessionId, turnIndex.Value) + 1;
            }

            string? statsJson = lastStats != null ? System.Text.Json.JsonSerializer.Serialize(lastStats, new System.Text.Json.JsonSerializerOptions { PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.CamelCase }) : null;
            var responseText = _pipeline.DecodeOnly(resultIds);
            _store.SaveTurn(sessionId, effectiveTurnIndex, prompt, responseText, imagePath, videoPath, maxFrames, resultIds, effectiveVersionIndex, statsJson);

            await Clients.Caller.SendAsync("GenerationComplete", effectiveTurnIndex, effectiveVersionIndex);
        }
        catch (OperationCanceledException) { await Clients.Caller.SendAsync("GenerationCancelled"); }
        catch (Exception ex) { await Clients.Caller.SendAsync("Error", $"推理失败: {ex.Message}"); }
        finally
        {
            entry.Lock.Release();
            ActiveCts.TryRemove(Context.ConnectionId, out _);
        }
    }

    private string? ResolveFilePath(string? name, string dirName)
    {
        if (string.IsNullOrEmpty(name)) return null;
        var path = Path.Combine(Directory.GetCurrentDirectory(), dirName, name);
        return File.Exists(path) ? path : null;
    }

    /// <summary>删除最后一轮对话（用于重试）</summary>
    public void DeleteLastTurn(string sessionId)
    {
        if (string.IsNullOrEmpty(sessionId)) return;
        _store.DeleteLastTurn(sessionId);
        if (ActiveSessions.TryRemove(sessionId, out var entry))
        {
            entry.Session.Dispose();
            entry.Lock.Dispose();
        }
    }

    public void CancelGeneration()
    {
        if (ActiveCts.TryGetValue(Context.ConnectionId, out var cts)) cts.Cancel();
    }

    /// <summary>撤回指定轮次及其后所有轮次，驱逐内存会话</summary>
    public bool RecallTurn(string sessionId, int turnIndex)
    {
        if (string.IsNullOrEmpty(sessionId)) return false;
        _store.DeleteTurnsFrom(sessionId, turnIndex);
        // 驱逐内存会话，强制下次 WarmUp
        if (ActiveSessions.TryRemove(sessionId, out var entry))
        {
            entry.Session.Dispose();
            entry.Lock.Dispose();
        }
        return true;
    }

    public List<string> GetTurnVersions(string sessionId, int turnIndex)
    {
        return _store.GetTurnVersions(sessionId, turnIndex);
    }

    public bool SaveSessionConfig(string sessionId, string configJson)
    {
        if (string.IsNullOrEmpty(sessionId)) return false;
        _store.SaveGenerationConfig(sessionId, configJson);
        return true;
    }

    public string? GetSessionConfig(string sessionId)
    {
        if (string.IsNullOrEmpty(sessionId)) return null;
        return _store.GetGenerationConfig(sessionId);
    }

    public override Task OnDisconnectedAsync(Exception? exception)
    {
        if (ActiveCts.TryRemove(Context.ConnectionId, out var cts)) { cts.Cancel(); cts.Dispose(); }
        ConnectionSessions.TryRemove(Context.ConnectionId, out _);
        return base.OnDisconnectedAsync(exception);
    }
}
