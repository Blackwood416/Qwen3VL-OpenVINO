using Microsoft.Data.Sqlite;
using System.Text.Json;

namespace Qwen3VL;

/// <summary>SQLite-based session persistence layer</summary>
public class SessionStore : IDisposable
{
    private readonly string _dbPath;
    private readonly SqliteConnection _conn;

    public SessionStore(string dbPath)
    {
        _dbPath = dbPath;
        _conn = new SqliteConnection($"Data Source={dbPath}");
        _conn.Open();
        InitDb();
    }

    private void InitDb()
    {
        using (var cmd = _conn.CreateCommand())
        {
            cmd.CommandText = @"
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;

                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    system_prompt TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    version_index INTEGER NOT NULL DEFAULT 0,
                    prompt TEXT NOT NULL,
                    response_text TEXT,
                    image_path TEXT,
                    video_path TEXT,
                    max_video_frames INTEGER DEFAULT 8,
                    generated_ids BLOB,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );
            ";
            cmd.ExecuteNonQuery();
        }

        // 迁移：安全地添加缺失列
        var existingCols = new HashSet<string>();
        using (var cmd = _conn.CreateCommand())
        {
            cmd.CommandText = "PRAGMA table_info(turns)";
            using var reader = cmd.ExecuteReader();
            while (reader.Read()) existingCols.Add(reader.GetString(1));
        }

        void AddColumnIfMissing(string table, string col, string def)
        {
            var colSet = table == "turns" ? existingCols : null;
            if (colSet == null)
            {
                colSet = new HashSet<string>();
                using var c = _conn.CreateCommand();
                c.CommandText = $"PRAGMA table_info({table})";
                using var r = c.ExecuteReader();
                while (r.Read()) colSet.Add(r.GetString(1));
            }
            if (!colSet.Contains(col))
            {
                using var c = _conn.CreateCommand();
                c.CommandText = $"ALTER TABLE {table} ADD COLUMN {col} {def}";
                c.ExecuteNonQuery();
            }
        }

        AddColumnIfMissing("turns", "version_index", "INTEGER NOT NULL DEFAULT 0");
        AddColumnIfMissing("turns", "stats_json", "TEXT");
        AddColumnIfMissing("sessions", "generation_config", "TEXT");

        using (var cmd = _conn.CreateCommand())
        {
            cmd.CommandText = "CREATE INDEX IF NOT EXISTS idx_turns_session_v ON turns(session_id, turn_index, version_index)";
            cmd.ExecuteNonQuery();
        }
    }

    // ═══════════════════════ Session CRUD ═══════════════════════

    public void SaveSession(string id, string? title, string? systemPrompt)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = @"
            INSERT INTO sessions (id, title, system_prompt, updated_at)
            VALUES ($id, $title, $sp, datetime('now'))
            ON CONFLICT(id) DO UPDATE SET
                title = $title,
                system_prompt = $sp,
                updated_at = datetime('now')
        ";
        cmd.Parameters.AddWithValue("$id", id);
        cmd.Parameters.AddWithValue("$title", (object?)title ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$sp", (object?)systemPrompt ?? DBNull.Value);
        cmd.ExecuteNonQuery();
    }

    public void UpdateTitle(string id, string title)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "UPDATE sessions SET title = $title, updated_at = datetime('now') WHERE id = $id";
        cmd.Parameters.AddWithValue("$id", id);
        cmd.Parameters.AddWithValue("$title", title);
        cmd.ExecuteNonQuery();
    }

    public List<SessionInfo> ListSessions()
    {
        var list = new List<SessionInfo>();
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "SELECT id, title, updated_at FROM sessions ORDER BY updated_at DESC";
        using var reader = cmd.ExecuteReader();
        while (reader.Read())
        {
            list.Add(new SessionInfo
            {
                Id = reader.GetString(0),
                Title = reader.IsDBNull(1) ? null : reader.GetString(1)
            });
        }
        return list;
    }

    public bool SessionExists(string id)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "SELECT COUNT(*) FROM sessions WHERE id = $id";
        cmd.Parameters.AddWithValue("$id", id);
        return Convert.ToInt64(cmd.ExecuteScalar()) > 0;
    }

    public string? GetSystemPrompt(string id)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "SELECT system_prompt FROM sessions WHERE id = $id";
        cmd.Parameters.AddWithValue("$id", id);
        var result = cmd.ExecuteScalar();
        return result == DBNull.Value ? null : result as string;
    }

    public bool DeleteSession(string id)
    {
        using var cmd = _conn.CreateCommand();
        // turns are cascade-deleted
        cmd.CommandText = "DELETE FROM sessions WHERE id = $id";
        cmd.Parameters.AddWithValue("$id", id);
        return cmd.ExecuteNonQuery() > 0;
    }

    // ═══════════════════════ Turn CRUD ═══════════════════════

    public void SaveTurn(string sessionId, int turnIndex, string prompt,
        string? responseText, string? imagePath, string? videoPath,
        int maxVideoFrames, List<int>? generatedIds, int versionIndex = 0, string? statsJson = null)
    {
        byte[]? idsBlob = null;
        if (generatedIds != null && generatedIds.Count > 0)
        {
            idsBlob = new byte[generatedIds.Count * 4];
            Buffer.BlockCopy(generatedIds.ToArray(), 0, idsBlob, 0, idsBlob.Length);
        }

        using var cmd = _conn.CreateCommand();
        cmd.CommandText = @"
            INSERT OR REPLACE INTO turns (session_id, turn_index, version_index, prompt, response_text, image_path, video_path, max_video_frames, generated_ids, stats_json)
            VALUES ($sid, $idx, $vidx, $prompt, $resp, $img, $vid, $frames, $ids, $stats)
        ";
        cmd.Parameters.AddWithValue("$sid", sessionId);
        cmd.Parameters.AddWithValue("$idx", turnIndex);
        cmd.Parameters.AddWithValue("$vidx", versionIndex);
        cmd.Parameters.AddWithValue("$prompt", prompt);
        cmd.Parameters.AddWithValue("$resp", (object?)responseText ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$img", (object?)imagePath ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$vid", (object?)videoPath ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$frames", maxVideoFrames);
        cmd.Parameters.AddWithValue("$ids", (object?)idsBlob ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$stats", (object?)statsJson ?? DBNull.Value);
        cmd.ExecuteNonQuery();

        // Update session timestamp
        using var upd = _conn.CreateCommand();
        upd.CommandText = "UPDATE sessions SET updated_at = datetime('now') WHERE id = $sid";
        upd.Parameters.AddWithValue("$sid", sessionId);
        upd.ExecuteNonQuery();
    }

    public bool DeleteLastTurn(string sessionId)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = @"
            DELETE FROM turns 
            WHERE session_id = $sid AND turn_index = (
                SELECT MAX(turn_index) FROM turns WHERE session_id = $sid
            )
        ";
        cmd.Parameters.AddWithValue("$sid", sessionId);
        return cmd.ExecuteNonQuery() > 0;
    }

    /// <summary>删除 turn_index >= fromIndex 的所有轮次</summary>
    public int DeleteTurnsFrom(string sessionId, int fromIndex)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "DELETE FROM turns WHERE session_id = $sid AND turn_index >= $idx";
        cmd.Parameters.AddWithValue("$sid", sessionId);
        cmd.Parameters.AddWithValue("$idx", fromIndex);
        return cmd.ExecuteNonQuery();
    }

    public List<HistoryItem> GetHistory(string sessionId)
    {
        var list = new List<HistoryItem>();
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = @"
            SELECT prompt, response_text, image_path, video_path, version_index, stats_json
            FROM turns t1
            WHERE session_id = $sid AND version_index = (
                SELECT MAX(version_index) FROM turns t2 
                WHERE t2.session_id = t1.session_id AND t2.turn_index = t1.turn_index
            )
            ORDER BY turn_index
        ";
        cmd.Parameters.AddWithValue("$sid", sessionId);
        using var reader = cmd.ExecuteReader();
        while (reader.Read())
        {
            list.Add(new HistoryItem
            {
                Prompt = reader.GetString(0),
                Response = reader.IsDBNull(1) ? "" : reader.GetString(1),
                ImagePath = reader.IsDBNull(2) ? null : Path.GetFileName(reader.GetString(2)),
                VideoPath = reader.IsDBNull(3) ? null : Path.GetFileName(reader.GetString(3)),
                VersionIndex = reader.GetInt32(4),
                Stats = reader.IsDBNull(5) ? null : reader.GetString(5)
            });
        }
        return list;
    }

    public List<string> GetTurnVersions(string sessionId, int turnIndex)
    {
        var list = new List<string>();
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "SELECT response_text FROM turns WHERE session_id = $sid AND turn_index = $idx ORDER BY version_index";
        cmd.Parameters.AddWithValue("$sid", sessionId);
        cmd.Parameters.AddWithValue("$idx", turnIndex);
        using var reader = cmd.ExecuteReader();
        while (reader.Read()) list.Add(reader.IsDBNull(0) ? "" : reader.GetString(0));
        return list;
    }

    public int GetMaxVersion(string sessionId, int turnIndex)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "SELECT MAX(version_index) FROM turns WHERE session_id = $sid AND turn_index = $idx";
        cmd.Parameters.AddWithValue("$sid", sessionId);
        cmd.Parameters.AddWithValue("$idx", turnIndex);
        var res = cmd.ExecuteScalar();
        return res == DBNull.Value ? -1 : Convert.ToInt32(res);
    }

    public List<ReplayTurn> GetTurnsForReplay(string sessionId)
    {
        var list = new List<ReplayTurn>();
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = @"
            SELECT prompt, image_path, video_path, max_video_frames, generated_ids, version_index
            FROM turns t1
            WHERE session_id = $sid AND version_index = (
                SELECT MAX(version_index) FROM turns t2 
                WHERE t2.session_id = t1.session_id AND t2.turn_index = t1.turn_index
            )
            ORDER BY turn_index
        ";
        cmd.Parameters.AddWithValue("$sid", sessionId);
        using var reader = cmd.ExecuteReader();
        while (reader.Read())
        {
            var turn = new ReplayTurn
            {
                Prompt = reader.GetString(0),
                ImagePath = reader.IsDBNull(1) ? null : reader.GetString(1),
                VideoPath = reader.IsDBNull(2) ? null : reader.GetString(2),
                MaxVideoFrames = reader.GetInt32(3),
                GeneratedIds = new List<int>()
            };

            if (!reader.IsDBNull(4))
            {
                byte[] blob = (byte[])reader[4];
                int[] ids = new int[blob.Length / 4];
                Buffer.BlockCopy(blob, 0, ids, 0, blob.Length);
                turn.GeneratedIds = ids.ToList();
            }
            turn.VersionIndex = reader.GetInt32(5);

            list.Add(turn);
        }
        return list;
    }

    public int GetTurnCount(string sessionId)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "SELECT COUNT(*) FROM turns WHERE session_id = $sid";
        cmd.Parameters.AddWithValue("$sid", sessionId);
        return Convert.ToInt32(cmd.ExecuteScalar());
    }

    // ═══════════════════════ Generation Config ═══════════════════════

    public void SaveGenerationConfig(string sessionId, string configJson)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "UPDATE sessions SET generation_config = $cfg, updated_at = datetime('now') WHERE id = $id";
        cmd.Parameters.AddWithValue("$id", sessionId);
        cmd.Parameters.AddWithValue("$cfg", configJson);
        cmd.ExecuteNonQuery();
    }

    public string? GetGenerationConfig(string sessionId)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "SELECT generation_config FROM sessions WHERE id = $id";
        cmd.Parameters.AddWithValue("$id", sessionId);
        var res = cmd.ExecuteScalar();
        return res == DBNull.Value || res == null ? null : res as string;
    }

    // ═══════════════════════ Migration ═══════════════════════

    /// <summary>从旧 sessions/ 目录迁移到 SQLite</summary>
    public int MigrateFromJsonDir(string sessionsDir, Qwen3VLPipeline pipeline)
    {
        if (!Directory.Exists(sessionsDir)) return 0;

        int migrated = 0;
        foreach (var dir in Directory.GetDirectories(sessionsDir))
        {
            string sessionId = Path.GetFileName(dir);
            if (SessionExists(sessionId)) continue; // Already migrated

            string jsonPath = Path.Combine(dir, "session.json");
            if (!File.Exists(jsonPath)) continue;

            // Check for .deleted marker
            if (File.Exists(Path.Combine(dir, ".deleted"))) continue;

            try
            {
                string json = File.ReadAllText(jsonPath);
                var state = JsonSerializer.Deserialize<MigrationSessionState>(json);
                if (state == null) continue;

                SaveSession(sessionId, state.Title, state.SystemPrompt);

                for (int i = 0; i < state.History.Count; i++)
                {
                    var h = state.History[i];
                    string responseText = pipeline.DecodeOnly(h.GeneratedIds);
                    SaveTurn(sessionId, i, h.Prompt, responseText,
                        h.ImagePath, h.VideoPath, h.MaxVideoFrames, h.GeneratedIds);
                }

                migrated++;
                Console.WriteLine($"[Migration] Migrated session: {sessionId} ({state.History.Count} turns)");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Migration] Failed to migrate {sessionId}: {ex.Message}");
            }
        }

        return migrated;
    }

    public void Dispose()
    {
        _conn?.Close();
        _conn?.Dispose();
    }

    // ═══════════════════════ DTOs ═══════════════════════

    public class SessionInfo
    {
        public string Id { get; set; } = "";
        public string? Title { get; set; }
    }

    public class HistoryItem
    {
        public string Prompt { get; set; } = "";
        public string Response { get; set; } = "";
        public string? ImagePath { get; set; }
        public string? VideoPath { get; set; }
        public int VersionIndex { get; set; }
        public string? Stats { get; set; }
    }

    public class ReplayTurn
    {
        public string Prompt { get; set; } = "";
        public string? ImagePath { get; set; }
        public string? VideoPath { get; set; }
        public int MaxVideoFrames { get; set; } = 8;
        public List<int> GeneratedIds { get; set; } = new();
        public int VersionIndex { get; set; }
    }

    // For JSON deserialization of old session.json
    private class MigrationSessionState
    {
        public string? Title { get; set; }
        public string? SystemPrompt { get; set; }
        public List<MigrationTurn> History { get; set; } = new();
    }

    private class MigrationTurn
    {
        public string Prompt { get; set; } = "";
        public string? ImagePath { get; set; }
        public string? VideoPath { get; set; }
        public int MaxVideoFrames { get; set; } = 8;
        public List<int> GeneratedIds { get; set; } = new();
    }
}
