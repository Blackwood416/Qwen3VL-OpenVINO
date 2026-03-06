using Qwen3VL;
using Microsoft.Extensions.FileProviders;
using Microsoft.AspNetCore.Builder;

var builder = WebApplication.CreateBuilder(args);

// 配置文件上传大小限制 (200MB)
builder.WebHost.ConfigureKestrel(options =>
{
    options.Limits.MaxRequestBodySize = 200 * 1024 * 1024;
});

builder.Services.AddSignalR(options =>
{
    options.MaximumReceiveMessageSize = 1024 * 1024; // 1MB
    options.EnableDetailedErrors = true;
    options.KeepAliveInterval = TimeSpan.FromSeconds(15);
    options.ClientTimeoutInterval = TimeSpan.FromSeconds(120);
    options.MaximumParallelInvocationsPerClient = 2; // 允许 CancelGeneration 与 SendMessage 并行
});
builder.Services.AddControllers();

// 注册 Pipeline 单例
string modelPath = builder.Configuration["ModelPath"]
    ?? Environment.GetEnvironmentVariable("QWEN3VL_MODEL_PATH")
    ?? @"C:\Alia_Models\Qwen3-VL-4B-Instruct-int4";
string device = builder.Configuration["Device"]
    ?? Environment.GetEnvironmentVariable("QWEN3VL_DEVICE")
    ?? "GPU";

Console.WriteLine($"[Startup] Model: {modelPath}, Device: {device}");
var pipeline = new Qwen3VLPipeline(modelPath, device);
builder.Services.AddSingleton(pipeline);

// 注册 SessionStore 并迁移旧数据
string dbPath = Path.Combine(Directory.GetCurrentDirectory(), "sessions.db");
string sessionsDir = Path.Combine(Directory.GetCurrentDirectory(), "sessions");
var store = new SessionStore(dbPath);
int migrated = store.MigrateFromJsonDir(sessionsDir, pipeline);
if (migrated > 0) Console.WriteLine($"[Startup] Successfully migrated {migrated} sessions to SQLite.");
builder.Services.AddSingleton(store);

var app = builder.Build();

// 强制自启动加载模型 (不等到第一个请求)
Console.WriteLine("[Startup] Waking up Qwen3VLPipeline...");
// Already initialized above

// 静态文件 (wwwroot)
app.UseDefaultFiles();
app.UseStaticFiles();

// 映射 uploads 目录
string uploadDir = Path.Combine(Directory.GetCurrentDirectory(), "uploads");
if (!Directory.Exists(uploadDir)) Directory.CreateDirectory(uploadDir);

app.UseStaticFiles(new StaticFileOptions
{
    FileProvider = new Microsoft.Extensions.FileProviders.PhysicalFileProvider(uploadDir),
    RequestPath = "/uploads"
});

app.UseRouting();
app.MapControllers();
app.MapHub<ChatHub>("/chatHub");

Console.WriteLine("[Startup] Server ready. Open http://localhost:5000 in your browser.");
app.Run("http://0.0.0.0:5000");