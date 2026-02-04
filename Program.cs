using System;
using System.Linq;
using System.Collections.Generic;

namespace Qwen3VL;

class Program
{
    static void Main(string[] args)
    {
        string modelPath = @"C:\YOUR_MODEL_PATH\Qwen3-VL-8B-Instruct-int4";

        Console.WriteLine("Initializing Qwen3-VL Pipeline...");
        try
        {
            using var pipeline = new Qwen3VLPipeline(modelPath, "GPU");
            Console.WriteLine("Pipeline initialized successfully.");

            // User recommended config for Vision
            var config = new GenerationConfig
            {
                MaxTokens = 2048,
                Greedy = false,
                TopK = 20,
                TopP = 0.8f,
                Temperature = 0.7f,
                RepetitionPenalty = 1.1f, // Keeping 1.1 for safety against repetition
                PresencePenalty = 0.5f,
                FrequencyPenalty = 0.0f,
                ContextWindow = 128000,
                StopSequences = new List<string> { "<|im_start|>", "<|im_end|>", "User:", "Assistant:" }
            };

            var session = pipeline.StartSession();
            Console.WriteLine("\n--- Qwen3-VL REPL ---");
            Console.WriteLine("Type 'exit' to quit. First message should include absolute image path if you want vision.");
            Console.WriteLine("Press Ctrl+C to interrupt output during generation.");

            var cts = new System.Threading.CancellationTokenSource();
            Console.CancelKeyPress += (s, e) =>
            {
                e.Cancel = true; // Prevents process exit
                cts.Cancel();
                // We'll reset it in the loop
            };

            while (true)
            {
                // Reset/Renew CTS if it was cancelled
                if (cts.IsCancellationRequested)
                {
                    cts.Dispose();
                    cts = new System.Threading.CancellationTokenSource();
                }

                Console.Write("\nUser: ");
                string? input = Console.ReadLine();
                if (string.IsNullOrEmpty(input)) break;
                if (input.ToLower() == "exit") break;

                // Handle Commands
                if (input.StartsWith("/sys "))
                {
                    string sysPrompt = input.Substring(5).Trim();
                    session.Dispose();
                    session = pipeline.StartSession();
                    try
                    {
                        session.InitSystemPrompt(sysPrompt);
                        Console.WriteLine("Session restarted with new system prompt.");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Error setting system prompt: {ex.Message}");
                    }
                    continue;
                }

                if (input.StartsWith("/save "))
                {
                    string path = input.Substring(6).Trim();
                    try
                    {
                        session.Save(path);
                        Console.WriteLine($"Session saved to {path}");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Error saving session: {ex.Message}");
                    }
                    continue;
                }

                if (input.StartsWith("/load "))
                {
                    string path = input.Substring(6).Trim();
                    // Load requires a fresh session for History Replay
                    session.Dispose();
                    session = pipeline.StartSession();
                    try
                    {
                        session.Load(path);
                        Console.WriteLine($"Session loaded from {path}");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Error loading session: {ex.Message}");
                    }
                    continue;
                }

                string? img = null;
                string prompt = input;

                string trimmed = input.Trim();
                if (trimmed.Contains("|"))
                {
                    var parts = trimmed.Split('|', 2);
                    img = parts[0].Trim();
                    prompt = parts[1].Trim();
                }
                else if (trimmed.StartsWith("["))
                {
                    int endIdx = trimmed.IndexOf(']');
                    if (endIdx > 1)
                    {
                        img = trimmed.Substring(1, endIdx - 1).Trim();
                        prompt = trimmed.Substring(endIdx + 1).Trim();
                    }
                }
                else
                {
                    // Heuristic: check if the first part looks like a path
                    string[] words = trimmed.Split(new char[] { ' ', '\t' }, 2);
                    if (words.Length > 0 && (words[0].ToLower().EndsWith(".jpg") || words[0].ToLower().EndsWith(".jpeg") || words[0].ToLower().EndsWith(".png")))
                    {
                        if (System.IO.File.Exists(words[0]))
                        {
                            img = words[0];
                            prompt = words.Length > 1 ? words[1] : "Describe this image.";
                        }
                    }
                }

                if (!string.IsNullOrEmpty(img))
                {
                    img = img.Trim('[', ']');
                    if (!System.IO.File.Exists(img))
                    {
                        Console.WriteLine($"[Warning] Image file not found: {img}");
                        img = null;
                    }
                    else
                    {
                        Console.WriteLine($"[Info] Attached image: {img}");
                    }
                }

                try
                {
                    session.Chat(prompt, img, config, ct: cts.Token);
                }
                catch (OperationCanceledException)
                {
                    Console.WriteLine("\n[Info] Output interrupted by user.");
                }
                catch (Exception exTurn)
                {
                    Console.WriteLine($"\n[Turn Error] {exTurn.Message}");
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Critical Error: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }
}