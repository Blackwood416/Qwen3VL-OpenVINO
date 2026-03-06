using System;
using System.Collections.Generic;

namespace Qwen3VL;

public class GenerationConfig
{
    public int MaxTokens { get; set; } = 512;
    public bool Greedy { get; set; } = false;
    public float TopP { get; set; } = 0.8f;
    public int TopK { get; set; } = 20;
    public float Temperature { get; set; } = 0.7f;
    public float RepetitionPenalty { get; set; } = 1.0f;
    public float PresencePenalty { get; set; } = 1.5f;
    public float FrequencyPenalty { get; set; } = 0.0f;
    public int ContextWindow { get; set; } = 4096;
    public List<string> StopSequences { get; set; } = new();
    public string Mode { get; set; } = "chat"; // Options: chat, computer_use, 2d_grounding, mobile_agent
}
