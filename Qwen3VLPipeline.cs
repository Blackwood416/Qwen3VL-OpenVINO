using System;
using System.IO;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Runtime.InteropServices;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;
using Tokenizers.HuggingFace.Tokenizer;

namespace Qwen3VL;

public class Qwen3VLPipeline : IDisposable
{
    private readonly string _modelDir;
    private nint _core;
    private readonly Tokenizer _tokenizer;

    // Model parts
    private nint _visionModel;
    private nint _visionPosModel;
    private nint _mergerModel;
    private nint _textEmbedModel;
    private nint _languageModel;

    // Compiled parts
    private nint _visionCompiled;
    private nint _visionPosCompiled;
    private nint _mergerCompiled;
    private nint _textEmbedCompiled;
    private nint _languageCompiled;

    // Persistent Infer Requests
    private nint _visionInfer;
    private nint _visionPosInfer;
    private nint _mergerInfer;
    private nint _textEmbedInfer;

    // Configurable sizes
    private int _hiddenSize = 2560; // Default fallback
    private int _visionHiddenSize = 1024; // Default fallback
    private int _visionRoPEDim = 32; // Default fallback (4B), 8B is 36
    private Random _rng = new Random();

    public struct GenerationStats
    {
        public double PrefillMs { get; set; }
        public double DecodeMs { get; set; }
        public int TokenCount { get; set; }
        public double TokensPerSecond => TokenCount > 0 && DecodeMs > 0 ? (TokenCount / (DecodeMs / 1000.0)) : 0;
    }

    // High Performance Sampling Buffers
    private struct LogitCandidate
    {
        public float Logit;
        public int Id;
    }
    private LogitCandidate[]? _sortBuffer;

    public Qwen3VLPipeline(string modelDir, string device = "CPU")
    {
        _modelDir = modelDir;

        string tokenizerPath = Path.Combine(modelDir, "tokenizer.json");
        _tokenizer = Tokenizer.FromFile(tokenizerPath);

        var status = NativeMethods.ov_core_create(out _core);
        if (status != OvStatus.OK) throw new Exception($"Core create failed: {status}");

        LoadModels(device);
    }

    private void LoadModels(string device)
    {
        _visionModel = LoadPart("openvino_vision_embeddings_model.xml");
        _visionHiddenSize = GetOutputDimension(_visionModel, 0);
        PrintSignatures(_visionModel, "Vision");
        _visionCompiled = CompilePart(_visionModel, device);
        CheckStatus(NativeMethods.ov_compiled_model_create_infer_request(_visionCompiled, out _visionInfer), "Create Vision infer");

        _visionPosModel = LoadPart("openvino_vision_embeddings_pos_model.xml");
        PrintSignatures(_visionPosModel, "VisionPos");
        _visionPosCompiled = CompilePart(_visionPosModel, device);
        CheckStatus(NativeMethods.ov_compiled_model_create_infer_request(_visionPosCompiled, out _visionPosInfer), "Create VisionPos infer");

        _mergerModel = LoadPart("openvino_vision_embeddings_merger_model.xml");
        _visionRoPEDim = GetInputDimensionByName(_mergerModel, "rotary_pos_emb");
        PrintSignatures(_mergerModel, "Merger");
        _mergerCompiled = CompilePart(_mergerModel, device);
        CheckStatus(NativeMethods.ov_compiled_model_create_infer_request(_mergerCompiled, out _mergerInfer), "Create Merger infer");

        _textEmbedModel = LoadPart("openvino_text_embeddings_model.xml");
        _hiddenSize = GetOutputDimension(_textEmbedModel, 0); // Dynamic hidden size
        PrintSignatures(_textEmbedModel, "TextEmbed");
        _textEmbedCompiled = CompilePart(_textEmbedModel, device);
        CheckStatus(NativeMethods.ov_compiled_model_create_infer_request(_textEmbedCompiled, out _textEmbedInfer), "Create TextEmbed infer");

        _languageModel = LoadPart("openvino_language_model.xml");
        PrintSignatures(_languageModel, "LLM");
        _languageCompiled = CompilePart(_languageModel, device, isLLM: true);

        Warmup();
    }

    private void Warmup()
    {
        Console.WriteLine("[Info] Warming up (pre-compiling GPU kernels)...");
        var sw = System.Diagnostics.Stopwatch.StartNew();
        try
        {
            // Minimal warmup like llama.cpp: push 1 token through text embedding
            // then 1 token through LLM to trigger GPU kernel compilation.
            float[] emb = GetEmbeddings(new int[] { 151643 });

            // LLM warmup: single-token prefill to compile kernels
            int hiddenSize = _hiddenSize;
            CheckStatus(NativeMethods.ov_compiled_model_create_infer_request(_languageCompiled, out var warmupInfer), "Warmup LLM infer");

            CheckStatus(NativeMethods.ov_shape_create(3, new long[] { 1, 1, hiddenSize }, out var eShape), "wE shape");
            var hE = System.Runtime.InteropServices.GCHandle.Alloc(emb, System.Runtime.InteropServices.GCHandleType.Pinned);
            CheckStatus(NativeMethods.ov_tensor_create_from_host_ptr(OvElementType.F32, eShape, hE.AddrOfPinnedObject(), out var tE), "wE tensor");

            CheckStatus(NativeMethods.ov_shape_create(3, new long[] { 3, 1, 1 }, out var pShape), "wP shape");
            var posIds = new long[] { 0, 0, 0 };
            var hP = System.Runtime.InteropServices.GCHandle.Alloc(posIds, System.Runtime.InteropServices.GCHandleType.Pinned);
            CheckStatus(NativeMethods.ov_tensor_create_from_host_ptr(OvElementType.I64, pShape, hP.AddrOfPinnedObject(), out var tP), "wP tensor");

            CheckStatus(NativeMethods.ov_shape_create(2, new long[] { 1, 1 }, out var mShape), "wM shape");
            var maskArr = new long[] { 1 };
            var hM = System.Runtime.InteropServices.GCHandle.Alloc(maskArr, System.Runtime.InteropServices.GCHandleType.Pinned);
            CheckStatus(NativeMethods.ov_tensor_create_from_host_ptr(OvElementType.I64, mShape, hM.AddrOfPinnedObject(), out var tM), "wM tensor");

            CheckStatus(NativeMethods.ov_shape_create(2, new long[] { 1, 1 }, out var vShape), "wV shape");
            var vmArr = new bool[] { false };
            var hV = System.Runtime.InteropServices.GCHandle.Alloc(vmArr, System.Runtime.InteropServices.GCHandleType.Pinned);
            CheckStatus(NativeMethods.ov_tensor_create_from_host_ptr(OvElementType.BOOL, vShape, hV.AddrOfPinnedObject(), out var tV), "wV tensor");

            CheckStatus(NativeMethods.ov_shape_create(3, new long[] { 3, 0, hiddenSize }, out var dShape), "wD shape");
            CheckStatus(NativeMethods.ov_tensor_create(OvElementType.F32, dShape, out var tD), "wD tensor");

            CheckStatus(NativeMethods.ov_shape_create(1, new long[] { 1 }, out var bShape), "wB shape");
            var beamArr = new int[] { 0 };
            var hB = System.Runtime.InteropServices.GCHandle.Alloc(beamArr, System.Runtime.InteropServices.GCHandleType.Pinned);
            CheckStatus(NativeMethods.ov_tensor_create_from_host_ptr(OvElementType.I32, bShape, hB.AddrOfPinnedObject(), out var tB), "wB tensor");

            CheckStatus(NativeMethods.ov_infer_request_set_tensor(warmupInfer, "inputs_embeds", tE), "wSet E");
            CheckStatus(NativeMethods.ov_infer_request_set_tensor(warmupInfer, "position_ids", tP), "wSet P");
            CheckStatus(HandleAttnMask(warmupInfer, tM), "wSet M");
            CheckStatus(NativeMethods.ov_infer_request_set_tensor(warmupInfer, "deepstack_visual_embeds", tD), "wSet D");
            CheckStatus(NativeMethods.ov_infer_request_set_tensor(warmupInfer, "visual_pos_masks", tV), "wSet V");
            CheckStatus(NativeMethods.ov_infer_request_set_tensor(warmupInfer, "beam_idx", tB), "wSet B");

            CheckStatus(NativeMethods.ov_infer_request_infer(warmupInfer), "Warmup LLM infer run");

            // Cleanup warmup resources
            hE.Free(); hP.Free(); hM.Free(); hV.Free(); hB.Free();
            NativeMethods.ov_tensor_free(tE); NativeMethods.ov_tensor_free(tP);
            NativeMethods.ov_tensor_free(tM); NativeMethods.ov_tensor_free(tV);
            NativeMethods.ov_tensor_free(tD); NativeMethods.ov_tensor_free(tB);
            NativeMethods.ov_infer_request_free(warmupInfer);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Warning] Warmup issue (non-fatal): {ex.Message}");
        }
        sw.Stop();
        Console.WriteLine($"[Info] Warmup complete in {sw.Elapsed.TotalMilliseconds:F0}ms.");
    }

    private nint LoadPart(string xmlName)
    {
        string path = Path.Combine(_modelDir, xmlName);
        var status = NativeMethods.ov_core_read_model(_core, path, null, out nint model);
        if (status != OvStatus.OK) throw new Exception($"Load {xmlName} failed: {status}");
        return model;
    }

    private nint CompilePart(nint model, string device, bool isLLM = false)
    {
        nint compiled;
        OvStatus status;
        if (device == "GPU")
        {
            if (!Directory.Exists("./cache")) Directory.CreateDirectory("./cache");

            if (isLLM)
            {
                // Reverting to direct GPU for max single-stream performance (22+ t/s).
                // Advanced 2025 hints like PERFORMANCE execution mode are retained.
                status = NativeMethods.ov_core_compile_model_props10(_core, model, device, 16, out compiled,
                    "PERFORMANCE_HINT", "LATENCY",
                    "INFERENCE_PRECISION_HINT", "f16",
                    "CACHE_DIR", "./cache",
                    "NUM_STREAMS", "1",
                    "ENABLE_MMAP", "YES",
                    "MODEL_PRIORITY", "HIGH",
                    "EXECUTION_MODE_HINT", "PERFORMANCE",
                    "PERFORMANCE_HINT_NUM_REQUESTS", "1",
                    "GPU_HOST_TASK_PRIORITY", "HIGH",
                    "KV_CACHE_PRECISION", "u8");
            }
            else
            {
                // Stable properties for sub-models (Vision/Merger/Embed)
                status = NativeMethods.ov_core_compile_model_props5(_core, model, device, 10, out compiled,
                    "PERFORMANCE_HINT", "LATENCY",
                    "INFERENCE_PRECISION_HINT", "f16",
                    "CACHE_DIR", "./cache",
                    "NUM_STREAMS", "1",
                    "ENABLE_MMAP", "YES");
            }
        }
        else
        {
            if (isLLM)
            {
                status = NativeMethods.ov_core_compile_model_props5(_core, model, device, 10, out compiled,
                    "PERFORMANCE_HINT", "LATENCY",
                    "INFERENCE_PRECISION_HINT", "f16",
                    "CACHE_DIR", "./cache",
                    "NUM_STREAMS", "1",
                    "KV_CACHE_PRECISION", "u8");
            }
            else
            {
                status = NativeMethods.ov_core_compile_model(_core, model, device, 0, out compiled);
            }
        }

        if (status != OvStatus.OK) CheckStatus(status, $"Compile model (isLLM={isLLM})");

        // Query optimal properties for hardware awareness
        try
        {
            nint propPtr;
            // Native property keys for 2025
            if (NativeMethods.ov_compiled_model_get_property(compiled, "OPTIMAL_NUMBER_OF_INFER_REQUESTS", out propPtr) == OvStatus.OK)
            {
                string? val = Marshal.PtrToStringUTF8(propPtr);
                Console.WriteLine($"[Info] {(isLLM ? "LLM" : "Sub")} Optimal Requests: {val ?? "N/A"}");
                NativeMethods.ov_free(propPtr);
            }
            if (NativeMethods.ov_compiled_model_get_property(compiled, "OPTIMAL_BATCH_SIZE", out propPtr) == OvStatus.OK)
            {
                string? val = Marshal.PtrToStringUTF8(propPtr);
                Console.WriteLine($"[Info] {(isLLM ? "LLM" : "Sub")} Optimal Batch Size: {val ?? "N/A"}");
                NativeMethods.ov_free(propPtr);
            }
        }
        catch { }

        return compiled;
    }

    public ChatSession StartSession()
    {
        return new ChatSession(this);
    }

    public class ChatSession : IDisposable
    {
        private readonly Qwen3VLPipeline _pipeline;
        private nint _llmInfer;
        private int _totalPhysical = 0;
        private int _totalTemporal = 0;
        private List<float[]> _accumulatedDeepstack = new List<float[]>();
        private List<bool> _sessionVisualMasks = new List<bool>();

        private string? _systemPrompt;
        private List<TurnData> _history = new List<TurnData>();
        public string? Title { get; set; }
        private HashSet<int> _uniqueIds = new HashSet<int>();

        public class TurnData
        {
            public string Prompt { get; set; } = "";
            public string? ImagePath { get; set; }
            public string? VideoPath { get; set; }
            public int MaxVideoFrames { get; set; } = 8;
            public List<int> GeneratedIds { get; set; } = new List<int>();
        }

        public ChatSession(Qwen3VLPipeline pipeline)
        {
            _pipeline = pipeline;
            CheckStatus(NativeMethods.ov_compiled_model_create_infer_request(_pipeline._languageCompiled, out _llmInfer), "Create LLM infer");
        }

        public List<int> Chat(string prompt, string? imagePath = null, string? videoPath = null, int maxVideoFrames = 8, GenerationConfig? config = null, bool silent = false, CancellationToken ct = default, Action<string>? onToken = null, Action<GenerationStats>? onStats = null)
        {
            var overallSw = System.Diagnostics.Stopwatch.StartNew();
            double visualMs = 0;
            double prefillMs = 0;
            config ??= new GenerationConfig();
            int maxTokens = config.MaxTokens;
            try
            {
                int hiddenSize = _pipeline._hiddenSize;
                nint visualEmbedsTensor = 0;
                nint deepstackFeaturesTensor = 0;
                int numVisualTokens = 0;
                int gridT = 0, gridH = 0, gridW = 0;
                int vIdBase = -1;
                if (videoPath != null && File.Exists(videoPath))
                {
                    var vsw = System.Diagnostics.Stopwatch.StartNew();
                    (visualEmbedsTensor, deepstackFeaturesTensor, gridT, gridH, gridW) = _pipeline.ProcessVideo(videoPath, maxVideoFrames, silent);
                    vsw.Stop();
                    visualMs = vsw.Elapsed.TotalMilliseconds;
                    if (!silent) Console.WriteLine($"[Info] ProcessVideo: {videoPath}, frames={gridT}");
                }
                else if (imagePath != null && File.Exists(imagePath))
                {
                    var vsw = System.Diagnostics.Stopwatch.StartNew();
                    (visualEmbedsTensor, deepstackFeaturesTensor, gridT, gridH, gridW) = _pipeline.ProcessImage(imagePath, silent);
                    vsw.Stop();
                    visualMs = vsw.Elapsed.TotalMilliseconds;
                }

                if (visualEmbedsTensor != 0)
                {
                    CheckStatus(NativeMethods.ov_tensor_get_shape(visualEmbedsTensor, out var vShape), "Get vShape");
                    long[] dims = _pipeline.GetShapeDims(vShape);
                    NativeMethods.ov_shape_free(ref vShape);
                    numVisualTokens = (int)(dims.Length == 3 ? dims[1] : dims[0]);
                    if (!silent) Console.WriteLine($"[Debug] ProcessVision: numVisualTokens={numVisualTokens}, gridT={gridT}, gridH={gridH}, gridW={gridW}");
                }

                // 1. Construct Turn Tokens
                List<int> turnIds = new List<int>();
                turnIds.AddRange(_pipeline._tokenizer.Encode("<|im_start|>user\n", addSpecialTokens: false).First().Ids.Select(x => (int)x));

                int visualStartIdx = -1;
                if (numVisualTokens > 0)
                {
                    visualStartIdx = turnIds.Count;
                    turnIds.AddRange(_pipeline._tokenizer.Encode("<|vision_start|>", addSpecialTokens: false).First().Ids.Select(x => (int)x));
                    int padToken = videoPath != null ? 151656 : 151655; // <|video_pad|> or <|image_pad|>
                    for (int i = 0; i < numVisualTokens; i++) turnIds.Add(padToken);
                    turnIds.AddRange(_pipeline._tokenizer.Encode("<|vision_end|>", addSpecialTokens: false).First().Ids.Select(x => (int)x));
                }

                turnIds.AddRange(_pipeline._tokenizer.Encode($"{prompt}<|im_end|>\n<|im_start|>assistant\n", addSpecialTokens: false).First().Ids.Select(x => (int)x));

                vIdBase = (numVisualTokens > 0 && visualStartIdx != -1) ? (visualStartIdx + _pipeline._tokenizer.Encode("<|vision_start|>", addSpecialTokens: false).First().Ids.Count) : -1;

                int totalTurnTokens = turnIds.Count;
                foreach (var id in turnIds) _uniqueIds.Add(id); float[] inputsEmbedsArr = _pipeline.GetEmbeddings(turnIds.ToArray());
                long[] positionIdsArr = new long[3 * totalTurnTokens];
                bool[] currentTurnVisualMasks = new bool[totalTurnTokens];

                if (visualEmbedsTensor != 0 && visualStartIdx != -1)
                {
                    CheckStatus(NativeMethods.ov_tensor_data(visualEmbedsTensor, out nint vPtr), "vPtr");
                    int copyLen = numVisualTokens * hiddenSize;
                    if (vIdBase >= 0 && (vIdBase + numVisualTokens) * hiddenSize <= inputsEmbedsArr.Length)
                    {
                        Marshal.Copy(vPtr, inputsEmbedsArr, vIdBase * hiddenSize, copyLen);
                    }

                    int llmGridH = gridH;
                    int llmGridW = gridW;
                    for (int i = 0; i < numVisualTokens; i++)
                    {
                        int idx = vIdBase + i;
                        if (idx >= 0 && idx < totalTurnTokens)
                        {
                            currentTurnVisualMasks[idx] = true;
                            positionIdsArr[0 * totalTurnTokens + idx] = _totalTemporal + (i / (llmGridH * llmGridW));
                            positionIdsArr[1 * totalTurnTokens + idx] = _totalTemporal + ((i / llmGridW) % llmGridH);
                            positionIdsArr[2 * totalTurnTokens + idx] = _totalTemporal + (i % llmGridW);
                        }
                    }
                }

                // Synchronize global visual masks - MOVED TO END of prefill to avoid duplication

                int currentT = _totalTemporal;
                bool inVision = false;
                int vTokensHandled = 0;

                for (int i = 0; i < totalTurnTokens; i++)
                {
                    if (currentTurnVisualMasks[i]) { /* Spatial already set */ }
                    else
                    {
                        positionIdsArr[0 * totalTurnTokens + i] = currentT;
                        positionIdsArr[1 * totalTurnTokens + i] = currentT;
                        positionIdsArr[2 * totalTurnTokens + i] = currentT;
                    }

                    if (i == vIdBase) inVision = true;
                    if (inVision)
                    {
                        vTokensHandled++;
                        if (vTokensHandled >= numVisualTokens)
                        {
                            inVision = false;
                            currentT += (gridT > 0 ? gridT : 1);
                        }
                    }
                    else currentT++;
                }

                if (!silent) Console.WriteLine($"[Debug] Prefill: totalPhysical={_totalPhysical + totalTurnTokens}, totalTemporal={currentT}, turnTokens={totalTurnTokens}");

                // 2. Chunked Pre-fill to bypass GPU memory limits (Intel 4GB buffer limit)
                const int chunkSize = 2048;
                float[] deepstackAllArr = null;
                if (deepstackFeaturesTensor != 0)
                {
                    NativeMethods.ov_tensor_get_size(deepstackFeaturesTensor, out nuint dSz);
                    deepstackAllArr = new float[(int)dSz];
                    NativeMethods.ov_tensor_data(deepstackFeaturesTensor, out nint dP);
                    Marshal.Copy(dP, deepstackAllArr, 0, deepstackAllArr.Length);
                    _accumulatedDeepstack.Add(deepstackAllArr);
                }


                for (int start = 0; start < totalTurnTokens; start += chunkSize)
                {
                    if (ct.IsCancellationRequested) throw new OperationCanceledException(ct);
                    int count = Math.Min(chunkSize, totalTurnTokens - start);

                    // A. Input Embeddings Slice
                    float[] chunkEmbeds = new float[count * hiddenSize];
                    Array.Copy(inputsEmbedsArr, start * hiddenSize, chunkEmbeds, 0, count * hiddenSize);
                    CheckStatus(NativeMethods.ov_shape_create((nuint)3, new long[] { 1, count, (long)hiddenSize }, out var ovEmbedShape), "Chunk Embed shape");
                    GCHandle hEmbed = GCHandle.Alloc(chunkEmbeds, GCHandleType.Pinned);
                    CheckStatus(NativeMethods.ov_tensor_create_from_host_ptr(OvElementType.F32, ovEmbedShape, hEmbed.AddrOfPinnedObject(), out var tEmbed), "Chunk Embed tensor");

                    // B. Position IDs Slice
                    long[] chunkPos = new long[3 * count];
                    for (int d = 0; d < 3; d++) Array.Copy(positionIdsArr, d * totalTurnTokens + start, chunkPos, d * count, count);
                    CheckStatus(NativeMethods.ov_shape_create((nuint)3, new long[] { 3, 1, count }, out var ovPosShape), "Chunk Pos shape");
                    GCHandle hPos = GCHandle.Alloc(chunkPos, GCHandleType.Pinned);
                    CheckStatus(NativeMethods.ov_tensor_create_from_host_ptr(OvElementType.I64, ovPosShape, hPos.AddrOfPinnedObject(), out var tPos), "Chunk Pos tensor");

                    // C. Attention Mask (Incremental)
                    int fullMaskLen = _totalPhysical + count;
                    CheckStatus(NativeMethods.ov_shape_create((nuint)2, new long[] { 1, fullMaskLen }, out var ovMaskShape), "Chunk Mask shape");
                    long[] fullMask = Enumerable.Repeat(1L, fullMaskLen).ToArray();
                    GCHandle hMask = GCHandle.Alloc(fullMask, GCHandleType.Pinned);
                    CheckStatus(NativeMethods.ov_tensor_create_from_host_ptr(OvElementType.I64, ovMaskShape, hMask.AddrOfPinnedObject(), out var tMask), "Chunk Mask tensor");

                    // D. Visual Masks Slice
                    bool[] chunkVMask = new bool[count];
                    Array.Copy(currentTurnVisualMasks, start, chunkVMask, 0, count);
                    CheckStatus(NativeMethods.ov_shape_create((nuint)2, new long[] { 1, count }, out var ovVMaskShape), "Chunk vMask shape");
                    GCHandle hVMask = GCHandle.Alloc(chunkVMask, GCHandleType.Pinned);
                    CheckStatus(NativeMethods.ov_tensor_create_from_host_ptr(OvElementType.BOOL, ovVMaskShape, hVMask.AddrOfPinnedObject(), out var tVMask), "Chunk vMask tensor");

                    // E. Deepstack Slice calculation
                    nint tDeepChunk = 0;
                    GCHandle hDeepChunk = default;
                    OvShape sDeepChunk = default;

                    int chunkVStart = Math.Max(start, vIdBase);
                    int chunkVEnd = Math.Min(start + count, vIdBase + (vIdBase != -1 ? numVisualTokens : 0));
                    int chunkVCount = (vIdBase != -1 && chunkVEnd > chunkVStart) ? (chunkVEnd - chunkVStart) : 0;

                    if (chunkVCount > 0 && deepstackAllArr != null)
                    {
                        int vOffsetInTurn = chunkVStart - vIdBase;
                        float[] chunkD = new float[3 * chunkVCount * hiddenSize];
                        for (int l = 0; l < 3; l++)
                        {
                            Array.Copy(deepstackAllArr, (l * numVisualTokens + vOffsetInTurn) * hiddenSize, chunkD, (l * chunkVCount) * hiddenSize, chunkVCount * hiddenSize);
                        }
                        hDeepChunk = GCHandle.Alloc(chunkD, GCHandleType.Pinned);
                        CheckStatus(NativeMethods.ov_shape_create(3, new long[] { 3, chunkVCount, hiddenSize }, out sDeepChunk), "Chunk D shape");
                        CheckStatus(NativeMethods.ov_tensor_create_from_host_ptr(OvElementType.F32, sDeepChunk, hDeepChunk.AddrOfPinnedObject(), out tDeepChunk), "Chunk D tensor");
                    }
                    else
                    {
                        CheckStatus(NativeMethods.ov_shape_create(3, new long[] { 3, 0, hiddenSize }, out sDeepChunk), "D empty shape");
                        CheckStatus(NativeMethods.ov_tensor_create(OvElementType.F32, sDeepChunk, out tDeepChunk), "D empty tensor");
                    }

                    CheckStatus(NativeMethods.ov_shape_create((nuint)1, new long[] { 1 }, out var ovBeamShape), "Beam shape");
                    GCHandle hBeam = GCHandle.Alloc(new int[] { 0 }, GCHandleType.Pinned);
                    CheckStatus(NativeMethods.ov_tensor_create_from_host_ptr(OvElementType.I32, ovBeamShape, hBeam.AddrOfPinnedObject(), out var tBeam), "Beam tensor");

                    CheckStatus(NativeMethods.ov_infer_request_set_tensor(_llmInfer, "inputs_embeds", tEmbed), "Set E");
                    CheckStatus(NativeMethods.ov_infer_request_set_tensor(_llmInfer, "position_ids", tPos), "Set P");
                    CheckStatus(_pipeline.HandleAttnMask(_llmInfer, tMask), "Set M");
                    CheckStatus(NativeMethods.ov_infer_request_set_tensor(_llmInfer, "deepstack_visual_embeds", tDeepChunk), "Set D");
                    CheckStatus(NativeMethods.ov_infer_request_set_tensor(_llmInfer, "visual_pos_masks", tVMask), "Set V");
                    CheckStatus(NativeMethods.ov_infer_request_set_tensor(_llmInfer, "beam_idx", tBeam), "Set B");

                    var chunkSw = System.Diagnostics.Stopwatch.StartNew();
                    CheckStatus(NativeMethods.ov_infer_request_infer(_llmInfer), "Chunk prefill infer");
                    chunkSw.Stop();
                    prefillMs += chunkSw.Elapsed.TotalMilliseconds;

                    _totalPhysical += count;

                    // Cleanup chunk tensors
                    NativeMethods.ov_tensor_free(tEmbed); NativeMethods.ov_shape_free(ref ovEmbedShape); hEmbed.Free();
                    NativeMethods.ov_tensor_free(tPos); NativeMethods.ov_shape_free(ref ovPosShape); hPos.Free();
                    NativeMethods.ov_tensor_free(tMask); NativeMethods.ov_shape_free(ref ovMaskShape); hMask.Free();
                    NativeMethods.ov_tensor_free(tDeepChunk); NativeMethods.ov_shape_free(ref sDeepChunk); if (hDeepChunk.IsAllocated) hDeepChunk.Free();
                    NativeMethods.ov_tensor_free(tVMask); NativeMethods.ov_shape_free(ref ovVMaskShape); hVMask.Free();
                    NativeMethods.ov_tensor_free(tBeam); NativeMethods.ov_shape_free(ref ovBeamShape); hBeam.Free();
                }

                _sessionVisualMasks.AddRange(currentTurnVisualMasks);
                _totalTemporal = currentT;

                var generatedIds = DecodeResponse(config, silent, prefillMs, visualMs, overallSw, ct, onToken, onStats);

                if (visualEmbedsTensor != 0) NativeMethods.ov_tensor_free(visualEmbedsTensor);
                if (deepstackFeaturesTensor != 0) NativeMethods.ov_tensor_free(deepstackFeaturesTensor);

                // Add to history
                _history.Add(new TurnData
                {
                    Prompt = prompt,
                    ImagePath = imagePath,
                    VideoPath = videoPath,
                    MaxVideoFrames = maxVideoFrames,
                    GeneratedIds = generatedIds
                });
                return generatedIds;
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Chat Error] {ex.Message}\n{ex.StackTrace}");
                return new List<int>();
            }
        }

        private List<int> DecodeResponse(GenerationConfig config, bool silent, double prefillMs, double visualMs, Stopwatch overallSw, CancellationToken ct, Action<string>? onToken = null, Action<GenerationStats>? onStats = null)
        {
            double ttft = 0;
            int hiddenSize = _pipeline._hiddenSize;
            CheckStatus(NativeMethods.ov_infer_request_get_tensor(_llmInfer, "logits", out nint initialLogits), "Get logits");
            List<int> generatedIds = new List<int>();
            int currentTokenId = _pipeline.Sample(initialLogits, _uniqueIds, config);
            overallSw.Stop();
            ttft = overallSw.Elapsed.TotalMilliseconds;

            var sw = System.Diagnostics.Stopwatch.StartNew();
            if (!silent) Console.Write("Assistant: ");
            int generatedCount = 0;
            int printedLength = 0;

            // Reuse buffers for the loop to avoid allocations
            long[] posArr = new long[3];
            long[] maskArr = new long[_totalPhysical + config.MaxTokens + 1];
            uint[] tokenBuffer = new uint[config.MaxTokens + 1];
            int[] beamArr = new int[] { 0 };
            bool[] vMaskArr = new bool[maskArr.Length];
            for (int j = 0; j < _sessionVisualMasks.Count; j++) vMaskArr[j] = _sessionVisualMasks[j];

            // Pre-fill mask with 1s for previous physical tokens
            for (int j = 0; j < _totalPhysical; j++) maskArr[j] = 1L;

            // Initialize vMask with existing session masks
            List<bool> currentVMasks = new List<bool>(_sessionVisualMasks);

            // Fixed shapes for single-token steps
            CheckStatus(NativeMethods.ov_shape_create(3, new long[] { 1, 1, (long)hiddenSize }, out var sEmbed), "sEmbed");
            CheckStatus(NativeMethods.ov_shape_create(3, new long[] { 3, 1, 1 }, out var sPos), "sPos");
            // vMask shape will be dynamic based on current history
            CheckStatus(NativeMethods.ov_shape_create(1, new long[] { 1 }, out var sBeam), "sBeam");
            CheckStatus(NativeMethods.ov_shape_create(2, new long[] { 1, 1 }, out var sIds), "sIds");

            // Optimized visual tensors for decoding (text only steps)
            CheckStatus(NativeMethods.ov_shape_create(3, new long[] { 3, 0, hiddenSize }, out var sDeepLoop), "D empty shape");
            CheckStatus(NativeMethods.ov_tensor_create(OvElementType.F32, sDeepLoop, out nint tDeep), "D empty tensor");

            // Detect if model supports input_ids (preferred for GPU speed)
            bool useInputIds = false;
            nint tIds = 0;
            long[] idArr = new long[1];
            GCHandle hIds = GCHandle.Alloc(idArr, GCHandleType.Pinned);
            CheckStatus(NativeMethods.ov_tensor_create_from_host_ptr(OvElementType.I64, sIds, hIds.AddrOfPinnedObject(), out tIds), "tIds create");
            if (NativeMethods.ov_infer_request_set_tensor(_llmInfer, "input_ids", tIds) == OvStatus.OK)
            {
                useInputIds = true;
                if (!silent) Console.WriteLine("[Info] input_ids supported. Using Async Pipeline.");
            }
            else
            {
                if (!silent) Console.WriteLine("[Info] input_ids NOT supported (set_tensor failed). Using inputs_embeds path.");
                NativeMethods.ov_tensor_free(tIds);
                hIds.Free();
            }

            // 1. Pre-allocate and Pin all loop buffers
            GCHandle hP = GCHandle.Alloc(posArr, GCHandleType.Pinned);
            GCHandle hM = GCHandle.Alloc(maskArr, GCHandleType.Pinned);
            GCHandle hB = GCHandle.Alloc(beamArr, GCHandleType.Pinned);

            // 2. Create tensors once
            CheckStatus(NativeMethods.ov_tensor_create_from_host_ptr(OvElementType.I64, sPos, hP.AddrOfPinnedObject(), out var tP), "tP");

            // vMask tensor: Simplified to [1, 1] always false for decoding tokens
            bool[] vMaskDec = new bool[] { false };
            CheckStatus(NativeMethods.ov_shape_create(2, new long[] { 1, 1 }, out OvShape sVDec), "sVDec");
            GCHandle hV = GCHandle.Alloc(vMaskDec, GCHandleType.Pinned);
            CheckStatus(NativeMethods.ov_tensor_create_from_host_ptr(OvElementType.BOOL, sVDec, hV.AddrOfPinnedObject(), out var tV), "tV create");

            CheckStatus(NativeMethods.ov_tensor_create_from_host_ptr(OvElementType.I32, sBeam, hB.AddrOfPinnedObject(), out var tB), "tB");

            // Attention mask tensor: allocate max size once
            CheckStatus(NativeMethods.ov_shape_create(2, new long[] { 1, maskArr.Length }, out var sMMax), "sMMax");
            CheckStatus(NativeMethods.ov_tensor_create_from_host_ptr(OvElementType.I64, sMMax, hM.AddrOfPinnedObject(), out var tM), "tM");

            // Pre-bind constant tensors to request
            CheckStatus(NativeMethods.ov_infer_request_set_tensor(_llmInfer, "position_ids", tP), "Set P");
            CheckStatus(NativeMethods.ov_infer_request_set_tensor(_llmInfer, "deepstack_visual_embeds", tDeep), "Set D");
            CheckStatus(NativeMethods.ov_infer_request_set_tensor(_llmInfer, "visual_pos_masks", tV), "Set V");
            CheckStatus(NativeMethods.ov_infer_request_set_tensor(_llmInfer, "beam_idx", tB), "Set B");

            // Stats for performance tracking
            double accInferMs = 0, accSampleMs = 0, accDeserMs = 0;
            int accCount = 0;

            for (int i = 0; i < config.MaxTokens; i++)
            {
                if (ct.IsCancellationRequested)
                {
                    if (!silent) Console.WriteLine(" [Cancelled]");
                    throw new OperationCanceledException();
                }
                if (currentTokenId == 151645 || currentTokenId == 151643) { if (!silent) Console.WriteLine(); break; }
                generatedIds.Add(currentTokenId); _uniqueIds.Add(currentTokenId);
                tokenBuffer[generatedCount] = (uint)currentTokenId;
                generatedCount++;

                // 3. Update host data for this step
                int physPos = _totalPhysical;
                int tempPos = _totalTemporal;
                posArr[0] = tempPos; posArr[1] = tempPos; posArr[2] = tempPos;
                maskArr[physPos] = 1L;
                // vMaskArr[physPos] = false; // No longer needed as tV is fixed
                // _sessionVisualMasks.Add(false); // No longer needed as tV is fixed

                // Increment counts immediately to stay in sync with KV Cache
                _totalPhysical++;
                _totalTemporal++;
                _sessionVisualMasks.Add(false); // Text token mask

                // Update masks shape to reflect current sequence length
                CheckStatus(NativeMethods.ov_shape_create(2, new long[] { 1, _totalPhysical }, out var sMCur), "sMCur");
                CheckStatus(NativeMethods.ov_tensor_set_shape(tM, sMCur), "Update M shape");
                NativeMethods.ov_shape_free(ref sMCur);

                var swInfer = System.Diagnostics.Stopwatch.StartNew();
                if (useInputIds)
                {
                    idArr[0] = currentTokenId;
                    CheckStatus(NativeMethods.ov_infer_request_set_tensor(_llmInfer, "input_ids", tIds), "Set I");
                    CheckStatus(_pipeline.HandleAttnMask(_llmInfer, tM), "Set M");
                    CheckStatus(NativeMethods.ov_infer_request_set_tensor(_llmInfer, "visual_pos_masks", tV), "Set V");

                    // ASYNC PIPELINE: Start next token GPU work
                    CheckStatus(NativeMethods.ov_infer_request_start_async(_llmInfer), "Async start");
                }
                else
                {
                    float[] nextEmbed = _pipeline.GetEmbedding(currentTokenId);
                    GCHandle hE = GCHandle.Alloc(nextEmbed, GCHandleType.Pinned);
                    CheckStatus(NativeMethods.ov_shape_create(3, new long[] { 1, 1, (long)hiddenSize }, out var sE), "sE");
                    CheckStatus(NativeMethods.ov_tensor_create_from_host_ptr(OvElementType.F32, sE, hE.AddrOfPinnedObject(), out var tE), "tE");
                    CheckStatus(NativeMethods.ov_infer_request_set_tensor(_llmInfer, "inputs_embeds", tE), "Set E");
                    CheckStatus(_pipeline.HandleAttnMask(_llmInfer, tM), "Set M");
                    CheckStatus(NativeMethods.ov_infer_request_set_tensor(_llmInfer, "visual_pos_masks", tV), "Set V");

                    CheckStatus(NativeMethods.ov_infer_request_infer(_llmInfer), "Sync fallback");

                    NativeMethods.ov_tensor_free(tE); NativeMethods.ov_shape_free(ref sE); hE.Free();
                }

                // 4. While GPU works (or after sync), do Decoding and Printing
                var swDeser = System.Diagnostics.Stopwatch.StartNew();

                uint[] currentTokens = new uint[generatedCount];
                Array.Copy(tokenBuffer, currentTokens, generatedCount);
                string fullDecoded = _pipeline._tokenizer.Decode(currentTokens, skipSpecialTokens: true);

                swDeser.Stop();
                accDeserMs += swDeser.Elapsed.TotalMilliseconds;

                if (fullDecoded.Length > printedLength)
                {
                    string newText = fullDecoded.Substring(printedLength);
                    if (config.StopSequences != null && config.StopSequences.Count > 0)
                    {
                        string matchedSeq = null;
                        int matchedStopIdx = -1;
                        bool matchesStop = false;
                        foreach (var seq in config.StopSequences)
                        {
                            if (newText.Contains(seq))
                            {
                                int stopIdx = newText.IndexOf(seq);
                                var stopText = newText.Substring(0, stopIdx);
                                if (!silent) Console.Write(stopText);
                                onToken?.Invoke(stopText);
                                matchesStop = true;
                                matchedSeq = seq;
                                matchedStopIdx = stopIdx;
                                break;
                            }
                        }
                        if (matchesStop)
                        {
                            if (!silent) Console.WriteLine();
                            // Truncate generatedIds to remove the tokens forming the stop sequence
                            _pipeline.PruneStopSequence(generatedIds, matchedSeq, printedLength + matchedStopIdx);
                            break;
                        }
                    }

                    if (fullDecoded.EndsWith("\uFFFD"))
                    {
                        string safePart = fullDecoded.Substring(0, fullDecoded.Length - 1);
                        if (safePart.Length > printedLength)
                        {
                            var partialText = safePart.Substring(printedLength);
                            if (!silent) Console.Write(partialText);
                            onToken?.Invoke(partialText);
                            printedLength = safePart.Length;
                        }
                    }
                    else
                    {
                        if (!silent) Console.Write(newText);
                        onToken?.Invoke(newText);
                        printedLength = fullDecoded.Length;
                    }
                }

                if (useInputIds)
                {
                    // WAIT for GPU to finish
                    CheckStatus(NativeMethods.ov_infer_request_wait(_llmInfer), "Async wait");
                }
                swInfer.Stop();
                accInferMs += swInfer.Elapsed.TotalMilliseconds;

                // 5. Sampling for the NEXT token
                var swSample = System.Diagnostics.Stopwatch.StartNew();
                CheckStatus(NativeMethods.ov_infer_request_get_tensor(_llmInfer, "logits", out nint nextLogits), "Next logits");
                currentTokenId = _pipeline.Sample(nextLogits, _uniqueIds, config);
                swSample.Stop();
                accSampleMs += swSample.Elapsed.TotalMilliseconds;
                accCount++;
            }

            double avgInfer = accCount > 0 ? accInferMs / accCount : 0;
            double avgSample = accCount > 0 ? accSampleMs / accCount : 0;
            if (!silent)
            {
                Console.Write($"\n[Perf] TTFT: {ttft:F2}ms | Prefill: {prefillMs:F2}ms | Visual: {visualMs:F2}ms");
                Console.Write($"\n       Infer(Avg): {avgInfer:F2}ms | Sample(Avg): {avgSample:F2}ms | Deser(Avg): {(accDeserMs / accCount):F2}ms\n");
            }

            // Cleanup
            NativeMethods.ov_tensor_free(tP);
            NativeMethods.ov_tensor_free(tV);
            NativeMethods.ov_tensor_free(tB);
            NativeMethods.ov_tensor_free(tM);
            NativeMethods.ov_tensor_free(tDeep); NativeMethods.ov_shape_free(ref sDeepLoop);
            hP.Free(); hM.Free(); hV.Free(); hB.Free();
            NativeMethods.ov_shape_free(ref sEmbed); NativeMethods.ov_shape_free(ref sPos); NativeMethods.ov_shape_free(ref sBeam); NativeMethods.ov_shape_free(ref sIds);
            NativeMethods.ov_shape_free(ref sMMax); NativeMethods.ov_shape_free(ref sVDec);
            if (useInputIds) { NativeMethods.ov_tensor_free(tIds); hIds.Free(); }

            // Final Flush
            uint[] finalTokens = new uint[generatedIds.Count];
            Array.Copy(tokenBuffer, finalTokens, generatedIds.Count);
            string finalDecoded = _pipeline._tokenizer.Decode(finalTokens, skipSpecialTokens: true);
            if (finalDecoded.Length > printedLength)
            {
                var remaining = finalDecoded.Substring(printedLength);
                if (!silent) Console.Write(remaining);
                onToken?.Invoke(remaining);
            }

            // Update stats
            sw.Stop();
            onStats?.Invoke(new GenerationStats
            {
                PrefillMs = prefillMs,
                DecodeMs = sw.Elapsed.TotalMilliseconds,
                TokenCount = generatedCount
            });

            if (!silent)
            {
                double tps = generatedCount / sw.Elapsed.TotalSeconds;
                Console.WriteLine($"\n[Stats] Generated: {generatedCount} tokens | Speed: {tps:F2} t/s");
            }

            return generatedIds;
        }


        public void InitSystemPrompt(string systemPrompt)
        {
            if (_totalPhysical > 0) throw new InvalidOperationException("Cannot set system prompt after chat started.");

            _systemPrompt = systemPrompt;

            // Encode <|im_start|>system\n{prompt}<|im_end|>\n
            var ids = new List<int>();
            ids.AddRange(_pipeline._tokenizer.Encode("<|im_start|>system\n", addSpecialTokens: false).First().Ids.Select(x => (int)x));
            ids.AddRange(_pipeline._tokenizer.Encode($"{systemPrompt}<|im_end|>\n", addSpecialTokens: false).First().Ids.Select(x => (int)x));

            RunPrefill(ids.ToArray(), 0, 0);

            Console.WriteLine($"[Info] System prompt applied. ({ids.Count} tokens)");
        }

        public void Save(string path)
        {
            if (!Directory.Exists(path)) Directory.CreateDirectory(path);

            var state = new SessionState
            {
                SystemPrompt = _systemPrompt,
                History = _history,
                Title = this.Title
            };

            string json = System.Text.Json.JsonSerializer.Serialize(state, new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(Path.Combine(path, "session.json"), json);
            Console.WriteLine($"[Info] Session saved to {path} ({_history.Count} turns)");
        }

        public void LoadLegacy(string path)
        {
            if (!Directory.Exists(path)) throw new DirectoryNotFoundException(path);

            // 1. Metadata
            string json = File.ReadAllText(Path.Combine(path, "metadata.json"));
            using var doc = System.Text.Json.JsonDocument.Parse(json);
            _totalPhysical = doc.RootElement.GetProperty("Physical").GetInt32();
            _totalTemporal = doc.RootElement.GetProperty("Temporal").GetInt32();

            // 2. States
            CheckStatus(NativeMethods.ov_infer_request_query_state(_llmInfer, out nint states, out nuint size), "Query states");
            nint[] statePtrs = new nint[size];
            Marshal.Copy(states, statePtrs, 0, (int)size);

            for (int i = 0; i < (int)size; i++)
            {
                nint state = statePtrs[i];
                CheckStatus(NativeMethods.ov_variable_state_get_name(state, out nint namePtr), "Get state name");
                string name = Marshal.PtrToStringAnsi(namePtr) ?? $"state_{i}";
                NativeMethods.ov_free(namePtr);

                string binPath = Path.Combine(path, $"{name}.bin");
                if (File.Exists(binPath))
                {
                    byte[] buffer = File.ReadAllBytes(binPath);
                    CheckStatus(NativeMethods.ov_variable_state_get_state(state, out nint currentTensor), "Get current state");
                    CheckStatus(NativeMethods.ov_tensor_get_shape(currentTensor, out var shape), "Get state shape");
                    CheckStatus(NativeMethods.ov_tensor_get_element_type(currentTensor, out var type), "Get state type");
                    NativeMethods.ov_tensor_free(currentTensor);

                    GCHandle h = GCHandle.Alloc(buffer, GCHandleType.Pinned);
                    CheckStatus(NativeMethods.ov_tensor_create_from_host_ptr(type, shape, h.AddrOfPinnedObject(), out nint newTensor), "Create persist tensor");

                    NativeMethods.ov_tensor_free(newTensor); // Free host-based one
                    h.Free();

                    // Correct way:
                    CheckStatus(NativeMethods.ov_tensor_create(type, shape, out newTensor), "Create state tensor");
                    CheckStatus(NativeMethods.ov_tensor_data(newTensor, out nint destPtr), "State data");
                    Marshal.Copy(buffer, 0, destPtr, buffer.Length);

                    CheckStatus(NativeMethods.ov_variable_state_set_state(state, newTensor), "Set state");
                    NativeMethods.ov_tensor_free(newTensor);
                    NativeMethods.ov_shape_free(ref shape);
                }
                NativeMethods.ov_variable_state_free(state);
            }
        }

        public void Load(string path)
        {
            if (!Directory.Exists(path)) throw new DirectoryNotFoundException(path);
            string jsonPath = Path.Combine(path, "session.json");
            if (!File.Exists(jsonPath)) throw new FileNotFoundException("session.json not found", jsonPath);

            string json = File.ReadAllText(jsonPath);
            var state = System.Text.Json.JsonSerializer.Deserialize<SessionState>(json);
            if (state == null) throw new Exception("Failed to deserialize session state.");

            Load(state.SystemPrompt, state.Title, state.History);
        }

        public void Load(string? systemPrompt, string? title, List<TurnData> history)
        {
            if (_totalPhysical > 0) throw new InvalidOperationException("Cannot load into a non-empty session.");

            // 1. Restore System Prompt
            if (!string.IsNullOrEmpty(systemPrompt))
            {
                InitSystemPrompt(systemPrompt);
            }
            this.Title = title;

            // 2. Replay History
            Console.WriteLine($"[Info] Replaying {history.Count} turns...");
            foreach (var turn in history)
            {
                Console.WriteLine($"  - Replaying: {turn.Prompt.Substring(0, Math.Min(20, turn.Prompt.Length))}...");
                ReplayTurn(turn, CancellationToken.None);
            }

            // 3. Restore history list
            _history = history;

            Console.WriteLine("[Info] Replay complete.");
        }

        private class SessionState
        {
            public string? Title { get; set; }
            public string? SystemPrompt { get; set; }
            public List<TurnData> History { get; set; } = new List<TurnData>();
        }

        private void ReplayTurn(TurnData turn, CancellationToken ct)
        {
            // Re-construct inputs (Logic similar to Chat but combined)
            int hiddenSize = _pipeline._hiddenSize;
            nint visualEmbedsTensor = 0;
            nint deepstackFeaturesTensor = 0;
            int numVisualTokens = 0;
            int gridT = 0, gridH = 0, gridW = 0;
            int vIdBase = -1;

            if (turn.VideoPath != null && File.Exists(turn.VideoPath))
            {
                (visualEmbedsTensor, deepstackFeaturesTensor, gridT, gridH, gridW) = _pipeline.ProcessVideo(turn.VideoPath, turn.MaxVideoFrames, silent: true);
                CheckStatus(NativeMethods.ov_tensor_get_shape(visualEmbedsTensor, out var vShape), "Get vShape");
                long[] dims = _pipeline.GetShapeDims(vShape);
                NativeMethods.ov_shape_free(ref vShape);
                numVisualTokens = (int)(dims.Length == 3 ? dims[1] : dims[0]);
            }
            else if (turn.ImagePath != null && File.Exists(turn.ImagePath))
            {
                (visualEmbedsTensor, deepstackFeaturesTensor, gridT, gridH, gridW) = _pipeline.ProcessImage(turn.ImagePath, silent: true);
                CheckStatus(NativeMethods.ov_tensor_get_shape(visualEmbedsTensor, out var vShape), "Get vShape");
                long[] dims = _pipeline.GetShapeDims(vShape);
                NativeMethods.ov_shape_free(ref vShape);
                numVisualTokens = (int)(dims.Length == 3 ? dims[1] : dims[0]);
            }

            List<int> turnIds = new List<int>();
            turnIds.AddRange(_pipeline._tokenizer.Encode("<|im_start|>user\n", addSpecialTokens: false).First().Ids.Select(x => (int)x));

            int visualStartIdx = -1;
            if (numVisualTokens > 0)
            {
                visualStartIdx = turnIds.Count;
                turnIds.AddRange(_pipeline._tokenizer.Encode("<|vision_start|>", addSpecialTokens: false).First().Ids.Select(x => (int)x));
                int padToken = turn.VideoPath != null ? 151656 : 151655;
                for (int i = 0; i < numVisualTokens; i++) turnIds.Add(padToken);
                turnIds.AddRange(_pipeline._tokenizer.Encode("<|vision_end|>", addSpecialTokens: false).First().Ids.Select(x => (int)x));
            }

            turnIds.AddRange(_pipeline._tokenizer.Encode($"{turn.Prompt}<|im_end|>\n<|im_start|>assistant\n", addSpecialTokens: false).First().Ids.Select(x => (int)x));

            // Append generated tokens (treating them as forced output)
            turnIds.AddRange(turn.GeneratedIds);

            int totalTurnTokens = turnIds.Count;
            foreach (var id in turnIds) _uniqueIds.Add(id); float[] inputsEmbedsArr = _pipeline.GetEmbeddings(turnIds.ToArray());
            long[] positionIdsArr = new long[3 * totalTurnTokens];
            bool[] visualPosMasksArr = new bool[totalTurnTokens];

            // Image Injection
            vIdBase = (numVisualTokens > 0) ? (visualStartIdx + _pipeline._tokenizer.Encode("<|vision_start|>", addSpecialTokens: false).First().Ids.Count) : -1;
            if (visualEmbedsTensor != 0 && vIdBase != -1)
            {
                CheckStatus(NativeMethods.ov_tensor_data(visualEmbedsTensor, out nint vPtr), "vPtr");
                int copyLen = numVisualTokens * hiddenSize;
                if (vIdBase >= 0 && (vIdBase + numVisualTokens) * hiddenSize <= inputsEmbedsArr.Length)
                {
                    Marshal.Copy(vPtr, inputsEmbedsArr, vIdBase * hiddenSize, copyLen);
                }

                int llmGridH = gridH;
                int llmGridW = gridW;
                for (int i = 0; i < numVisualTokens; i++)
                {
                    int idx = vIdBase + i;
                    if (idx >= 0 && idx < totalTurnTokens)
                    {
                        visualPosMasksArr[idx] = true;
                        positionIdsArr[0 * totalTurnTokens + idx] = _totalTemporal + (i / (llmGridH * llmGridW));
                        positionIdsArr[1 * totalTurnTokens + idx] = _totalTemporal + ((i / llmGridW) % llmGridH);
                        positionIdsArr[2 * totalTurnTokens + idx] = _totalTemporal + (i % llmGridW);
                    }
                }
            }

            // Position IDs
            int currentT = _totalTemporal;
            bool inVision = false;
            int vTokensHandled = 0;

            for (int i = 0; i < totalTurnTokens; i++)
            {
                if (!visualPosMasksArr[i])
                {
                    positionIdsArr[0 * totalTurnTokens + i] = currentT;
                    positionIdsArr[1 * totalTurnTokens + i] = currentT;
                    positionIdsArr[2 * totalTurnTokens + i] = currentT;
                }

                if (i == vIdBase) inVision = true;
                if (inVision)
                {
                    vTokensHandled++;
                    if (vTokensHandled >= numVisualTokens)
                    {
                        inVision = false;
                        currentT += (gridT > 0 ? gridT : 1);
                    }
                }
                else currentT++;
            }

            // 2. Chunked Pre-fill to bypass GPU memory limits (Intel 4GB buffer limit)
            const int chunkSize = 2048;
            float[] deepstackAllArr = null;
            if (deepstackFeaturesTensor != 0)
            {
                NativeMethods.ov_tensor_get_size(deepstackFeaturesTensor, out nuint dSz);
                deepstackAllArr = new float[(int)dSz];
                NativeMethods.ov_tensor_data(deepstackFeaturesTensor, out nint dP);
                Marshal.Copy(dP, deepstackAllArr, 0, deepstackAllArr.Length);
                _accumulatedDeepstack.Add(deepstackAllArr);
            }

            for (int start = 0; start < totalTurnTokens; start += chunkSize)
            {
                int count = Math.Min(chunkSize, totalTurnTokens - start);

                // A. Input Embeddings Slice
                float[] chunkEmbeds = new float[count * hiddenSize];
                Array.Copy(inputsEmbedsArr, start * hiddenSize, chunkEmbeds, 0, count * hiddenSize);
                CheckStatus(NativeMethods.ov_shape_create((nuint)3, new long[] { 1, count, (long)hiddenSize }, out var ovEmbedShape), "Chunk Embed shape");
                GCHandle hEmbed = GCHandle.Alloc(chunkEmbeds, GCHandleType.Pinned);
                CheckStatus(NativeMethods.ov_tensor_create_from_host_ptr(OvElementType.F32, ovEmbedShape, hEmbed.AddrOfPinnedObject(), out var tEmbed), "Chunk Embed tensor");

                // B. Position IDs Slice
                long[] chunkPos = new long[3 * count];
                for (int d = 0; d < 3; d++) Array.Copy(positionIdsArr, d * totalTurnTokens + start, chunkPos, d * count, count);
                CheckStatus(NativeMethods.ov_shape_create((nuint)3, new long[] { 3, 1, count }, out var ovPosShape), "Chunk Pos shape");
                GCHandle hPos = GCHandle.Alloc(chunkPos, GCHandleType.Pinned);
                CheckStatus(NativeMethods.ov_tensor_create_from_host_ptr(OvElementType.I64, ovPosShape, hPos.AddrOfPinnedObject(), out var tPos), "Chunk Pos tensor");

                // C. Attention Mask (Incremental)
                int fullMaskLen = _totalPhysical + count;
                CheckStatus(NativeMethods.ov_shape_create((nuint)2, new long[] { 1, fullMaskLen }, out var ovMaskShape), "Chunk Mask shape");
                long[] fullMask = Enumerable.Repeat(1L, fullMaskLen).ToArray();
                GCHandle hMask = GCHandle.Alloc(fullMask, GCHandleType.Pinned);
                CheckStatus(NativeMethods.ov_tensor_create_from_host_ptr(OvElementType.I64, ovMaskShape, hMask.AddrOfPinnedObject(), out var tMask), "Chunk Mask tensor");

                // D. Visual Masks Slice
                bool[] chunkVMask = new bool[count];
                Array.Copy(visualPosMasksArr, start, chunkVMask, 0, count);
                CheckStatus(NativeMethods.ov_shape_create((nuint)2, new long[] { 1, count }, out var ovVMaskShape), "Chunk vMask shape");
                GCHandle hVMask = GCHandle.Alloc(chunkVMask, GCHandleType.Pinned);
                CheckStatus(NativeMethods.ov_tensor_create_from_host_ptr(OvElementType.BOOL, ovVMaskShape, hVMask.AddrOfPinnedObject(), out var tVMask), "Chunk vMask tensor");

                // E. Deepstack Slice calculation
                nint tDeepChunk = 0;
                GCHandle hDeepChunk = default;
                OvShape sDeepChunk = default;

                int chunkVStart = Math.Max(start, vIdBase);
                int chunkVEnd = Math.Min(start + count, vIdBase + (vIdBase != -1 ? numVisualTokens : 0));
                int chunkVCount = (vIdBase != -1 && chunkVEnd > chunkVStart) ? (chunkVEnd - chunkVStart) : 0;

                if (chunkVCount > 0 && deepstackAllArr != null)
                {
                    int vOffsetInTurn = chunkVStart - vIdBase;
                    float[] chunkD = new float[3 * chunkVCount * hiddenSize];
                    for (int l = 0; l < 3; l++)
                    {
                        Array.Copy(deepstackAllArr, (l * numVisualTokens + vOffsetInTurn) * hiddenSize, chunkD, (l * chunkVCount) * hiddenSize, chunkVCount * hiddenSize);
                    }
                    hDeepChunk = GCHandle.Alloc(chunkD, GCHandleType.Pinned);
                    CheckStatus(NativeMethods.ov_shape_create(3, new long[] { 3, chunkVCount, hiddenSize }, out sDeepChunk), "Chunk D shape");
                    CheckStatus(NativeMethods.ov_tensor_create_from_host_ptr(OvElementType.F32, sDeepChunk, hDeepChunk.AddrOfPinnedObject(), out tDeepChunk), "Chunk D tensor");
                }
                else
                {
                    CheckStatus(NativeMethods.ov_shape_create(3, new long[] { 3, 0, hiddenSize }, out sDeepChunk), "D empty shape");
                    CheckStatus(NativeMethods.ov_tensor_create(OvElementType.F32, sDeepChunk, out tDeepChunk), "D empty tensor");
                }

                CheckStatus(NativeMethods.ov_shape_create((nuint)1, new long[] { 1 }, out var ovBeamShape), "Beam shape");
                GCHandle hBeam = GCHandle.Alloc(new int[] { 0 }, GCHandleType.Pinned);
                CheckStatus(NativeMethods.ov_tensor_create_from_host_ptr(OvElementType.I32, ovBeamShape, hBeam.AddrOfPinnedObject(), out var tBeam), "Beam tensor");

                CheckStatus(NativeMethods.ov_infer_request_set_tensor(_llmInfer, "inputs_embeds", tEmbed), "Set E");
                CheckStatus(NativeMethods.ov_infer_request_set_tensor(_llmInfer, "position_ids", tPos), "Set P");
                CheckStatus(_pipeline.HandleAttnMask(_llmInfer, tMask), "Set M");
                CheckStatus(NativeMethods.ov_infer_request_set_tensor(_llmInfer, "deepstack_visual_embeds", tDeepChunk), "Set D");
                CheckStatus(NativeMethods.ov_infer_request_set_tensor(_llmInfer, "visual_pos_masks", tVMask), "Set V");
                CheckStatus(NativeMethods.ov_infer_request_set_tensor(_llmInfer, "beam_idx", tBeam), "Set B");

                CheckStatus(NativeMethods.ov_infer_request_infer(_llmInfer), "Chunk replay infer");

                _totalPhysical += count;

                // Cleanup chunk tensors
                NativeMethods.ov_tensor_free(tEmbed); NativeMethods.ov_shape_free(ref ovEmbedShape); hEmbed.Free();
                NativeMethods.ov_tensor_free(tPos); NativeMethods.ov_shape_free(ref ovPosShape); hPos.Free();
                NativeMethods.ov_tensor_free(tMask); NativeMethods.ov_shape_free(ref ovMaskShape); hMask.Free();
                NativeMethods.ov_tensor_free(tDeepChunk); NativeMethods.ov_shape_free(ref sDeepChunk); if (hDeepChunk.IsAllocated) hDeepChunk.Free();
                NativeMethods.ov_tensor_free(tVMask); NativeMethods.ov_shape_free(ref ovVMaskShape); hVMask.Free();
                NativeMethods.ov_tensor_free(tBeam); NativeMethods.ov_shape_free(ref ovBeamShape); hBeam.Free();
            }

            _sessionVisualMasks.AddRange(visualPosMasksArr);
            _totalTemporal = currentT;

            if (visualEmbedsTensor != 0) NativeMethods.ov_tensor_free(visualEmbedsTensor);
            if (deepstackFeaturesTensor != 0) NativeMethods.ov_tensor_free(deepstackFeaturesTensor);
        }

        private nint CreateCombinedDeepstack(out GCHandle handle, out OvShape shape)
        {
            int L = 3;
            int H = _pipeline._hiddenSize;
            int totalN = 0;
            foreach (var arr in _accumulatedDeepstack) totalN += arr.Length / (L * H);

            if (totalN == 0)
            {
                handle = default;
                CheckStatus(NativeMethods.ov_shape_create(3, new long[] { L, 0, H }, out shape), "Create empty D shape");
                CheckStatus(NativeMethods.ov_tensor_create(OvElementType.F32, shape, out nint t), "Create empty D tensor");
                return t;
            }

            float[] combined = new float[L * totalN * H];
            for (int l = 0; l < L; l++)
            {
                int layerOffsetCombined = l * totalN * H;
                int nProcessed = 0;
                foreach (var arr in _accumulatedDeepstack)
                {
                    int imgN = arr.Length / (L * H);
                    int layerOffsetImg = l * imgN * H;
                    Array.Copy(arr, layerOffsetImg, combined, layerOffsetCombined + nProcessed * H, imgN * H);
                    nProcessed += imgN;
                }
            }

            handle = GCHandle.Alloc(combined, GCHandleType.Pinned);
            CheckStatus(NativeMethods.ov_shape_create(3, new long[] { L, totalN, H }, out shape), "Create combined D shape");
            CheckStatus(NativeMethods.ov_tensor_create_from_host_ptr(OvElementType.F32, shape, handle.AddrOfPinnedObject(), out nint tensor), "Create combined D tensor");
            return tensor;
        }

        public List<TurnData> GetHistory() => _history;

        public void Dispose()
        {
            if (_llmInfer != 0) NativeMethods.ov_infer_request_free(_llmInfer);
        }

        private void RunPrefill(int[] turnIds, nint visualEmbedsTensor, int numVisualTokens)
        {
            int totalTurnTokens = turnIds.Length;
            float[] inputsEmbedsArr = _pipeline.GetEmbeddings(turnIds);
            bool[] visualPosMasksArr = new bool[totalTurnTokens]; // This array is not used for visual tokens in this method, but kept for consistency with Chat/ReplayTurn

            // Position IDs
            long[] positionIdsArr = new long[3 * totalTurnTokens];
            int currentT = _totalTemporal;
            for (int i = 0; i < totalTurnTokens; i++)
            {
                positionIdsArr[0 * totalTurnTokens + i] = currentT;
                positionIdsArr[1 * totalTurnTokens + i] = currentT;
                positionIdsArr[2 * totalTurnTokens + i] = currentT;
                currentT++;
            }

            int realHidden = _pipeline._hiddenSize;

            CheckStatus(NativeMethods.ov_shape_create((nuint)3, new long[] { 1, totalTurnTokens, (long)realHidden }, out var ovEmbedShape), "Embed shape");
            GCHandle hEmbed = GCHandle.Alloc(inputsEmbedsArr, GCHandleType.Pinned);
            CheckStatus(NativeMethods.ov_tensor_create_from_host_ptr(OvElementType.F32, ovEmbedShape, hEmbed.AddrOfPinnedObject(), out var tEmbed), "Embed tensor");

            CheckStatus(NativeMethods.ov_shape_create((nuint)3, new long[] { 3, 1, totalTurnTokens }, out var ovPosShape), "Pos shape");
            GCHandle hPos = GCHandle.Alloc(positionIdsArr, GCHandleType.Pinned);
            CheckStatus(NativeMethods.ov_tensor_create_from_host_ptr(OvElementType.I64, ovPosShape, hPos.AddrOfPinnedObject(), out var tPos), "Pos tensor");

            int fullMaskLen = _totalPhysical + totalTurnTokens;
            CheckStatus(NativeMethods.ov_shape_create((nuint)2, new long[] { 1, fullMaskLen }, out var ovMaskShape), "Mask shape");
            long[] fullMask = Enumerable.Repeat(1L, fullMaskLen).ToArray();
            GCHandle hMask = GCHandle.Alloc(fullMask, GCHandleType.Pinned);
            CheckStatus(NativeMethods.ov_tensor_create_from_host_ptr(OvElementType.I64, ovMaskShape, hMask.AddrOfPinnedObject(), out var tMask), "Mask tensor");

            CheckStatus(NativeMethods.ov_shape_create((nuint)2, new long[] { 1, totalTurnTokens }, out var ovVMaskShape), "vMask shape");
            GCHandle hVMask = GCHandle.Alloc(visualPosMasksArr, GCHandleType.Pinned);
            CheckStatus(NativeMethods.ov_tensor_create_from_host_ptr(OvElementType.BOOL, ovVMaskShape, hVMask.AddrOfPinnedObject(), out var tVMask), "vMask tensor");

            CheckStatus(NativeMethods.ov_shape_create((nuint)1, new long[] { 1 }, out var ovBeamShape), "Beam shape");
            GCHandle hBeam = GCHandle.Alloc(new int[] { 0 }, GCHandleType.Pinned);
            CheckStatus(NativeMethods.ov_tensor_create_from_host_ptr(OvElementType.I32, ovBeamShape, hBeam.AddrOfPinnedObject(), out var tBeam), "Beam tensor");

            CheckStatus(NativeMethods.ov_infer_request_set_tensor(_llmInfer, "inputs_embeds", tEmbed), "Set E");
            CheckStatus(NativeMethods.ov_infer_request_set_tensor(_llmInfer, "position_ids", tPos), "Set P");
            CheckStatus(_pipeline.HandleAttnMask(_llmInfer, tMask), "Set M");

            nint tDeepCombined = CreateCombinedDeepstack(out GCHandle hDeepComb, out var sDeepComb);
            CheckStatus(NativeMethods.ov_infer_request_set_tensor(_llmInfer, "deepstack_visual_embeds", tDeepCombined), "Set D");
            CheckStatus(NativeMethods.ov_infer_request_set_tensor(_llmInfer, "visual_pos_masks", tVMask), "Set V");
            CheckStatus(NativeMethods.ov_infer_request_set_tensor(_llmInfer, "beam_idx", tBeam), "Set B");

            CheckStatus(NativeMethods.ov_infer_request_infer(_llmInfer), "RunPrefill infer");

            _totalPhysical += totalTurnTokens;
            _totalTemporal = currentT;

            NativeMethods.ov_tensor_free(tEmbed); NativeMethods.ov_shape_free(ref ovEmbedShape); hEmbed.Free();
            NativeMethods.ov_tensor_free(tPos); NativeMethods.ov_shape_free(ref ovPosShape); hPos.Free();
            NativeMethods.ov_tensor_free(tMask); NativeMethods.ov_shape_free(ref ovMaskShape); hMask.Free();
            NativeMethods.ov_tensor_free(tDeepCombined); NativeMethods.ov_shape_free(ref sDeepComb); if (hDeepComb.IsAllocated) hDeepComb.Free();
            NativeMethods.ov_tensor_free(tVMask); NativeMethods.ov_shape_free(ref ovVMaskShape); hVMask.Free();
            NativeMethods.ov_tensor_free(tBeam); NativeMethods.ov_shape_free(ref ovBeamShape); hBeam.Free();
        }
    }

    private OvStatus HandleAttnMask(nint infer, nint mask)
    {
        return NativeMethods.ov_infer_request_set_tensor(infer, "attention_mask", mask);
    }

    private void PruneStopSequence(List<int> tokens, string stopSeq, int targetLength)
    {
        if (tokens == null || tokens.Count == 0) return;

        while (tokens.Count > 0)
        {
            var sub = tokens.Select(x => (uint)x).ToArray();
            string decoded = _tokenizer.Decode(sub, skipSpecialTokens: true);
            if (decoded.Length > targetLength)
            {
                tokens.RemoveAt(tokens.Count - 1);
            }
            else break;
        }
    }

    private float[] GetEmbeddings(int[] tokenIds)
    {
        int count = tokenIds.Length;
        int hiddenSize = _hiddenSize;
        float[] result = new float[count * hiddenSize];

        long[] idsLong = new long[count];
        for (int i = 0; i < count; i++) idsLong[i] = (long)tokenIds[i];

        CheckStatus(NativeMethods.ov_shape_create(2, new long[] { 1, count }, out var shape), "Embed shape");
        GCHandle handle = GCHandle.Alloc(idsLong, GCHandleType.Pinned);
        CheckStatus(NativeMethods.ov_tensor_create_from_host_ptr(OvElementType.I64, shape, handle.AddrOfPinnedObject(), out var tensor), "Embed tensor");

        CheckStatus(NativeMethods.ov_infer_request_set_tensor(_textEmbedInfer, "input", tensor), "Set embed input");
        CheckStatus(NativeMethods.ov_infer_request_infer(_textEmbedInfer), "Infer embed");
        CheckStatus(NativeMethods.ov_infer_request_get_tensor(_textEmbedInfer, "inputs_embeds", out nint outTensor), "Get embed output");

        CheckStatus(NativeMethods.ov_tensor_data(outTensor, out nint dataPtr), "Get embed data");
        Marshal.Copy(dataPtr, result, 0, result.Length);

        NativeMethods.ov_tensor_free(tensor);
        NativeMethods.ov_shape_free(ref shape);
        handle.Free();

        return result;
    }

    private float[] GetEmbedding(int tokenId)
    {
        return GetEmbeddings(new int[] { tokenId });
    }

    private unsafe int Sample(nint logitsTensor, HashSet<int> historySet, GenerationConfig config)
    {
        CheckStatus(NativeMethods.ov_tensor_get_shape(logitsTensor, out var shape), "Get logits shape");
        long seqLen = Marshal.ReadInt64(shape.Dims + 1 * sizeof(long));
        long vocabSize = Marshal.ReadInt64(shape.Dims + 2 * sizeof(long));
        int vocabLen = (int)vocabSize;

        CheckStatus(NativeMethods.ov_tensor_data(logitsTensor, out nint dataPtr), "Get logits data");
        float* logitsPtr = (float*)dataPtr;
        long offset = (seqLen - 1) * vocabSize * sizeof(float);
        logitsPtr = (float*)((byte*)logitsPtr + offset);

        // Explicit Greedy path check
        if (config.Greedy)
        {
            int maxIdx = 0;
            float maxVal = logitsPtr[0];
            float greedyRepP = config.RepetitionPenalty;
            float greedyFreqP = config.FrequencyPenalty;

            bool applyGreedyPenalty = (greedyRepP != 1.0f || config.PresencePenalty != 0.0f || greedyFreqP != 0.0f) && historySet != null && historySet.Count > 0;

            for (int i = 0; i < vocabLen; i++)
            {
                float val = logitsPtr[i];
                if (applyGreedyPenalty && historySet!.Contains(i) && i < 151643)
                {
                    if (val < 0) val *= greedyRepP; else val /= greedyRepP;
                    val -= config.PresencePenalty;
                    val -= greedyFreqP;
                }
                if (i == 0 || val > maxVal) { maxVal = val; maxIdx = i; }
            }
            return maxIdx;
        }


        float tempInv = 1.0f / config.Temperature; // Ensure sort buffer

        // Fused loop: Apply temperature AND fill buffer in one pass
        float repPenalty = config.RepetitionPenalty;
        float freqPenalty = config.FrequencyPenalty;
        bool applyPenalty = (repPenalty != 1.0f || config.PresencePenalty != 0.0f || freqPenalty != 0.0f) && historySet != null && historySet.Count > 0;

        // Optimization: Single-pass TopK selection using a small array
        // Increased for better accuracy in non-greedy sampling
        // Optimization: Single-pass fast Top-K selection on the main thread.
        // Parallel.For on 150K items has massive overhead (15-20ms) which dominates the token time.
        // A simple scalar loop here takes < 0.5ms.
        int k = (config.TopK > 0 && config.TopK < 256) ? config.TopK * 2 : 128;
        if (_sortBuffer == null || _sortBuffer.Length < k) _sortBuffer = new LogitCandidate[k];

        int candidatesCount = 0;
        float minScoreInTopK = -float.MaxValue;
        int minScoreIdx = -1;

        for (int i = 0; i < vocabLen; i++)
        {
            float val = logitsPtr[i];
            if (applyPenalty && historySet!.Contains(i) && i < 151643)
            {
                if (val < 0) val *= repPenalty; else val /= repPenalty;
                val -= config.PresencePenalty;
                val -= freqPenalty;
            }
            float score = val * tempInv;

            if (candidatesCount < k)
            {
                _sortBuffer[candidatesCount].Id = i;
                _sortBuffer[candidatesCount].Logit = score;
                if (score < minScoreInTopK || minScoreIdx == -1) { minScoreInTopK = score; minScoreIdx = candidatesCount; }
                candidatesCount++;
            }
            else if (score > minScoreInTopK)
            {
                _sortBuffer[minScoreIdx].Id = i;
                _sortBuffer[minScoreIdx].Logit = score;

                minScoreInTopK = _sortBuffer[0].Logit;
                minScoreIdx = 0;
                for (int j = 1; j < k; j++)
                {
                    if (_sortBuffer[j].Logit < minScoreInTopK)
                    {
                        minScoreInTopK = _sortBuffer[j].Logit;
                        minScoreIdx = j;
                    }
                }
            }
        }

        // Final Sort of the top K elements
        Array.Sort(_sortBuffer, 0, candidatesCount, Comparer<LogitCandidate>.Create((a, b) => b.Logit.CompareTo(a.Logit)));

        Span<LogitCandidate> topCandidates = _sortBuffer.AsSpan(0, candidatesCount);

        // Softmax on top N only with Numerical Stability (Subtract Max Logit)
        float maxL = topCandidates[0].Logit;
        double sum = 0;
        for (int i = 0; i < topCandidates.Length; i++)
        {
            // Use double for intermediate sum to avoid precision loss
            double p = Math.Exp((double)topCandidates[i].Logit - maxL);
            topCandidates[i].Logit = (float)p;
            sum += p;
        }

        if (sum < 1e-9) // Fallback for extreme cases
        {
            for (int i = 0; i < topCandidates.Length; i++) topCandidates[i].Logit = 1.0f / topCandidates.Length;
        }
        else
        {
            float invSum = (float)(1.0 / sum);
            for (int i = 0; i < topCandidates.Length; i++) topCandidates[i].Logit *= invSum;
        }

        // TopK/TopP
        int count = topCandidates.Length;
        if (config.TopK > 0 && config.TopK < count) count = config.TopK;

        if (config.TopP > 0 && config.TopP < 1.0f)
        {
            double cumSum = 0;
            for (int i = 0; i < count; i++)
            {
                cumSum += topCandidates[i].Logit;
                if (cumSum >= config.TopP) { count = i + 1; break; }
            }
        }

        // Final Sample
        double newSum = 0;
        for (int i = 0; i < count; i++) newSum += topCandidates[i].Logit;

        double rnd = _rng.NextDouble() * newSum;
        double acc = 0;
        for (int i = 0; i < count; i++)
        {
            acc += topCandidates[i].Logit;
            if (rnd <= acc) return topCandidates[i].Id;
        }

        if (count <= 0) return topCandidates[0].Id;
        return topCandidates[count - 1].Id;
    }

    private int ArgMax(float[] logits)
    {
        int maxIdx = 0;
        float maxVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (logits[i] > maxVal) { maxVal = logits[i]; maxIdx = i; }
        }
        return maxIdx;
    }

    private int GetOutputDimension(nint model, nuint index)
    {
        NativeMethods.ov_model_output_by_index(model, index, out nint port);
        NativeMethods.ov_port_get_partial_shape(port, out var shape);

        long rank = shape.Rank.Min;
        if (rank > 0)
        {
            IntPtr dimsPtr = shape.Dims;
            // Last dim is at index rank-1
            IntPtr lastDimPtr = dimsPtr + (int)(rank - 1) * Marshal.SizeOf<OvDimension>();
            OvDimension dim = Marshal.PtrToStructure<OvDimension>(lastDimPtr);
            return (int)dim.Min;
        }
        return 0;
    }

    private int GetInputDimensionByName(nint model, string name)
    {
        // Search inputs by name
        nuint count;
        NativeMethods.ov_model_inputs_size(model, out count);
        for (nuint i = 0; i < count; i++)
        {
            NativeMethods.ov_model_input_by_index(model, i, out nint port);
            NativeMethods.ov_port_get_any_name(port, out nint namePtr);
            string portName = Marshal.PtrToStringAnsi(namePtr) ?? "";
            NativeMethods.ov_free(namePtr);
            if (portName == name)
            {
                NativeMethods.ov_port_get_partial_shape(port, out var shape);
                long rank = shape.Rank.Min;
                if (rank > 0)
                {
                    IntPtr dimsPtr = shape.Dims;
                    IntPtr lastDimPtr = dimsPtr + (int)(rank - 1) * Marshal.SizeOf<OvDimension>();
                    OvDimension dim = Marshal.PtrToStructure<OvDimension>(lastDimPtr);
                    return (int)dim.Min;
                }
            }
        }
        return 0;
    }

    private long[] GetShapeDims(OvShape shape)
    {
        int rank = (int)shape.Rank;
        long[] dims = new long[rank];
        for (int i = 0; i < rank; i++)
        {
            dims[i] = Marshal.ReadInt64(shape.Dims + i * sizeof(long));
        }
        return dims;
    }

    public static void CheckStatus(OvStatus status, string message)
    {
        if (status != OvStatus.OK)
        {
            nint msgPtr = NativeMethods.ov_get_last_err_msg();
            string detail = (msgPtr != nint.Zero) ? (Marshal.PtrToStringUTF8(msgPtr) ?? "No message details") : "No detail pointer";
            throw new Exception($"[OpenVINO Error] {message} (Status: {status}): {detail}");
        }
    }

    /// <summary>
    /// Matches official Qwen3VL smart_resize: preserves aspect ratio, rounds to factor multiples.
    /// factor = patch_size(16) * merge_size(2) = 32
    /// </summary>
    private static (int h, int w) SmartResize(int height, int width,
        int factor = 32, int minPixels = 256 * 256, int maxPixels = 1280 * 28 * 28)
    {
        int hBar = (int)Math.Round((double)height / factor) * factor;
        int wBar = (int)Math.Round((double)width / factor) * factor;
        if (hBar < factor) hBar = factor;
        if (wBar < factor) wBar = factor;
        if ((long)hBar * wBar > maxPixels)
        {
            double beta = Math.Sqrt((double)height * width / maxPixels);
            hBar = Math.Max(factor, (int)Math.Floor(height / beta / factor) * factor);
            wBar = Math.Max(factor, (int)Math.Floor(width / beta / factor) * factor);
        }
        else if ((long)hBar * wBar < minPixels)
        {
            double beta = Math.Sqrt((double)minPixels / (height * width));
            hBar = (int)Math.Ceiling(height * beta / factor) * factor;
            wBar = (int)Math.Ceiling(width * beta / factor) * factor;
        }
        return (hBar, wBar);
    }

    private (nint embeds, nint deepstack, int t, int h, int w) ProcessImage(string path, bool silent = false)
    {
        using var rawImage = Image.Load<Rgb24>(path);

        // 1. Dynamic Smart Resize (preserves aspect ratio, multiples of factor=32)
        var (targetH, targetW) = SmartResize(rawImage.Height, rawImage.Width);
        rawImage.Mutate(x => x.Resize(targetW, targetH));

        int gridT = 1;
        int gridH = targetH / 16;
        int gridW = targetW / 16;
        int numPatches = gridT * gridH * gridW;

        // 2. Prepare Name for Vision Input
        NativeMethods.ov_model_input_by_index(_visionModel, 0, out nint visionInputPort);
        NativeMethods.ov_port_get_any_name(visionInputPort, out nint visionInputNamePtr);
        string visionInputName = Marshal.PtrToStringAnsi(visionInputNamePtr) ?? "hidden_states";
        NativeMethods.ov_free(visionInputNamePtr);

        // 3. Variables for Pooling and Interpolation
        int pH = gridH / 2;
        int pW = gridW / 2;
        int numGridPerSide = 48;

        float[] pixelValues = new float[numPatches * 1536];
        long[] posIndices = new long[4 * numPatches];
        float[] posWeights = new float[4 * numPatches];

        float[] mean = { 0.48145466f, 0.4578275f, 0.40821073f };
        float[] std = { 0.26862954f, 0.26130258f, 0.27577711f };

        float[] hIdxsGrid = new float[gridH];
        float[] wIdxsGrid = new float[gridW];
        for (int i = 0; i < gridH; i++) hIdxsGrid[i] = i * (numGridPerSide - 1.0f) / (gridH - 1.0f);
        for (int i = 0; i < gridW; i++) wIdxsGrid[i] = i * (numGridPerSide - 1.0f) / (gridW - 1.0f);

        // Block-Major (2x2) Patching Loop
        for (int hb = 0; hb < pH; hb++)
        {
            for (int wb = 0; wb < pW; wb++)
            {
                for (int hi = 0; hi < 2; hi++)
                {
                    for (int wi = 0; wi < 2; wi++)
                    {
                        int h = hb * 2 + hi;
                        int w = wb * 2 + wi;
                        int currentPatch = (hb * pW + wb) * 4 + (hi * 2 + wi);

                        // A. Fill Pixel Values
                        int patchBase = currentPatch * 1536;
                        for (int c = 0; c < 3; c++)
                        {
                            for (int py = 0; py < 16; py++)
                            {
                                for (int px = 0; px < 16; px++)
                                {
                                    int y = h * 16 + py;
                                    int x = w * 16 + px;
                                    var pixel = rawImage[x, y];
                                    float val = c == 0 ? pixel.R : (c == 1 ? pixel.G : pixel.B);
                                    float normVal = (val / 255.0f - mean[c]) / std[c];

                                    // Layout [C, T, H, W] where T=2 (Qwen2-VL 3D patch embedding)
                                    pixelValues[patchBase + c * (2 * 16 * 16) + 0 * (16 * 16) + py * 16 + px] = normVal;
                                    pixelValues[patchBase + c * (2 * 16 * 16) + 1 * (16 * 16) + py * 16 + px] = normVal;
                                }
                            }
                        }

                        // B. Interpolation Weights
                        float fh = hIdxsGrid[h];
                        float fw = wIdxsGrid[w];
                        int hFloor = (int)Math.Floor(fh);
                        int wFloor = (int)Math.Floor(fw);
                        int hCeil = Math.Min(numGridPerSide - 1, hFloor + 1);
                        int wCeil = Math.Min(numGridPerSide - 1, wFloor + 1);
                        float dh = fh - hFloor;
                        float dw = fw - wFloor;

                        posIndices[0 * numPatches + currentPatch] = hFloor * numGridPerSide + wFloor;
                        posIndices[1 * numPatches + currentPatch] = hFloor * numGridPerSide + wCeil;
                        posIndices[2 * numPatches + currentPatch] = hCeil * numGridPerSide + wFloor;
                        posIndices[3 * numPatches + currentPatch] = hCeil * numGridPerSide + wCeil;

                        posWeights[0 * numPatches + currentPatch] = (1 - dh) * (1 - dw);
                        posWeights[1 * numPatches + currentPatch] = (1 - dh) * dw;
                        posWeights[2 * numPatches + currentPatch] = dh * (1 - dw);
                        posWeights[3 * numPatches + currentPatch] = dh * dw;
                    }
                }
            }
        }

        // 4. Vision Encoder Execution
        CheckStatus(NativeMethods.ov_shape_create(2, new long[] { numPatches, 1536 }, out var ovShape), "Vision shape");
        GCHandle handle = GCHandle.Alloc(pixelValues, GCHandleType.Pinned);
        CheckStatus(NativeMethods.ov_tensor_create_from_host_ptr(OvElementType.F32, ovShape, handle.AddrOfPinnedObject(), out var inputTensor), "Vision tensor");
        CheckStatus(NativeMethods.ov_compiled_model_create_infer_request(_visionCompiled, out nint visionInfer), "Vision infer create");
        CheckStatus(NativeMethods.ov_infer_request_set_tensor(visionInfer, visionInputName, inputTensor), "Vision set input");
        CheckStatus(NativeMethods.ov_infer_request_infer(visionInfer), "Vision execution");
        CheckStatus(NativeMethods.ov_infer_request_get_tensor(visionInfer, "last_hidden_state", out nint hiddenStates), "Vision get result");

        // 5. Positional Embedding Summation
        CheckStatus(NativeMethods.ov_shape_create(2, new long[] { 4, numPatches }, out var ovPosShape), "Pos shape");
        GCHandle posHandle = GCHandle.Alloc(posIndices, GCHandleType.Pinned);
        CheckStatus(NativeMethods.ov_tensor_create_from_host_ptr(OvElementType.I64, ovPosShape, posHandle.AddrOfPinnedObject(), out var posInputTensor), "Pos tensor");
        CheckStatus(NativeMethods.ov_compiled_model_create_infer_request(_visionPosCompiled, out nint posInfer), "Pos infer create");
        CheckStatus(NativeMethods.ov_infer_request_set_tensor(posInfer, "input", posInputTensor), "Pos set tensor");
        CheckStatus(NativeMethods.ov_infer_request_infer(posInfer), "Pos execution");
        CheckStatus(NativeMethods.ov_infer_request_get_output_tensor_by_index(posInfer, 0, out nint posEmbedsTensor), "Pos result");

        CheckStatus(NativeMethods.ov_tensor_data(hiddenStates, out nint hPtr), "hPtr");
        CheckStatus(NativeMethods.ov_tensor_data(posEmbedsTensor, out nint pPtr), "pPtr");
        int hiddenSize = _visionHiddenSize;
        float[] hArr = new float[numPatches * hiddenSize];
        float[] pArr = new float[4 * numPatches * hiddenSize];
        Marshal.Copy(hPtr, hArr, 0, hArr.Length);
        Marshal.Copy(pPtr, pArr, 0, pArr.Length);
        for (int p = 0; p < numPatches; p++)
        {
            for (int k = 0; k < 4; k++)
            {
                float wP = posWeights[k * numPatches + p];
                for (int d = 0; d < hiddenSize; d++) hArr[p * hiddenSize + d] += pArr[k * (numPatches * hiddenSize) + p * hiddenSize + d] * wP;
            }
        }
        Marshal.Copy(hArr, 0, hPtr, hArr.Length);

        // 6. Merger (RoPE & Attention)
        int ropeDim = _visionRoPEDim;
        float[] rotaryPosEmbArr = CalculateVisionRoPE(numPatches, gridW, ropeDim);
        CheckStatus(NativeMethods.ov_shape_create(2, new long[] { numPatches, ropeDim }, out var ovRopeShape), "Rope shape");
        GCHandle ropeHandle = GCHandle.Alloc(rotaryPosEmbArr, GCHandleType.Pinned);
        CheckStatus(NativeMethods.ov_tensor_create_from_host_ptr(OvElementType.F32, ovRopeShape, ropeHandle.AddrOfPinnedObject(), out var rotaryPosEmb), "Rope tensor");

        float[] mergerMask = new float[numPatches * numPatches];
        CheckStatus(NativeMethods.ov_shape_create(3, new long[] { 1, numPatches, numPatches }, out var ovMaskShape), "Mask shape");
        GCHandle maskHandle = GCHandle.Alloc(mergerMask, GCHandleType.Pinned);
        CheckStatus(NativeMethods.ov_tensor_create_from_host_ptr(OvElementType.F32, ovMaskShape, maskHandle.AddrOfPinnedObject(), out var mergerMaskTensor), "Mask tensor");

        CheckStatus(NativeMethods.ov_compiled_model_create_infer_request(_mergerCompiled, out nint mergerInfer), "Merger infer");
        CheckStatus(NativeMethods.ov_infer_request_set_tensor(mergerInfer, "hidden_states", hiddenStates), "Merger hidden");
        CheckStatus(NativeMethods.ov_infer_request_set_tensor(mergerInfer, "attention_mask", mergerMaskTensor), "Merger mask");
        CheckStatus(NativeMethods.ov_infer_request_set_tensor(mergerInfer, "rotary_pos_emb", rotaryPosEmb), "Merger rope");
        CheckStatus(NativeMethods.ov_infer_request_infer(mergerInfer), "Merger execution");
        CheckStatus(NativeMethods.ov_infer_request_get_tensor(mergerInfer, "last_hidden_state", out nint visualEmbeds), "Get result");
        CheckStatus(NativeMethods.ov_infer_request_get_tensor(mergerInfer, "deepstack_feature_lists", out nint deepstackFeatures), "Get deepstack");

        // 7. Cleanup
        NativeMethods.ov_infer_request_free(visionInfer); NativeMethods.ov_tensor_free(inputTensor); NativeMethods.ov_shape_free(ref ovShape); handle.Free();
        NativeMethods.ov_infer_request_free(posInfer); NativeMethods.ov_tensor_free(posInputTensor); NativeMethods.ov_shape_free(ref ovPosShape); posHandle.Free();
        NativeMethods.ov_tensor_free(rotaryPosEmb); NativeMethods.ov_shape_free(ref ovRopeShape); ropeHandle.Free();
        NativeMethods.ov_tensor_free(mergerMaskTensor); NativeMethods.ov_shape_free(ref ovMaskShape); maskHandle.Free();

        return (visualEmbeds, deepstackFeatures, gridT, pH, pW);
    }

    private double GetVideoDuration(string path)
    {
        try
        {
            var startInfo = new ProcessStartInfo
            {
                FileName = "ffprobe",
                Arguments = $"-v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 \"{path}\"",
                RedirectStandardOutput = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };
            using var process = Process.Start(startInfo);
            if (process == null) return 1.0;
            string output = process.StandardOutput.ReadToEnd();
            process.WaitForExit();
            if (double.TryParse(output.Trim(), out double duration)) return duration;
        }
        catch { }
        return 1.0; // Fallback
    }

    private (nint embeds, nint deepstack, int t, int h, int w) ProcessVideo(string videoPath, int maxFrames = 8, bool silent = false)
    {
        string tmpDir = Path.Combine(Path.GetTempPath(), "Qwen3VL_Vid_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tmpDir);
        try
        {
            double duration = GetVideoDuration(videoPath);
            double fps = maxFrames / Math.Max(0.1, duration);
            if (!silent) Console.WriteLine($"[Info] Extracting {maxFrames} frames (duration: {duration:F2}s, fps: {fps:F2})...");
            var startInfo = new ProcessStartInfo
            {
                FileName = "ffmpeg",
                Arguments = $"-y -i \"{videoPath}\" -vf \"fps={fps:F4}\" -vframes {maxFrames} \"{tmpDir}/f_%03d.jpg\"",
                RedirectStandardError = false,
                UseShellExecute = false,
                CreateNoWindow = true
            };
            using (var process = Process.Start(startInfo))
            {
                if (process == null) throw new Exception("Failed to start FFmpeg.");
                process.WaitForExit();
            }

            string[] frames = Directory.GetFiles(tmpDir, "*.jpg").OrderBy(f => f).ToArray();
            if (frames.Length == 0) throw new Exception("FFmpeg frame extraction failed.");

            List<float[]> allE = new();
            List<float[]> allD = new();
            int gH = 0, gW = 0, hidden = _hiddenSize, visHidden = _visionHiddenSize;

            int frameIdx = 1;
            foreach (var f in frames)
            {
                if (!silent) Console.WriteLine($"  - Processing frame {frameIdx}/{frames.Length}...");
                var (eT, dT, _, h, w) = ProcessImage(f, silent);
                gH = h; gW = w;
                frameIdx++;

                NativeMethods.ov_tensor_get_size(eT, out nuint eSz);
                float[] eA = new float[(int)eSz];
                NativeMethods.ov_tensor_data(eT, out nint eP);
                Marshal.Copy(eP, eA, 0, eA.Length);
                allE.Add(eA);

                NativeMethods.ov_tensor_get_size(dT, out nuint dSz);
                float[] dA = new float[(int)dSz];
                NativeMethods.ov_tensor_data(dT, out nint dP);
                Marshal.Copy(dP, dA, 0, dA.Length);
                allD.Add(dA);

                NativeMethods.ov_tensor_free(eT);
                NativeMethods.ov_tensor_free(dT);
            }

            // Combine Embeds
            int totalTokens = allE.Sum(a => a.Length / hidden);
            float[] combE = new float[totalTokens * hidden];
            int offE = 0;
            foreach (var a in allE) { Array.Copy(a, 0, combE, offE, a.Length); offE += a.Length; }

            CheckStatus(NativeMethods.ov_shape_create(3, new long[] { 1, totalTokens, hidden }, out var sE), "Vid eShape");
            CheckStatus(NativeMethods.ov_tensor_create(OvElementType.F32, sE, out nint tE), "Vid eTensor");
            NativeMethods.ov_tensor_data(tE, out nint pE);
            Marshal.Copy(combE, 0, pE, combE.Length);

            // Combine Deepstack
            // The deepstack features from Merger have the SAME hidden size as the LLM (hidden), not visHidden.
            int totalVisTokens = allD.Sum(a => a.Length / (3 * hidden));
            if (!silent) Console.WriteLine($"[Debug] Combined Deepstack: totalVisTokens={totalVisTokens}, hidden={hidden}");

            float[] combD = new float[3 * totalVisTokens * hidden];
            for (int l = 0; l < 3; l++)
            {
                int offDL = l * totalVisTokens * hidden;
                int nDone = 0;
                foreach (var a in allD)
                {
                    int n = a.Length / (3 * hidden);
                    Array.Copy(a, l * n * hidden, combD, offDL + nDone * hidden, n * hidden);
                    nDone += n;
                }
            }

            CheckStatus(NativeMethods.ov_shape_create(3, new long[] { 3, totalVisTokens, hidden }, out var sD), "Vid dShape");
            CheckStatus(NativeMethods.ov_tensor_create(OvElementType.F32, sD, out nint tD), "Vid dTensor");
            NativeMethods.ov_tensor_data(tD, out nint pD);
            Marshal.Copy(combD, 0, pD, combD.Length);

            return (tE, tD, frames.Length, gH, gW);
        }
        finally
        {
            try { Directory.Delete(tmpDir, true); } catch { }
        }
    }

    private float[] CalculateVisionRoPE(int numPatches, int unpooledW, int dim)
    {
        int halfDim = dim / 2;
        int quarterDim = halfDim / 2; // For 32 is 8, for 36 is 9
        float theta = 10000.0f;
        float[] freqs = new float[numPatches * dim];

        int pW = unpooledW / 2;

        for (int hb = 0; hb < (numPatches / 4) / pW; hb++)
        {
            for (int wb = 0; wb < pW; wb++)
            {
                for (int hi = 0; hi < 2; hi++)
                {
                    for (int wi = 0; wi < 2; wi++)
                    {
                        int h = hb * 2 + hi;
                        int w = wb * 2 + wi;
                        int i = (hb * pW + wb) * 4 + (hi * 2 + wi);

                        for (int d = 0; d < halfDim; d++)
                        {
                            float invFreq = 1.0f / (float)Math.Pow(theta, (double)(2 * (d % quarterDim)) / (double)halfDim);
                            freqs[i * dim + d] = h * invFreq;
                        }
                        for (int d = 0; d < halfDim; d++)
                        {
                            float invFreq = 1.0f / (float)Math.Pow(theta, (double)(2 * (d % quarterDim)) / (double)halfDim);
                            freqs[i * dim + halfDim + d] = w * invFreq;
                        }
                    }
                }
            }
        }
        return freqs;
    }

    private void PrintSignatures(nint model, string name)
    {
        nuint count; NativeMethods.ov_model_inputs_size(model, out count);
        Console.WriteLine($"{name} inputs ({count}):");
        for (nuint i = 0; i < count; i++) { nint port; NativeMethods.ov_model_input_by_index(model, i, out port); PrintPort(port, i); }
        NativeMethods.ov_model_outputs_size(model, out count);
        Console.WriteLine($"{name} outputs ({count}):");
        for (nuint i = 0; i < count; i++) { nint port; NativeMethods.ov_model_output_by_index(model, i, out port); PrintPort(port, i); }
    }

    private void PrintPort(nint port, nuint index)
    {
        nint namePtr; NativeMethods.ov_port_get_any_name(port, out namePtr);
        string? portName = Marshal.PtrToStringAnsi(namePtr);
        OvElementType type; NativeMethods.ov_port_get_element_type(port, out type);
        OvPartialShape shape; NativeMethods.ov_port_get_partial_shape(port, out shape);
        string shapeStr = ParseShape(shape);
        Console.WriteLine($"  [{index}] {portName} (Type: {type}, Shape: {shapeStr})");
        NativeMethods.ov_free(namePtr);
    }

    private string ParseShape(OvPartialShape shape)
    {
        if (shape.Rank.Min < 0 || shape.Dims == IntPtr.Zero) return "Dynamic";
        if (shape.Rank.Min == 0) return "[]";
        long rank = shape.Rank.Min;
        OvDimension[] dims = new OvDimension[rank];
        for (int i = 0; i < rank; i++) dims[i] = Marshal.PtrToStructure<OvDimension>(shape.Dims + i * Marshal.SizeOf<OvDimension>());
        return "[" + string.Join(", ", dims.Select(d => d.Min == d.Max ? d.Min.ToString() : (d.Max == -1 ? $"{d.Min}..?" : $"{d.Min}..{d.Max}"))) + "]";
    }

    public string DecodeOnly(List<int> ids)
    {
        if (ids == null || ids.Count == 0) return "";
        uint[] uids = ids.Select(x => (uint)x).ToArray();
        return _tokenizer.Decode(uids, skipSpecialTokens: true);
    }

    public void Dispose()
    {
        if (_visionInfer != 0) NativeMethods.ov_infer_request_free(_visionInfer);
        if (_visionPosInfer != 0) NativeMethods.ov_infer_request_free(_visionPosInfer);
        if (_mergerInfer != 0) NativeMethods.ov_infer_request_free(_mergerInfer);
        if (_textEmbedInfer != 0) NativeMethods.ov_infer_request_free(_textEmbedInfer);

        if (_visionModel != 0) NativeMethods.ov_model_free(_visionModel);
        if (_visionPosModel != 0) NativeMethods.ov_model_free(_visionPosModel);
        if (_mergerModel != 0) NativeMethods.ov_model_free(_mergerModel);
        if (_textEmbedModel != 0) NativeMethods.ov_model_free(_textEmbedModel);
        if (_languageModel != 0) NativeMethods.ov_model_free(_languageModel);

        if (_visionCompiled != 0) NativeMethods.ov_compiled_model_free(_visionCompiled);
        if (_visionPosCompiled != 0) NativeMethods.ov_compiled_model_free(_visionPosCompiled);
        if (_mergerCompiled != 0) NativeMethods.ov_compiled_model_free(_mergerCompiled);
        if (_textEmbedCompiled != 0) NativeMethods.ov_compiled_model_free(_textEmbedCompiled);
        if (_languageCompiled != 0) NativeMethods.ov_compiled_model_free(_languageCompiled);

        if (_core != 0) NativeMethods.ov_core_free(_core);
    }
}
