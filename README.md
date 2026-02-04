# Qwen3VL-OpenVINO

OpenVINO inference pipeline for Qwen3VL series models, using OpenVINO C API, written in C#.

## Features

- **OpenVINO C API**: Leverages the high-performance OpenVINO C API for inference.
- **Qwen3VL Support**: Specifically optimized for the Qwen3VL series models.
- **Precision**: Tested on Qwen3VL 4B/8B **int4** quantized models.
- **Language**: Written in **C#**, providing a seamless integration for .NET developers.

## Prerequisites

- [OpenVINO Toolkit](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html)
- .NET SDK (Compatible with the project version)
- Qwen3VL models (converted to OpenVINO format)

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/Blackwood416/Qwen3VL-OpenVINO.git
   ```
2. Configure your model paths in the code.
3. Build and run the project.

## Acknowledgements

- [Qwen Team](https://github.com/QwenLM) for the Qwen3VL models.
- [Intel OpenVINO](https://github.com/openvinotoolkit/openvino) for the inference engine.
