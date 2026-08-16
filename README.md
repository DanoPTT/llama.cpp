# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## About this fork

This is an **experimental fork** built for one specific purpose: serving a Qwen3.6-27B
dense hybrid SSM/attention model ("f711") and a Qwen3.8-27B vision model in production
on an **AMD Radeon AI PRO R9700** (32 GB, gfx1201/RDNA4) via ROCm/HIP, driving
[Claude Code](https://github.com/anthropics/claude-code) through the `/v1/messages`
endpoint.

It combines two upstream sources:

1. [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — upstream, tracked
   close to `master`.
2. [stew675/llama.cpp:rdna-boosts](https://github.com/stew675/llama.cpp/tree/rdna-boosts) —
   a set of RDNA-specific kernel fusions (`MUL_MAT`, `FLASH_ATTN_EXT`, `RMS_NORM`,
   `MUL`) ported on top of (1).

...plus a single-target CI workflow (`.github/workflows/f711-rocm.yml`, ROCm 7.14,
`AMDGPU_TARGETS=gfx1201` only) so a build finishes in minutes instead of covering
every GPU target upstream CI builds for.

### Why

Upstream `llama.cpp` on this GPU/architecture combination (gfx1201/RDNA4, a dense
Qwen3.5/3.6-family hybrid SSM+attention model) left a lot of prefill throughput on
the table. `rdna-boosts` closes most of that gap. Before deploying it we ran two
independent correctness gates (not just a speed benchmark): `test-backend-ops` on
every op the branch touches, and a perplexity/code-quality comparison against a
control build differing only by those commits — see [Results](#results) below.

### Results

Measured on production hardware, 15–16 Aug 2026. **Hardware:** AMD Radeon AI PRO
R9700 (32 GB VRAM, gfx1201/RDNA4), ROCm 7.14. **Model:** f711-AMD-Q6_K
(Qwen3.6-27B, dense, `n_expert = 0`, hybrid SSM/gated-delta-net + MTP head).

**Speed** — real 100k-token prompt, production `-c 131072`:

| ctx | arm | prefill | TTFT | gen | shared VRAM (spill) |
|---|---|---|---|---|---|
| 131072 | base | 312.4 t/s | 5m22s | 18.4 t/s | 242 MiB |
| 131072 | **rdna-boosts** | **573.9 t/s (+84%)** | **2m55s** | 18.5 t/s | 244 MiB |
| 147456 | base | *collapses* (>10 min) | — | — | 768 MiB (over the spill threshold) |
| 147456 | **rdna-boosts** | **519.6 t/s** | 3m13s | 18.6 t/s | 542 MiB (still under threshold) |

`rdna-boosts` is also more VRAM-efficient at high context — it raises the
spill-collapse threshold instead of only being faster at the same one.

**Correctness / quality:**

- `test-backend-ops` on the ops the branch changes: **1194/1194** (`MUL_MAT`) and
  **4552/4552** (`FLASH_ATTN_EXT`) passing on gfx1201.
- Perplexity, production config (`-fa on`, fusions on): base **2.6898** →
  rdna-boosts **2.6992** (+0.35%). Bisection isolated the cause to one commit,
  `6e478a115` ("fuse IMRoPE + set-rows for BF16 KV cache"), which mis-fires on
  plain f16 KV cache where it shouldn't apply — reported upstream, kept deployed
  anyway since it doesn't regress generated-code quality and the speed win is
  large. As an independent cross-check, the same model on the **Vulkan** backend
  (which shares none of these kernels) measures PPL **2.7124** — i.e. the
  base→rdna-boosts gap is smaller than the ordinary HIP↔Vulkan backend gap.
- Code-generation quality benchmark (21 runs: generate C#/.NET code, `dotnet build`
  + regex assertions against a pre-validated skeleton): base 16/21 build-ok /
  98.3% asserts vs. rdna-boosts 15/21 / 99.2% asserts — no measurable regression
  (the one-build difference is within run-to-run noise at temperature 0.7).

**Status:** deployed in production since 16 Aug 2026 (branch `f711-rdna`, build tag
`rdna-20260815`), serving both models above. Two later upstream commits from the
same branch (`1b009339e`, `7955770b2`) were evaluated on 16 Aug and **not**
deployed — correct, but no measurable speed gain (−0.33%, within noise) on this
model/GPU; not worth the extra rebase-conflict surface.

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
