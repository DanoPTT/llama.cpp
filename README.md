> [!NOTE]
> **This is the DanoPTT fork of llama.cpp for one AMD Radeon AI PRO R9700** (gfx1201, RDNA4, ROCm/HIP, Windows 11).
> It is not an official llama.cpp build. The upstream README continues below the fork section.

## Fork: rollback build r149

**Branch `f711-rdna-r149-stack1027` @ `2a09574f5`** was the production build from 2026-10-27 13:22 to 2026-10-28 17:57. It is now kept as the **rollback** for the production branch `f711-rdna-r160-stack1028` (the default branch of this fork; see its README for the current production state).

- upstream llama.cpp `6050b467c`
- [stew675/llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts) release `v15-6050b467c-r149` (tree `423a59a6`), commit `faf81eb51`
- 9 carry commits on top (the same set as r160):

| commit | change | upstream |
|---|---|---|
| `531c47787` | server: clear stale state when reusing LRU-selected slots | #27624 |
| `dd69cb41b` | server: consult the prompt cache when the outgoing state is not worth saving | port of #28992 |
| `4177dbaf4`, `89632a8eb` | server: emit tool results before user text in Anthropic `/v1/messages` conversion (+ test) | #29482 |
| `0b1827c53` | server: don't store the draft KV in context checkpoints | #29509 |
| `9ffdbe1d4` | cuda: keep the decode-only SSM fusions off by default | #28768 |
| `fa3fedc90` | server: fall back to full prompt re-processing when memory rollback fails | #28007 |
| `2a09574f5` | cuda: `__vsub4` in the non-HIP Q6_K mmvq VDR=2 kernel | follow-up to #29478 |
| `16dbe3320` | ci: `f711-rocm` workflow (Windows, ROCm 10.0.0, gfx1201, `workflow_dispatch` only) | – |

### Runtime configuration (as it ran in production)

- Model: Qwen3.8-27B UD-Q5_K_XL (dense hybrid, Gated Delta Net + attention) with built-in MTP head, mmproj F16 on CPU (`--no-mmproj-offload`)
- `-c 262144 -np 2 --kv-unified -fa on -ub 512 -b 1024 --cache-type-k q8_0 --cache-type-v q8_0`
- Speculative decoding: `--spec-type draft-mtp --spec-draft-n-max 3`, **draft KV `-ctkd f16 -ctvd f16`**. On r149, a q8_0 draft KV allocates its own ~1 GiB FA staging arena (net +544 MiB, [stew675 #237](https://github.com/stew675/llama-cpp-rdna-boosts/issues/237), fixed in stew r156/r159).
- No `GGML_CUDA_FA_STAGE_*` variables (they were added in stew r159).
- `--cache-ram 10240 --ctx-checkpoints 8 --jinja` with a patched Qwen chat template, sampling `--temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 --presence-penalty 1.5`
- Client: Claude Code via the Anthropic-compatible `/v1/messages` endpoint

To roll back from r160: use the r149 binary with `-ctkd f16 -ctvd f16` and without the `GGML_CUDA_FA_STAGE_*` variables.

### Measured in production (2026-10-27 13:32 → 2026-10-28 15:07, 6 server runs, 827 requests)

| metric | value |
|---|---|
| prefill (prompts > 5k tokens) | 416–1017 t/s, median 877 t/s |
| decode | median 46.0 t/s |
| MTP draft acceptance | median 0.723 |
| longest context | 202,834 tokens |
| WDDM shared memory | grew by ~500 MiB/h (1,953 MiB after 2.7 h); r160 stays flat at ~1.26 GB |
| errors | no crashes; 2× `failed to prepare attention ubatches`, the server kept running |

Release gate (2026-10-27) against r100: PPL 1.8683 (r100 1.8680), greedy output hashes identical, MTP decode 51.4 t/s at 90/109 accepted, two slots: slot B after a long slot A +21 % tg. The only failing `test-backend-ops` case is SOLVE_TRI, a known test artifact (ggml-org/llama.cpp#31108).

### Building

Run the `f711-rocm` workflow (Actions → f711-rocm → Run workflow) on this branch. The defaults are ROCm 10.0.0 and `AMDGPU_TARGETS=gfx1201`.

---

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

## Quick start

A few options to get `llama.cpp` installed on your machine:

```bash
# curl
curl -LsSf https://llama.app/install.sh | sh

# powershell
irm https://llama.app/install.ps1 | iex
```

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
