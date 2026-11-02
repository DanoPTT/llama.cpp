> [!NOTE]
> **This is the DanoPTT fork of llama.cpp for one AMD Radeon AI PRO R9700** (gfx1201, RDNA4, ROCm/HIP, Windows 11).
> It is not an official llama.cpp build. The upstream README continues below the fork section.

## Fork: production build (as of 2026-11-02)

**Branch `f711-rdna-r205-stack1102`** (the default branch of this fork, in production since 2026-11-02; the binary is built from `09370b1ed`, the commit after it only adds this README section):

- upstream llama.cpp `6050b467c`
- [stew675/llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts) release `v16-6050b467c-r205` (tree `d6e71901`), commit `0bb009678`
- 9 carry commits on top:

| commit | change | upstream |
|---|---|---|
| `894faacbd` | server: clear stale state when reusing LRU-selected slots | #27624 |
| `56f4b979d` | server: consult the prompt cache when the outgoing state is not worth saving | port of #28992 |
| `cbbcc69c4`, `a2535a539` | server: emit tool results before user text in Anthropic `/v1/messages` conversion (+ test) | #29482 |
| `bf1507822` | server: don't store the draft KV in context checkpoints | #29509 |
| `4eae46888` | cuda: keep the decode-only SSM fusions off by default | #28768 |
| `d156e268a` | server: fall back to full prompt re-processing when memory rollback fails | #28007 |
| `09370b1ed` | cuda: `__vsub4` in the non-HIP Q6_K mmvq VDR=2 kernel | follow-up to #29478 |
| `7c83de802` | ci: `f711-rocm` workflow (Windows, ROCm 10.0.0, gfx1201, `workflow_dispatch` only) | – |

Rollback branch: `f711-rdna-r160-stack1028` (stew r160, same carry set).

New against r160, relevant for this setup: the MTP draft KV window (r193, sized per sequence since r200 after our report stew675/llama-cpp-rdna-boosts#264), the softplus fix in the SSM gate/beta fusions (r201, after stew675/llama-cpp-rdna-boosts#265), GDN decode folds (r183) and q8_0 KV rotation fusions (r187). Since r201 the SSM fusions are byte-identical on and off and speed-neutral here, so the #28768 carry is due to be dropped with the next stack.

### Runtime configuration

- Model: Qwen3.8-27B UD-Q5_K_XL (dense hybrid, Gated Delta Net + attention) with built-in MTP head, mmproj F16 on CPU (`--no-mmproj-offload`)
- `-c 262144 -np 2 --kv-unified -fa on -ub 512 -b 1024 --cache-type-k q8_0 --cache-type-v q8_0`
- Speculative decoding: `--spec-type draft-mtp --spec-draft-n-max 3`, draft KV `-ctkd q8_0 -ctvd q8_0`, default MTP draft window (64k tokens per sequence, 132 096 draft KV cells)
- `--cache-ram 10240 --ctx-checkpoints 8 --jinja` with a patched Qwen chat template, sampling `--temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 --presence-penalty 1.5`
- Environment: `GGML_CUDA_ASYNC_INPUT=1`, `GGML_CUDA_FA_STAGE_PREALLOC=2`, `GGML_CUDA_FA_STAGE_MAX_MB=256` (without the last two, pp at 100k drops ~24 % and WDDM shared memory grows to ~2.9 GB)
- Client: Claude Code via the Anthropic-compatible `/v1/messages` endpoint

### Release gate (2026-11-02, against r160)

Greedy requests on the production launcher, fresh server per run, arms interleaved.

| metric | r160 | r205 |
|---|---|---|
| prefill, 100k-token prompt | 771–778 t/s | 765–812 t/s |
| decode with MTP, 100k-token prompt | 50.8–51.6 t/s | 50.6–51.4 t/s |
| MTP acceptance, 100k / 161k tokens | 90/109 / 369/425 | 90/111 / 369/425 |
| VRAM dedicated / WDDM shared after a 100k request | 31 728 / 910 MiB | 31 234 / 646 MiB |
| perplexity (Q6_K, q8_0 KV, 8 chunks) | 1.8683 | 1.8683 |

Greedy output hashes are identical to r160 at 100k and 161k. The draft window costs no acceptance at 161k tokens (identical to the window switched off). Two-slot, cache-restore, vision and tool-use checks pass (29/29 requests reproducible). Without MTP, r205 decodes ~1 % slower than r160. The only failing `test-backend-ops` case is SOLVE_TRI, a known test artifact (ggml-org/llama.cpp#31108). Production throughput figures will follow after a day of traffic.

### Building

Run the `f711-rocm` workflow (Actions → f711-rocm → Run workflow) on the production branch. The defaults are ROCm 10.0.0 and `AMDGPU_TARGETS=gfx1201`.

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
