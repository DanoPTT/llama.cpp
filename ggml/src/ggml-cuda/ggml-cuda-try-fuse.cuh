// The delivery's CUDA graph fusion dispatcher (matchers + per-concern try-fuse arms).
// Split out of ggml-cuda.cu (an upstream hub); declared here, defined in ggml-cuda-try-fuse.cu.
#pragma once

#include "common.cuh"

int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i);

// Also used by the graph optimizer in ggml-cuda.cu.
bool ggml_cuda_gdn_state_gather_deferrable(const ggml_cgraph * cgraph, int node_idx);

// One pass over the graph answering ggml_cuda_gdn_state_gather_deferrable for every gather: the per-gather walk
// above visits every later node (~48 gathers x ~1900 nodes per decode graph on a 64-layer GDN model, ~1.3 ms host
// per token).  Built lazily once per graph evaluation (GGML_CUDA_GDN_GATHER_INDEX=0: per-gather walk).
struct ggml_cuda_gdn_gather_index {
    struct gather_info {
        int  gdn          = -1;    // index of the GATED_DELTA_NET whose state (src[5]) roots at the gather
        int  n_gdn        = 0;
        bool other_reader = false; // a non-view node other than that GDN reads the gathered rows
    };
    std::unordered_map<const ggml_tensor *, gather_info>      gathers; // keyed by the candidate GET_ROWS
    std::unordered_map<const ggml_tensor *, std::vector<int>> writers; // cache root -> non-view nodes rooted at it
    bool built = false;
};
void ggml_cuda_gdn_gather_index_build(const ggml_cgraph * cgraph, ggml_cuda_gdn_gather_index & idx);
bool ggml_cuda_gdn_state_gather_deferrable(const ggml_cgraph * cgraph, int node_idx, const ggml_cuda_gdn_gather_index & idx);
struct ggml_cuda_topk_moe_args;
bool ggml_cuda_topk_moe_fusion(const struct ggml_cgraph * cgraph, int node_idx, struct ggml_cuda_topk_moe_args & args);
