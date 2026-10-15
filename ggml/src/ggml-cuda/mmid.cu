#include "common.cuh"
#include "mmid.cuh"

// Qwen4exp 512-expert / 10-active fast path (ported from the halo-box tree). The generic helper
// below launches one warp per expert and every warp scans all token slots (O(n_experts*n_tokens*
// n_used)). Here one 1024-thread block histograms the slots, computes the 512 bounds, and
// scatters the two maps in O(n_tokens*n_used). Row ordering inside an expert is irrelevant:
// ids_dst maps every compact MMQ row back to its exact destination and there is no cross-row
// reduction, so results are identical to the generic kernel by construction.  Every occurrence of
// a repeated id is its own slot here, matching the generic kernel's duplicate handling.
__launch_bounds__(1024, 1)
static __global__ void mm_ids_helper_512_10(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst,
        int32_t * __restrict__ expert_bounds, const int n_tokens, const int nchannels_y,
        const int si1, const int sis1, const bool write_inverse) {
    __shared__ int counts[512];
    __shared__ int cursors[512];

    const int tid = threadIdx.x;
    if (tid < 512) {
        counts[tid] = 0;
        cursors[tid] = 0;
    }
    __syncthreads();

    const int n_slots = n_tokens * 10;
    for (int idx = tid; idx < n_slots; idx += blockDim.x) {
        const int token = idx / 10;
        const int iex   = idx - token * 10;
        atomicAdd(&counts[ids[token * si1 + iex]], 1);
    }
    __syncthreads();

    if (tid == 0) {
        int total = 0;
        expert_bounds[0] = 0;
        for (int e = 0; e < 512; ++e) {
            total += counts[e];
            expert_bounds[e + 1] = total;
        }
    }
    __syncthreads();

    for (int idx = tid; idx < n_slots; idx += blockDim.x) {
        const int token = idx / 10;
        const int iex   = idx - token * 10;
        const int expert = ids[token * si1 + iex];
        const int dst = expert_bounds[expert] + atomicAdd(&cursors[expert], 1);
        ids_dst[dst] = idx;
        if (write_inverse) {
            ids_src1[idx] = dst;
        } else {
            ids_src1[dst] = token * sis1 + iex % nchannels_y;
        }
    }
}

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
//
// An expert id may repeat within a token's row; every occurrence is an independent destination column and gets its own
// compact row (the Vulkan backend handles the same case in its prepass).  Pass 1 counts the columns of the lower-indexed
// experts (nex_prev), pass 2 writes this expert's columns in (token, iex) order directly to the compact arrays.  No
// per-token staging buffer is used, so the kernel's shared-memory need does not grow with n_tokens*n_expert_used.
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int expert = blockIdx.x;
    const int lane   = threadIdx.x;

    // Pass 1: number of compact columns for the experts with a lower index.
    int nex_prev = 0;
    for (int it = 0; it < n_tokens; ++it) {
        for (int iex = lane; iex < n_expert_used; iex += warp_size) {
            nex_prev += ids[it*si1 + iex] < expert;
        }
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);

    // Pass 2: write this expert's columns, one per matching (token, iex) pair, in (token, iex) order.
    int it_compact = 0;
    for (int it = 0; it < n_tokens; ++it) {
        int lane_count = 0;
        for (int iex = lane; iex < n_expert_used; iex += warp_size) {
            lane_count += ids[it*si1 + iex] == expert;
        }
        const int tok_count = warp_reduce_sum<warp_size>(lane_count);
        const int lane_base = warp_prefix_inclusive_sum<int, warp_size>(lane_count) - lane_count;

        int j = 0;
        for (int iex = lane; iex < n_expert_used; iex += warp_size) {
            if (ids[it*si1 + iex] == expert) {
                const int pos = nex_prev + it_compact + lane_base + j;
                ids_dst[pos] = it*n_expert_used + iex;
                // ids_src1 holds the forward map, or the inverse map (token slot -> compact row) for quant dedup
                if (write_inverse) {
                    ids_src1[it*n_expert_used + iex] = pos;
                } else {
                    ids_src1[pos] = it*sis1 + iex % nchannels_y;
                }
                ++j;
            }
        }
        it_compact += tok_count;
    }

    if (lane == 0) {
        expert_bounds[expert] = nex_prev;
        if (expert == static_cast<int>(gridDim.x) - 1) {
            expert_bounds[gridDim.x] = nex_prev + it_compact;
        }
    }
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    static const bool mmid_512_disabled = getenv("GGML_CUDA_DISABLE_MMID_512") != nullptr;
    // mm_ids_helper_512_10: one 1024-thread block replaces the 512 one-warp expert blocks of the
    // generic helper for the 512-expert/10-used shape.  RDNA3_5 (source of record) + RDNA4
    // (2026-09-06 gfx1201 validation: prefill +4.5% pp16384 / +5.2% pp2048, same-seed text
    // byte-identical; decode unaffected - the helper feeds the mmq prefill path only).
    if ((GGML_CUDA_CC_IS_RDNA3_5(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) && n_experts == 512 && n_expert_used == 10 && !mmid_512_disabled) {
        mm_ids_helper_512_10<<<1, 1024, 0, stream>>>(
            ids, ids_src1, ids_dst, expert_bounds, n_tokens, nchannels_y, si1, sis1, write_inverse);
        return;
    }

    GGML_ASSERT(n_tokens >= 0 && n_expert_used >= 0);
    GGML_ASSERT((int64_t) n_tokens * n_expert_used < (int64_t) 1 << 31);

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    mm_ids_helper<<<num_blocks, block_size, 0, stream>>>(
        ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse);
}
