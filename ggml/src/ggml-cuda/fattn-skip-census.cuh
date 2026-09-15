#pragma once

// Gate 0 census for PR #28943 (skip fully masked KV tiles in WMMA FA).
//
// Host-only instrumentation: it does NOT touch the kernel, so a binary built
// with this is the production kernel plus a host-side mask scan. The mask is
// identical in both measurement arms, so the census describes the opportunity
// that #28943 could exploit, independently of whether the patch is present.
//
// Enable with GGML_CUDA_FA_SKIP_CENSUS=1. Each distinct shape is scanned once
// (capped, see kMaxEntries) and one line is printed to stderr; everything after
// that is free. The scan is O(n_q * n_kv) per shape and syncs the stream, so
// this build must never be used for timing.
//
// Reported per shape:
//   tiles     total (jt, kb0) tiles over the full K length
//   skip      tiles where every relevant mask element is exactly -INF
//   tail      skippable tiles in the trailing run that flash_attn_mask_to_KV_max
//             already skips today (fattn-common.cuh:666 walks back from the end
//             and returns a scalar)
//   interior  skip - tail == what #28943 would skip ON TOP of what we do today.
//             THIS IS THE GATE 0 NUMBER. If it is ~0 the patch has nothing to
//             win here and a null speed result is explained, not ambiguous.
//   kvmax     whether the KV_max helper is even launched for this shape
//             (fattn-common.cuh:1094: Q->ne[1] >= 1024 || Q->ne[3] > 1)

#include "common.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <tuple>
#include <vector>

static bool ggml_cuda_fa_skip_census_enabled() {
    static const bool enabled = getenv("GGML_CUDA_FA_SKIP_CENSUS") != nullptr;
    return enabled;
}

// One scan per distinct shape; the cap keeps a long run from turning into a log flood.
static const size_t ggml_cuda_fa_skip_census_max_entries = 64;

static void ggml_cuda_fa_skip_census(
        const ggml_tensor * dst, const int DKQ, const int ncols1, const int ncols2,
        const int nbatch_fa, cudaStream_t stream) {
    if (!ggml_cuda_fa_skip_census_enabled()) {
        return;
    }

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * mask = dst->src[3];

    if (!mask) {
        return; // no mask -> nothing to skip, and the kernel takes the mask_h == nullptr path
    }

    using key_t = std::tuple<int, int, int64_t, int64_t, int64_t>;
    static std::set<key_t> seen;

    const key_t key = { DKQ, nbatch_fa, Q->ne[1], K->ne[1], Q->ne[3] };
    if (seen.count(key)) {
        return;
    }
    if (seen.size() >= ggml_cuda_fa_skip_census_max_entries) {
        return;
    }
    seen.insert(key);

    const size_t nbytes = ggml_nbytes(mask);
    if (nbytes > (size_t) 1024*1024*1024) {
        fprintf(stderr, "FA-CENSUS DKQ=%d n_q=%5lld n_kv=%7lld: mask too large (%zu B), skipped\n",
                DKQ, (long long) Q->ne[1], (long long) K->ne[1], nbytes);
        fflush(stderr);
        return;
    }

    std::vector<char> host(nbytes);
    CUDA_CHECK(cudaMemcpyAsync(host.data(), mask->data, nbytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int64_t ne01     = Q->ne[1];
    const int64_t n_kv     = K->ne[1];
    const int64_t ntiles_x = (ne01 + ncols1 - 1) / ncols1;
    const int64_t ntiles_k = n_kv / nbatch_fa; // partial trailing tile is not counted
    const int64_t nseq     = mask->ne[3];

    int64_t tiles = 0, skip = 0, tail = 0;

    for (int64_t s3 = 0; s3 < nseq; ++s3) {
        for (int64_t jt = 0; jt < ntiles_x; ++jt) {
            std::vector<char> tile_skippable(ntiles_k, 0);

            for (int64_t kb0 = 0; kb0 < ntiles_k; ++kb0) {
                bool all_inf = true;

                for (int64_t jj = 0; jj < ncols1 && all_inf; ++jj) {
                    // matches fastmodulo(jt*ncols1 + jj, ne01) in the kernel
                    const int64_t j = (jt*ncols1 + jj) % ne01;

                    const char * row = host.data() + j*mask->nb[1] + s3*mask->nb[3];

                    for (int64_t i = 0; i < nbatch_fa; ++i) {
                        // The kernel tests __half2float(x) == -INFINITY, which is true for exactly
                        // one fp16 bit pattern, so comparing the raw bits is equivalent and avoids
                        // a conversion call per element (the scan is O(n_q * n_kv)).
                        const uint16_t v = *((const uint16_t *) (row + (kb0*nbatch_fa + i)*mask->nb[0]));
                        if (v != 0xFC00u) {
                            all_inf = false;
                            break;
                        }
                    }
                }

                tiles += 1;
                if (all_inf) {
                    skip += 1;
                    tile_skippable[kb0] = 1;
                }
            }

            // Trailing run: exactly what flash_attn_mask_to_KV_max collapses into its scalar today.
            for (int64_t kb0 = ntiles_k - 1; kb0 >= 0 && tile_skippable[kb0]; --kb0) {
                tail += 1;
            }
        }
    }

    const int64_t interior = skip - tail;
    const bool    kvmax    = (ne01 >= 1024 || Q->ne[3] > 1);

    fprintf(stderr,
            "FA-CENSUS DKQ=%d ncols1=%d ncols2=%d nbatch_fa=%d n_q=%5lld n_kv=%7lld nseq=%lld | "
            "tiles=%8lld skip=%8lld (%5.1f%%) tail=%8lld interior=%8lld (%5.1f%%) kvmax=%s\n",
            DKQ, ncols1, ncols2, nbatch_fa,
            (long long) ne01, (long long) n_kv, (long long) nseq,
            (long long) tiles, (long long) skip, tiles ? 100.0*skip/tiles : 0.0,
            (long long) tail, (long long) interior, tiles ? 100.0*interior/tiles : 0.0,
            kvmax ? "yes" : "no");
    fflush(stderr);
}
