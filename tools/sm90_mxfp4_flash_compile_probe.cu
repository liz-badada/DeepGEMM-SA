#define DG_NVLINK_BARRIER_TRAP_ONLY_TIMEOUT 1
#include <deep_gemm/impls/sm90_mxfp4_mega_moe_h200_fused.cuh>

#ifndef DG_PROBE_EPW
#define DG_PROBE_EPW 16
#endif
#ifndef DG_PROBE_NUM_RANKS
#define DG_PROBE_NUM_RANKS 8
#endif
#ifndef DG_PROBE_BLOCK_M
#define DG_PROBE_BLOCK_M 8
#endif
#ifndef DG_PROBE_BLOCK_N
#define DG_PROBE_BLOCK_N 256
#endif
#ifndef DG_PROBE_STAGES
#define DG_PROBE_STAGES 4
#endif
#ifndef DG_PROBE_SWAP_AB
#define DG_PROBE_SWAP_AB true
#endif
#ifndef DG_PROBE_SINGLE_DISPATCH
#define DG_PROBE_SINGLE_DISPATCH true
#endif
#ifndef DG_PROBE_MODE2_ROW
#define DG_PROBE_MODE2_ROW true
#endif

using namespace deep_gemm;

static void instantiate_flash_kernel() {
    auto ptr = reinterpret_cast<void*>(
        &sm90_mxfp4_mega_moe_h200_fused_impl<
            /* kNumSMs */ 78,
            /* kNumRanks */ DG_PROBE_NUM_RANKS,
            /* kHidden */ 4096,
            /* kIntermediateHidden */ 2048,
            /* kNumExperts */ 256,
            /* kNumTopk */ 6,
            /* kNumMaxTokensPerRank */ 8192,
            /* kNumExpertsPerWave */ DG_PROBE_EPW,
            /* BLOCK_M */ DG_PROBE_BLOCK_M,
            /* BLOCK_N */ DG_PROBE_BLOCK_N,
            /* kNumMaxPoolTokens */ 399360,
            /* kNumPaddedSFPoolTokens */ 6389760,
            /* kNumStages */ DG_PROBE_STAGES,
            /* kActivationClamp */ 0x1.4p+3f,
            /* kFastMath */ true,
            /* kSwapABRequested */ DG_PROBE_SWAP_AB,
            /* kSingleActiveDispatchWarp */ DG_PROBE_SINGLE_DISPATCH,
            /* kUseMode2RowDecoder */ DG_PROBE_MODE2_ROW,
            /* kUseInterleavedScheduler */ true>);
    (void)ptr;
}
