#define DG_NVLINK_BARRIER_TRAP_ONLY_TIMEOUT 1
#include <deep_gemm/impls/sm90_mxfp4_mega_moe_h200_fused.cuh>

using namespace deep_gemm;

static void instantiate_flash_m8_kernel() {
    auto ptr = reinterpret_cast<void*>(
        &sm90_mxfp4_mega_moe_h200_fused_impl<
            /* kNumSMs */ 78,
            /* kHidden */ 4096,
            /* kIntermediateHidden */ 2048,
            /* kNumExperts */ 256,
            /* kNumTopk */ 6,
            /* kNumMaxTokensPerRank */ 8192,
            /* kNumExpertsPerWave */ 16,
            /* BLOCK_M */ 8,
            /* BLOCK_N */ 256,
            /* kNumMaxPoolTokens */ 399360,
            /* kNumPaddedSFPoolTokens */ 6389760,
            /* kNumStages */ 4,
            /* kActivationClamp */ 0x1.4p+3f,
            /* kFastMath */ true,
            /* kSwapABRequested */ true,
            /* kSingleActiveDispatchWarp */ true,
            /* kUseMode2RowDecoder */ true,
            /* kUseInterleavedScheduler */ true>);
    (void)ptr;
}
