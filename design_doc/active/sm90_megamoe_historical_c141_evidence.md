# SM90 MegaMoE historical C141 evidence

_Read-only seed evidence for the DeepSeek-V4-Flash H20 CAKE campaign, captured
2026-09-13_

## Scope and provenance

The retained Kernel Factory archive on `H20-GPU-05` contains a later,
independently evolved SM90 NVFP4 MegaMoE line. Its formal C141 champion is not
the current MXFP4 source and ran on eight H200 GPUs, so its timings are not an
H20 baseline. It is nevertheless strong mechanism evidence because its Flash
workload has the same model geometry used by this campaign: hidden 4096,
intermediate hidden 2048, 256 experts, top-k 6, and EP8.

The authoritative archive root is:

`/lustre/raplab/client/jinyanc/workspace/jinyanc/github/DeepGEMM-aichen/.worktrees/kf-megamoe/.kf-work/megamoe`

Important retained identities are:

- C141 solution SHA-256:
  `cf9b6a09b146762e727d7a356a5ec0a6b83f94c7de13bc84e85d520c18751213`
- C141 fused-body SHA-256:
  `0e698d895ac3c71f5b145e20c35386999c980ad5a99ffea8988dae68d7d23e77`
- C141 kernel-header SHA-256:
  `bce850c9371f08220a830869c952499efdc8c2822a90cc2bbac3843fbd63eb22`
- C141 quantization/prepack SHA-256:
  `a10c557c8fa86e41efee0d15b68b2e43a1113e2a8bceb2a3f5a2add701db5139`
- Full-24 audit:
  `manual/c175-topology-tail-map/full24-audit.json`

## Exact Flash measurements on H200

The full-24 audit records the frozen C141 control and the paired FP8 control.
All values below are true EP8 system latency from the exact Flash geometry.

| M | C141 NVFP4 | Frozen FP8 | FP8 / C141 |
| ---: | ---: | ---: | ---: |
| 1 | 107.504 us | 123.008 us | 1.1442x |
| 8 | 239.216 us | 248.512 us | 1.0389x |
| 16 | 253.040 us | 274.680 us | 1.0855x |
| 32 | 259.640 us | 286.736 us | 1.1044x |
| 64 | 261.536 us | 299.760 us | 1.1462x |
| 128 | 267.840 us | 297.208 us | 1.1096x |
| 256 | 285.656 us | 296.848 us | 1.0392x |
| 512 | 387.480 us | 446.672 us | 1.1528x |

The eight-row geometric-mean speedup is 1.1018x. This proves that compact FP4
decode can beat the FP8 kernel on SM90 for this exact model geometry, but it
does not prove the result on H20 or for M above 512. In particular, M=8 and
M=256 do not meet this campaign's stricter per-row 1.05x floor.

## Transferable mechanisms and negative evidence

### High-confidence split-M accumulator transfer

The strongest directly matching experiment is C121, which became part of the
C141 champion. Flash M=512 uses the same structural topology as the current
MXFP4 M=512+ tier: `BM128/BN128/BK128`, with two M64N128 math warpgroups
sharing one exactly recovered B tile. C121 changed only the compile-time
eligibility of this topology from eight F32-accumulator WGMMAs to eight packed
F16-accumulator WGMMAs. On a same-allocation 8xH200 comparison it reduced
latency from 452.320 to 407.048 us (-10.01%), beat the paired FP8 control by
9.46%, and passed exact eight-rank correctness. Registers remained 168 with
zero stack/local memory; 192 scalar FFMAs became 96 HFMA2 plus 128 HADD2.

An identical-flags SASS comparison confirms that the current Flash M=512
MXFP4 specialization still emits eight
`QGMMA.64x128x32.F32.E4M3.E4M3`, whereas the retained C141 cubin emits eight
`QGMMA.64x128x32.F16.E4M3.E4M3`. Both use 168 registers and have no
stack/local memory. The current specialization has 6,295 static instructions,
33 `SYNCS.EXCH.64`, 82 phase-check try-waits, and 26 `STS.128`; C141 has 6,455,
25, 74, and 17 respectively. This makes a bounded packed-F16 accumulator lane
the highest-confidence large-M transfer. It must remain topology-gated rather
than M-gated and must pass the MXFP4 numerical contract before timing.

The authoritative C121 inputs are retained at:

- Seed solution:
  `packages/ep8/c121-splitm-f16-accum-dev1/seed/seed_solution.json`
- Measured candidate:
  `candidates/ep8/c121-splitm-f16-accum-3083055/candidate_solution.json`
- Promotion candidate and receipts:
  `candidates/ep8/c121-splitm-f16-accum-3083214/`

The serialized seed-to-candidate source diff adds only
`kSplitMDecodedWeightReuse` to the packed-F16 eligibility predicate. That
one-line delta is not directly applicable to the current MXFP4 tree: the C121
seed already contained `FP8MMAF16M64N128K32`, packed `uint32_t` accumulators,
`__half2` scaled promotion, F16 L1/L2 mainloop branches, and final conversion
to float for the unchanged epilogue. The current MXFP4 implementation contains
none of that infrastructure. A faithful CAKE port must therefore carry all of
the following as one coherent topology-gated change:

1. Add the exact `m64n128k32.f16.e4m3.e4m3` wrapper locally in the MXFP4
   kernel header and include `cuda_fp16.h`; do not alter the shared MMA ABI.
2. Enable it only when `kSplitMDecodedWeightReuse` is true and swap-AB is
   inactive, with static M/N/K shape assertions against the existing F32
   selector.
3. Accumulate WGMMA output in 32 packed `uint32_t` registers, apply activation
   scales through `__half2`/`__hfma2`, and retain the existing two per-K64 L2
   scale groups.
4. Convert the packed final accumulator to the existing float layout before
   the current SwiGLU/scatter epilogues, leaving all routing, barriers, decoded
   weight sharing, and public formats unchanged.
5. Prove the generated M512 cubin actually contains eight F16 QGMMAs and no
   F32 QGMMA, with 168 registers and zero stack/local/spill, before consuming
   the eight-GPU correctness/performance gate.

### Other evidence

1. C141 uses a private sign-swizzled fused-weight cache while preserving the
   public Marlin ABI. Its isolated recovery core removes two fixed sign PRMTs
   and compiles to eight hot instructions. The current MXFP4 `dequant_word`
   already has an eight-operation selector/shift/PRMT/sign-merge structure, so
   CAKE must compare generated SASS before attempting this transformation; a
   blind NVFP4 transplant is likely redundant.
2. C122 pairs independent LUT loads before dependent decode and improved Flash
   M=512 by 0.6--2.4% in same-allocation tests. MXFP4 constructs E8M0-scaled
   tables in registers and has no shared LUT load, so the reusable principle is
   to expose independent scale/decode work early, not to copy the LUT code.
3. C175's topology-aware tail map changed the full-24 geometric mean by only
   -0.24% versus C141. C177 reduced Pro M=512 by about 0.62% but was neutral on
   the full suite. These are composable tail optimizations, not primary seeds.
4. The retained C177 NCU sample is barrier-bound: barrier stalls 41.79%, long
   scoreboard stalls 19.68%, tensor-pipe active 27.06%, and only 0.73 eligible
   warps/cycle. This supports testing the current barrier-striped startup and
   then reducing mainloop publication/wait latency without changing WGMMA
   ordering.
5. C179's register/shared WGMMA experiment is a negative control. Although it
   removed dense decoded-B shared-memory stages and remained bit exact, a wait
   after every fragment made Flash M=8 9.73% slower than C177. Do not retry an
   RS path unless multiple fragments are legally grouped between commit/wait
   operations and the decode is overlapped with the in-flight group.

### Interleaved-scheduler negative control

The retained three-way promotion artifact
`manual/interleave-allm-fixed-20260802/threeway-3421552.json` is a direct
warning against inferring a scheduler win from fewer atomics alone. On 8xH200
with the MiMo geometry (`H=6144`, `IH=2048`, `E=384`, top-k 8), the optimized
interleaved scheduler improved the existing NVFP4 branch by 6.19% at M=16 and
4.76% at M=32. The same comparison passed correctness and used two runs per
implementation, eight retained system repetitions, maximum latency across all
eight ranks, and an 8 GB L2 flush per sample per rank.

This is not the campaign's H20 DeepSeek-V4-Flash geometry and therefore cannot
reject a shape-specific static-scheduler candidate by itself. It does mean
that disabling interleaving at M=16/M=32 is a high-risk reversal of a measured
optimization. Require paired H20 evidence; do not promote it merely because
static SASS reduces `ATOMG` or `SYNCS.EXCH` counts.

## CAKE priorities

Use this ordering after the eight-GPU H20 gate becomes available:

1. Measure the barrier-striped candidate across the complete M=8..8192 Flash
   denominator.
2. Test packed-F16 accumulation only for the existing split-M decoded-weight
   reuse topology, with M=512 correctness as the first gate and all eleven rows
   as the promotion gate.
3. Treat any M=16/M=32 static-scheduler candidate as a falsification test for
   the H200 interleave result and require paired H20 evidence before promotion.
4. If startup improves but the large-M rows remain decode-bound, test bounded
   overlap of independent E8M0 table construction/decode with the current
   TMA/WGMMA pipeline.
5. Use partial-A TMA, tail mapping, or grouped register-fed decode only as
   isolated lanes with exact eight-rank correctness and paired controls.
6. Reject per-M kernel copies, extra public formats, dense offline FP8 weights,
   per-fragment WGMMA waits, and any result inferred from H200 timings alone.
