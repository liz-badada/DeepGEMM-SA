# Humming and Marlin SASS evidence for SM90 MXFP4 MegaMoE

_DeepSeek-V4-Flash H20 evidence note, captured 2026-09-13_

## Scope

This note converts Humming and Marlin implementations into bounded CAKE search
inputs for the exact `H=4096`, `IH=2048`, `E=256`, `topk=6`, `world_size=8`
campaign. It distinguishes reusable machine-code mechanisms from incompatible
kernel semantics. All clones, compilers, caches, cubins, disassemblies, and
logs were confined to
`/lustre/raplab/client/jinyanc/workspace/jinyanc/campaigns/sm90-mxfp4-megamoe`
on H20-GPU-05.

The inspected revisions are:

- Humming `ba6ed5b36c43a8253ecce514f6404355af39041e`.[^1]
- vLLM `82a85dc1d2d5b3ad4453eaa1bd7596de888a66b3`, including its current
  Marlin MXFP4 specializations.[^2]
- Original Marlin `1f25790bdd49fba53106164a24666dade68d7c90`.[^3]

## Applicability gates

Humming has the closest arithmetic body. The controlled compile request used
FP8 E4M3 activations, E2M1 weights, fused E8M0 group scales with group width
32, `N=K=4096`, and 256 indexed experts. Upstream nevertheless leaves
`Sm90H20Heuristics.b4_allowed_dtypes` empty. Temporarily enabling E2M1 allowed
JIT compilation. Its benchmark helper first added a stale dimension to the
already two-dimensional dynamic-token scale, and its SM90 checked tuning then
forced scale-major `(1, M)` storage even though the public indexed compute
configuration remained row-major. An evidence-only wrapper corrected those
two layout mismatches without changing the quantized values or kernel.

The corrected single-GPU H20 probe executed the indexed kernel for small M at
`N=K=4096`, 256 experts, and `topk=6`:

| M | 8 | 16 | 32 | 64 | 128 | 256 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Humming latency (ms) | 0.1570 | 0.2603 | 0.4223 | 0.6749 | 0.8443 | 0.9889 |

An isolated same-GPU M=8 A/B measured 0.1585 ms for the selected
`BN512/BK64`, three-stage, three-CTA-per-SM body and 0.1902 ms for the
64-register `BN128/BK256`, four-stage, two-CTA-per-SM body. The wider-N,
shallower-K body is about 16.7% lower latency in Humming despite using more
registers. This rejects register count alone as an occupancy proxy and
strengthens the separate BN512/BK64 search axis, without predicting its
effect inside DeepGEMM's distributed persistent protocol.

This establishes that the selected cubins are dynamically executable on H20. It is not a
correctness result and is not comparable with the eight-rank persistent
DeepGEMM contract, so it remains mechanism evidence rather than a campaign
latency denominator. Humming's generic result saver also attempted an
unrelated `m256n64k32` synthetic WGMMA TOPS kernel that SM90 rejects; the
evidence wrapper deliberately reports only the measured GEMM row.

Current vLLM Marlin has genuine E2M1 plus E8M0/group32 instantiations. Its
public MXFP4 linear wrapper is weight-only, however, and its generator omits
FP8 activations on SM90. The generator explains that the SM90 FP8
`mma.sync.m16n8k32` form is simulated with FP16 MMA and provides no
acceleration. A forced SM90 compile below confirms that expansion. This path
must not replace Hopper WGMMA.

Original Marlin is FP16 times INT4 and describes Ampere/Ada rather than
Hopper. Its SM90 compile is useful only for cache-policy, pipeline, and
partitioning evidence.

## Static SASS comparison

Counts are static occurrences in one selected specialization. They are not
dynamic instruction counts. The DeepGEMM row is the current exact Flash M=8
MXFP4 body (`BM8`, `BN256`, four stages); the Humming rows use the same
`N=K=4096`, FP8/E2M1/E8M0 types. Marlin rows have different standalone GEMM
semantics and are deliberately labeled.

| Body | Tile and schedule | REG / stack | Tensor-core body | PRMT | LOP3 | IMAD | IDP.4A | global-to-shared | LDS / STS | BAR / SYNCS |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| DeepGEMM current M=8 | `8x256`, 4-stage persistent | `168 / 0` | 16 QGMMA | 194 | 428 | 782 | 0 | 40 `UTMALDG.2D` | 56 / 61 | 32 / 200 |
| Humming selected M=8 | `8x512x64`, stream-K, 3-stage, 3 CTA/SM | `80 / 0` | 40 QGMMA | 161 | 284 | 475 | 80 | 48 LDGSTS | 100 / 16 | 26 / 0 |
| Humming low-register probe | `8x128x256`, stream-K, 4-stage, 2 CTA/SM | `64 / 0` | 28 QGMMA | 113 | 218 | 453 | 56 | 44 LDGSTS | 110 / 10 | 32 / 0 |
| vLLM Marlin MXFP4 BF16 M=8 | 128 threads, 4-stage | `94 / 32` | 32 HMMA | 38 | 338 | 528 | 0 | 43 LDGSTS | 112 / 37 | 18 / 0 |
| vLLM Marlin forced FP8/MXFP4 M=16 | 128 threads, 4-stage | `120 / 0` | 64 HMMA | 17 | 246 | 596 | 0 | 49 LDGSTS | 130 / 17 | 20 / 0 |
| Original Marlin FP16/INT4 M=16 | 256 threads, 4-stage, group128 | `126 / 0` | 64 HMMA | 0 | 160 | 185 | 0 | 44 LDGSTS | 116 / 32 | 12 / 0 |

The forced vLLM FP8/MXFP4 body also contains 128 FFMA operations. It is direct
machine-code evidence of the SM90 emulation cost and a rejection signal, not a
candidate.

## Transferable mechanisms

### Register-resident E2M1 and E8M0 fusion

Humming's `datatype/dequant_fused.cuh` extracts one of four E8M0 bytes with
DP4A, builds E4M3 exponent bytes arithmetically, selects E2M1 magnitudes with
PRMT, restores signs with LOP3, and swaps the middle packed words for WGMMA's
register operand layout. Its selected body emits 80 `IDP.4A.U8.U8`, compared
with none in the current DeepGEMM M=8 body.

DeepGEMM currently builds a saturating, underflow-safe scaled lookup table for
each E8M0 byte and then uses PRMT/LOP3. Humming's shorter formulation must not
be copied until an exhaustive one-word test proves equality over all packed
E2M1 values and all accepted E8M0 codes, including zero, underflow, overflow,
and the campaign's exact-dequant tolerance. The useful CAKE axis is an
isolated, equivalence-tested selector/scale-construction replacement.

### Small-M topology and occupancy

Humming's selected small-M specialization spends only 80 registers and uses a
wide `BN512`, shallow `BK64`, three-stage stream-K pipeline at three CTA/SM.
The alternative `BN128/BK256` body reaches 64 registers but issues fewer
QGMMA operations and more LDS. This is evidence to search a small-M-specific
topology rather than applying the rejected packed-F16 accumulator change to
all large-M work.

DeepGEMM must preserve one persistent launch, symmetric-buffer ownership, two
GEMM layers, SwiGLU/scatter, and eight-rank synchronization. A CAKE worker may
therefore change only one of `BN/BK`, CTA residency, or work partitioning per
candidate, and must retain the existing routing protocol. Humming's standalone
indexed GEMM timing is not a denominator.

### Streaming weight loads and cache policy

Humming's selected body emits 43
`LDGSTS.E.BYPASS.LTC256B.128` operations. Original Marlin's explicit
`createpolicy.fractional.L2::evict_first` weight loads compile on SM90 to
`LDGSTS.E.BYPASS.LTC128B.128`; its M=16 specialization contains 44 LDGSTS.
This is strong evidence that one-pass packed weights should avoid displacing
reused activations and output/reduction state.

The current persistent body moves packed weights with TMA, whose SASS is 40
plain `UTMALDG.2D` operations. Cache policy cannot be inferred to be
equivalent across CP.ASYNC and TMA. CAKE should first inspect whether the SM90
TMA descriptor/API exposes a supported eviction hint. If not, an isolated
CP.ASYNC packed-weight lane is permissible only when it retains the same
shared-memory layout and overlaps decode without increasing synchronization.

### Striped and stream-K work assignment

Both Marlin variants partition output columns and K slices across CTAs, reduce
partial outputs through shared memory, and use narrow global synchronization
only when a K slice is split. This supports a scheduler experiment that
reduces idle persistent CTAs for sparse experts. It does not justify adopting
Marlin's serial L2 reduction: the campaign must retain DeepGEMM's distributed
combine and scatter semantics.

## CAKE search order

1. Finish live eight-H20 evaluation of the already isolated barrier-striped
   candidate. Do not combine it with a new arithmetic or tile mutation.
2. Add an exhaustive host/device equivalence test for a Humming-style
   DP4A/PRMT E2M1+E8M0 decoder. Materialize a decoder-only candidate only if
   it reduces the exact M=8 SASS arithmetic body without adding registers,
   spills, stack, or divergent branches.
3. Materialize one small-M-only `BN512/BK64` scheduling candidate, preserving
   the current F32 WGMMA accumulator and distributed protocol. Statically gate
   register use and dynamic shared memory before timing.
4. Separately test packed-weight cache behavior: supported TMA eviction policy
   first, CP.ASYNC substitution second. Reject any version that increases the
   existing barrier/SYNCS body without a measured memory-stall reduction.
5. Promote only after exact-dequant correctness and paired same-process
   FP8-versus-MXFP4 timing across all eleven M values. The already measured
   packed-F16 candidate (`0.713x` minimum, `0.780x` geometric mean) remains a
   negative result and must not be used as the next base.

## Artifact inventory

The H20 workspace contains reproducible cubins, SASS, resources, compiler
signatures, and source snapshots under:

- `evidence/humming-sm90/selected-m8` and `lowreg-bk256`.
- `evidence/humming-sm90/probe_humming_h20.py`, `probe-m8-run.log`, and
  `probe-small-m-run.log` for the
  evidence-only single-GPU H20 compatibility run.
- `evidence/humming-sm90/probe_humming_h20_ab.py`,
  `probe-m8-bn512-bk64-ab.log`, and `probe-m8-lowreg-bk256.log` for the
  controlled M=8 topology A/B.
- `evidence/deepgemm-sm90-current-m8`.
- `evidence/vllm-marlin-sm90`, including BF16 MXFP4 and forced FP8/MXFP4.
- `evidence/original-marlin-sm90/fp16-int4-m16-group128.sass`.

[^1]: vLLM Project. "Humming." https://github.com/vllm-project/humming
[^2]: vLLM Project. "vLLM Marlin quantization sources." https://github.com/vllm-project/vllm/tree/main/csrc/quantization/marlin
[^3]: Frantar et al. "Marlin." https://github.com/IST-DASLab/marlin
