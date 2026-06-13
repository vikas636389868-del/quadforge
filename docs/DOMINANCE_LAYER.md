# QuadForge Dominance Layer

> Implementation of Roadmap Part XI (Competitive Hardening) + Part XIII
> (Dominance Layer Additions).

The Dominance Layer is an opt-in wrapper around the core QuadForge
pipeline that turns "mathematically good" into "production-ready" on
real-world meshes. It is enabled with a single checkbox
(**N-panel → QuadForge → Dominance Layer → Dominance Mode**) and
leaves the core pipeline completely unchanged when disabled.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│  run_dominance_pipeline(input_mesh, params, settings)          │
│                                                                │
│  ┌───────────┐ ┌─────────┐ ┌────────┐ ┌──────────┐ ┌────────┐  │
│  │ repair_   │→│artist_  │→│mesh_   │→│gated     │→│quality_│  │
│  │ mesh      │ │intelli- │ │classi- │ │retry     │ │optim-  │  │
│  │           │ │gence    │ │fier    │ │loop      │ │izer    │  │
│  │§11.2/13.7 │ │§11.4/   │ │§11.3/  │ │§11.6/    │ │§13.3   │  │
│  │           │ │13.2     │ │13.5    │ │13.9      │ │        │  │
│  └───────────┘ └─────────┘ └────────┘ └──────────┘ └────────┘  │
│         │           │          │           │           │      │
│         ▼           ▼          ▼           ▼           ▼      │
│  ┌───────────────────────────────────────────────────────┐    │
│  │              edge_flow_refine (§13.10)                │    │
│  └──────────────────────────┬────────────────────────────┘    │
│                             ▼                                  │
│                    DominanceResult                             │
│              (mesh + confidence + all reports)                 │
└────────────────────────────────────────────────────────────────┘
```

Each arrow is a module in `QuadForge/engine/`. Each stage is independent
and can be disabled without affecting the others.

## Stage reference

### Mesh Repair (`mesh_repair.py`)

Pre-flight sanitization that runs before any other geometry processing.

- **Strip NaN/Inf** vertex positions (clamped to origin).
- **Adaptive-tolerance welding** via spatial hash + union-find. Tolerance
  defaults to `1e-4 × bbox_diagonal`.
- **Degenerate face removal** (topological + area-below-threshold).
- **Duplicate face removal** (winding-agnostic).
- **Isolated vertex removal.**
- **BFS winding consistency** from the largest-area face.
- **Non-manifold vertex splitting** (split each multi-fan vertex into
  one copy per fan).
- **Genus estimate** via Euler characteristic.
- **Voxel rescue path** (`voxel_rescue_remesh`) for severely corrupted
  inputs that fail everything above.

Produces a `RepairReport` with counts, warnings, and a 0–1 confidence
score that propagates into the final Confidence Report.

### Determinism (`determinism.py`)

Ensures bit-identical outputs across repeated runs of the same input.

```python
from QuadForge.engine import DeterministicContext

with DeterministicContext(seed=42):
    verts, faces = run_pipeline(...)
```

Pins Python/NumPy RNGs, forces `OMP_NUM_THREADS=1` and
`OPENBLAS_NUM_THREADS=1`, and provides:

- `canonicalize_mesh(verts, faces)` — lexicographic vertex + face
  ordering so inputs with shuffled vertices produce identical outputs.
- `mesh_fingerprint(verts, faces)` — SHA-256 hex digest of the
  canonical form; used by the benchmark harness to detect silent
  regression drift.
- `deterministic_sum`, `stable_argsort`, `deterministic_choose` — reduction
  helpers that stay bit-stable regardless of thread count.

### Artist Intelligence (`artist_intelligence.py`)

Semantic layer that biases the solver toward artist-friendly edge flow.

- `classify_mesh` → `MeshClass` with category (`organic`, `hard_surface`,
  `cad`, `scan`, `damaged`) and sub-category (`head`, `body`, `creature`,
  `panel`, `mechanical`, `planar`, `assembly`, `noisy`).
- `detect_semantic_features` → `SemanticFeatures` with cylindrical,
  tubular, flat-patch, concave-ring, and panel-edge detectors. Returns
  per-vertex **singularity attractor** and **repeller** maps that the
  singularity stage consumes as soft constraints.
- Topology template registry: Human Head (eye loops / mouth loops /
  nose bridge / ear rings, +8 suggested singularities), Humanoid Body
  (shoulder / elbow / knee / hip / finger rings, +24), Hard-Surface Panel
  (panel routing / bevel bands / cylinder rings, singularity weight 0.5),
  Architectural Plane (rectilinear grid, singularity weight 0.2),
  Photogrammetry Scan (curvature alignment, +20 singularities), etc.

### Quality Optimizer (`quality_optimizer.py`)

Composite quality score (weighted over quad %, valence error, feature
alignment, symmetry deviation, angle quality, flipped-face penalties)
driving a bounded parameter sweep.

Default 9-candidate sweep:

| Label            | What it changes                              |
|------------------|----------------------------------------------|
| baseline         | (none)                                       |
| denser           | target quad count ×1.15                      |
| sparser          | target quad count ×0.88                      |
| more_adaptive    | curvature adaptivity +0.15                   |
| less_adaptive    | curvature adaptivity −0.15                   |
| harder_snap      | feature snap distance ×1.5                   |
| softer_snap      | feature snap distance ×0.6                   |
| more_smoothing   | smoothing iterations +10                     |
| less_smoothing   | smoothing iterations −5                      |

Each candidate is bounded by `time_budget_seconds`; the optimizer stops
early when the budget is exhausted and automatically rolls back to the
previous best if a candidate fails or degrades the score.

**Note on the signed scaled Jacobian:** `_scaled_jacobian_quad` returns
values in `[-1, +1]`. Bowtie / self-intersecting quads return negative
values at the inverted corner, which is the signal the quality gate
uses to reject flipped output. The sign is derived from the alignment
of each corner's wedge cross product with a reference normal computed
as the sum of the two triangle normals of the quad.

### Confidence & Quality Gates (`confidence.py`)

```python
gates = QualityGates(
    min_quad_percentage=0.95,
    max_flipped_ratio=0.001,
    max_mean_valence_error=0.35,
    max_feature_alignment_error=0.05,
    max_symmetry_deviation=0.08,
    min_scaled_jacobian=0.1,
    reject_non_manifold=True,
)
```

`compute_confidence` returns a 0–100 score weighted as
`15% repair + 55% quality + 30% gates`, plus a per-face **risk map**
(for the UI heatmap overlay), a plain-language summary, and a
recommendation. `plan_retry` maps each failing gate to a concrete
parameter override (e.g. low quad % → denser target + more smoothing;
feature alignment failure → stronger snap) so the retry loop can
converge toward the target confidence.

### Edge Flow Refinement (`edge_flow_refine.py`)

Final polish pass that runs after the main solve but before projection:

1. **Irregular cluster detection** — group irregular vertices
   (|valence − 4| ≥ 1) that are within `0.05 × bbox_diagonal` of each
   other, via union-find.
2. **Doublet removal** — a valence-2 vertex has exactly 2 incident faces
   and both must share the same 2 edges at that vertex, so they share
   3 vertices (the doublet + both neighbours). The merged quad is the
   outer 4-cycle with the doublet removed; winding is preserved by
   inspecting which of {doublet, outer} follows the first neighbour
   in the original face cycle.
3. **Small-loop collapse** — quads shorter than 0.1% of bbox diagonal
   in any direction, or whose shortest edge is less than 20% of the
   longest edge, are collapsed.
4. **Feature-preserving relaxation** — umbrella Laplacian with
   `strength=0.4` for 2 iterations, leaving feature vertices pinned.

Reports before/after valence error and signed min-SJ so the caller
can verify the pass didn't regress the quality score.

### Progressive Preview (`progressive_preview.py`)

Three-tier preview with a `PreviewCache` keyed by a
mesh-and-params fingerprint. Re-running with the same inputs skips
straight to the cached tier; changing any parameter invalidates the
cache.

- **Tier 1 (instant):** cross-field direction lines (`(V, 2, 3)`
  segments), singularity markers, density heatmap, feature edges.
  Runs in < 200 ms even on million-face meshes.
- **Tier 2 (~2 s):** voxel surface-nets extraction at a target quad
  spacing derived from `sqrt(surface_area / target_quad_count)`.
  Not final topology but enough to judge edge flow and density
  before committing to the full solve.
- **Tier 3:** the real pipeline output.

Each tier produces a `PreviewOverlays` bundle that the viewport draw
handler can render.

### Mesh Classifier / Dispatcher (`mesh_classifier.py`)

```python
decision = decide_tier(
    mesh_class,
    num_vertices=V, num_faces=F,
    repair_confidence=repair_report.confidence,
    time_budget_s=60.0,
)
```

Rules (in priority order):

1. User explicitly requested fast preview → **TERTIARY**.
2. Repair confidence < 0.55 → **SECONDARY**.
3. Mesh category is `scan` or `damaged` → **SECONDARY**.
4. More than 5M faces → **SECONDARY** (primary won't finish).
5. More than 2M faces AND primary ETA > time budget → **SECONDARY**.
6. Otherwise → **PRIMARY**.

`dispatch_with_fallback` walks the fallback chain on any exception,
empty output, or unsuccessful tier run, so the user is never left
without output as long as the repair stage produced anything at all.

Primary ETA model: `O(F log F)` with a ~1.8 µs/face constant on a
modern desktop. Secondary runs at ~35% of primary, tertiary at ~5%.

### Dominance Pipeline Integrator (`dominance_pipeline.py`)

```python
from QuadForge.engine import run_dominance_pipeline

result = run_dominance_pipeline(
    input_mesh, params,
    settings=scene.quadforge_settings,
    progress_cb=my_callback,
)
# result: DominanceResult with
#   .vertices, .faces
#   .repair_report, .mesh_class, .semantic_features
#   .dispatch_decision, .tier_attempts
#   .quality_breakdown, .confidence_report
#   .optimizer_result, .edge_flow_report
#   .retries_used, .elapsed_seconds
```

Control flow:

```
repair → canonicalize (if reproducible) → classify →
dispatch → primary/secondary/tertiary solve →
compute quality → check gates →
  if passed or retries exhausted:
    break
  else:
    plan_retry → apply overrides → relax gates → retry
→ (optional) auto_optimizer → edge_flow_refine →
recompute confidence → write result to settings
```

The `_LocalQFParams` dataclass duck-types `bridge.QFParams`
attribute-for-attribute so the dispatcher can hand a real
attribute-access params object to `core_pipeline.run_pipeline`
without having to import `bpy` at module load time (which would
break standalone testing).

## Properties

All Dominance Layer settings live on `scene.quadforge_settings`:

| Property | Type | Default | Roadmap |
|---|---|---|---|
| `enable_dominance_mode` | bool | True | — |
| `enable_mesh_repair` | bool | True | §11.2 |
| `enable_artist_intelligence` | bool | True | §13.2 |
| `enable_auto_optimizer` | bool | False | §13.3 |
| `enable_edge_flow_refine` | bool | True | §13.10 |
| `reproducible_mode` | bool | False | §13.1 |
| `reproducible_seed` | int | 0xC0FFEE | §13.1 |
| `confidence_target` | float 0–100 | 75.0 | §13.9 |
| `time_budget_seconds` | float | 60.0 | §11.10 |

Plus read-only `last_confidence_score`, `last_mesh_category`,
`last_solver_tier`, `last_retries_used`, `last_repair_welded`,
`last_repair_fixed`, `last_gate_status` populated after each solve.

## Testing

The Dominance Layer modules are standalone-importable without `bpy`:

```python
import sys, types
qf = types.ModuleType('QuadForge'); qf.__path__ = ['QuadForge']
sys.modules['QuadForge'] = qf
from QuadForge.engine import (
    repair_mesh, classify_mesh, compute_quality_score,
    compute_confidence, QualityGates, refine_edge_flow,
    decide_tier, run_progressive_preview,
)
```

Smoke tests verified in v41.0.0:

- Cube with near-duplicate vertex: welded=1, confidence=0.99.
- Broken 7v/7f mesh (inconsistent winding + duplicates + degenerates):
  reduced to 5v/3f, welded=1, degen=2, dup=2, confidence=0.89.
- Icosphere: zero changes, confidence=1.00, classified as
  organic/creature with elongation=1.00.
- Unit quad: quality composite=0.800, min_sj=1.000, flipped=0.
- Bowtie quad: **min_sj=-1.000, flipped=100%** (verifies the signed-SJ
  fix — this was broken in the initial draft).
- `_LocalQFParams` with overrides: target_quad_count override,
  symmetry tuple construction, every attribute access works.
