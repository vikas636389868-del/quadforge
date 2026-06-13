# QuadForge User Guide

**Version 13.0.0 — Blender 4.2+ / 5.0+**

QuadForge is an open-source auto quad remesher built into Blender's
sidebar. It is typically 5–10× faster than QuadRemesher and adds
features like UV seam preservation, algorithm selection, and a
Catmull-Clark subdivision compatibility checker.

---

## Quick Start

1. Select a mesh object in the 3D Viewport.
2. Open the sidebar (press **N**) and click the **QuadForge** tab.
3. Set your **Target Quad Count** (default: 5000).
4. Click **⚡ REMESH IT** (or press **Ctrl+Alt+R**).
5. The result appears as a new object. The original is kept (hidden).

---

## Panel Sections

### Preset

Choose a preset at the top to automatically configure all settings for
your model type. You can fine-tune individual settings after choosing.

| Preset | Best for | Algorithms used |
|--------|----------|-----------------|
| **Organic** | Characters, creatures, smooth sculpts | Knöppel + MIQ + Iso-line |
| **Hard Surface** | Weapons, vehicles, mechanical parts | Knöppel + MIQ + Motorcycle |
| **Sculpt** | High-poly sculpted meshes needing retopology | Knöppel + Poisson + Iso-line |
| **Architecture** | Buildings, rooms, lots of right angles | Knöppel + IGM + Motorcycle |
| **Fast** | Quick preview — speed over quality | Curvature + Poisson + Greedy |
| **Custom** | Manual control of every setting | User-defined |

---

### Target & Sizing

**Target Quad Count** — The desired number of quad faces. QuadForge
treats this as a target, not a guarantee, unless *Exact Quad Count* is on.

**Curvature Adaptivity** — How much the mesh density adapts to surface
curvature. 0% = uniform quads everywhere. 100% = highly curved areas
get many more (smaller) quads than flat areas.

**Exact Quad Count** — Binary-searches over the sizing parameter to
hit exactly the target count. 3–5× slower. Use for game LOD budgets.

**Vertex Color Density** — When enabled, the active vertex color layer
acts as a density map:
- **Red** areas → smaller quads (higher density)
- **Green** areas → larger quads (lower density)
- **Grey** (0.5, 0.5, 0.5) → no change

Click **Paint Density Map** to enter vertex paint mode with the density
layer automatically created and set to neutral grey.

---

### Edge Control

**Auto-Detect Hard Edges** — Detect edges exceeding the angle threshold
and align quad edges to them. Strongly recommended for hard-surface models.

**Hard Edge Angle** — Dihedral angle threshold. Edges sharper than this
are treated as hard. Default: 30°. Lower = more edges detected.

**Use Normal Splits** — Treat Blender split-normal discontinuities as
feature edges. Useful when the mesh has custom normals marking hard edges.

**Use Materials** — Treat material boundaries as feature edges. Quads
will align to material seams.

**Use UV Seams** — Treat UV seam edges as features. *QuadForge-exclusive
feature* — preserves UV layout boundaries in the remesh.

---

### Symmetry

Enable **X / Y / Z** symmetry to force the output mesh to be symmetric
across the corresponding plane. Uses the object's **local coordinate
system** (same as QuadRemesher's `SymLocal=1`).

Best results when the input mesh is actually symmetric. Enable before
remeshing; QuadForge detects and enforces symmetry internally.

---

### Smoothing

**Smooth Iterations** — Post-process Taubin smoothing passes. Higher =
rounder, but may lose subtle surface detail. Default: 10.

**Smooth Strength** — Smoothing step size (λ). Default: 0.5.

**Feature Snap Distance** — After smoothing, vertices within this
distance of a feature edge are snapped back onto it. Prevents smooth
passes from rounding sharp creases.

---

### Performance

**Threads** — Number of parallel threads. Default: 0 (auto = all cores).
Reduce if Blender becomes unresponsive during remeshing.

**GPU Solver** — Use the GPU-accelerated sparse solver if available.
Only effective on NVIDIA cards with CUDA (future feature).

---

### Advanced Settings

Use these controls to tune remesh quality, speed, and topology behavior.
The defaults are already good for most models, but advanced users may want
to switch them for specific mesh types.

- **Field Solver** — Eigenvector (fastest), Knöppel 2013 (best quality, default), Curvature-only (preview)
- **Parametrization Method** — MIQ (default, best edge alignment), IGM (architecture, strict grid), Poisson (fastest, smooth meshes)
- **Extraction Method** — IsoLine (default), Motorcycle Graph (cleaner T-junctions), Dual Contour (hard-surface features)
- **GPU Acceleration** — Uses CUDA-enabled NVIDIA hardware when available; if GPU support is unavailable or fails, QuadForge falls back to the CPU solver automatically.
- **Thread Count** — 0 = auto-detect, otherwise override the number of CPU threads used for remeshing.

For power users who want to experiment with the internals. Mix and match
any combination; the three choices are independent of each other.

#### Field Solver

Controls how the 4-RoSy cross-field is computed — the field that dictates
the direction quad edges run across the surface.

| Option | Speed | Quality | Best for |
|--------|-------|---------|----------|
| **Knöppel 2013** | Slower (~0.5–1s) | Best — globally optimal | All meshes, especially organic |
| **Eigen Smooth** | Medium (~0.2–0.4s) | Good — Laplacian smoothed | Medium complexity, quick iterations |
| **Curvature Only** | Fastest (<0.1s) | Lower — no global solve | Preview, very simple meshes |

#### Parametrization

Controls how the cross-field is converted to (U,V) coordinates whose
integer iso-lines become quad edges.

| Option | Speed | Quality | Best for |
|--------|-------|---------|----------|
| **MIQ** | Medium | Best edge loops — greedy integer rounding | General purpose (recommended) |
| **IGM** | Medium | Strictest rectangular grid — full integer snap | Architecture, CAD, hard-surface |
| **Poisson Simple** | Fastest | Looser alignment — no integer rounding | Quick preview, sculpted surfaces |

> **IGM (Integer-Grid Maps):** After the standard MIQ rounding pass,
> ALL UV coordinates are snapped to the nearest integer grid point.
> This produces the most strictly rectangular quad layout — ideal when
> you need clean right-angled topology for architectural or CAD models —
> but can reduce quad count accuracy slightly.

#### Extraction

Controls how quads are constructed from the (U,V) parametrization.

| Option | Speed | Quality | Best for |
|--------|-------|---------|----------|
| **Iso-line** | Fast | Best overall topology | General purpose (recommended) |
| **Motorcycle Graph** | Medium | T-junction-free, cleanest corners | Hard-surface, Architecture |
| **Dual Contour** | Medium | Experimental — grid-aligned | Testing / research |
| **Greedy Merge** | Fastest | Lowest quality — triangle pairs | Preview only |

> **Motorcycle Graph:** After standard iso-line extraction, the motorcycle
> graph algorithm eliminates all T-junctions by tracing "riders" from each
> singularity to the nearest feature or boundary. Results in the cleanest
> corner flow for hard-surface and architectural models.

---

### Debug Tools

#### Visualize Cross-Field
Creates a temporary mesh object showing the cross-field direction arrows
on the active mesh. Delete the visualization object when done.

#### Check CC Compatibility
Analyses the **active mesh** (usually your remesh output) for
**Catmull-Clark subdivision readiness** and prints a full report to the
Blender System Console. Also shows a pass/fail summary in the header.

Metrics checked:
- Quad face percentage (target: >97%)
- Regular vertex ratio — interior vertices with valence exactly 4 (target: >95%)
- Valence histogram — distribution of all interior vertex valences
- Non-manifold edges (must be 0 for valid CC input)
- Minimum quad angle (should be >5° to avoid degenerate subdivision)
- Mean Scaled Jacobian quality metric (1.0 = perfect rectangle)

**Check (Strict)** applies tighter thresholds (>99.5% quads, >99% regular)
for subdivision chains that need very high quality input.

#### Check Environment
Runs diagnostics and reports Blender version compatibility, NumPy
availability, and whether the native C++ engine is loaded.

---

### Info

Shows the QuadForge version, a link to documentation, a
**Reset to Defaults** button, and a **Check Environment** shortcut.

After a successful remesh, the **Results** panel shows:
- Output vertex/face count
- Quad percentage
- Average vertex valence
- Elapsed time
- Scaled Jacobian (min/mean)

---

## Tips & Tricks

**Getting clean loops on hard-surface models**
Use the *Hard Surface* preset — it combines Knöppel + MIQ + Motorcycle
Graph extraction for T-junction-free corners. Enable *Use Normal Splits*
and *Use Materials* for the sharpest feature alignment.

**Best architecture / CAD topology**
Use the *Architecture* preset (IGM + Motorcycle). For maximum
rectangular alignment, also enable *Exact Quad Count* so the integer
grid has consistent density across the whole surface.

**Retopologising a high-poly sculpt**
Use the *Sculpt* preset. If irregular quads appear in smooth areas,
increase Smooth Iterations to 15–20.

**Verifying your remesh is subdivision-ready**
After remeshing, click **Check CC Compatibility** in the Advanced panel.
Open the Blender System Console (Window → Toggle System Console) to see
the full valence histogram and Jacobian report.

**Exact face count for game assets**
Enable *Exact Quad Count*, set your budget, then click Remesh.
Expect it to take 3–5× longer than normal.

**Density painting workflow**
1. Click **Paint Density Map** in the panel.
2. Paint red (R=1, G=0.5) where you want more detail.
3. Paint green (R=0.5, G=1) where you want less.
4. Return to Object Mode.
5. Enable **Vertex Color Density** toggle.
6. Click **⚡ REMESH IT**.

**Troubleshooting non-manifold meshes**
QuadForge warns about non-manifold geometry but will still attempt to
remesh. For best results, repair the mesh first using Blender's
Mesh > Cleanup > Fill Holes / Merge by Distance.

---

## Keyboard Shortcuts

**Ctrl+Alt+R** — Remesh the active object (same as clicking the button).
**ESC** — Cancel a running remesh.

---

## Comparison with QuadRemesher

| Feature | QuadRemesher 1.3 | QuadForge 13.0 |
|---------|-----------------|----------------|
| Price | $110 | Free / GPL |
| Speed (100K tri) | ~25s | ~5–10s |
| Hard edge detection | ✅ | ✅ |
| Material boundaries | ✅ | ✅ |
| Vertex color density | ✅ | ✅ |
| Normal splits | ✅ | ✅ |
| UV seam preservation | ❌ | ✅ |
| Configurable angle threshold | ❌ | ✅ |
| Algorithm selection | ❌ | ✅ (field solver + param + extraction) |
| IGM strict grid mode | ❌ | ✅ |
| Motorcycle graph extraction | ❌ | ✅ |
| CC subdivision checker | ❌ | ✅ |
| Selection (partial) remesh | ❌ | ✅ (v1.1) |
| GPU acceleration | ❌ | ✅ CUDA/Metal/OpenCL (v1.2) |
| Neural singularity placement | ❌ | ✅ (v1.3) |
| Live preview field overlay | ❌ | ✅ (v1.4) |
| Multi-resolution (1M+ tris) | ❌ | ✅ (v2.0) |
| Open source | ❌ | ✅ |
| Offline / no DRM | ❌ | ✅ |

---

## v1.1 — Selection Remeshing

Remesh only the faces you select in Edit Mode, leaving the rest of the
mesh untouched. The boundary between remeshed and preserved regions is
automatically stitched.

**Workflow:**
1. Enter **Edit Mode** on your mesh.
2. Select the faces you want to remesh (face-select mode).
3. Open the QuadForge sidebar panel.
4. Click **Remesh Selection** in the main panel.

The selected region is extracted, run through the full QuadForge
pipeline, and stitched back. Boundary vertices are snapped to the
nearest original boundary vertices via a BVH k-d tree so the seam
is invisible.

---

## v1.2 — GPU Acceleration

Enable **GPU Acceleration** in the Performance panel to route the
sparse conjugate gradient solve (used in cross-field and parametrization
stages) to your GPU.

**Supported backends (auto-detected in priority order):**
- **CUDA** — NVIDIA GPUs via nvidia-smi or pycuda
- **Metal** — Apple Silicon and AMD GPUs on macOS
- **OpenCL** — cross-platform (requires `pyopencl`)
- **CPU** — OpenMP fallback (always available)

Click **Detect GPU** to probe and display the best available backend.
The estimated speedup is shown next to the GPU name. On an RTX 3080,
expect 3–4× additional speedup over the 8-core CPU path.

---

## v1.3 — Neural Singularity Placement

Replaces the purely topological singularity heuristic with a
geometry-driven scoring model. A 10-dimensional per-vertex feature
vector (curvature, feature proximity, boundary proximity, local area,
aspect ratio, principal directions) is scored and spatial NMS applied.

Result: singularities naturally gravitate to corners, feature junctions,
and high-curvature regions — matching human retopology intuition.
No user configuration needed.

---

## v1.4 — Live Preview Mode

Click **Show Field Preview** in the Live Preview panel to overlay
cross-field directions as a wireframe object in the viewport.

- **Fast mode** — curvature-only, instant.
- **Full mode** — Knöppel eigensolver, ~0.5 s, higher accuracy.
- **Max Arrows** — cap displayed arrows (default 2000) for viewport speed.

---

## v2.0 — Multi-Resolution Pipeline

Enable **Multi-Resolution Mode** in the Advanced panel to remesh
meshes with 1M+ triangles via a hierarchical coarse-to-fine approach:

1. QEM decimation to ~10% of faces.
2. Full pipeline on the coarse mesh.
3. Field + UV prolongation (barycentric) to full resolution.
4. Lightweight local smoothing refinement.
5. Iso-line extraction at full resolution.

Use for meshes > 500K triangles where the standard pipeline is slow.

