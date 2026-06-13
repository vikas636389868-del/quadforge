# QuadForge Algorithm Reference

Mathematical foundations of every pipeline stage.

---

## Stage 1 — Preprocessing & Feature Detection

### 1.1 Triangulation
Non-triangle faces are split into triangles using ear-clipping: O(F).
Original polygon structure is preserved for material ID transfer.

### 1.2 Half-Edge Construction
Build a half-edge data structure in O(V + F) time using a hash map
from edge pairs (v_lo, v_hi) to half-edge indices.

### 1.3 Feature Edge Detection
For each interior edge shared by triangles t1, t2 with face normals n1, n2:

```
θ = acos(clamp(n1 · n2, -1, 1))   [dihedral angle, radians]
is_feature = θ > threshold
```

The threshold defaults to 30° = π/6 radians.

### 1.4 Sizing Field
Base edge length for uniform remeshing:

```
L_base = sqrt(total_area / target_quad_count)
```

Curvature-adaptive sizing at vertex v with max curvature κ(v) and
adaptivity parameter α ∈ [0,1]:

```
L(v) = L_base / (1 + α · κ(v) · L_base)
```

Vertex color density modulation (matching QuadRemesher's formula):
- Red channel r > 0.5 → `density_mult = 4^(2r - 1)` (finer)
- Green channel g > 0.5 → `density_mult = 0.25^(2g - 1)` (coarser)

---

## Stage 2 — Cross-Field Computation (Knöppel 2013)

### 2.1 4-RoSy Representation
Each face f carries a complex number:

```
u_f = r_f · exp(4i · θ_f)
```

where θ_f is the angle of one cross arm in a local face reference frame,
and r_f is the confidence weight (usually 1.0 for unconstrained faces).

The 4-fold symmetry means exp(4i·(θ + kπ/2)) = exp(4iθ) for integer k,
encoding the 4 arms of the cross without distinguishing them.

### 2.2 Smoothness Energy
The 4-RoSy smoothness energy sums over all interior edges (f1, f2):

```
E = Σ_edges |u_f1 - u_f2 · exp(4i · φ_{f1,f2})|²
```

where φ_{f1,f2} is the parallel transport angle — the rotation needed
to align f2's reference frame to f1's across their shared edge.

Computing φ: for edge e shared by faces f1, f2 with half-edge he:

```
φ_{f1,f2} = angle(frame_f1(e)) - angle(frame_f2(e))
```

This is the **connection Laplacian** — a complex sparse matrix of
size F × F. Minimising E is equivalent to finding the null space of L.

### 2.3 Globally Optimal Solve (Knöppel 2013)
Rather than minimising E with fixed BCs (local minimum prone), we find
the **smallest eigenvector** of L:

```
L · u = λ_min · u
```

This is the globally smoothest possible 4-RoSy field for the given
mesh topology. Numerical method: shift-invert Lanczos iteration
(Spectra library or Eigen's SelfAdjointEigenSolver for small meshes).

For F faces, the solve takes O(F^1.5) time with sparse Cholesky.

### 2.4 Feature Constraints
Hard constraint: face f aligned to feature edge e with direction d:

Add penalty term to the diagonal of L:
```
L[f,f] += λ_penalty · |u_f - d_complex|²
```

where d_complex = exp(4i · atan2(d.y, d.x)).
With λ_penalty = 10^6, this effectively pins u_f to d.

### 2.5 Singularity Detection
For each interior vertex v, compute the **holonomy** — total field
rotation around v's face ring:

```
holonomy(v) = Σ_{faces around v} (field rotation across each face pair)
```

A vertex is singular if |holonomy(v) mod 2π| ≠ 0.
Index = holonomy(v) / (2π), rounded to nearest ±0.25.

By Poincaré-Hopf: Σ_i index(i) = χ(M) / 4

---

## Stage 3 — Global Parametrization

### 3.1 Field Combing
The 4-fold ambiguity is resolved by a BFS from a seed face. At each
edge, choose the matching (out of 4) that minimises the jump:

```
matching = argmin_k |θ_f2 + k·π/2 - θ_f1|
```

Inconsistencies (holonomy ≠ 0 around seam cycles) require seam cuts.

### 3.2 Poisson Parametrization
Given combed field directions X (U) and Y (V) per triangle, find
scalar functions U, V minimising:

```
E[U] = Σ_f area(f) · |∇U_f - X_f|²
E[V] = Σ_f area(f) · |∇V_f - Y_f|²
```

This is two independent Poisson equations. After discretisation
(cotangent weights):

```
L_cot · u = b_U
L_cot · v = b_V
```

where L_cot is the cotangent-weight Laplacian (symmetric positive
semi-definite). Solved with sparse Cholesky (SuiteSparse CHOLMOD
when available, else Eigen SparseLU).

**Adaptive metric**: multiply each triangle's area weight by
`(L_base / L(v))²` where L(v) is the sizing field — this adjusts
the iso-line spacing to match the desired quad size.

### 3.3 Mixed-Integer Rounding (MIQ)
The transition functions across seam cuts must be integer translations.
For each seam edge e with period jump variable p_e:

1. Solve Poisson with current integer constraints.
2. Find p_e closest to an integer (most confident).
3. Round it: `p_e ← round(p_e)`.
4. Update RHS with rank-1 Cholesky update.
5. Repeat until all p_e are integer.

Greedy rounding is O(S) solves where S = number of seam variables.

---

## Stage 4 — Quad Extraction

### 4.1 Iso-Line Tracing
For each triangle t with UV values (u0,v0), (u1,v1), (u2,v2):
- Find all integer U values k between min(u0,u1,u2) and max(u0,u1,u2)
- For each k: find the two triangle edges where U crosses k,
  interpolate to get crossing points, connect them as a segment
- Repeat for V

Adjacent segments are chained into polylines (U-lines and V-lines).

### 4.2 Quad Construction
Each rectangular cell bounded by consecutive U-lines u=k, u=k+1
and V-lines v=j, v=j+1 becomes a quad face.

### 4.3 Motorcycle Graph
Trace riders from each singularity along the field direction until
they crash into a boundary, another singularity, or another rider's
track. The resulting paths resolve all T-junctions:

- Index +1/4 singularity (valence 3) → 3 riders depart
- Index -1/4 singularity (valence 5) → 5 riders depart

---

## Stage 5 — Post-Processing

### 5.1 Taubin λ/μ Smoothing
Standard Laplacian update with weight λ:

```
v'_i = v_i + λ · (1/|N(i)|) · Σ_{j ∈ N(i)} (v_j - v_i)
```

This shrinks the mesh. The Taubin correction applies a second step
with negative weight μ (typically μ = −λ / (λ·k_PB − 1) where
k_PB ≈ 0.1 is the passband cutoff):

```
v''_i = v'_i + μ · Laplacian(v'_i)
```

With λ=0.5, μ=−0.53, the mesh retains its volume while becoming
smoother. The computation is embarrassingly parallel (each vertex
update is independent within a pass).

### 5.2 Surface Projection
After smoothing, project vertex i back to:

```
v_i ← argmin_{p ∈ surface} |v_i − p|
```

Using the BVH, this is O(log F) per vertex. All projections are
computed in parallel with OpenMP.

---

## Quality Metrics

### Scaled Jacobian
For each quad face with corners p0, p1, p2, p3, compute the bilinear
Jacobian J at all four corners:

```
J_corner = [p1−p0, p3−p0] (at corner 0)
```

Scaled Jacobian = det(J) / (||col1|| · ||col2||)

Range: [-1, 1]. Perfect rectangle = 1.0. Degenerate = 0.0.
Inverted quad < 0.

**Target:** min SJ > 0.1, mean SJ > 0.7.

### Feature Alignment Error
For each output edge near a detected feature curve, compute the angle
between the edge direction and the nearest feature curve tangent:

```
error = acos(|e_dir · t_feature|)   [radians]
```

**Target:** mean < 3°, max < 10°.

---

## Integer-Grid Maps (IGM) — Strict Parametrization Mode

The **IGM** parametrization method extends MIQ with a full-mesh integer
snap pass. After the standard greedy MIQ rounding of seam vertices, ALL
UV coordinates are rounded to the nearest integer:

```
u_igm[i] = round(u_miq[i])
v_igm[i] = round(v_miq[i])
```

This enforces the strictest possible integer-grid constraint. Every quad
in the output will be bounded by exact integer iso-lines, producing the
most rectangular topology. The trade-off is a slight reduction in quad
count accuracy (the exact target count is harder to hit when UVs are
fully quantized), which is why *Exact Quad Count* mode pairs well with IGM.

IGM is recommended for architecture, CAD parts, and any mesh where
right-angled quads and strict grid alignment matter more than organic flow.

---

## Catmull-Clark Subdivision Compatibility

The `engine/subdiv.py` module implements a post-remesh compatibility
analyser that checks whether the output quad mesh is suitable as input
to a Catmull-Clark subdivision surface modifier.

### Compatibility Criteria

A quad mesh is fully CC-compatible when:
1. **100% quad faces** — CC subdivision requires all-quad input; any
   triangles or pentagons will degenerate or produce artifacts.
2. **Regular vertices** — Interior vertices with valence ≠ 4 produce
   *extraordinary points* under CC subdivision. These are unavoidable
   at singularities (required by Poincaré-Hopf), but their count should
   be minimised (target: <5% of interior vertices).
3. **No non-manifold edges** — Edges shared by more than two faces
   cause the CC rule to produce undefined results.
4. **No degenerate quads** — Quads with a minimum interior angle
   below 5° will collapse or self-intersect after the first
   subdivision step.
5. **Positive Scaled Jacobian** — All quad face Jacobians must be
   positive (>0) to avoid inverted faces after subdivision.

### QuadForge Quality Targets (Normal Mode)
- Quad percentage ≥ 97%
- Regular vertex ratio ≥ 95%
- Minimum quad angle ≥ 5°
- Scaled Jacobian ≥ 0 (no inverted quads)

### Strict Mode Thresholds
- Quad percentage ≥ 99.5%
- Regular vertex ratio ≥ 99%

Access via the *Check CC Compatibility* button in the Advanced panel,
or call `engine.check_subdiv_compatibility(vertices, faces)` directly.

---

## Motorcycle Graph Extraction

The **Motorcycle** extraction mode drives the `resolve_t_junctions()`
call during extraction (Stage 7) rather than deferring it to Stage 8.
This means the motorcycle graph receives 4 resolution passes immediately
after iso-line tracing, before any cleanup. The result is a mesh whose
T-junctions are resolved as tightly as possible — at the cost of a
slightly higher edge count near singularities.

The Stage 8 pass is skipped when Motorcycle mode was used, avoiding
redundant double-processing.

Recommended for: Hard Surface preset, Architecture preset, any mesh
where clean quad flow around features is the primary concern.
