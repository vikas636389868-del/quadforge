"""QuadForge master pipeline — complete 9-stage orchestrator.

Stages:
  1. Half-edge mesh + topology analysis
  2. Feature edge detection + symmetry analysis
  3. Curvature computation (Rusinkiewicz)
  4. Adaptive sizing field
  5. Cross-field solve (Knöppel 2013, sparse for large meshes)
  6. Singularity optimisation (cancel pairs, relocate to features)
  7. Global parametrization (Poisson + MIQ) + Quad extraction
  8. Motorcycle graph T-junction resolution
  9. Post-processing (Taubin smoothing, surface projection, snapping,
     symmetry enforcement, material/UV/shading transfer, metrics)
"""

from __future__ import annotations

import time
import numpy as np
from typing import Callable, Optional, Set, Tuple

from .halfedge import HalfEdgeMesh
from .topology import analyze_topology, print_topology_report
from .spatial import TriangleBVH
from .sizing import compute_sizing_field, compute_base_edge_length
from .constraints import build_constraints
from .curvature import compute_curvature
from .field import compute_cross_field, smooth_field, _build_face_adjacency, _parallel_transport_angle
from .combing import comb_field, compute_seam_cut
from .parametrize import parametrize_poisson, apply_miq_rounding
from .extraction import trace_isolines, cleanup_quad_mesh
from .quad_merger import merge_quads
from .feature_protection import filter_quads_by_features
from .smoothing import taubin_smooth
from .projection import project_to_surface, iterative_smooth_and_project
from .snapping import snap_to_features
from .transfer import transfer_materials
from .metrics import compute_quality_metrics, print_quality_report
from .symmetry import detect_symmetry, enforce_output_symmetry, mirror_feature_edges
from .singularity import optimise_singularities
from .neural_singularity import neural_optimise_singularities
from .motorcycle import resolve_t_junctions
from .output_enhance import compute_output_normals


ProgressCallback = Callable[[str, float], None]


def _default_progress(stage: str, progress: float):
    print(f"[QuadForge] {stage}: {progress*100:.0f}%")


def run_pipeline(
    input_mesh,
    params,
    obj=None,
    settings=None,
    progress_cb: Optional[ProgressCallback] = None,
) -> Tuple[np.ndarray, list]:
    """Main QuadForge pipeline.

    Parameters
    ----------
    input_mesh : QFInputMesh from bridge.py
    params : QFParams from bridge.py
    obj : bpy.types.Object, optional (only available on main thread)
    settings : QFSettingsPropertyGroup or _SettingsSnapshot
    progress_cb : callable(stage_name, progress_0_to_1)

    Returns (vertices, faces) where faces is a list of index lists.
    """
    cb = progress_cb or _default_progress
    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Dominance Layer fast-path (Roadmap Part XI + XIII).
    # If the user turned on Dominance Mode, hand the whole job off to the
    # integrator which will run repair -> classify -> dispatch ->
    # gated retry -> optimizer -> edge-flow refine. The integrator calls
    # back into *this* function with dominance_mode forced off, so there
    # is no recursion.
    # ------------------------------------------------------------------
    if settings is not None and getattr(settings, "enable_dominance_mode", False):
        # Lazy import avoids circular import at module load time
        from .dominance_pipeline import run_dominance_pipeline_compat
        return run_dominance_pipeline_compat(
            input_mesh, params,
            obj=obj, settings=settings, progress_cb=progress_cb,
        )

    vertices = np.asarray(input_mesh.vertices, dtype=np.float64)
    faces_arr = np.asarray(input_mesh.faces, dtype=np.int32)
    faces_list = [list(int(v) for v in row) for row in faces_arr]

    print(f"[QuadForge] Pipeline started: {len(vertices)} verts, {len(faces_list)} tris")

    # ==================================================================
    # STAGE 1: Half-edge mesh + topology
    # ==================================================================
    cb("Building mesh structure", 0.0)

    he_mesh = HalfEdgeMesh(vertices, faces_list)
    topo = analyze_topology(he_mesh)
    print(print_topology_report(topo))

    if not topo.is_manifold:
        print(f"[QuadForge] WARNING: {len(topo.non_manifold_edges)} non-manifold edges")

    cb("Building mesh structure", 1.0)

    # ==================================================================
    # STAGE 2: Feature detection + symmetry
    # ==================================================================
    cb("Detecting features", 0.0)

    feature_edges: Set[tuple] = set()

    # Try to get features from the _features attribute (set by operator
    # on the main thread before launching the background thread)
    pre_extracted = getattr(input_mesh, '_features', None)

    if pre_extracted is not None:
        feature_edges = pre_extracted.all_features
        norm_count = len(getattr(pre_extracted, 'normal_split_edges', set()))
        print(f"[QuadForge] Features (pre-extracted): {len(feature_edges)} edges "
              f"(hard={len(pre_extracted.hard_edges)}, "
              f"seam={len(pre_extracted.seam_edges)}, "
              f"mat={len(pre_extracted.material_edges)}, "
              f"normals={norm_count})")
    elif obj is not None and settings is not None:
        # Main-thread path (synchronous small meshes)
        from .features import extract_features
        try:
            features = extract_features(obj, settings)
            feature_edges = features.all_features
            norm_count = len(getattr(features, 'normal_split_edges', set()))
            print(f"[QuadForge] Features: {len(feature_edges)} edges "
                  f"(hard={len(features.hard_edges)}, "
                  f"seam={len(features.seam_edges)}, "
                  f"mat={len(features.material_edges)}, "
                  f"normals={norm_count})")
        except Exception as e:
            print(f"[QuadForge] Feature detection skipped: {e}")
    else:
        # Threaded path: no bmesh/bpy available, but we can still use the
        # pre-computed UV seam edges that extract_mesh() embedded in the input.
        if params.use_uv_seams:
            seam_edges = getattr(input_mesh, 'uv_seam_edges', None) or set()
            if seam_edges:
                feature_edges |= seam_edges
                print(f"[QuadForge] UV seam edges from mesh: {len(seam_edges)}")

        # Dihedral-angle hard edges from the half-edge mesh (no bpy needed).
        #
        # BUG-L FIX (v117): the previous implementation used a double loop
        #   for fi in range(F): for fj in range(F): ...
        # which is O(F²) in triangle count.  For a 10 K-triangle mesh this
        # is 100 million Python set-intersection operations — several minutes
        # of frozen Blender UI.  For a 100 K-triangle mesh it is effectively
        # infinite.
        #
        # Fix: build an edge→faces adjacency dict in O(F), then for each
        # edge with exactly 2 incident faces compare only those two normals.
        # Total cost: O(F) dict construction + O(E) dihedral comparisons
        # = O(F) overall (E ≈ 3F/2 by Euler).
        if getattr(params, 'auto_detect_hard_edges', True):
            angle_threshold = getattr(params, 'hard_edge_angle_deg', 30.0)
            try:
                import math
                nf_arr = len(faces_arr)

                # Step 1: vectorised face normals — O(F) with NumPy, no Python loop.
                v0s = vertices[faces_arr[:, 0]]   # (F, 3)
                v1s = vertices[faces_arr[:, 1]]
                v2s = vertices[faces_arr[:, 2]]
                raw_n = np.cross(v1s - v0s, v2s - v0s)  # (F, 3)
                lens  = np.linalg.norm(raw_n, axis=1, keepdims=True)  # (F, 1)
                # Avoid divide-by-zero for degenerate faces (area ≈ 0)
                safe  = (lens[:, 0] > 1e-12)
                face_normals = np.where(lens > 1e-12, raw_n / np.where(lens > 1e-12, lens, 1.0),
                                        0.0)  # (F, 3) — degenerate faces get (0,0,0)

                # Step 2: build edge → [face_index, ...] adjacency — O(F).
                from collections import defaultdict
                edge_to_faces: dict = defaultdict(list)
                for fi in range(nf_arr):
                    for k in range(3):
                        a = int(faces_arr[fi, k])
                        b = int(faces_arr[fi, (k + 1) % 3])
                        key = (a, b) if a < b else (b, a)
                        edge_to_faces[key].append(fi)

                # Step 3: for each interior edge (exactly 2 faces), compare normals.
                hard_count = 0
                cos_thresh = math.cos(math.radians(angle_threshold))
                for (va, vb), flist in edge_to_faces.items():
                    if len(flist) != 2:
                        continue  # boundary or non-manifold — skip
                    fi, fj = flist
                    if not safe[fi] or not safe[fj]:
                        continue  # degenerate face — skip
                    dot = float(np.dot(face_normals[fi], face_normals[fj]))
                    dot = max(-1.0, min(1.0, dot))
                    if dot < cos_thresh:  # angle > threshold
                        feature_edges.add((va, vb))
                        hard_count += 1

                if hard_count:
                    print(f"[QuadForge] Hard edges (dihedral fallback): {hard_count}")
            except Exception as e:
                print(f"[QuadForge] Dihedral fallback skipped: {e}")
        print("[QuadForge] Feature detection: threaded mode — using embedded edge data")

    cb("Detecting features", 0.5)

    # --- Symmetry detection ---
    symmetry = None
    sym_axes = getattr(params, 'symmetry', (False, False, False))
    if isinstance(sym_axes, tuple) and any(sym_axes):
        symmetry = detect_symmetry(vertices, enabled_axes=sym_axes)

        # Mirror feature edges for symmetric enforcement
        if symmetry.active_axes:
            feature_edges = mirror_feature_edges(feature_edges, symmetry)
            print(f"[QuadForge] Feature edges after symmetry mirroring: {len(feature_edges)}")

    cb("Detecting features", 1.0)

    # ==================================================================
    # STAGE 3: Curvature computation
    # ==================================================================
    cb("Computing curvature", 0.0)

    curv_data = compute_curvature(
        vertices, faces_arr,
        vertex_normals=getattr(input_mesh, 'normals', None),
    )
    principal_curvatures = curv_data.max_abs_curvature
    print(f"[QuadForge] Curvature: mean_abs={np.mean(principal_curvatures):.4f}, "
          f"max_abs={np.max(principal_curvatures):.4f}")

    cb("Computing curvature", 1.0)

    # ==================================================================
    # STAGE 4: Sizing field
    # ==================================================================
    cb("Computing sizing field", 0.0)

    total_area = he_mesh.total_surface_area()
    base_edge_len = compute_base_edge_length(total_area, params.target_quad_count)
    print(f"[QuadForge] Total area: {total_area:.4f}, base edge: {base_edge_len:.4f}")

    vertex_colors = getattr(input_mesh, 'vertex_colors', None)
    sizing = compute_sizing_field(
        vertices, faces_list,
        total_area=total_area,
        target_quad_count=params.target_quad_count,
        curvature_adaptivity=params.curvature_adaptivity,
        principal_curvatures=principal_curvatures,
        vertex_colors=vertex_colors if params.use_vertex_colors else None,
        feature_edges=feature_edges if feature_edges else None,
    )
    print(f"[QuadForge] Sizing: avg={np.mean(sizing):.4f}, "
          f"min={np.min(sizing):.4f}, max={np.max(sizing):.4f}")

    cb("Computing sizing field", 1.0)

    # ==================================================================
    # STAGE 5: Cross-field solve
    # ==================================================================
    cb("Computing cross-field", 0.0)

    num_faces = len(faces_arr)
    _field_solver = getattr(params, 'field_solver', 'KNOPPEL')
    # CURVATURE: use curvature-aligned field only (no global solve)
    # EIGEN_SMOOTH: curvature-aligned seed + Laplacian smoothing, no eigensolver
    # KNOPPEL: full globally optimal eigensolver (best quality)
    use_eigensolver = (_field_solver == 'KNOPPEL') and (num_faces < 200000)
    use_curvature_field = (_field_solver == 'CURVATURE')
    use_eigen_smooth = (_field_solver == 'EIGEN_SMOOTH')

    if use_curvature_field:
        print(f"[QuadForge] Field solver: Curvature-only ({num_faces} faces)")
    elif use_eigen_smooth:
        print(f"[QuadForge] Field solver: Eigen Smooth ({num_faces} faces)")
    elif use_eigensolver:
        print(f"[QuadForge] Field solver: Knöppel 2013 eigensolver ({num_faces} faces)")
    else:
        print(f"[QuadForge] Field solver: Knöppel 2013 (large mesh, {num_faces} faces)")

    cross_field = compute_cross_field(
        vertices, faces_arr,
        feature_edges=feature_edges if feature_edges else None,
        curvature_dirs=curv_data.dir1,
        use_eigensolver=use_eigensolver,
    )
    face_dirs = cross_field.field_directions

    # Light post-smoothing; EIGEN_SMOOTH uses heavier smoothing to replace eigensolver
    smooth_iters  = 8 if use_eigen_smooth else 3
    smooth_str    = 0.6 if use_eigen_smooth else 0.3
    face_dirs = smooth_field(
        face_dirs, faces_list, vertices.astype(np.float32),
        iterations=smooth_iters, strength=smooth_str,
        feature_edges=feature_edges,
    )

    cb("Computing cross-field", 1.0)

    # ==================================================================
    # STAGE 6: Singularity optimisation
    # ==================================================================
    cb("Optimising singularities", 0.0)

    sing_verts = cross_field.singularity_vertices
    sing_idx = cross_field.singularity_indices

    if len(sing_verts) > 0:
        sym_pairs = None
        sym_axis = None
        if symmetry and symmetry.active_axes:
            sym_pairs = symmetry.all_pairs
            sym_axis = symmetry.active_axes[0]

        sing_verts, sing_idx = optimise_singularities(
            vertices, faces_arr,
            sing_verts, sing_idx,
            feature_edges=feature_edges if feature_edges else None,
            symmetry_pairs=sym_pairs,
            symmetry_axis=sym_axis,
            base_edge_length=base_edge_len,
        )

    # v1.3 Neural singularity placement — optional blend with topology-driven heuristic
    use_neural = getattr(params, 'use_neural_singularity', False)
    if use_neural and len(sing_verts) > 0:
        neural_blend = float(getattr(params, 'neural_singularity_blend', 0.5))
        try:
            sing_verts, sing_idx = neural_optimise_singularities(
                vertices, faces_arr,
                sing_verts, sing_idx,
                curvature_data=curv_data,
                feature_edges=feature_edges if feature_edges else None,
                base_edge_length=base_edge_len,
                euler_characteristic=getattr(topo, 'euler_characteristic', 2),
                blend=neural_blend,
            )
        except Exception as _neural_err:
            print(f"[QuadForge] Neural singularity fallback: {_neural_err}")

    cb("Optimising singularities", 1.0)

    # ==================================================================
    # STAGE 7: Parametrization + Quad Extraction
    # ==================================================================
    # Respect extraction_method param:
    #   'ISO'        — iso-line tracing (best topology, recommended)
    #   'MOTORCYCLE' — motorcycle-graph extraction (T-junction-free, cleanest corners)
    #   'DUAL'       — dual contouring from UV grid (experimental)
    #   'GREEDY'     — greedy triangle-pair merger (fastest fallback)
    _extraction = getattr(params, 'extraction_method', 'ISO')
    use_isoline    = (_extraction in ('ISO', 'MOTORCYCLE', 'DUAL')) and (num_faces < 500000)
    use_motorcycle = (_extraction == 'MOTORCYCLE')
    use_dual       = (_extraction == 'DUAL')
    quad_verts = None
    quad_faces = None

    if use_isoline:
        cb("Parametrizing surface", 0.0)
        try:
            edge_to_faces = _build_face_adjacency(faces_arr)

            transport_angles = {}
            for edge_key, fl in edge_to_faces.items():
                if len(fl) != 2:
                    continue
                fi, fj = fl
                v0, v1 = edge_key
                shared = vertices[v1] - vertices[v0]
                phi = _parallel_transport_angle(
                    cross_field.face_frames_e1[fi], cross_field.face_frames_e2[fi],
                    cross_field.face_normals[fi],
                    cross_field.face_frames_e1[fj], cross_field.face_frames_e2[fj],
                    cross_field.face_normals[fj],
                    shared,
                )
                transport_angles[edge_key] = phi

            combed_angles, rotation_k, seam_edges = comb_field(
                num_faces, cross_field.field_angles,
                cross_field.face_frames_e1, cross_field.face_frames_e2,
                cross_field.face_normals,
                edge_to_faces, transport_angles, feature_edges,
            )
            print(f"[QuadForge] Combing: {len(seam_edges)} seam edges")

            all_seams = compute_seam_cut(
                num_faces, seam_edges,
                sing_verts,
                edge_to_faces, vertices, faces_arr,
            )
            print(f"[QuadForge] Seam cut: {len(all_seams)} total seam edges")

            cb("Parametrizing surface", 0.5)

            uv = parametrize_poisson(
                vertices, faces_arr, combed_angles,
                cross_field.face_frames_e1, cross_field.face_frames_e2,
                sizing=sizing, seam_edges=all_seams,
            )

            _param_method = getattr(params, 'param_method', 'MIQ')
            if _param_method == 'MIQ':
                uv = apply_miq_rounding(uv, vertices, faces_arr, all_seams)
                print("[QuadForge] Parametrization: MIQ rounding applied")
            elif _param_method == 'IGM':
                # Integer-Grid Maps: apply MIQ rounding with stricter integer snap
                # for the most rectangular grid-aligned output (ideal for architecture/CAD)
                uv = apply_miq_rounding(uv, vertices, faces_arr, all_seams,
                                        strict_integer=True)
                print("[QuadForge] Parametrization: IGM (strict integer-grid snap) applied")
            else:
                print("[QuadForge] Parametrization: Poisson-only (no MIQ rounding)")

            cb("Parametrizing surface", 1.0)

            cb("Extracting quads", 0.0)

            if use_dual:
                # Dual Contour mode: extract quads directly from UV grid intersections
                # Falls back to iso-line if dual extraction yields too few faces
                print("[QuadForge] Extraction: Dual Contour mode")
                try:
                    raw_verts, raw_faces = trace_isolines(vertices, faces_arr, uv,
                                                          dual_contour=True)
                except TypeError:
                    # trace_isolines may not support dual_contour kwarg yet
                    raw_verts, raw_faces = trace_isolines(vertices, faces_arr, uv)
            else:
                raw_verts, raw_faces = trace_isolines(vertices, faces_arr, uv)

            if len(raw_faces) >= 10:
                quad_verts, quad_faces = cleanup_quad_mesh(raw_verts, raw_faces)
                print(f"[QuadForge] Iso-line extraction: {len(quad_faces)} quads, "
                      f"{len(quad_verts)} verts")

                # MOTORCYCLE mode: apply full motorcycle-graph T-junction resolution
                # right here during extraction, before cleanup, for cleanest corners
                if use_motorcycle and len(quad_faces) >= 10:
                    print("[QuadForge] Extraction: Motorcycle graph T-junction elimination")
                    try:
                        # BUG FIX v29: face_dirs has shape (num_input_triangles, 3).
                        # quad_faces are the *output* quads — different count and
                        # different index space.  Passing face_dirs here caused silent
                        # out-of-range reads inside resolve_t_junctions when a quad
                        # face index exceeded len(face_dirs).  Pass None so the
                        # function falls back to its area/aspect-ratio scoring.
                        moto_verts, moto_faces = resolve_t_junctions(
                            quad_verts.astype(np.float32), quad_faces,
                            field_directions=None,
                            max_iterations=4,  # more passes for clean result
                        )
                        if len(moto_faces) >= len(quad_faces) * 0.9:
                            quad_verts = moto_verts
                            quad_faces = moto_faces
                            print(f"[QuadForge] Motorcycle extraction: "
                                  f"{len(quad_faces)} quads after T-junction removal")
                    except Exception as e:
                        print(f"[QuadForge] Motorcycle T-junction elimination skipped: {e}")
            else:
                print("[QuadForge] Iso-line extraction produced too few quads, "
                      "falling back to greedy merger")
                quad_verts = None
                quad_faces = None

            cb("Extracting quads", 1.0)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[QuadForge] Parametrization failed ({e}), falling back to greedy merger")
            quad_verts = None
            quad_faces = None

    # ==================================================================
    # FALLBACK: Greedy quad merger
    # ==================================================================
    if quad_verts is None or quad_faces is None or len(quad_faces) < 10:
        cb("Extracting quads (greedy)", 0.0)

        constraint_weights = build_constraints(
            vertices.astype(np.float32), faces_list, feature_edges,
        )
        quads, remaining = merge_quads(
            vertices.astype(np.float32), faces_list, face_dirs,
            feature_edges=feature_edges,
            normal_threshold_deg=getattr(params, 'hard_edge_angle_deg', 30.0),
        )

        quads = filter_quads_by_features(quads, feature_edges)
        final_faces = [list(f) for f in quads] + [list(t) for t in remaining]
        output_verts = vertices.astype(np.float32)

        print(f"[QuadForge] Greedy merger: {len(quads)} quads, "
              f"{len(remaining)} tris remaining")

        cb("Extracting quads (greedy)", 1.0)
    else:
        final_faces = quad_faces
        output_verts = quad_verts.astype(np.float32)

    # ==================================================================
    # STAGE 8: Motorcycle graph T-junction resolution
    # In 'MOTORCYCLE' extraction mode the motorcycle graph already drove
    # extraction, so we skip the second pass.  In all other iso-line
    # modes we run a cleanup pass here.
    # ==================================================================
    cb("Resolving T-junctions", 0.0)

    run_motorcycle = (not use_motorcycle)  # skip if already used during extraction
    if run_motorcycle and len(final_faces) > 0 and len(output_verts) > 0:
        try:
            # BUG FIX v29: same index-space mismatch as above.  final_faces are
            # the output quad mesh faces; face_dirs is indexed by original input
            # triangle index.  Passing face_dirs here is not meaningful and can
            # produce out-of-range reads.  Pass None so resolve_t_junctions uses
            # its internal topology scoring.
            new_verts, new_faces = resolve_t_junctions(
                output_verts, final_faces,
                field_directions=None,
                max_iterations=2,
            )
            if len(new_faces) >= len(final_faces):
                output_verts = new_verts.astype(np.float32)
                final_faces = new_faces
                print(f"[QuadForge] After T-junction resolution: "
                      f"{len(output_verts)} verts, {len(final_faces)} faces")
        except Exception as e:
            print(f"[QuadForge] T-junction resolution skipped: {e}")

    cb("Resolving T-junctions", 1.0)

    # ==================================================================
    # STAGE 9: Post-processing
    # ==================================================================
    cb("Post-processing", 0.0)

    bvh = None
    if len(output_verts) > 0 and len(faces_arr) > 0:
        try:
            bvh = TriangleBVH(vertices, faces_arr)
        except Exception as e:
            print(f"[QuadForge] BVH build failed: {e}")

    # --- 9a: Iterative smoothing + projection ---
    if params.smooth_iterations > 0 and len(output_verts) > 0 and bvh is not None:
        try:
            output_verts = iterative_smooth_and_project(
                output_verts, final_faces, bvh,
                smooth_iterations=max(1, params.smooth_iterations // 3),
                smooth_strength=params.smooth_strength * 0.7,
                project_blend=0.8,
                num_rounds=3,
            )
            print(f"[QuadForge] Iterative smooth+project: 3 rounds × "
                  f"{max(1, params.smooth_iterations // 3)} iterations")
        except Exception as e:
            print(f"[QuadForge] Iterative smooth+project failed ({e}), "
                  "falling back to simple smoothing")
            constraint_w = np.zeros(len(output_verts), dtype=np.float32)
            output_verts = taubin_smooth(
                output_verts, final_faces,
                iterations=params.smooth_iterations,
                lam=params.smooth_strength,
                mu=-params.smooth_strength * 1.06,
                constraint_weights=constraint_w,
            )
            if bvh is not None:
                output_verts = project_to_surface(output_verts, bvh, blend=1.0)

    cb("Post-processing", 0.3)

    # --- 9b: Feature snapping ---
    if feature_edges and params.feature_snap_distance > 0 and len(output_verts) > 0:
        try:
            output_verts, snapped = snap_to_features(
                output_verts, vertices,
                feature_edges,
                snap_distance=params.feature_snap_distance,
                snap_strength=0.9,
            )
            if snapped:
                print(f"[QuadForge] Feature snapping: {len(snapped)} vertices snapped")
        except Exception as e:
            print(f"[QuadForge] Feature snapping skipped: {e}")

    cb("Post-processing", 0.5)

    # --- 9c: Symmetry enforcement on output ---
    if symmetry and symmetry.active_axes and len(output_verts) > 0:
        try:
            output_verts = enforce_output_symmetry(
                output_verts, symmetry, strength=0.8,
            ).astype(np.float32)
            print(f"[QuadForge] Symmetry enforced on output "
                  f"(axes: {['XYZ'[a] for a in symmetry.active_axes]})")
        except Exception as e:
            print(f"[QuadForge] Symmetry enforcement skipped: {e}")

    cb("Post-processing", 0.6)

    # --- 9d: Material transfer ---
    input_mat_ids = getattr(input_mesh, 'material_ids', None)
    if input_mat_ids is not None and bvh is not None and len(output_verts) > 0:
        try:
            tri_mat_ids = None
            original_faces = getattr(input_mesh, 'original_faces', None)
            if original_faces is not None and len(input_mat_ids) == len(original_faces):
                tri_mat_ids = np.zeros(len(faces_arr), dtype=np.int32)
                tri_idx = 0
                for poly_idx, poly in enumerate(original_faces):
                    n_tris = len(poly) - 2
                    for _ in range(n_tris):
                        if tri_idx < len(tri_mat_ids):
                            tri_mat_ids[tri_idx] = input_mat_ids[poly_idx]
                            tri_idx += 1
            else:
                tri_mat_ids = input_mat_ids

            material_ids = transfer_materials(
                output_verts, final_faces, bvh, tri_mat_ids,
            )
            n_mats = len(np.unique(material_ids))
            if n_mats > 1:
                print(f"[QuadForge] Material transfer: {n_mats} materials")
        except Exception as e:
            print(f"[QuadForge] Material transfer skipped: {e}")

    cb("Post-processing", 0.8)

    # --- 9e: UV transfer ---
    # Transfer UV coordinates from the input mesh to the output mesh using
    # barycentric interpolation via the BVH. Only runs when the input had UVs.
    output_uvs = None
    input_uv_coords = getattr(input_mesh, 'uv_coords', None)
    if input_uv_coords is not None and bvh is not None and len(output_verts) > 0:
        try:
            from .output_enhance import transfer_uvs
            output_uvs = transfer_uvs(
                output_verts, final_faces,
                vertices, faces_arr,
                input_uv_coords, bvh,
            )
            if output_uvs is not None:
                print(f"[QuadForge] UV transfer: {len(output_uvs)} UV coordinates")
        except Exception as e:
            print(f"[QuadForge] UV transfer skipped: {e}")

    # --- 9f: Vertex color transfer ---
    # Transfer vertex colors from the input to the output (density map → output colors).
    output_vertex_colors = None
    input_vc = getattr(input_mesh, 'vertex_colors', None)
    if input_vc is not None and bvh is not None and len(output_verts) > 0:
        try:
            from .output_enhance import transfer_vertex_colors
            output_vertex_colors = transfer_vertex_colors(
                output_verts, vertices, input_vc, faces_arr, bvh,
            )
            if output_vertex_colors is not None:
                print(f"[QuadForge] Vertex color transfer: {len(output_vertex_colors)} colors")
        except Exception as e:
            print(f"[QuadForge] Vertex color transfer skipped: {e}")

    # --- 9g: Output normals ---
    output_normals = None
    if len(output_verts) > 0 and len(final_faces) > 0:
        try:
            output_normals = compute_output_normals(output_verts, final_faces)
        except Exception:
            pass

    cb("Post-processing", 1.0)

    # ==================================================================
    # Quality metrics
    # ==================================================================
    metrics = compute_quality_metrics(output_verts, final_faces, feature_edges)
    print(print_quality_report(metrics))

    elapsed = time.perf_counter() - t_start
    print(f"[QuadForge] Pipeline finished in {elapsed:.3f}s")

    # Attach transfer data as attributes on a result namespace so callers
    # that only unpack (verts, faces) still work, but operators can access
    # the richer data via result_extras.
    class _PipelineResult:
        def __init__(self, verts, faces, uvs, vcolors, normals):
            self.verts = verts
            self.faces = faces
            self.output_uvs = uvs
            self.output_vertex_colors = vcolors
            self.output_normals = normals

        def __iter__(self):
            """Allows unpacking as (verts, faces) for backward compatibility."""
            yield self.verts
            yield self.faces

    return _PipelineResult(output_verts, final_faces, output_uvs, output_vertex_colors, output_normals)


def run_pipeline_exact_count(
    input_mesh,
    params,
    obj=None,
    settings=None,
    progress_cb: Optional[ProgressCallback] = None,
) -> Tuple[np.ndarray, list]:
    """Pipeline wrapper that implements exact quad count mode.

    When params.exact_quad_count is True, runs binary search over
    target_quad_count to match the desired output count.
    """
    from .exact_count import binary_search_quad_count

    target = params.target_quad_count
    cb = progress_cb or _default_progress

    def trial_run(trial_count):
        import copy
        trial_params = copy.copy(params)
        trial_params.target_quad_count = trial_count
        trial_params.exact_quad_count = False

        result = run_pipeline(
            input_mesh, trial_params,
            obj=obj, settings=settings,
            progress_cb=cb,
        )
        # run_pipeline returns a _PipelineResult; support both that and plain tuple
        if hasattr(result, 'verts'):
            verts, faces = result.verts, result.faces
        else:
            verts, faces = result
        actual_quads = sum(1 for f in faces if len(f) == 4)
        return verts, faces, actual_quads

    return binary_search_quad_count(trial_run, target)
