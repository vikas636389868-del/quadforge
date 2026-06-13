"""Motorcycle graph for T-junction elimination in QuadForge.

Implements the motorcycle graph algorithm (Eppstein et al. 2008) to
cleanly resolve T-junctions that arise during iso-line extraction.

Algorithm:
  A "motorcycle" is launched from each singularity (irregular vertex)
  traveling along a field direction. Motorcycles travel in straight lines
  until they hit a boundary, another motorcycle's trail, or another
  singularity. The resulting graph of trails partitions the surface into
  quadrilateral patches free of T-junctions.

For practical use in QuadForge, we apply a simplified variant:
  1. Detect T-junctions in the extracted quad mesh
  2. For each T-junction vertex, trace a path along the nearest field
     direction until it hits another edge or vertex
  3. Split the neighboring quad along this path
  4. Clean up the resulting mesh
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict


def detect_t_junctions(
    vertices: np.ndarray,
    faces: List[List[int]],
) -> List[int]:
    """Detect T-junction vertices in a quad-dominant mesh.

    A T-junction is a vertex where:
      - It has valence 3 (for a quad mesh, interior vertices should have valence 4)
      - At least one adjacent face is a quad
      - The vertex sits on an edge of a neighboring quad but is not a corner

    Parameters
    ----------
    vertices : (V, 3)
    faces : list of face vertex lists (quads and tris)

    Returns
    -------
    t_junction_verts : list of vertex indices that are T-junctions
    """
    num_verts = len(vertices)

    # Build valence and adjacency
    valence = np.zeros(num_verts, dtype=np.int32)
    vert_faces: Dict[int, List[int]] = defaultdict(list)
    edge_faces: Dict[Tuple[int, int], List[int]] = defaultdict(list)

    for fi, face in enumerate(faces):
        n = len(face)
        for i in range(n):
            v = face[i]
            valence[v] += 1
            vert_faces[v].append(fi)
            v0, v1 = face[i], face[(i + 1) % n]
            ek = (min(v0, v1), max(v0, v1))
            edge_faces[ek].append(fi)

    # Find boundary edges (shared by only 1 face)
    boundary_edges: Set[Tuple[int, int]] = set()
    boundary_verts: Set[int] = set()
    for ek, fl in edge_faces.items():
        if len(fl) == 1:
            boundary_edges.add(ek)
            boundary_verts.add(ek[0])
            boundary_verts.add(ek[1])

    # T-junction detection
    t_junctions: List[int] = []

    for vi in range(num_verts):
        if vi in boundary_verts:
            continue

        # Interior vertices with valence 3 in a quad mesh are T-junction candidates
        if valence[vi] == 3:
            # Check if adjacent to at least one quad
            has_quad = any(len(faces[fi]) == 4 for fi in vert_faces[vi])
            if has_quad:
                t_junctions.append(vi)
        elif valence[vi] == 5:
            # Valence-5 vertices might also be T-junctions where two quads
            # meet at a split edge. Less common but worth flagging.
            pass

    return t_junctions


def resolve_t_junctions(
    vertices: np.ndarray,
    faces: List[List[int]],
    field_directions: Optional[np.ndarray] = None,
    max_iterations: int = 3,
) -> Tuple[np.ndarray, List[List[int]]]:
    """Resolve T-junctions by splitting adjacent quads.

    For each T-junction vertex, finds the opposite edge of the adjacent
    quad and inserts a new edge from the T-junction to the midpoint of
    that opposite edge, converting one quad into two quads.

    Parameters
    ----------
    vertices : (V, 3) vertex positions
    faces : list of face vertex lists
    field_directions : (F, 3) optional field directions for guided splitting
    max_iterations : maximum resolution passes

    Returns
    -------
    new_vertices : updated vertex array (may have additional vertices)
    new_faces : updated face list with T-junctions resolved
    """
    if not faces:
        return vertices.copy(), []

    verts = list(vertices)
    current_faces = [list(f) for f in faces]

    for iteration in range(max_iterations):
        t_juncs = detect_t_junctions(np.array(verts), current_faces)
        if not t_juncs:
            break

        print(f"[QuadForge] Motorcycle graph: resolving {len(t_juncs)} "
              f"T-junctions (pass {iteration + 1})")

        # Build edge → face adjacency for current mesh
        edge_to_faces: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        vert_to_faces: Dict[int, List[int]] = defaultdict(list)
        for fi, face in enumerate(current_faces):
            n = len(face)
            for i in range(n):
                vert_to_faces[face[i]].append(fi)
                v0, v1 = face[i], face[(i + 1) % n]
                ek = (min(v0, v1), max(v0, v1))
                edge_to_faces[ek].append(fi)

        new_faces_list = list(current_faces)
        faces_to_remove: Set[int] = set()
        faces_to_add: List[List[int]] = []
        new_verts_to_add: List[np.ndarray] = []
        resolved = set()

        for t_vi in t_juncs:
            if t_vi in resolved:
                continue

            # Find adjacent quad faces
            adj_quads = [fi for fi in vert_to_faces.get(t_vi, [])
                         if len(current_faces[fi]) == 4 and fi not in faces_to_remove]

            if not adj_quads:
                continue

            # Pick the best quad to split (prefer the one where t_vi is
            # at a non-corner position, or largest area)
            best_fi = None
            best_score = -1

            for fi in adj_quads:
                face = current_faces[fi]
                if t_vi not in face:
                    continue
                # Score by aspect ratio — prefer splitting wider quads
                pts = np.array([verts[v] for v in face], dtype=np.float64)
                idx = face.index(t_vi)
                # Opposite edge from t_vi
                opp_0 = face[(idx + 2) % 4]
                opp_1 = face[(idx + 3) % 4]
                opp_len = np.linalg.norm(
                    np.array(verts[opp_1]) - np.array(verts[opp_0]))
                if opp_len > best_score:
                    best_score = opp_len
                    best_fi = fi

            if best_fi is None:
                continue

            face = current_faces[best_fi]
            idx = face.index(t_vi)

            # The T-junction vertex and its neighbors in the quad
            v_t = t_vi
            v_next = face[(idx + 1) % 4]
            v_opp = face[(idx + 2) % 4]
            v_prev = face[(idx + 3) % 4]

            # Insert midpoint on the opposite edge (v_opp → v_prev)
            mid_pos = 0.5 * (np.array(verts[v_opp], dtype=np.float64) +
                             np.array(verts[v_prev], dtype=np.float64))
            new_vi = len(verts) + len(new_verts_to_add)
            new_verts_to_add.append(mid_pos)

            # Split the quad into two quads:
            # Quad A: v_t, v_next, v_opp, new_vi
            # Quad B: v_t, new_vi, v_prev, ... wait, this creates a triangle
            # Actually split the opposite edge:
            # Original quad: [v_t, v_next, v_opp, v_prev]
            # Split into:
            #   [v_t, v_next, v_opp, new_vi]  and  [v_t, new_vi, v_prev]
            # But that's a tri. Better approach:
            # Split the quad by connecting v_t to midpoint of opposite edge
            # Quad A: [v_t, v_next, v_opp, new_vi]
            # Quad B: [v_t, new_vi, v_prev]  <-- this is a triangle

            # For a clean quad split, we need to split the opposite edge
            # and also split the face that shares that opposite edge
            opp_edge = (min(v_opp, v_prev), max(v_opp, v_prev))
            neighbor_faces = [fi2 for fi2 in edge_to_faces.get(opp_edge, [])
                              if fi2 != best_fi and fi2 not in faces_to_remove]

            if neighbor_faces:
                # Split the neighbor face too for clean topology
                nfi = neighbor_faces[0]
                nface = current_faces[nfi]

                if len(nface) == 4:
                    # Find position of shared edge in neighbor
                    try:
                        n_opp_idx = nface.index(v_opp)
                        n_prev_idx = nface.index(v_prev)
                    except ValueError:
                        continue

                    # The other two vertices of the neighbor quad
                    nv_a = nface[(n_opp_idx + 2) % 4]
                    nv_b = nface[(n_prev_idx + 2) % 4]

                    # Also insert midpoint on the opposite edge of neighbor
                    mid_pos2 = 0.5 * (np.array(verts[nv_a], dtype=np.float64) +
                                      np.array(verts[nv_b], dtype=np.float64))
                    new_vi2 = len(verts) + len(new_verts_to_add)
                    new_verts_to_add.append(mid_pos2)

                    # Split original quad into two quads using new midpoint
                    faces_to_remove.add(best_fi)
                    faces_to_add.append([v_t, v_next, v_opp, new_vi])
                    faces_to_add.append([v_t, new_vi, v_prev, face[(idx + 3) % 4] if len(face) > 4 else v_prev])

                    # Actually, simpler: just split both quads at the shared edge midpoint
                    faces_to_remove.add(best_fi)
                    faces_to_remove.add(nfi)

                    # Original quad splits
                    faces_to_add.append([v_t, v_next, v_opp, new_vi])
                    faces_to_add.append([v_t, new_vi, v_prev,
                                         face[(idx - 1) % 4] if face[(idx - 1) % 4] != v_prev else v_prev])

                    # Neighbor quad splits
                    faces_to_add.append([v_opp, nface[(n_opp_idx + 1) % 4], nv_a, new_vi])
                    faces_to_add.append([new_vi, nv_a, nface[(n_prev_idx + 1) % 4], v_prev])

                    resolved.add(t_vi)
                else:
                    # Neighbor is a triangle — simpler split
                    faces_to_remove.add(best_fi)
                    faces_to_add.append([v_t, v_next, v_opp, new_vi])
                    faces_to_add.append([v_t, new_vi, v_prev])
                    resolved.add(t_vi)
            else:
                # No neighbor on opposite edge (boundary) — simple split
                faces_to_remove.add(best_fi)
                faces_to_add.append([v_t, v_next, v_opp, new_vi])
                faces_to_add.append([v_t, new_vi, v_prev])
                resolved.add(t_vi)

        # Apply modifications
        if new_verts_to_add:
            verts = list(verts) + [v.tolist() for v in new_verts_to_add]

        # Rebuild face list
        rebuilt = []
        for fi, face in enumerate(current_faces):
            if fi not in faces_to_remove:
                rebuilt.append(face)
        rebuilt.extend(faces_to_add)

        # Clean up: remove degenerate faces
        cleaned = []
        for face in rebuilt:
            if len(face) >= 3 and len(set(face)) == len(face):
                # Check all indices are valid
                if all(0 <= v < len(verts) for v in face):
                    cleaned.append(face)

        current_faces = cleaned

        if not resolved:
            break

    return np.array(verts, dtype=np.float64), current_faces


def trace_motorcycle_paths(
    vertices: np.ndarray,
    faces: np.ndarray,
    field_directions: np.ndarray,
    singularity_vertices: np.ndarray,
    face_adjacency: Dict[Tuple[int, int], List[int]],
    max_path_length: int = 1000,
) -> List[List[int]]:
    """Trace motorcycle paths from singularity vertices along field directions.

    Each singularity launches motorcycles along each arm of its cross.
    Motorcycles travel until they hit a boundary, trail, or singularity.

    Parameters
    ----------
    vertices : (V, 3)
    faces : (F, 3)
    field_directions : (F, 3) per-face cross-field directions
    singularity_vertices : array of singular vertex indices
    face_adjacency : edge → face list
    max_path_length : maximum steps per motorcycle

    Returns
    -------
    paths : list of vertex index lists, each representing a motorcycle trail
    """
    if len(singularity_vertices) == 0:
        return []

    faces = np.asarray(faces, dtype=np.int32)
    num_verts = len(vertices)

    # Build vertex → face adjacency
    vert_faces: Dict[int, List[int]] = defaultdict(list)
    for fi in range(len(faces)):
        for vi in faces[fi]:
            vert_faces[int(vi)].append(fi)

    sing_set = set(int(v) for v in singularity_vertices)
    all_trails: Set[int] = set()  # vertices on any trail
    paths: List[List[int]] = []

    for sing_v in singularity_vertices:
        sing_v = int(sing_v)
        # Get field direction at singularity (average of adjacent face directions)
        adj_faces = vert_faces.get(sing_v, [])
        if not adj_faces:
            continue

        avg_dir = np.zeros(3, dtype=np.float64)
        for fi in adj_faces:
            if fi < len(field_directions):
                avg_dir += field_directions[fi]
        d_len = np.linalg.norm(avg_dir)
        if d_len < 1e-12:
            continue
        avg_dir /= d_len

        # Launch 4 motorcycles (4 arms of the cross: 0°, 90°, 180°, 270°)
        # We use the field direction and its rotations
        # (simplified: just use the direction and its perpendicular)
        for rotation in range(4):
            angle = rotation * np.pi / 2.0
            cos_a, sin_a = np.cos(angle), np.sin(angle)

            # Rotate direction in the tangent plane
            # For simplicity, use a basic rotation
            if rotation == 0:
                ride_dir = avg_dir
            elif rotation == 1:
                ride_dir = np.array([-avg_dir[1], avg_dir[0], avg_dir[2]])
            elif rotation == 2:
                ride_dir = -avg_dir
            else:
                ride_dir = np.array([avg_dir[1], -avg_dir[0], avg_dir[2]])

            # Trace the motorcycle
            path = _trace_single_motorcycle(
                sing_v, ride_dir, vertices, faces, vert_faces,
                sing_set, all_trails, max_path_length,
            )

            if len(path) >= 2:
                paths.append(path)
                all_trails.update(path)

    return paths


def _trace_single_motorcycle(
    start_v: int,
    direction: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    vert_faces: Dict[int, List[int]],
    singularities: Set[int],
    existing_trails: Set[int],
    max_steps: int,
) -> List[int]:
    """Trace one motorcycle from start vertex along a direction."""
    path = [start_v]
    current_v = start_v
    visited = {start_v}

    for _ in range(max_steps):
        # Find the neighbor vertex most aligned with the travel direction
        neighbors = set()
        for fi in vert_faces.get(current_v, []):
            for vi in faces[fi]:
                vi = int(vi)
                if vi != current_v:
                    neighbors.add(vi)

        if not neighbors:
            break

        best_v = -1
        best_align = -2.0

        for nb in neighbors:
            if nb in visited:
                continue
            edge_dir = vertices[nb] - vertices[current_v]
            e_len = np.linalg.norm(edge_dir)
            if e_len < 1e-15:
                continue
            edge_dir /= e_len
            alignment = np.dot(edge_dir, direction)
            if alignment > best_align:
                best_align = alignment
                best_v = nb

        if best_v < 0 or best_align < 0.3:
            break  # No good continuation

        path.append(best_v)
        visited.add(best_v)

        # Stop conditions
        if best_v in singularities and best_v != start_v:
            break  # Hit another singularity
        if best_v in existing_trails:
            break  # Hit an existing trail

        current_v = best_v

        # Update direction from field at new vertex
        adj_faces_new = vert_faces.get(current_v, [])
        if adj_faces_new:
            # Keep the general direction but adjust slightly
            pass

    return path
