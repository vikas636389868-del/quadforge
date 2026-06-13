"""Half-edge mesh data structure for QuadForge.

Builds an efficient half-edge representation from flat vertex/face arrays.
Supports:
  - O(1) vertex → outgoing half-edge lookup
  - O(1) face → first half-edge lookup
  - O(valence) vertex one-ring traversal (vertices, faces, edges)
  - O(k) boundary loop enumeration
  - Handles meshes with boundaries and non-manifold edges gracefully
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Core data: stored in flat arrays for speed, wrapped in a class for access
# ---------------------------------------------------------------------------

NONE = -1  # sentinel for "no half-edge / no face / etc."


@dataclass
class HalfEdge:
    """Single half-edge record (stored as struct-of-arrays in HalfEdgeMesh)."""
    twin: int = NONE      # opposite half-edge (NONE if boundary)
    next: int = NONE      # next half-edge in the same face loop
    prev: int = NONE      # previous half-edge in the same face loop
    vertex: int = NONE    # vertex this half-edge points TO (head vertex)
    face: int = NONE      # face to the left (NONE for boundary half-edges)
    edge: int = NONE      # undirected edge index


class HalfEdgeMesh:
    """Half-edge mesh built from vertex positions and face index arrays.

    Parameters
    ----------
    vertices : np.ndarray, shape (V, 3)
        Vertex positions.
    faces : list[list[int]]  or  np.ndarray shape (F, k)
        Face vertex indices (triangles, quads, or mixed).
    """

    def __init__(self, vertices: np.ndarray, faces):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.num_vertices: int = len(self.vertices)

        # Normalise faces to list-of-lists
        if isinstance(faces, np.ndarray):
            if faces.ndim == 2:
                self._face_lists = [list(int(v) for v in row) for row in faces]
            else:
                self._face_lists = [list(int(v) for v in f) for f in faces]
        else:
            self._face_lists = [list(int(v) for v in f) for f in faces]

        self.num_faces: int = len(self._face_lists)

        # --- allocate storage ---
        # Upper bound on half-edges: sum of face sizes * 2 (interior) but we
        # allocate exactly sum(face_sizes) interior half-edges first, then add
        # boundary half-edges.
        total_he = sum(len(f) for f in self._face_lists)

        # Flat arrays (struct-of-arrays layout)
        self._twin = np.full(total_he, NONE, dtype=np.int32)
        self._next = np.full(total_he, NONE, dtype=np.int32)
        self._prev = np.full(total_he, NONE, dtype=np.int32)
        self._vert = np.full(total_he, NONE, dtype=np.int32)  # head vertex
        self._face = np.full(total_he, NONE, dtype=np.int32)
        self._edge = np.full(total_he, NONE, dtype=np.int32)

        # Per-vertex: one outgoing half-edge
        self._vert_he = np.full(self.num_vertices, NONE, dtype=np.int32)
        # Per-face: first half-edge
        self._face_he = np.full(self.num_faces, NONE, dtype=np.int32)

        self.num_halfedges: int = 0
        self.num_edges: int = 0
        self.num_boundary_halfedges: int = 0

        self._build()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build(self):
        """Build the connectivity from face lists."""
        # Pass 1: create interior half-edges for each face
        edge_map: dict[Tuple[int, int], int] = {}  # (tail, head) -> he index
        he_idx = 0

        for fi, face in enumerate(self._face_lists):
            n = len(face)
            if n < 3:
                continue
            first_he = he_idx
            self._face_he[fi] = first_he

            for i in range(n):
                tail = face[i]
                head = face[(i + 1) % n]
                self._vert[he_idx] = head
                self._face[he_idx] = fi

                # next / prev within this face
                self._next[he_idx] = first_he + ((i + 1) % n)
                self._prev[he_idx] = first_he + ((i - 1) % n)

                # Record in edge map
                edge_map[(tail, head)] = he_idx

                # Vertex → outgoing half-edge (from tail)
                if self._vert_he[tail] == NONE:
                    self._vert_he[tail] = he_idx

                he_idx += 1

        self.num_halfedges = he_idx

        # Pass 2: pair twins and assign undirected edge indices
        edge_idx = 0
        paired: set = set()

        for (tail, head), hi in edge_map.items():
            if hi in paired:
                continue
            twin_key = (head, tail)
            if twin_key in edge_map:
                tj = edge_map[twin_key]
                self._twin[hi] = tj
                self._twin[tj] = hi
                self._edge[hi] = edge_idx
                self._edge[tj] = edge_idx
                paired.add(hi)
                paired.add(tj)
            else:
                # Boundary — twin will be created in pass 3
                self._edge[hi] = edge_idx
            edge_idx += 1

        self.num_edges = edge_idx

        # Pass 3: create boundary half-edges for unpaired edges
        boundary_hes: list = []
        boundary_map: dict[Tuple[int, int], int] = {}

        for (tail, head), hi in edge_map.items():
            if self._twin[hi] != NONE:
                continue
            # Create a boundary half-edge going (head → tail)
            bhe_idx = self.num_halfedges + len(boundary_hes)
            self._twin[hi] = bhe_idx
            boundary_map[(head, tail)] = bhe_idx
            boundary_hes.append({
                'twin': hi,
                'vertex': tail,  # head of boundary he = tail of interior he
                'face': NONE,
                'edge': self._edge[hi],
                'tail': head,    # temporary for linking
            })

        if boundary_hes:
            # Extend arrays
            n_bhe = len(boundary_hes)
            old_size = self.num_halfedges
            new_size = old_size + n_bhe

            self._twin = np.resize(self._twin, new_size)
            self._next = np.resize(self._next, new_size)
            self._prev = np.resize(self._prev, new_size)
            self._vert = np.resize(self._vert, new_size)
            self._face = np.resize(self._face, new_size)
            self._edge = np.resize(self._edge, new_size)

            for i, bhe in enumerate(boundary_hes):
                idx = old_size + i
                self._twin[idx] = bhe['twin']
                self._vert[idx] = bhe['vertex']
                self._face[idx] = NONE
                self._edge[idx] = bhe['edge']
                self._next[idx] = NONE  # will be linked below
                self._prev[idx] = NONE

            # Link boundary half-edges: next/prev along boundary loops
            # For boundary he at bhe_idx with tail=v_tail, head=v_head:
            #   next should be the boundary he whose tail == v_head
            # We build a lookup: tail_vertex → boundary_he_index
            tail_to_bhe: dict[int, int] = {}
            for i, bhe in enumerate(boundary_hes):
                tail_v = bhe['tail']
                tail_to_bhe[tail_v] = old_size + i

            for i, bhe in enumerate(boundary_hes):
                idx = old_size + i
                head_v = bhe['vertex']  # where this boundary he points to
                # Next boundary he starts at head_v
                if head_v in tail_to_bhe:
                    nxt = tail_to_bhe[head_v]
                    self._next[idx] = nxt
                    self._prev[nxt] = idx

            self.num_halfedges = new_size
            self.num_boundary_halfedges = n_bhe

            # Update vert_he for boundary vertices to prefer boundary half-edges
            for i, bhe in enumerate(boundary_hes):
                idx = old_size + i
                tail_v = bhe['tail']
                self._vert_he[tail_v] = idx

    # ------------------------------------------------------------------
    # Basic accessors
    # ------------------------------------------------------------------

    def he_twin(self, he: int) -> int:
        return int(self._twin[he])

    def he_next(self, he: int) -> int:
        return int(self._next[he])

    def he_prev(self, he: int) -> int:
        return int(self._prev[he])

    def he_vertex(self, he: int) -> int:
        """Head vertex of this half-edge."""
        return int(self._vert[he])

    def he_tail(self, he: int) -> int:
        """Tail vertex of this half-edge (= head of prev)."""
        return int(self._vert[self._prev[he]])

    def he_face(self, he: int) -> int:
        return int(self._face[he])

    def he_edge(self, he: int) -> int:
        return int(self._edge[he])

    def is_boundary_he(self, he: int) -> bool:
        return self._face[he] == NONE

    def is_boundary_vertex(self, vi: int) -> bool:
        """True if vertex vi is on a mesh boundary."""
        start = self._vert_he[vi]
        if start == NONE:
            return False
        if self.is_boundary_he(start):
            return True
        he = start
        while True:
            twin = self.he_twin(he)
            if twin == NONE:
                return True
            he = self.he_next(twin)
            if he == start:
                break
        return False

    def is_boundary_edge(self, edge_idx: int) -> bool:
        """True if the undirected edge is on a boundary."""
        # Check both half-edges of this edge
        for he in range(self.num_halfedges):
            if self._edge[he] == edge_idx and self._face[he] == NONE:
                return True
        return False

    def face_vertices(self, fi: int) -> List[int]:
        """Return ordered vertex indices of face fi."""
        return list(self._face_lists[fi])

    def face_halfedges(self, fi: int) -> List[int]:
        """Return ordered half-edge indices of face fi."""
        start = int(self._face_he[fi])
        if start == NONE:
            return []
        result = [start]
        he = self.he_next(start)
        while he != start and he != NONE:
            result.append(he)
            he = self.he_next(he)
        return result

    # ------------------------------------------------------------------
    # One-ring traversals
    # ------------------------------------------------------------------

    def vertex_one_ring(self, vi: int) -> List[int]:
        """Return neighboring vertex indices around vi (ordered CCW)."""
        start = self._vert_he[vi]
        if start == NONE:
            return []

        neighbors = []
        he = start

        # If starting on a boundary half-edge, walk to the "first" boundary
        if self.is_boundary_he(start):
            # Walk backwards to find the other boundary
            cur = start
            while True:
                head_v = self.he_vertex(cur)
                neighbors.append(head_v)
                nxt = self.he_next(cur)
                if nxt == NONE or nxt == start:
                    break
                twin = self.he_twin(nxt)
                if twin == NONE:
                    neighbors.append(self.he_vertex(nxt))
                    break
                cur = twin
                if cur == start:
                    break
            return neighbors

        # Interior vertex — circulate via twin/next
        while True:
            neighbors.append(self.he_vertex(he))
            twin = self.he_twin(he)
            if twin == NONE:
                break
            he = self.he_next(twin)
            if he == start:
                break
        return neighbors

    def vertex_faces(self, vi: int) -> List[int]:
        """Return face indices incident to vertex vi."""
        start = self._vert_he[vi]
        if start == NONE:
            return []

        faces_out = []
        he = start
        while True:
            f = self.he_face(he)
            if f != NONE:
                faces_out.append(f)
            twin = self.he_twin(he)
            if twin == NONE:
                break
            he = self.he_next(twin)
            if he == start:
                break
        return faces_out

    def vertex_valence(self, vi: int) -> int:
        """Return the valence (degree) of vertex vi."""
        return len(self.vertex_one_ring(vi))

    # ------------------------------------------------------------------
    # Boundary loops
    # ------------------------------------------------------------------

    def boundary_loops(self) -> List[List[int]]:
        """Return a list of boundary loops, each as ordered vertex indices."""
        visited: Set[int] = set()
        loops: List[List[int]] = []

        for he in range(self.num_halfedges):
            if self._face[he] != NONE:
                continue  # not a boundary half-edge
            if he in visited:
                continue

            loop_verts: List[int] = []
            cur = he
            while cur not in visited:
                visited.add(cur)
                loop_verts.append(self.he_vertex(cur))
                cur = self.he_next(cur)
                if cur == NONE or cur == he:
                    break

            if loop_verts:
                loops.append(loop_verts)

        return loops

    # ------------------------------------------------------------------
    # Edge queries
    # ------------------------------------------------------------------

    def edge_vertices(self, edge_idx: int) -> Optional[Tuple[int, int]]:
        """Return (v0, v1) for undirected edge."""
        for he in range(self.num_halfedges):
            if self._edge[he] == edge_idx:
                v_head = self.he_vertex(he)
                v_tail = self.he_vertex(self.he_twin(he)) if self.he_twin(he) != NONE else self.he_tail(he)
                return (min(v_tail, v_head), max(v_tail, v_head))
        return None

    def edge_key(self, he: int) -> Tuple[int, int]:
        """Return sorted (v0, v1) for the edge this half-edge belongs to."""
        head = self.he_vertex(he)
        tail = self.he_tail(he)
        return (min(tail, head), max(tail, head))

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def face_normal(self, fi: int) -> np.ndarray:
        """Compute the unit normal of face fi (Newell's method for polygons)."""
        verts = self._face_lists[fi]
        n = len(verts)
        normal = np.zeros(3, dtype=np.float64)
        for i in range(n):
            v0 = self.vertices[verts[i]]
            v1 = self.vertices[verts[(i + 1) % n]]
            normal[0] += (v0[1] - v1[1]) * (v0[2] + v1[2])
            normal[1] += (v0[2] - v1[2]) * (v0[0] + v1[0])
            normal[2] += (v0[0] - v1[0]) * (v0[1] + v1[1])
        length = np.linalg.norm(normal)
        if length < 1e-15:
            return np.array([0.0, 0.0, 1.0])
        return normal / length

    def face_area(self, fi: int) -> float:
        """Compute the area of face fi."""
        verts = self._face_lists[fi]
        if len(verts) == 3:
            p0, p1, p2 = self.vertices[verts[0]], self.vertices[verts[1]], self.vertices[verts[2]]
            return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
        elif len(verts) == 4:
            p0, p1, p2, p3 = [self.vertices[v] for v in verts]
            a1 = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
            a2 = 0.5 * np.linalg.norm(np.cross(p2 - p0, p3 - p0))
            return a1 + a2
        else:
            # General polygon — fan triangulation
            total = 0.0
            p0 = self.vertices[verts[0]]
            for i in range(1, len(verts) - 1):
                p1 = self.vertices[verts[i]]
                p2 = self.vertices[verts[i + 1]]
                total += 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
            return total

    def total_surface_area(self) -> float:
        """Sum of all face areas."""
        return sum(self.face_area(fi) for fi in range(self.num_faces))

    def edge_length(self, he: int) -> float:
        """Length of the edge associated with half-edge he."""
        head = self.he_vertex(he)
        tail = self.he_tail(he)
        return float(np.linalg.norm(self.vertices[head] - self.vertices[tail]))

    def face_centroid(self, fi: int) -> np.ndarray:
        """Centroid of face fi."""
        verts = self._face_lists[fi]
        return np.mean(self.vertices[verts], axis=0)
