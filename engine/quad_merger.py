import numpy as np


def _quad_normal(pts):
    n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        return None
    return n / norm


def is_convex_quad(pts):
    """Check if a quad is convex by projecting to the dominant plane."""
    if len(pts) != 4:
        return False

    normal = _quad_normal(pts)
    if normal is None:
        return False

    axis = int(np.argmax(np.abs(normal)))
    proj = np.delete(pts, axis, axis=1)

    signs = []
    for i in range(4):
        a = proj[i]
        b = proj[(i + 1) % 4]
        c = proj[(i + 2) % 4]
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(cross) < 1e-12:
            return False
        signs.append(np.sign(cross))

    return all(s > 0 for s in signs) or all(s < 0 for s in signs)


def merge_quads(
    vertices: np.ndarray,
    faces: list,
    face_directions: np.ndarray,
    feature_edges: set | None = None,
    normal_threshold_deg: float = 30.0,
    area_ratio_threshold: float = 2.0,
):
    num_tris = len(faces)
    if num_tris == 0:
        return [], []

    feature_edges = feature_edges or set()
    tri_normals = np.zeros((num_tris, 3), dtype=np.float32)
    tri_areas = np.zeros(num_tris, dtype=np.float32)

    for i, tri in enumerate(faces):
        v0, v1, v2 = vertices[tri]
        cross = np.cross(v1 - v0, v2 - v0)
        area = 0.5 * np.linalg.norm(cross)
        tri_areas[i] = area
        if area > 1e-12:
            tri_normals[i] = cross / (2.0 * area)
        else:
            tri_normals[i] = (0.0, 0.0, 1.0)

    edge_to_tris = {}
    for i, tri in enumerate(faces):
        for j in range(3):
            v0, v1 = tri[j], tri[(j + 1) % 3]
            key = tuple(sorted((v0, v1)))
            edge_to_tris.setdefault(key, []).append(i)

    candidates = []
    normal_threshold = np.radians(normal_threshold_deg)

    for edge, tris in edge_to_tris.items():
        if len(tris) != 2:
            continue
        if edge in feature_edges:
            continue

        t0, t1 = tris
        a0, a1 = tri_areas[t0], tri_areas[t1]
        if min(a0, a1) < 1e-8:
            continue
        if max(a0, a1) / min(a0, a1) > area_ratio_threshold:
            continue

        dot = np.dot(tri_normals[t0], tri_normals[t1])
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        if angle > normal_threshold:
            continue

        tri0 = faces[t0]
        tri1 = faces[t1]
        vA, vB = edge
        third0 = [v for v in tri0 if v not in edge][0]
        third1 = [v for v in tri1 if v not in edge][0]

        quad_options = [
            [vA, third0, vB, third1],
            [vA, third1, vB, third0],
        ]

        best_quad = None
        best_score = -1.0

        for quad in quad_options:
            pts = vertices[quad]
            if not is_convex_quad(pts):
                continue

            edges = np.roll(pts, -1, axis=0) - pts
            lens = np.linalg.norm(edges, axis=1)
            if np.min(lens) < 1e-6:
                continue

            ratio = np.max(lens) / np.min(lens)
            d0 = face_directions[t0]
            d1 = face_directions[t1]
            if np.dot(d0, d1) < 0:
                d1 = -d1
            avg_dir = d0 + d1
            avg_dir /= (np.linalg.norm(avg_dir) + 1e-12)
            longest = edges[np.argmax(lens)]
            longest /= (np.linalg.norm(longest) + 1e-12)
            align = abs(np.dot(longest, avg_dir))
            score = align * (1.0 / ratio)

            if score > best_score:
                best_score = score
                best_quad = quad

        if best_quad is not None:
            candidates.append((best_score, t0, t1, best_quad))

    candidates.sort(key=lambda x: x[0], reverse=True)

    used = set()
    quads = []
    for _, t0, t1, quad in candidates:
        if t0 in used or t1 in used:
            continue
        quads.append(quad)
        used.add(t0)
        used.add(t1)

    remaining = [faces[i] for i in range(num_tris) if i not in used]
    return quads, remaining
