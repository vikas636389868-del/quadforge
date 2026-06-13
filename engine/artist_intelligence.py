"""QuadForge — Artist Intelligence Layer (Roadmap §13.2 + §11.4).

Provides semantic detectors and topology templates that bias the cross-
field solver and singularity placement toward artist-friendly edge flow.

The layer is organised into three tiers:

    1. **Mesh Classifier** — coarse category detection (character / head /
       hard-surface / flat / cloth / scan). Cheap, runs on every mesh.

    2. **Feature Detectors** — locate specific structures (concave rings,
       cylindrical bands, panel cuts, flat patches, tubular limbs).
       Produce soft constraints for the cross-field solver.

    3. **Topology Templates** — when the classifier is confident about a
       category (e.g. "human face"), inject prior singularity locations
       and edge-flow loops (eye loops, mouth loops, shoulder rings…).

All detectors return *soft* constraints — they never override hard user
constraints and they are always safe to ignore if the pipeline is time-
budgeted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ===========================================================================
# 1. Mesh classifier
# ===========================================================================

@dataclass
class MeshClass:
    """Output of the lightweight mesh classifier."""
    category: str = "unknown"        # organic | hard_surface | cad | scan | damaged | unknown
    sub_category: str = ""           # head | body | panel | mechanical | terrain | ...
    confidence: float = 0.0          # 0..1

    # Supporting statistics
    curvature_mean: float = 0.0
    curvature_std: float = 0.0
    flat_fraction: float = 0.0       # fraction of faces below curvature threshold
    sharp_fraction: float = 0.0      # fraction of edges with dihedral > 50°
    elongation: float = 1.0          # bbox aspect ratio
    symmetry_axes: int = 0           # 0..3

    def describe(self) -> str:
        parts = [f"{self.category}"]
        if self.sub_category:
            parts.append(f"/{self.sub_category}")
        parts.append(f" (conf={self.confidence:.2f})")
        return "".join(parts)


def classify_mesh(vertices: np.ndarray,
                  faces: np.ndarray,
                  *,
                  curvature: Optional[np.ndarray] = None,
                  ) -> MeshClass:
    """Classify a mesh into a coarse category using geometric statistics.

    This is deliberately heuristic and cheap — it's meant to pick a
    pipeline, not to do surgical semantic labeling.
    """
    result = MeshClass()

    if len(vertices) == 0 or len(faces) == 0:
        return result

    V = len(vertices)
    F = len(faces)

    # ---- Bounding box + elongation ----
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    extent = bbox_max - bbox_min
    extent_sorted = np.sort(extent)
    if extent_sorted[0] > 1e-9:
        result.elongation = float(extent_sorted[2] / extent_sorted[0])
    else:
        result.elongation = 10.0

    # ---- Face normals & dihedrals ----
    va = vertices[faces[:, 0]]
    vb = vertices[faces[:, 1]]
    vc = vertices[faces[:, 2]]
    n = np.cross(vb - va, vc - va)
    nl = np.linalg.norm(n, axis=1) + 1e-20
    face_normals = n / nl[:, None]

    # Build edge -> face map
    edge_face: Dict[Tuple[int, int], List[int]] = {}
    for fi, tri in enumerate(faces):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            k = (int(min(a, b)), int(max(a, b)))
            edge_face.setdefault(k, []).append(fi)

    sharp = 0
    total_manifold_edges = 0
    dihedrals: List[float] = []
    for flist in edge_face.values():
        if len(flist) != 2:
            continue
        total_manifold_edges += 1
        n0 = face_normals[flist[0]]
        n1 = face_normals[flist[1]]
        cos_a = float(np.clip(np.dot(n0, n1), -1.0, 1.0))
        angle = math.degrees(math.acos(cos_a))
        dihedrals.append(angle)
        if angle > 50.0:
            sharp += 1

    if total_manifold_edges > 0:
        result.sharp_fraction = sharp / total_manifold_edges

    # ---- Curvature-based flat fraction ----
    if curvature is not None and len(curvature) == V:
        k = np.abs(curvature)
        result.curvature_mean = float(np.mean(k))
        result.curvature_std = float(np.std(k))
        thresh = 0.05 * (result.curvature_mean + 1e-9)
        result.flat_fraction = float(np.mean(k < thresh))
    elif dihedrals:
        # Use dihedral stats as curvature proxy
        darr = np.asarray(dihedrals)
        result.curvature_mean = float(np.mean(darr)) / 90.0
        result.curvature_std = float(np.std(darr)) / 90.0
        result.flat_fraction = float(np.mean(darr < 5.0))

    # ---- Symmetry axis count (cheap test: bbox-centred mirror distance) ----
    center = 0.5 * (bbox_min + bbox_max)
    centred = vertices - center
    result.symmetry_axes = _count_symmetry_axes(centred, extent)

    # ---- Category decision ----
    if result.sharp_fraction > 0.08 and result.flat_fraction > 0.3:
        result.category = "hard_surface"
        result.confidence = min(1.0, 0.5 + result.sharp_fraction * 2.0)
        if result.flat_fraction > 0.6:
            result.sub_category = "panel"
        elif result.elongation > 2.5:
            result.sub_category = "mechanical"
    elif result.sharp_fraction > 0.15:
        result.category = "cad"
        result.sub_category = "assembly"
        result.confidence = min(1.0, 0.6 + result.sharp_fraction)
    elif result.flat_fraction > 0.85:
        result.category = "cad"
        result.sub_category = "planar"
        result.confidence = 0.8
    else:
        # Organic default
        result.category = "organic"
        result.confidence = 0.5 + (1.0 - result.sharp_fraction) * 0.4
        if result.elongation < 1.4 and result.symmetry_axes >= 1:
            result.sub_category = "head"
        elif result.elongation > 1.8 and result.symmetry_axes >= 1:
            result.sub_category = "body"
        else:
            result.sub_category = "creature"

    # Scan / damaged override: very high curvature variance usually
    # indicates photogrammetry scan noise
    if result.curvature_std > 4.0 * max(result.curvature_mean, 1e-6):
        result.category = "scan"
        result.sub_category = "noisy"
        result.confidence = 0.6

    return result


def _count_symmetry_axes(centred: np.ndarray, extent: np.ndarray) -> int:
    """Count axes that look approximately mirror-symmetric."""
    if len(centred) < 20:
        return 0
    axes = 0
    scale = float(np.linalg.norm(extent)) + 1e-9
    # Downsample for speed
    n = len(centred)
    stride = max(1, n // 2000)
    sample = centred[::stride]
    for axis in range(3):
        mirrored = sample.copy()
        mirrored[:, axis] = -mirrored[:, axis]
        # For each sampled point, find nearest neighbour in mirrored set
        # via a coarse grid hash (good enough for a heuristic)
        d = _mean_nearest_distance(sample, mirrored, scale)
        if d < 0.02:
            axes += 1
    return axes


def _mean_nearest_distance(a: np.ndarray, b: np.ndarray, scale: float) -> float:
    """Approximate mean nearest-neighbour distance normalised by scale."""
    # Grid hash with cell = scale * 0.05
    cell = scale * 0.05 + 1e-12
    grid: Dict[Tuple[int, int, int], List[int]] = {}
    for i, p in enumerate(b):
        k = (int(p[0] / cell), int(p[1] / cell), int(p[2] / cell))
        grid.setdefault(k, []).append(i)

    total = 0.0
    count = 0
    for p in a:
        ki = int(p[0] / cell)
        kj = int(p[1] / cell)
        kk = int(p[2] / cell)
        best = float("inf")
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    key = (ki + di, kj + dj, kk + dk)
                    if key not in grid:
                        continue
                    for j in grid[key]:
                        d = np.linalg.norm(p - b[j])
                        if d < best:
                            best = float(d)
        if best < float("inf"):
            total += best
            count += 1
    if count == 0:
        return 1.0
    return total / count / scale


# ===========================================================================
# 2. Feature detectors
# ===========================================================================

@dataclass
class SemanticFeatures:
    """Soft constraints produced by the artist intelligence layer."""
    cylindrical_vertices: List[int] = field(default_factory=list)
    tubular_vertices: List[int] = field(default_factory=list)
    flat_patch_vertices: List[int] = field(default_factory=list)
    concave_ring_vertices: List[int] = field(default_factory=list)
    panel_edges: List[Tuple[int, int]] = field(default_factory=list)

    # Bias map: per-vertex [-1..+1] score that pushes the field toward
    # certain orientations when combined with cross_field constraints
    field_bias: Optional[np.ndarray] = None  # (V, 3) direction hint

    # Singularity priors: per-vertex weight used by singularity relocation
    singularity_attractor: Optional[np.ndarray] = None  # (V,) in 0..1
    singularity_repeller: Optional[np.ndarray] = None   # (V,) in 0..1


def detect_semantic_features(vertices: np.ndarray,
                             faces: np.ndarray,
                             *,
                             curvature_k1: Optional[np.ndarray] = None,
                             curvature_k2: Optional[np.ndarray] = None,
                             mesh_class: Optional[MeshClass] = None,
                             ) -> SemanticFeatures:
    """Run all lightweight feature detectors.

    Curvature data is optional; if present it enables the cylindrical /
    flat detectors. Without it the function returns an empty feature set.
    """
    features = SemanticFeatures()
    V = len(vertices)

    attractor = np.zeros(V, dtype=np.float64)
    repeller = np.zeros(V, dtype=np.float64)

    if curvature_k1 is not None and curvature_k2 is not None:
        k1 = np.asarray(curvature_k1, dtype=np.float64)
        k2 = np.asarray(curvature_k2, dtype=np.float64)

        # Cylindrical band: one principal curvature ~ 0, the other large
        # and of consistent sign across the neighborhood.
        abs_k1 = np.abs(k1)
        abs_k2 = np.abs(k2)
        ratio = abs_k2 / (abs_k1 + 1e-9)
        cyl_mask = (abs_k1 > 0.05) & (ratio < 0.1)
        features.cylindrical_vertices = np.nonzero(cyl_mask)[0].tolist()

        # Tubular: small loops have both curvatures similar and positive
        tub_mask = (abs_k1 > 0.1) & (abs_k2 > 0.1) & (np.abs(k1 - k2) < 0.3 * (abs_k1 + abs_k2))
        features.tubular_vertices = np.nonzero(tub_mask)[0].tolist()

        # Flat patches: both curvatures near zero
        flat_thresh = 0.02 * float(np.mean(abs_k1 + abs_k2) + 1e-9)
        flat_mask = (abs_k1 < flat_thresh) & (abs_k2 < flat_thresh)
        features.flat_patch_vertices = np.nonzero(flat_mask)[0].tolist()
        # Flat regions REPEL singularities (ugly to have irregulars in flat areas)
        repeller[flat_mask] += 1.0

        # Concave rings: k1 and k2 both negative and large (e.g. eye sockets)
        concave_mask = (k1 < -0.1) & (k2 < -0.1)
        features.concave_ring_vertices = np.nonzero(concave_mask)[0].tolist()
        # Concave rings ATTRACT singularities (characters expect irregulars here)
        attractor[concave_mask] += 0.7

    # Hard-surface panel edges: run separately below via dihedral scan
    if mesh_class is None or mesh_class.category in ("hard_surface", "cad"):
        features.panel_edges = _detect_panel_edges(vertices, faces)
        # Panels repel singularities along the edge but attract them at
        # panel intersections (corners). The repeller weight is modest
        # because the main constraint system already pins features.
        for a, b in features.panel_edges:
            repeller[a] += 0.3
            repeller[b] += 0.3

    features.singularity_attractor = attractor
    features.singularity_repeller = repeller
    return features


def _detect_panel_edges(vertices: np.ndarray, faces: np.ndarray,
                        *, angle_threshold_deg: float = 35.0) -> List[Tuple[int, int]]:
    """Return all edges whose dihedral angle exceeds the threshold — these
    are the panel lines of hard-surface models."""
    va = vertices[faces[:, 0]]
    vb = vertices[faces[:, 1]]
    vc = vertices[faces[:, 2]]
    n = np.cross(vb - va, vc - va)
    nl = np.linalg.norm(n, axis=1) + 1e-20
    normals = n / nl[:, None]

    edge_face: Dict[Tuple[int, int], List[int]] = {}
    for fi, tri in enumerate(faces):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            k = (int(min(a, b)), int(max(a, b)))
            edge_face.setdefault(k, []).append(fi)

    thresh_cos = math.cos(math.radians(180.0 - angle_threshold_deg))
    panel: List[Tuple[int, int]] = []
    for edge, flist in edge_face.items():
        if len(flist) != 2:
            continue
        cos_a = float(np.clip(np.dot(normals[flist[0]], normals[flist[1]]), -1.0, 1.0))
        if cos_a < thresh_cos:
            panel.append(edge)
    return panel


# ===========================================================================
# 3. Topology templates
# ===========================================================================

@dataclass
class TopologyTemplate:
    """Prior topology pattern — describes where singularities and edge
    loops should ideally sit for a given class of mesh."""
    name: str
    loop_hints: List[str] = field(default_factory=list)
    suggested_singularities: int = 0
    singularity_weighting: float = 1.0  # multiplier for attractor weights


_TEMPLATES: Dict[Tuple[str, str], TopologyTemplate] = {
    ("organic", "head"): TopologyTemplate(
        name="Human Head",
        loop_hints=["eye_loops", "mouth_loops", "nose_bridge", "ear_rings"],
        suggested_singularities=8,
        singularity_weighting=1.2,
    ),
    ("organic", "body"): TopologyTemplate(
        name="Humanoid Body",
        loop_hints=["shoulder_rings", "elbow", "knee", "hip", "finger_rings"],
        suggested_singularities=24,
        singularity_weighting=1.0,
    ),
    ("organic", "creature"): TopologyTemplate(
        name="Generic Creature",
        loop_hints=["joint_loops", "contour_alignment"],
        suggested_singularities=12,
        singularity_weighting=0.9,
    ),
    ("hard_surface", "panel"): TopologyTemplate(
        name="Hard-Surface Panel",
        loop_hints=["panel_routing", "bevel_bands", "cylinder_rings"],
        suggested_singularities=4,
        singularity_weighting=0.5,  # far fewer irregulars allowed
    ),
    ("hard_surface", "mechanical"): TopologyTemplate(
        name="Mechanical Assembly",
        loop_hints=["cylinder_rings", "bolt_holes", "panel_routing"],
        suggested_singularities=6,
        singularity_weighting=0.6,
    ),
    ("cad", "planar"): TopologyTemplate(
        name="Architectural Plane",
        loop_hints=["rectilinear_grid"],
        suggested_singularities=0,
        singularity_weighting=0.2,
    ),
    ("cad", "assembly"): TopologyTemplate(
        name="CAD Assembly",
        loop_hints=["panel_routing", "corner_rings"],
        suggested_singularities=4,
        singularity_weighting=0.5,
    ),
    ("scan", "noisy"): TopologyTemplate(
        name="Photogrammetry Scan",
        loop_hints=["curvature_alignment"],
        suggested_singularities=20,
        singularity_weighting=1.3,  # scans often need more irregulars
    ),
}


def lookup_template(mesh_class: MeshClass) -> Optional[TopologyTemplate]:
    key = (mesh_class.category, mesh_class.sub_category)
    tpl = _TEMPLATES.get(key)
    if tpl is not None:
        return tpl
    # Fall back by category alone
    for (cat, sub), t in _TEMPLATES.items():
        if cat == mesh_class.category:
            return t
    return None


def apply_template_bias(template: TopologyTemplate,
                        features: SemanticFeatures) -> SemanticFeatures:
    """Scale the attractor/repeller maps by the template's weighting and
    return the features object (mutated in place) so the caller can chain.
    """
    if template is None:
        return features
    w = template.singularity_weighting
    if features.singularity_attractor is not None:
        features.singularity_attractor = features.singularity_attractor * w
    if features.singularity_repeller is not None:
        features.singularity_repeller = features.singularity_repeller * (2.0 - w)
    return features


# ===========================================================================
# Top-level convenience
# ===========================================================================

def run_artist_intelligence(vertices: np.ndarray,
                            faces: np.ndarray,
                            *,
                            curvature_k1: Optional[np.ndarray] = None,
                            curvature_k2: Optional[np.ndarray] = None,
                            ) -> Tuple[MeshClass, SemanticFeatures,
                                       Optional[TopologyTemplate]]:
    """Full pipeline: classify + detect + lookup template + apply bias."""
    mesh_class = classify_mesh(vertices, faces,
                               curvature=(curvature_k1 if curvature_k1 is not None
                                          else None))
    features = detect_semantic_features(vertices, faces,
                                        curvature_k1=curvature_k1,
                                        curvature_k2=curvature_k2,
                                        mesh_class=mesh_class)
    template = lookup_template(mesh_class)
    if template is not None:
        features = apply_template_bias(template, features)
    return mesh_class, features, template
