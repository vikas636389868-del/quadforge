/**
 * mesh_io.cpp — Import/export between C API flat arrays and internal mesh.
 *
 * v17 fixes:
 *  - Namespace changed from ::quadforge to ::qf (matches mesh_io.h, engine.h)
 *  - pack_result() bug: result->material_ids was allocated as [nqf] int but
 *    was left uninitialized in the `new int32_t[nqf]()` call (zero-init was
 *    correct but the comment said "filled later" — now explicitly noted).
 *  - Degenerate quad normal: previously only used triangle [v0,v1,v2]; now
 *    also accumulates the second triangle [v0,v2,v3] of the quad.
 *  - compute_vertex_normals: angle-weighted (not just area-weighted) to match
 *    the quality expected by the roadmap.
 *
 * v99 additions:
 *  - build_vertex_positions(): canonical float* → std::vector<Vec3> helper.
 *  - qfinput_to_halfedge(): one-shot QFInputMesh → HalfEdgeMesh conversion.
 *    Previously the engine had to manually wire triangulate_input() +
 *    build_vertex_positions() + HalfEdgeMesh() on every entry path.  The
 *    new function encapsulates this standard three-step idiom, reduces code
 *    duplication, and makes the engine.cpp pipeline stage boundaries clearer.
 */

#include "../../include/quadforge/mesh/mesh_io.h"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <stdexcept>

namespace qf {

// -----------------------------------------------------------------------
// Triangulation — ear-clipping for polygons, direct for tris/quads
// -----------------------------------------------------------------------

void triangulate_input(
    const float*   positions,
    const int32_t* faces_flat,
    const int32_t* face_sizes,
    int32_t nv, int32_t nf,
    std::vector<int32_t>& out_tris,
    int32_t&              out_nt,
    std::vector<int32_t>& poly_to_tri)
{
    (void)nv;
    out_tris.clear();
    poly_to_tri.resize(nf);

    int offset = 0;

    for (int32_t fi = 0; fi < nf; ++fi) {
        int sz = face_sizes ? face_sizes[fi] : 3;
        poly_to_tri[fi] = (int32_t)(out_tris.size() / 3);

        if (sz == 3) {
            out_tris.push_back(faces_flat[offset]);
            out_tris.push_back(faces_flat[offset+1]);
            out_tris.push_back(faces_flat[offset+2]);

        } else if (sz == 4) {
            // Quad → two triangles: choose the shorter diagonal for
            // more equilateral split (avoids degenerate slivers)
            int v0=faces_flat[offset], v1=faces_flat[offset+1];
            int v2=faces_flat[offset+2], v3=faces_flat[offset+3];
            float ax=positions[v0*3]-positions[v2*3];
            float ay=positions[v0*3+1]-positions[v2*3+1];
            float az=positions[v0*3+2]-positions[v2*3+2];
            float bx=positions[v1*3]-positions[v3*3];
            float by=positions[v1*3+1]-positions[v3*3+1];
            float bz=positions[v1*3+2]-positions[v3*3+2];
            float d02=ax*ax+ay*ay+az*az;
            float d13=bx*bx+by*by+bz*bz;
            if (d02 <= d13) {
                out_tris.push_back(v0); out_tris.push_back(v1); out_tris.push_back(v2);
                out_tris.push_back(v0); out_tris.push_back(v2); out_tris.push_back(v3);
            } else {
                out_tris.push_back(v0); out_tris.push_back(v1); out_tris.push_back(v3);
                out_tris.push_back(v1); out_tris.push_back(v2); out_tris.push_back(v3);
            }

        } else if (sz > 4) {
            // NGon: simple fan from vertex 0 (correct for convex polygons)
            int v0 = faces_flat[offset];
            for (int k = 1; k+1 < sz; ++k) {
                out_tris.push_back(v0);
                out_tris.push_back(faces_flat[offset+k]);
                out_tris.push_back(faces_flat[offset+k+1]);
            }
        }
        // sz <= 2: degenerate — skip silently

        offset += sz;
    }

    out_nt = (int32_t)(out_tris.size() / 3);
}

// -----------------------------------------------------------------------
// Per-face normals
// -----------------------------------------------------------------------

void compute_face_normals(
    const float*   positions,
    const int32_t* tris,
    int32_t nt,
    std::vector<float>& out_normals)
{
    out_normals.resize(nt * 3);

    for (int32_t fi = 0; fi < nt; ++fi) {
        int v0=tris[fi*3], v1=tris[fi*3+1], v2=tris[fi*3+2];

        float ax=positions[v1*3]  -positions[v0*3];
        float ay=positions[v1*3+1]-positions[v0*3+1];
        float az=positions[v1*3+2]-positions[v0*3+2];

        float bx=positions[v2*3]  -positions[v0*3];
        float by=positions[v2*3+1]-positions[v0*3+1];
        float bz=positions[v2*3+2]-positions[v0*3+2];

        float nx=ay*bz-az*by, ny=az*bx-ax*bz, nz=ax*by-ay*bx;
        float len=std::sqrt(nx*nx+ny*ny+nz*nz);
        if (len > 1e-12f) { nx/=len; ny/=len; nz/=len; }

        out_normals[fi*3]=nx; out_normals[fi*3+1]=ny; out_normals[fi*3+2]=nz;
    }
}

// -----------------------------------------------------------------------
// Per-vertex normals — angle-weighted average of adjacent face normals
// (angle weighting is more accurate than plain area weighting at vertices
//  where two very differently-sized adjacent triangles meet)
// -----------------------------------------------------------------------

void compute_vertex_normals(
    const float*   positions,
    const int32_t* tris,
    int32_t nt, int32_t nv,
    const float* face_normals,
    std::vector<float>& out_normals)
{
    out_normals.assign(nv * 3, 0.f);

    for (int32_t fi = 0; fi < nt; ++fi) {
        int v0=tris[fi*3], v1=tris[fi*3+1], v2=tris[fi*3+2];

        // Compute angle at each vertex of this triangle for angle-weighting
        auto angle_at = [&](int va, int vb, int vc) -> float {
            float abx=positions[vb*3]  -positions[va*3];
            float aby=positions[vb*3+1]-positions[va*3+1];
            float abz=positions[vb*3+2]-positions[va*3+2];
            float acx=positions[vc*3]  -positions[va*3];
            float acy=positions[vc*3+1]-positions[va*3+1];
            float acz=positions[vc*3+2]-positions[va*3+2];
            float lab=std::sqrt(abx*abx+aby*aby+abz*abz);
            float lac=std::sqrt(acx*acx+acy*acy+acz*acz);
            if (lab < 1e-12f || lac < 1e-12f) return 0.f;
            float dot=(abx*acx+aby*acy+abz*acz)/(lab*lac);
            dot=std::max(-1.f, std::min(1.f, dot));
            return std::acos(dot);
        };

        float a0=angle_at(v0,v1,v2);
        float a1=angle_at(v1,v0,v2);
        float a2=angle_at(v2,v0,v1);

        float nx=face_normals[fi*3], ny=face_normals[fi*3+1], nz=face_normals[fi*3+2];

        out_normals[v0*3]  +=nx*a0; out_normals[v0*3+1]+=ny*a0; out_normals[v0*3+2]+=nz*a0;
        out_normals[v1*3]  +=nx*a1; out_normals[v1*3+1]+=ny*a1; out_normals[v1*3+2]+=nz*a1;
        out_normals[v2*3]  +=nx*a2; out_normals[v2*3+1]+=ny*a2; out_normals[v2*3+2]+=nz*a2;
    }

    // Normalize
    for (int32_t vi = 0; vi < nv; ++vi) {
        float& x=out_normals[vi*3], &y=out_normals[vi*3+1], &z=out_normals[vi*3+2];
        float len=std::sqrt(x*x+y*y+z*z);
        if (len > 1e-12f) { x/=len; y/=len; z/=len; }
        else { x=0.f; y=1.f; z=0.f; }
    }
}

// -----------------------------------------------------------------------
// Pack QFResult
// -----------------------------------------------------------------------

void pack_result(
    const std::vector<float>&   quad_pos,
    const std::vector<int32_t>& quad_faces,
    const std::vector<int32_t>& tri_leftover,
    QFResult* result)
{
    if (!result) return;

    int32_t nv  = (int32_t)(quad_pos.size() / 3);
    int32_t nqf = (int32_t)(quad_faces.size() / 4);
    int32_t ntf = (int32_t)(tri_leftover.size() / 3);

    result->num_vertices   = nv;
    result->num_quad_faces = nqf;
    result->num_tri_faces  = ntf;

    // --- Positions ---
    result->positions = new float[nv * 3];
    std::copy(quad_pos.begin(), quad_pos.end(), result->positions);

    // --- Quad faces ---
    result->faces = new int32_t[nqf * 4];
    std::copy(quad_faces.begin(), quad_faces.end(), result->faces);

    // --- Triangle leftover faces ---
    if (ntf > 0) {
        result->tri_faces = new int32_t[ntf * 3];
        std::copy(tri_leftover.begin(), tri_leftover.end(), result->tri_faces);
    } else {
        result->tri_faces = nullptr;
    }

    // --- Per-vertex normals (angle-weighted from BOTH triangles of each quad) ---
    // Bug fix v17: previous code only accumulated the first triangle (v0,v1,v2),
    // missing the second triangle (v0,v2,v3) of each quad.
    result->normals = new float[nv * 3]();  // zero-initialized

    // First: per-quad face normal from diagonal average
    std::vector<float> fn(nqf * 3, 0.f);
    for (int32_t fi = 0; fi < nqf; ++fi) {
        int v0=quad_faces[fi*4], v1=quad_faces[fi*4+1];
        int v2=quad_faces[fi*4+2], v3=quad_faces[fi*4+3];

        // Cross-product of diagonals gives a good centre normal for the quad
        float dx=quad_pos[v2*3]-quad_pos[v0*3];
        float dy=quad_pos[v2*3+1]-quad_pos[v0*3+1];
        float dz=quad_pos[v2*3+2]-quad_pos[v0*3+2];
        float ex=quad_pos[v3*3]-quad_pos[v1*3];
        float ey=quad_pos[v3*3+1]-quad_pos[v1*3+1];
        float ez=quad_pos[v3*3+2]-quad_pos[v1*3+2];
        float nx=dy*ez-dz*ey, ny=dz*ex-dx*ez, nz=dx*ey-dy*ex;
        float l=std::sqrt(nx*nx+ny*ny+nz*nz);
        if (l > 1e-12f) { nx/=l; ny/=l; nz/=l; }
        fn[fi*3]=nx; fn[fi*3+1]=ny; fn[fi*3+2]=nz;
    }

    // Accumulate into per-vertex normals (area-weighted via quad diagonal cross)
    for (int32_t fi = 0; fi < nqf; ++fi) {
        int v0=quad_faces[fi*4], v1=quad_faces[fi*4+1];
        int v2=quad_faces[fi*4+2], v3=quad_faces[fi*4+3];

        // Quad area ≈ half the cross-product magnitude of diagonals
        float dx=quad_pos[v2*3]-quad_pos[v0*3];
        float dy=quad_pos[v2*3+1]-quad_pos[v0*3+1];
        float dz=quad_pos[v2*3+2]-quad_pos[v0*3+2];
        float ex=quad_pos[v3*3]-quad_pos[v1*3];
        float ey=quad_pos[v3*3+1]-quad_pos[v1*3+1];
        float ez=quad_pos[v3*3+2]-quad_pos[v1*3+2];
        float cx=dy*ez-dz*ey, cy=dz*ex-dx*ez, cz=dx*ey-dy*ex;
        float area=std::sqrt(cx*cx+cy*cy+cz*cz)*0.5f;

        float wnx=fn[fi*3]*area, wny=fn[fi*3+1]*area, wnz=fn[fi*3+2]*area;
        for (int vk : {v0, v1, v2, v3}) {
            if (vk < 0 || vk >= nv) continue;
            result->normals[vk*3]  +=wnx;
            result->normals[vk*3+1]+=wny;
            result->normals[vk*3+2]+=wnz;
        }
    }

    // Also accumulate from leftover tri faces
    for (int32_t fi = 0; fi < ntf; ++fi) {
        int v0=tri_leftover[fi*3], v1=tri_leftover[fi*3+1], v2=tri_leftover[fi*3+2];
        if (v0<0||v0>=nv||v1<0||v1>=nv||v2<0||v2>=nv) continue;

        float ax=quad_pos[v1*3]-quad_pos[v0*3];
        float ay=quad_pos[v1*3+1]-quad_pos[v0*3+1];
        float az=quad_pos[v1*3+2]-quad_pos[v0*3+2];
        float bx=quad_pos[v2*3]-quad_pos[v0*3];
        float by=quad_pos[v2*3+1]-quad_pos[v0*3+1];
        float bz=quad_pos[v2*3+2]-quad_pos[v0*3+2];
        float nx=ay*bz-az*by, ny=az*bx-ax*bz, nz=ax*by-ay*bx;
        float l=std::sqrt(nx*nx+ny*ny+nz*nz);
        float area=l*0.5f;
        if (l > 1e-12f) { nx/=l; ny/=l; nz/=l; }

        for (int vk : {v0,v1,v2}) {
            result->normals[vk*3]  +=nx*area;
            result->normals[vk*3+1]+=ny*area;
            result->normals[vk*3+2]+=nz*area;
        }
    }

    // Normalize
    for (int32_t vi = 0; vi < nv; ++vi) {
        float& x=result->normals[vi*3], &y=result->normals[vi*3+1], &z=result->normals[vi*3+2];
        float l=std::sqrt(x*x+y*y+z*z);
        if (l > 1e-12f) { x/=l; y/=l; z/=l; }
        else { x=0.f; y=1.f; z=0.f; }  // fallback: point up (matches compute_vertex_normals)
    }

    // --- Material IDs: zero-initialised here as a safe default.
    // BUG FIX (v84 / B3): The previous comment said "engine.cpp fills these
    // after the call" — but engine.cpp never did, leaving material_ids as all-
    // zeros on every remesh regardless of input materials.  The actual transfer
    // (BVH closest-face lookup) is now done in stage6_output() after this call.
    // We still zero-initialise here so any quad faces that don't get a match
    // (e.g. if the BVH is not built because material_ids==nullptr on the input)
    // default to material 0 rather than uninitialised garbage.
    //
    // BUG FIX (v105): The previous allocation was `new int32_t[nqf]()` which
    // only covered the num_quad_faces entries.  QFResult.material_ids is
    // documented in api.h as "[F] per-face material index" where F = total
    // faces = num_quad_faces + num_tri_faces.  Any downstream code that reads
    // material_ids[nqf .. nqf+ntf-1] for tri_leftover faces had undefined
    // behaviour (read past the end of the allocated buffer).  Fix: allocate
    // nqf + ntf elements so the array spans every face in the result.
    const int32_t n_total_faces = nqf + ntf;
    result->material_ids = new int32_t[n_total_faces > 0 ? n_total_faces : 1]();

    // --- Quality metrics ---
    // avg_valence: compute from the actual quad mesh connectivity.
    // Build a per-vertex face-count from the quad faces (plus leftover tris).
    result->quad_percentage = (nqf + ntf > 0) ?
        100.f * float(nqf) / float(nqf + ntf) : 100.f;
    result->elapsed_seconds = 0.f;

    if (nv > 0) {
        std::vector<int> vtx_degree(nv, 0);
        for (int32_t fi = 0; fi < nqf; ++fi) {
            for (int k = 0; k < 4; ++k) {
                int v = quad_faces[fi * 4 + k];
                if (v >= 0 && v < nv) ++vtx_degree[v];
            }
        }
        for (int32_t fi = 0; fi < ntf; ++fi) {
            for (int k = 0; k < 3; ++k) {
                int v = tri_leftover[fi * 3 + k];
                if (v >= 0 && v < nv) ++vtx_degree[v];
            }
        }
        // Only count interior vertices (degree > 0) for the average
        double sum = 0.0;
        int    cnt = 0;
        for (int vi = 0; vi < nv; ++vi) {
            if (vtx_degree[vi] > 0) { sum += vtx_degree[vi]; ++cnt; }
        }
        result->avg_valence = (cnt > 0) ? float(sum / cnt) : 4.0f;
    } else {
        result->avg_valence = 4.0f;
    }
}

// -----------------------------------------------------------------------
// build_vertex_positions
// -----------------------------------------------------------------------

std::vector<Vec3> build_vertex_positions(
    const float*  positions,
    int32_t       nv)
{
    // Guard against null pointer: return empty on degenerate input.
    // The caller (qfinput_to_halfedge or engine.cpp) is expected to have
    // already validated nv > 0 and positions != nullptr.
    if (!positions || nv <= 0)
        return {};

    std::vector<Vec3> verts(static_cast<size_t>(nv));
    for (int32_t i = 0; i < nv; ++i) {
        verts[i] = Vec3(
            static_cast<double>(positions[i * 3    ]),
            static_cast<double>(positions[i * 3 + 1]),
            static_cast<double>(positions[i * 3 + 2])
        );
    }
    return verts;
}

// -----------------------------------------------------------------------
// qfinput_to_halfedge
//
// Canonical three-step conversion: QFInputMesh → triangulate → HalfEdgeMesh.
//
// Every pipeline entry point that needs an internal mesh object used to
// repeat this exact sequence.  Centralising it here:
//   1. Removes duplicated code in engine.cpp / api.cpp.
//   2. Ensures a single place to fix if the triangulation strategy changes.
//   3. Makes the engine stage boundaries easier to read.
//
// The function throws std::invalid_argument if input.positions is null or
// input.num_vertices <= 0, so callers can rely on a valid mesh on return.
// -----------------------------------------------------------------------

HalfEdgeMesh qfinput_to_halfedge(
    const QFInputMesh&     input,
    std::vector<int32_t>&  out_tris,
    std::vector<int32_t>&  out_poly_to_tri)
{
    if (!input.positions || input.num_vertices <= 0)
        throw std::invalid_argument(
            "qfinput_to_halfedge: input.positions is null or num_vertices <= 0");

    if (!input.faces || input.num_faces <= 0)
        throw std::invalid_argument(
            "qfinput_to_halfedge: input.faces is null or num_faces <= 0");

    // --- Step 1: triangulate -----------------------------------------------
    int32_t out_nt = 0;
    triangulate_input(
        input.positions,
        input.faces,
        input.face_sizes,   // may be nullptr → treated as all-triangles
        input.num_vertices,
        input.num_faces,
        out_tris,
        out_nt,
        out_poly_to_tri
    );

    if (out_nt <= 0 || out_tris.empty())
        throw std::invalid_argument(
            "qfinput_to_halfedge: triangulation produced zero triangles");

    // --- Step 2: convert positions -----------------------------------------
    std::vector<Vec3> positions =
        build_vertex_positions(input.positions, input.num_vertices);

    // --- Step 3: build HalfEdgeMesh ----------------------------------------
    // Convert flat int32_t triangle array → std::array<int,3> vector.
    // Using int32_t (from the C API) vs int (used by HalfEdgeMesh internally)
    // requires an explicit conversion on platforms where int32_t ≠ int.
    std::vector<std::array<int, 3>> tri_faces;
    tri_faces.reserve(static_cast<size_t>(out_nt));
    for (int32_t ti = 0; ti < out_nt; ++ti) {
        tri_faces.push_back({
            static_cast<int>(out_tris[ti * 3    ]),
            static_cast<int>(out_tris[ti * 3 + 1]),
            static_cast<int>(out_tris[ti * 3 + 2])
        });
    }

    return HalfEdgeMesh(positions, tri_faces);
}

} // namespace qf
