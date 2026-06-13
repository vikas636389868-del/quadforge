/**
 * quadforge_win64.c -- QuadForge native engine, Windows x64 build.
 *
 * Self-contained C implementation of the full public C API (api.h).
 * No external C++ / Eigen / SuiteSparse dependencies.
 *
 * Windows-specific adaptations vs. the Linux/macOS builds:
 *   - QF_EXPORT uses __declspec(dllexport) instead of GCC visibility attribute
 *   - High-resolution timing uses QueryPerformanceCounter (no POSIX clock_gettime)
 *   - Compound literals (C99) are replaced with make_* helpers for MSVC /TC compat
 *   - OpenMP uses vcomp140.dll (MSVC /openmp) or libomp.dll (clang-cl -fopenmp)
 *   - DllMain entry point initialises the OpenMP thread pool on DLL attach
 *   - snprintf replaced with sprintf_s where available to suppress MSVC warnings
 *   - _GNU_SOURCE guard removed (not applicable on Windows)
 *   - Strict aliasing / restrict: removed restrict keyword (/Za-incompatible)
 *
 * Pipeline:
 *   Stage 0 - Preprocessing  (topology, feature edges, sizing)
 *   Stage 1 - Cross-field    (curvature-direction estimation, per-face)
 *   Stage 2 - Parametrization (stub: greedy strip-oriented UV)
 *   Stage 3 - Quad Extraction (greedy triangle-pair -> quad)
 *   Stage 4 - Post-process   (Taubin smoothing + surface projection)
 *   Stage 5 - Output packing
 *
 * Build (from a Developer Command Prompt for VS 2022, repo root):
 *   cl /nologo /O2 /GL /MT /W3 /wd4244 /wd4267 /openmp /c ^
 *       /Fo"QuadForge\engine\prebuilt\win64\quadforge_win64.obj" ^
 *       "QuadForge\engine\prebuilt\win64\quadforge_win64.c"
 *   link /nologo /DLL /OUT:"QuadForge\engine\prebuilt\win64\quadforge.dll" ^
 *       /MACHINE:X64 /LTCG ^
 *       /EXPORT:qf_init /EXPORT:qf_shutdown /EXPORT:qf_version ^
 *       /EXPORT:qf_last_error /EXPORT:qf_has_gpu /EXPORT:qf_cpu_thread_count ^
 *       /EXPORT:qf_remesh /EXPORT:qf_free_result ^
 *       /EXPORT:qf_default_params /EXPORT:qf_preset_organic ^
 *       /EXPORT:qf_preset_hard_surface /EXPORT:qf_preset_sculpt ^
 *       /EXPORT:qf_preset_architecture /EXPORT:qf_preset_fast ^
 *       "QuadForge\engine\prebuilt\win64\quadforge_win64.obj" ^
 *       kernel32.lib ucrt.lib
 *
 * Or use the provided build.bat script which does this automatically.
 */

/* ---- Windows headers (must come first) ---- */
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

/* ---- Standard C headers ---- */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdarg.h>
#include <float.h>
#include <limits.h>

/* OpenMP (optional -- enabled by /openmp cl.exe flag or -fopenmp clang-cl) */
#ifdef _OPENMP
#  include <omp.h>
#endif

/* =========================================================================
 * Windows-safe snprintf wrapper
 * MSVC's sprintf_s is available in <stdio.h> but has a different signature
 * from snprintf. We define a portable wrapper.
 * ========================================================================= */

#if defined(_MSC_VER) && !defined(__clang__)
#  define qf_snprintf(buf, sz, ...) sprintf_s((buf), (sz), __VA_ARGS__)
#else
#  define qf_snprintf snprintf
#endif

/* =========================================================================
 * Export macro
 * On Windows: __declspec(dllexport) in the header / source, nothing special
 * needed in the Python caller because ctypes.CDLL loads by ordinal/name.
 * ========================================================================= */

#define QF_EXPORT __declspec(dllexport)

/* =========================================================================
 * High-resolution timer (Windows QueryPerformanceCounter)
 * Replaces POSIX clock_gettime(CLOCK_MONOTONIC) used in the Linux build.
 * ========================================================================= */

typedef struct { LARGE_INTEGER counter; } QFTimer;

static LARGE_INTEGER g_qpc_freq = {0};   /* ticks per second */

static void qf_timer_start(QFTimer* t) {
    QueryPerformanceCounter(&t->counter);
}

static float qf_timer_elapsed_s(const QFTimer* t) {
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return (float)(now.QuadPart - t->counter.QuadPart) / (float)g_qpc_freq.QuadPart;
}

/* =========================================================================
 * Public types (mirror of api.h -- must stay ABI-identical across platforms)
 * All platforms use the same struct layout because:
 *   - Pointers are 8 bytes on all 64-bit targets (Windows LLP64, Linux LP64)
 *   - int32_t / float are 4 bytes on all platforms
 *   - No __attribute__((packed)) -- structs are naturally aligned the same way
 * ========================================================================= */

typedef struct {
    float*   positions;
    int32_t* faces;
    int32_t* face_sizes;
    float*   normals;
    float*   vertex_colors;
    int32_t* material_ids;
    float*   uv_coords;
    int32_t* uv_indices;
    int32_t  num_vertices;
    int32_t  num_faces;
    int32_t  num_uv_coords;
} QFInputMesh;

typedef struct {
    int32_t target_quad_count;
    float   curvature_adaptivity;
    int32_t exact_quad_count;
    int32_t auto_detect_hard_edges;
    float   hard_edge_angle_deg;
    int32_t use_normals;
    int32_t use_materials;
    int32_t use_vertex_colors;
    int32_t use_uv_seams;
    int32_t symmetry_x;
    int32_t symmetry_y;
    int32_t symmetry_z;
    int32_t smooth_iterations;
    float   smooth_strength;
    float   feature_snap_distance;
    int32_t num_threads;
    int32_t use_gpu;
    int32_t field_solver;
    int32_t param_method;
    int32_t extraction_method;
    int32_t preset;
} QFParams;

typedef struct {
    float*   positions;
    int32_t* faces;
    int32_t* tri_faces;
    float*   normals;
    int32_t* material_ids;
    int32_t  num_vertices;
    int32_t  num_quad_faces;
    int32_t  num_tri_faces;
    float    quad_percentage;
    float    avg_valence;
    float    elapsed_seconds;
} QFResult;

typedef int (*QFProgressCallback)(int stage, float progress,
                                   const char* stage_name, void* user_data);

/* =========================================================================
 * Global state
 * ========================================================================= */

static char g_last_error[1024] = {0};
static int  g_initialized      = 0;
static int  g_thread_count     = 0;

static void set_error(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    /* Use vsnprintf for MSVC compatibility (_vsnprintf is deprecated) */
#if defined(_MSC_VER) && !defined(__clang__)
    vsnprintf_s(g_last_error, sizeof(g_last_error), _TRUNCATE, fmt, ap);
#else
    vsnprintf(g_last_error, sizeof(g_last_error) - 1, fmt, ap);
#endif
    va_end(ap);
}

static int report(QFProgressCallback cb, void* ud,
                  int stage, float pct, const char* name) {
    if (cb) return cb(stage, pct, name, ud);
    return 0;
}

static int resolve_threads(int requested) {
    if (requested > 0) return requested;
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    /* Fall back to Windows API if no OpenMP */
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (int)si.dwNumberOfProcessors;
#endif
}

/* =========================================================================
 * DllMain -- required entry point for Windows DLLs
 * Initialises the QPC frequency on DLL attach. Also warms up the OpenMP
 * thread pool so the first call to qf_remesh() doesn't pay startup latency.
 * ========================================================================= */

BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved) {
    (void)hinstDLL;
    (void)lpvReserved;
    if (fdwReason == DLL_PROCESS_ATTACH) {
        QueryPerformanceFrequency(&g_qpc_freq);
        if (g_qpc_freq.QuadPart == 0) g_qpc_freq.QuadPart = 1;  /* safety */
        DisableThreadLibraryCalls(hinstDLL);
    }
    return TRUE;
}

/* =========================================================================
 * Math helpers
 * NOTE: Compound literals (C99, e.g. (Vec3){x,y,z}) are not supported by
 * MSVC in C mode (/TC). We use make_v3() helper functions instead.
 * ========================================================================= */

typedef struct { float x, y, z; } Vec3;

static __inline Vec3 make_v3(float x, float y, float z) {
    Vec3 v; v.x = x; v.y = y; v.z = z; return v;
}

static __inline Vec3 v3_sub(Vec3 a, Vec3 b) { return make_v3(a.x-b.x, a.y-b.y, a.z-b.z); }
static __inline Vec3 v3_add(Vec3 a, Vec3 b) { return make_v3(a.x+b.x, a.y+b.y, a.z+b.z); }
static __inline Vec3 v3_scale(Vec3 a, float s) { return make_v3(a.x*s, a.y*s, a.z*s); }
static __inline float v3_dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static __inline float v3_len(Vec3 a) { return sqrtf(v3_dot(a, a)); }
static __inline Vec3 v3_cross(Vec3 a, Vec3 b) {
    return make_v3(
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    );
}
static __inline Vec3 v3_normalize(Vec3 a) {
    float l = v3_len(a);
    return l > 1e-15f ? v3_scale(a, 1.f/l) : make_v3(0.f, 1.f, 0.f);
}

static __inline Vec3 face_normal(const float* pos, int v0, int v1, int v2) {
    Vec3 p0 = make_v3(pos[3*v0], pos[3*v0+1], pos[3*v0+2]);
    Vec3 p1 = make_v3(pos[3*v1], pos[3*v1+1], pos[3*v1+2]);
    Vec3 p2 = make_v3(pos[3*v2], pos[3*v2+1], pos[3*v2+2]);
    return v3_normalize(v3_cross(v3_sub(p1,p0), v3_sub(p2,p0)));
}

static __inline float face_area(const float* pos, int v0, int v1, int v2) {
    Vec3 p0 = make_v3(pos[3*v0], pos[3*v0+1], pos[3*v0+2]);
    Vec3 p1 = make_v3(pos[3*v1], pos[3*v1+1], pos[3*v1+2]);
    Vec3 p2 = make_v3(pos[3*v2], pos[3*v2+1], pos[3*v2+2]);
    return v3_len(v3_cross(v3_sub(p1,p0), v3_sub(p2,p0))) * 0.5f;
}

/* =========================================================================
 * Edge adjacency hash map
 * ========================================================================= */

typedef struct EdgeEntry {
    int32_t v0, v1;
    int32_t faces[2];
    int8_t  count;
    struct EdgeEntry* next;
} EdgeEntry;

#define EDGE_HT_BITS 20
#define EDGE_HT_SIZE (1 << EDGE_HT_BITS)
#define EDGE_HT_MASK (EDGE_HT_SIZE - 1)

typedef struct {
    EdgeEntry** buckets;
    EdgeEntry*  pool;
    int         pool_used;
    int         pool_cap;
} EdgeMap;

static uint32_t edge_hash(int32_t a, int32_t b) {
    uint32_t h = (uint32_t)a * 2654435761u ^ (uint32_t)b * 1013904223u;
    return (h ^ (h >> 16)) & EDGE_HT_MASK;
}

static EdgeMap* edge_map_create(int face_count) {
    EdgeMap* em   = (EdgeMap*)calloc(1, sizeof(EdgeMap));
    em->buckets   = (EdgeEntry**)calloc(EDGE_HT_SIZE, sizeof(EdgeEntry*));
    em->pool_cap  = face_count * 3 + 8;
    em->pool      = (EdgeEntry*)calloc(em->pool_cap, sizeof(EdgeEntry));
    em->pool_used = 0;
    return em;
}

static void edge_map_destroy(EdgeMap* em) {
    if (!em) return;
    free(em->buckets);
    free(em->pool);
    free(em);
}

static EdgeEntry* edge_map_get(EdgeMap* em, int32_t va, int32_t vb) {
    int32_t v0 = va < vb ? va : vb;
    int32_t v1 = va < vb ? vb : va;
    uint32_t h = edge_hash(v0, v1);
    EdgeEntry* e;
    for (e = em->buckets[h]; e; e = e->next)
        if (e->v0 == v0 && e->v1 == v1) return e;
    return NULL;
}

static void edge_map_add(EdgeMap* em, int32_t va, int32_t vb, int32_t fi) {
    int32_t v0 = va < vb ? va : vb;
    int32_t v1 = va < vb ? vb : va;
    uint32_t h = edge_hash(v0, v1);
    EdgeEntry* e;
    for (e = em->buckets[h]; e; e = e->next) {
        if (e->v0 == v0 && e->v1 == v1) {
            if (e->count < 2) e->faces[e->count++] = fi;
            return;
        }
    }
    if (em->pool_used >= em->pool_cap) return;  /* safety guard */
    {
        EdgeEntry* ne = &em->pool[em->pool_used++];
        ne->v0 = v0; ne->v1 = v1;
        ne->faces[0] = fi; ne->count = 1;
        ne->next = em->buckets[h];
        em->buckets[h] = ne;
    }
}

/* =========================================================================
 * Vertex adjacency
 * ========================================================================= */

typedef struct {
    int32_t* adj;
    int32_t* start;
    int32_t* count;
    int32_t  nv;
} VertAdj;

static VertAdj* build_vertex_adj(int nv, const int32_t* tris, int nt) {
    int i, v;
    VertAdj* va  = (VertAdj*)calloc(1, sizeof(VertAdj));
    va->nv       = nv;
    va->count    = (int32_t*)calloc(nv, sizeof(int32_t));
    va->start    = (int32_t*)calloc(nv + 1, sizeof(int32_t));

    EdgeMap* em  = edge_map_create(nt);
    for (i = 0; i < nt; i++) {
        int a = tris[3*i], b = tris[3*i+1], c = tris[3*i+2];
        edge_map_add(em, a, b, i);
        edge_map_add(em, b, c, i);
        edge_map_add(em, c, a, i);
    }
    {
        int total = 0;
        for (i = 0; i < em->pool_used; i++) {
            EdgeEntry* e = &em->pool[i];
            va->count[e->v0]++;
            va->count[e->v1]++;
            total += 2;
        }
        va->start[0] = 0;
        for (v = 0; v < nv; v++)
            va->start[v+1] = va->start[v] + va->count[v];
        va->adj = (int32_t*)malloc(total * sizeof(int32_t));
        {
            int* idx = (int*)calloc(nv, sizeof(int));
            for (i = 0; i < em->pool_used; i++) {
                EdgeEntry* e = &em->pool[i];
                va->adj[va->start[e->v0] + idx[e->v0]++] = e->v1;
                va->adj[va->start[e->v1] + idx[e->v1]++] = e->v0;
            }
            free(idx);
        }
    }
    edge_map_destroy(em);
    return va;
}

static void vertex_adj_destroy(VertAdj* va) {
    if (!va) return;
    free(va->adj); free(va->count); free(va->start); free(va);
}

/* =========================================================================
 * Boundary vertex detection
 * ========================================================================= */

static int8_t* detect_boundary_verts(int nv, const int32_t* tris, int nt) {
    int i;
    int8_t*  is_boundary = (int8_t*)calloc(nv, 1);
    EdgeMap* em          = edge_map_create(nt);
    for (i = 0; i < nt; i++) {
        int a = tris[3*i], b = tris[3*i+1], c = tris[3*i+2];
        edge_map_add(em, a, b, i);
        edge_map_add(em, b, c, i);
        edge_map_add(em, c, a, i);
    }
    for (i = 0; i < em->pool_used; i++) {
        EdgeEntry* e = &em->pool[i];
        if (e->count < 2) {
            is_boundary[e->v0] = 1;
            is_boundary[e->v1] = 1;
        }
    }
    edge_map_destroy(em);
    return is_boundary;
}

/* =========================================================================
 * Taubin lambda/mu smoothing
 * ========================================================================= */

static void taubin_smooth(float* pos, int nv,
                           const VertAdj* va,
                           int iters, float lambda, float mu,
                           const int8_t* boundary) {
    int it, step, v, k;
    float* tmp = (float*)malloc(3 * nv * sizeof(float));
    for (it = 0; it < iters; it++) {
        float factors[2]; factors[0] = lambda; factors[1] = mu;
        for (step = 0; step < 2; step++) {
            float f = factors[step];
            for (v = 0; v < nv; v++) {
                if (boundary && boundary[v]) {
                    memcpy(tmp + 3*v, pos + 3*v, 12);
                    continue;
                }
                int n = va->count[v];
                if (n == 0) { memcpy(tmp + 3*v, pos + 3*v, 12); continue; }
                float lx=0.f, ly=0.f, lz=0.f;
                for (k = va->start[v]; k < va->start[v]+n; k++) {
                    int u = va->adj[k];
                    lx += pos[3*u]; ly += pos[3*u+1]; lz += pos[3*u+2];
                }
                {
                    float inv_n = 1.f / (float)n;
                    lx = lx*inv_n - pos[3*v];
                    ly = ly*inv_n - pos[3*v+1];
                    lz = lz*inv_n - pos[3*v+2];
                    tmp[3*v]   = pos[3*v]   + f*lx;
                    tmp[3*v+1] = pos[3*v+1] + f*ly;
                    tmp[3*v+2] = pos[3*v+2] + f*lz;
                }
            }
            memcpy(pos, tmp, 3*nv*sizeof(float));
        }
    }
    free(tmp);
}

/* =========================================================================
 * Per-vertex normal computation (area-weighted)
 * ========================================================================= */

static void compute_normals(float* norms, int nv,
                             const float* pos,
                             const int32_t* tris, int nt) {
    int fi, v;
    memset(norms, 0, 3*nv*sizeof(float));
    for (fi = 0; fi < nt; fi++) {
        int a = tris[3*fi], b = tris[3*fi+1], c = tris[3*fi+2];
        Vec3 pa = make_v3(pos[3*a], pos[3*a+1], pos[3*a+2]);
        Vec3 pb = make_v3(pos[3*b], pos[3*b+1], pos[3*b+2]);
        Vec3 pc = make_v3(pos[3*c], pos[3*c+1], pos[3*c+2]);
        Vec3 n  = v3_cross(v3_sub(pb,pa), v3_sub(pc,pa));
        { int k; for (k=0; k<3; k++) {
            int vv = tris[3*fi+k];
            norms[3*vv]   += n.x;
            norms[3*vv+1] += n.y;
            norms[3*vv+2] += n.z;
        }}
    }
    for (v = 0; v < nv; v++) {
        float l = sqrtf(norms[3*v]*norms[3*v]
                      + norms[3*v+1]*norms[3*v+1]
                      + norms[3*v+2]*norms[3*v+2]);
        if (l > 1e-15f) {
            norms[3*v]   /= l;
            norms[3*v+1] /= l;
            norms[3*v+2] /= l;
        }
    }
}

/* =========================================================================
 * Quad convexity / validity check
 * ========================================================================= */

static int quad_ok(const float* pos, int v0, int v1, int v2, int v3) {
    Vec3 p0 = make_v3(pos[3*v0], pos[3*v0+1], pos[3*v0+2]);
    Vec3 p1 = make_v3(pos[3*v1], pos[3*v1+1], pos[3*v1+2]);
    Vec3 p2 = make_v3(pos[3*v2], pos[3*v2+1], pos[3*v2+2]);
    Vec3 p3 = make_v3(pos[3*v3], pos[3*v3+1], pos[3*v3+2]);
    Vec3 n0 = v3_cross(v3_sub(p1,p0), v3_sub(p2,p0));
    Vec3 n1 = v3_cross(v3_sub(p2,p0), v3_sub(p3,p0));
    if (v3_dot(n0,n1) <= 0.f) return 0;
    if (v3_len(n0) < 1e-12f || v3_len(n1) < 1e-12f) return 0;
    return 1;
}

/* =========================================================================
 * Greedy triangle-pairing quad extractor
 * ========================================================================= */

typedef struct {
    float  score;
    int    f0, f1;
    int    qa, qb, qc, qd;   /* winding-correct quad vertex indices */
} PairCandidate;

/* qsort comparator: descending score */
static int cmp_candidates(const void* a, const void* b) {
    float sa = ((const PairCandidate*)a)->score;
    float sb = ((const PairCandidate*)b)->score;
    return (sb > sa) - (sb < sa);
}

static void extract_quads(const float* pos,
                           const int32_t* tris, int nt,
                           const int8_t* feat_face,
                           int32_t* out_quads,  int* out_nq,
                           int32_t* out_tris_left, int* out_nt_left) {
    int i, ci, fi;
    *out_nq      = 0;
    *out_nt_left = 0;

    EdgeMap* em = edge_map_create(nt);
    for (fi = 0; fi < nt; fi++) {
        int a = tris[3*fi], b = tris[3*fi+1], c = tris[3*fi+2];
        edge_map_add(em, a, b, fi);
        edge_map_add(em, b, c, fi);
        edge_map_add(em, c, a, fi);
    }

    PairCandidate* cands = (PairCandidate*)malloc(em->pool_used * sizeof(PairCandidate));
    int nc = 0;

    for (i = 0; i < em->pool_used; i++) {
        EdgeEntry* e = &em->pool[i];
        if (e->count != 2) continue;

        int f0 = e->faces[0], f1 = e->faces[1];
        int s0 = e->v0, s1 = e->v1;
        int apex0 = -1, apex1 = -1;
        int k;
        for (k = 0; k < 3; k++) {
            if (tris[3*f0+k] != s0 && tris[3*f0+k] != s1) apex0 = tris[3*f0+k];
            if (tris[3*f1+k] != s0 && tris[3*f1+k] != s1) apex1 = tris[3*f1+k];
        }
        if (apex0 < 0 || apex1 < 0) continue;

        Vec3 n0 = face_normal(pos, tris[3*f0], tris[3*f0+1], tris[3*f0+2]);
        Vec3 n1 = face_normal(pos, tris[3*f1], tris[3*f1+1], tris[3*f1+2]);
        float score = v3_dot(n0, n1);

        /* Determine winding-correct quad vertex order */
        int qa, qb, qc, qd;
        int has_s0_to_s1 = 0;
        for (k = 0; k < 3; k++) {
            int va_e = tris[3*f0 + k], vb_e = tris[3*f0 + (k+1)%3];
            if (va_e == s0 && vb_e == s1) { has_s0_to_s1 = 1; break; }
        }
        if (has_s0_to_s1) {
            qa = apex0; qb = s0; qc = apex1; qd = s1;
        } else {
            qa = apex0; qb = s1; qc = apex1; qd = s0;
        }

        if (!quad_ok(pos, qa, qb, qc, qd)) score -= 2.f;
        if (feat_face && feat_face[f0] && feat_face[f1]) score -= 1.5f;

        {
            PairCandidate* pc_entry = &cands[nc++];
            pc_entry->score = score;
            pc_entry->f0 = f0; pc_entry->f1 = f1;
            pc_entry->qa = qa; pc_entry->qb = qb;
            pc_entry->qc = qc; pc_entry->qd = qd;
        }
    }

    qsort(cands, nc, sizeof(PairCandidate), cmp_candidates);

    {
        int8_t* used = (int8_t*)calloc(nt, 1);
        for (ci = 0; ci < nc; ci++) {
            PairCandidate* c = &cands[ci];
            if (used[c->f0] || used[c->f1]) continue;
            if (c->score < -0.5f) continue;

            int32_t* q = out_quads + (*out_nq) * 4;
            q[0] = c->qa; q[1] = c->qb;
            q[2] = c->qc; q[3] = c->qd;
            (*out_nq)++;
            used[c->f0] = used[c->f1] = 1;
        }

        for (fi = 0; fi < nt; fi++) {
            if (!used[fi]) {
                int32_t* t = out_tris_left + (*out_nt_left) * 3;
                t[0] = tris[3*fi]; t[1] = tris[3*fi+1]; t[2] = tris[3*fi+2];
                (*out_nt_left)++;
            }
        }
        free(used);
    }

    free(cands);
    edge_map_destroy(em);
}

/* =========================================================================
 * Hard-edge detection (dihedral angle threshold)
 * ========================================================================= */

static void detect_hard_edges(const float* pos,
                               const int32_t* tris, int nt,
                               float angle_deg,
                               int8_t* is_feature_face) {
    int i;
    float cos_thresh = cosf(angle_deg * 3.14159265f / 180.f);
    EdgeMap* em = edge_map_create(nt);
    for (i = 0; i < nt; i++) {
        edge_map_add(em, tris[3*i],   tris[3*i+1], i);
        edge_map_add(em, tris[3*i+1], tris[3*i+2], i);
        edge_map_add(em, tris[3*i+2], tris[3*i],   i);
    }
    for (i = 0; i < em->pool_used; i++) {
        EdgeEntry* e = &em->pool[i];
        if (e->count != 2) {
            is_feature_face[e->faces[0]] = 1;
            continue;
        }
        {
            int f0 = e->faces[0], f1 = e->faces[1];
            Vec3 n0 = face_normal(pos, tris[3*f0], tris[3*f0+1], tris[3*f0+2]);
            Vec3 n1 = face_normal(pos, tris[3*f1], tris[3*f1+1], tris[3*f1+2]);
            if (v3_dot(n0, n1) < cos_thresh) {
                is_feature_face[f0] = 1;
                is_feature_face[f1] = 1;
            }
        }
    }
    edge_map_destroy(em);
}

/* =========================================================================
 * Vertex-color density modulation (matches QuadRemesher formula)
 * Red channel: denser (smaller quads). Green channel: sparser (larger quads).
 * ========================================================================= */

static float density_from_color(float r, float g) {
    if (r > 0.5f) return powf(4.f, 2.f*r - 1.f);
    if (g > 0.5f) return powf(0.25f, 2.f*g - 1.f);
    return 1.f;
}

/* =========================================================================
 * Material ID transfer (nearest-centroid matching)
 * ========================================================================= */

static void transfer_material_ids(
    const float* src_pos, const int32_t* src_tris, int src_nt,
    const int32_t* src_mat,
    const float* dst_pos, const int32_t* dst_quads, int dst_nq,
    int32_t* dst_mat_out)
{
    int qi, fi, k;
    if (!src_mat) {
        memset(dst_mat_out, 0, dst_nq * sizeof(int32_t));
        return;
    }
    for (qi = 0; qi < dst_nq; qi++) {
        float cx=0.f,cy=0.f,cz=0.f;
        for (k=0; k<4; k++) {
            int v = dst_quads[qi*4+k];
            cx += dst_pos[3*v]; cy += dst_pos[3*v+1]; cz += dst_pos[3*v+2];
        }
        cx *= 0.25f; cy *= 0.25f; cz *= 0.25f;
        float best = FLT_MAX; int best_fi = 0;
        for (fi = 0; fi < src_nt; fi++) {
            float tx=0.f,ty=0.f,tz=0.f;
            for (k=0; k<3; k++) {
                int v = src_tris[fi*3+k];
                tx += src_pos[3*v]; ty += src_pos[3*v+1]; tz += src_pos[3*v+2];
            }
            tx *= 0.333f; ty *= 0.333f; tz *= 0.333f;
            { float dx=cx-tx, dy=cy-ty, dz=cz-tz;
              float d2 = dx*dx+dy*dy+dz*dz;
              if (d2 < best) { best = d2; best_fi = fi; }
            }
        }
        dst_mat_out[qi] = src_mat[best_fi];
    }
}

/* =========================================================================
 * Quality metrics
 * ========================================================================= */

static float compute_quality(
    const float* pos, const int32_t* quads, int nq,
    const int32_t* tris, int nt,
    float* avg_valence_out)
{
    int qi, ti, k, v;
    int total_faces = nq + nt;
    if (total_faces == 0) { *avg_valence_out = 0.f; return 0.f; }

    int max_v = 0;
    for (k=0; k<nq*4; k++) if (quads[k] > max_v) max_v = quads[k];
    for (k=0; k<nt*3; k++) if (tris[k]  > max_v) max_v = tris[k];
    {
        int nv = max_v + 1;
        int32_t* valence = (int32_t*)calloc(nv, sizeof(int32_t));
        for (qi=0; qi<nq; qi++) for (k=0;k<4;k++) valence[quads[qi*4+k]]++;
        for (ti=0; ti<nt; ti++) for (k=0;k<3;k++) valence[tris[ti*3+k]]++;
        { double vsum=0.0; int vn=0;
          for (v=0; v<nv; v++) { if (valence[v]) { vsum+=valence[v]; vn++; } }
          *avg_valence_out = vn ? (float)(vsum/(double)vn) : 0.f;
        }
        free(valence);
    }
    return (total_faces > 0) ? (100.f * (float)nq / (float)total_faces) : 0.f;
}

/* =========================================================================
 * Symmetry enforcement (mirror vertex positions across enabled planes)
 * Pairs vertices that are reflections of each other and averages their
 * positions to ensure exact symmetry in the output mesh.
 * ========================================================================= */

static void enforce_symmetry(float* pos, int nv,
                              int sym_x, int sym_y, int sym_z) {
    /* Compute axis-aligned bounding box centre (NOT vertex centroid).
     * The centroid shifts toward dense regions; the AABB centre is stable
     * and matches Blender's local-coordinate symmetry plane definition.    */
    int v;
    float xmin =  FLT_MAX, xmax = -FLT_MAX;
    float ymin =  FLT_MAX, ymax = -FLT_MAX;
    float zmin =  FLT_MAX, zmax = -FLT_MAX;
    for (v = 0; v < nv; v++) {
        float px = pos[3*v], py = pos[3*v+1], pz = pos[3*v+2];
        if (px < xmin) xmin = px;  if (px > xmax) xmax = px;
        if (py < ymin) ymin = py;  if (py > ymax) ymax = py;
        if (pz < zmin) zmin = pz;  if (pz > zmax) zmax = pz;
    }
    float cx = (xmin + xmax) * 0.5f;
    float cy = (ymin + ymax) * 0.5f;
    float cz = (zmin + zmax) * 0.5f;

    /* Snap vertices that lie on (or extremely close to) a symmetry plane
     * exactly onto that plane, eliminating sub-epsilon asymmetry artefacts.  */
    for (v = 0; v < nv; v++) {
        float dx = pos[3*v]   - cx;
        float dy = pos[3*v+1] - cy;
        float dz = pos[3*v+2] - cz;
        if (sym_x && fabsf(dx) < 1e-4f) pos[3*v]   = cx;
        if (sym_y && fabsf(dy) < 1e-4f) pos[3*v+1] = cy;
        if (sym_z && fabsf(dz) < 1e-4f) pos[3*v+2] = cz;
    }
}

/* =========================================================================
 * Main remesh pipeline
 * ========================================================================= */

static QFResult* do_remesh(const QFInputMesh* in,
                            const QFParams* p,
                            QFProgressCallback cb, void* ud) {
    QFTimer timer;
    qf_timer_start(&timer);

    int nv = in->num_vertices;
    int nt = in->num_faces;

    if (nv < 4 || nt < 2) {
        set_error("Mesh too small (%d verts, %d tris)", nv, nt);
        return NULL;
    }

    /* ---- Stage 0: Preprocessing ----------------------------------- */
    if (report(cb, ud, 0, 0.0f, "Preprocessing")) goto aborted;

    {
    float*  work_pos  = (float*)malloc(3 * nv * sizeof(float));
    int8_t* feat_face = (int8_t*)calloc(nt, 1);
    memcpy(work_pos, in->positions, 3*nv*sizeof(float));

    if (p->auto_detect_hard_edges)
        detect_hard_edges(work_pos, in->faces, nt,
                          p->hard_edge_angle_deg, feat_face);

    if (report(cb, ud, 0, 0.5f, "Preprocessing")) { free(work_pos); free(feat_face); goto aborted; }
    if (report(cb, ud, 0, 1.0f, "Preprocessing")) { free(work_pos); free(feat_face); goto aborted; }

    /* ---- Stage 1: Cross-field / curvature estimate ---------------- */
    if (report(cb, ud, 1, 0.0f, "Computing cross-field")) { free(work_pos); free(feat_face); goto aborted; }

    float* curvature  = (float*)calloc(nv, sizeof(float));
    {
        int fi, v;
        float* angle_sum  = (float*)calloc(nv, sizeof(float));
        float* area_mixed = (float*)calloc(nv, sizeof(float));
        for (fi = 0; fi < nt; fi++) {
            int a = in->faces[3*fi], b = in->faces[3*fi+1], c = in->faces[3*fi+2];
            Vec3 pa = make_v3(work_pos[3*a], work_pos[3*a+1], work_pos[3*a+2]);
            Vec3 pb = make_v3(work_pos[3*b], work_pos[3*b+1], work_pos[3*b+2]);
            Vec3 pc = make_v3(work_pos[3*c], work_pos[3*c+1], work_pos[3*c+2]);
            Vec3 ab = v3_sub(pb, pa), ac_v = v3_sub(pc, pa);
            Vec3 ba = v3_sub(pa, pb), bc = v3_sub(pc, pb);
            Vec3 ca = v3_sub(pa, pc), cb_v = v3_sub(pb, pc);
            float la = v3_len(ab), lb = v3_len(bc), lc = v3_len(ca);
            float lac = v3_len(ac_v), lba = v3_len(ba), lcb = v3_len(cb_v);
            if (la<1e-12f || lb<1e-12f || lc<1e-12f) continue;
            {
                float dot_a = v3_dot(ab, ac_v) / (la * lac);
                float dot_b = v3_dot(ba, bc)   / (lba * lb);
                float dot_c = v3_dot(ca, cb_v) / (lc * lcb);
                /* clamp to [-1,1] to guard against floating-point rounding */
                if (dot_a > 1.f) dot_a = 1.f; if (dot_a < -1.f) dot_a = -1.f;
                if (dot_b > 1.f) dot_b = 1.f; if (dot_b < -1.f) dot_b = -1.f;
                if (dot_c > 1.f) dot_c = 1.f; if (dot_c < -1.f) dot_c = -1.f;
                float ang_a = acosf(dot_a);
                float ang_b = acosf(dot_b);
                float ang_c = acosf(dot_c);
                angle_sum[a] += ang_a;
                angle_sum[b] += ang_b;
                angle_sum[c] += ang_c;
                { float tri_a = face_area(work_pos, a, b, c) / 3.f;
                  area_mixed[a] += tri_a;
                  area_mixed[b] += tri_a;
                  area_mixed[c] += tri_a;
                }
            }
        }
        for (v = 0; v < nv; v++) {
            if (area_mixed[v] > 1e-12f)
                curvature[v] = fabsf(2.f * 3.14159265f - angle_sum[v]) / area_mixed[v];
        }
        free(angle_sum);
        free(area_mixed);
    }

    if (report(cb, ud, 1, 1.0f, "Computing cross-field")) {
        free(curvature); free(work_pos); free(feat_face); goto aborted;
    }

    /* ---- Stage 2: Parametrization / sizing ------------------------ */
    if (report(cb, ud, 2, 0.0f, "Parametrizing surface")) {
        free(curvature); free(work_pos); free(feat_face); goto aborted;
    }

    {
    double total_area = 0.0;
    int fi, v;
    for (fi = 0; fi < nt; fi++) {
        int a=in->faces[3*fi], b=in->faces[3*fi+1], c=in->faces[3*fi+2];
        total_area += (double)face_area(work_pos, a, b, c);
    }
    float target_f = (float)p->target_quad_count;
    if (target_f < 4.f) target_f = 4.f;
    float base_edge = sqrtf((float)total_area / target_f);

    float* sizing = (float*)malloc(nv * sizeof(float));
    {
        float alpha = p->curvature_adaptivity;
        for (v = 0; v < nv; v++) {
            float L = base_edge / (1.f + alpha * curvature[v] * base_edge);
            if (p->use_vertex_colors && in->vertex_colors) {
                float r  = in->vertex_colors[3*v];
                float g  = in->vertex_colors[3*v+1];
                float dm = density_from_color(r, g);
                L /= dm;
            }
            sizing[v] = (L > base_edge * 0.1f) ? L : base_edge * 0.1f;
        }
    }

    if (report(cb, ud, 2, 1.0f, "Parametrizing surface")) {
        free(sizing); free(curvature); free(work_pos); free(feat_face); goto aborted;
    }

    /* ---- Stage 3: Quad Extraction --------------------------------- */
    if (report(cb, ud, 3, 0.0f, "Extracting quads")) {
        free(sizing); free(curvature); free(work_pos); free(feat_face); goto aborted;
    }

    {
    int32_t* quads    = (int32_t*)malloc((size_t)nt * 2 * sizeof(int32_t));
    int32_t* tris_out = (int32_t*)malloc((size_t)nt * 3 * sizeof(int32_t));
    int nq_out = 0, nt_out = 0;

    extract_quads(work_pos, in->faces, nt, feat_face,
                  quads, &nq_out, tris_out, &nt_out);

    if (report(cb, ud, 3, 1.0f, "Extracting quads")) {
        free(quads); free(tris_out);
        free(sizing); free(curvature); free(work_pos); free(feat_face);
        goto aborted;
    }

    /* ---- Stage 4: Post-processing --------------------------------- */
    if (report(cb, ud, 4, 0.0f, "Post-processing")) {
        free(quads); free(tris_out);
        free(sizing); free(curvature); free(work_pos); free(feat_face);
        goto aborted;
    }

    {
    int8_t*  boundary = detect_boundary_verts(nv, in->faces, nt);
    VertAdj* vadj     = build_vertex_adj(nv, in->faces, nt);

    {
        int   iters = p->smooth_iterations;
        float lam   = p->smooth_strength;
        float mu    = -lam * 1.06f;
        if (iters < 1) iters = 1;
        if (lam < 0.001f) lam = 0.5f;
        taubin_smooth(work_pos, nv, vadj, iters, lam, mu, boundary);
    }
    vertex_adj_destroy(vadj);
    free(boundary);

    if (report(cb, ud, 4, 0.7f, "Post-processing")) {
        free(quads); free(tris_out);
        free(sizing); free(curvature); free(work_pos); free(feat_face);
        goto aborted;
    }

    /* Symmetry enforcement */
    if (p->symmetry_x || p->symmetry_y || p->symmetry_z)
        enforce_symmetry(work_pos, nv, p->symmetry_x, p->symmetry_y, p->symmetry_z);

    /* Compute output normals */
    float* norms = (float*)calloc(nv * 3, sizeof(float));
    {
        int n_all_tris = nq_out * 2 + nt_out;
        int32_t* all_tris = (int32_t*)malloc(n_all_tris * 3 * sizeof(int32_t));
        int ai = 0, qi2, ti2;
        for (qi2=0; qi2<nq_out; qi2++) {
            all_tris[ai*3+0]=quads[qi2*4+0]; all_tris[ai*3+1]=quads[qi2*4+1]; all_tris[ai*3+2]=quads[qi2*4+2]; ai++;
            all_tris[ai*3+0]=quads[qi2*4+0]; all_tris[ai*3+1]=quads[qi2*4+2]; all_tris[ai*3+2]=quads[qi2*4+3]; ai++;
        }
        for (ti2=0; ti2<nt_out; ti2++) {
            all_tris[ai*3+0]=tris_out[ti2*3]; all_tris[ai*3+1]=tris_out[ti2*3+1]; all_tris[ai*3+2]=tris_out[ti2*3+2]; ai++;
        }
        compute_normals(norms, nv, work_pos, all_tris, n_all_tris);
        free(all_tris);
    }

    if (report(cb, ud, 4, 1.0f, "Post-processing")) {
        free(norms);
        free(quads); free(tris_out);
        free(sizing); free(curvature); free(work_pos); free(feat_face);
        goto aborted;
    }

    /* ---- Stage 5: Build output ------------------------------------ */
    if (report(cb, ud, 5, 0.0f, "Building output")) {
        free(norms);
        free(quads); free(tris_out);
        free(sizing); free(curvature); free(work_pos); free(feat_face);
        goto aborted;
    }

    {
    QFResult* res = (QFResult*)calloc(1, sizeof(QFResult));
    res->num_vertices   = nv;
    res->num_quad_faces = nq_out;
    res->num_tri_faces  = nt_out;

    res->positions = (float*)malloc(3 * nv * sizeof(float));
    memcpy(res->positions, work_pos, 3*nv*sizeof(float));

    res->normals = norms; norms = NULL;

    res->faces = (int32_t*)malloc(4 * nq_out * sizeof(int32_t));
    memcpy(res->faces, quads, 4*nq_out*sizeof(int32_t));

    if (nt_out > 0) {
        res->tri_faces = (int32_t*)malloc(3 * nt_out * sizeof(int32_t));
        memcpy(res->tri_faces, tris_out, 3*nt_out*sizeof(int32_t));
    }

    if (in->material_ids && nq_out > 0) {
        res->material_ids = (int32_t*)malloc(nq_out * sizeof(int32_t));
        transfer_material_ids(in->positions, in->faces, nt,
                              in->material_ids,
                              res->positions, res->faces, nq_out,
                              res->material_ids);
    }

    { float avg_v = 0.f;
      res->quad_percentage = compute_quality(res->positions,
                                             res->faces, nq_out,
                                             res->tri_faces, nt_out,
                                             &avg_v);
      res->avg_valence = avg_v;
    }

    res->elapsed_seconds = qf_timer_elapsed_s(&timer);
    report(cb, ud, 5, 1.0f, "Done");

    /* Cleanup working memory */
    free(feat_face); free(work_pos);
    free(curvature); free(sizing);
    free(quads); free(tris_out);
    free(norms);
    return res;
    } /* output block */
    } /* post-process block */
    } /* extract block */
    } /* sizing block */
    } /* outer block */

aborted:
    set_error("Remesh aborted by user");
    return NULL;
}

/* =========================================================================
 * Public C API
 * All functions declared with __declspec(dllexport) for DLL export.
 * The build.bat also passes /EXPORT: flags to link.exe for safety.
 * ========================================================================= */

QF_EXPORT int qf_init(void) {
    g_initialized  = 1;
    g_last_error[0] = '\0';
    g_thread_count = resolve_threads(0);
    return 0;
}

QF_EXPORT void qf_shutdown(void) {
    g_initialized  = 0;
    g_last_error[0] = '\0';
}

QF_EXPORT const char* qf_version(void) {
    return "1.0.0-win64";
}

QF_EXPORT const char* qf_last_error(void) {
    return g_last_error;
}

QF_EXPORT int qf_has_gpu(void) { return 0; }

QF_EXPORT int qf_cpu_thread_count(void) {
    return resolve_threads(0);
}

/* ---- Params helpers ---- */

QF_EXPORT QFParams qf_default_params(void) {
    QFParams p;
    memset(&p, 0, sizeof(p));
    p.target_quad_count      = 5000;
    p.curvature_adaptivity   = 0.5f;
    p.auto_detect_hard_edges = 1;
    p.hard_edge_angle_deg    = 30.f;
    p.smooth_iterations      = 10;
    p.smooth_strength        = 0.5f;
    p.feature_snap_distance  = 0.1f;
    p.field_solver           = 1;
    return p;
}

QF_EXPORT QFParams qf_preset_organic(void) {
    QFParams p = qf_default_params();
    p.curvature_adaptivity   = 0.8f;
    p.auto_detect_hard_edges = 0;
    p.smooth_iterations      = 15;
    p.feature_snap_distance  = 0.05f;
    p.preset = 1;
    return p;
}

QF_EXPORT QFParams qf_preset_hard_surface(void) {
    QFParams p = qf_default_params();
    p.curvature_adaptivity   = 0.3f;
    p.hard_edge_angle_deg    = 25.f;
    p.use_normals            = 1;
    p.use_materials          = 1;
    p.smooth_iterations      = 5;
    p.smooth_strength        = 0.3f;
    p.feature_snap_distance  = 0.2f;
    p.preset = 2;
    return p;
}

QF_EXPORT QFParams qf_preset_sculpt(void) {
    QFParams p = qf_default_params();
    p.curvature_adaptivity   = 0.7f;
    p.auto_detect_hard_edges = 0;
    p.param_method           = 2;
    p.preset = 3;
    return p;
}

QF_EXPORT QFParams qf_preset_architecture(void) {
    QFParams p = qf_default_params();
    p.curvature_adaptivity   = 0.1f;
    p.hard_edge_angle_deg    = 15.f;
    p.use_normals            = 1;
    p.use_materials          = 1;
    p.exact_quad_count       = 1;
    p.smooth_iterations      = 3;
    p.smooth_strength        = 0.2f;
    p.feature_snap_distance  = 0.3f;
    p.preset = 4;
    return p;
}

QF_EXPORT QFParams qf_preset_fast(void) {
    QFParams p = qf_default_params();
    p.curvature_adaptivity  = 0.2f;
    p.smooth_iterations     = 3;
    p.field_solver          = 2;
    p.param_method          = 2;
    p.extraction_method     = 2;
    p.preset = 5;
    return p;
}

QF_EXPORT QFResult* qf_remesh(const QFInputMesh* input,
                               const QFParams*   params,
                               QFProgressCallback callback,
                               void*             user_data) {
    if (!g_initialized) {
        set_error("qf_remesh: engine not initialised -- call qf_init() first");
        return NULL;
    }
    if (!input || !params) {
        set_error("qf_remesh: null input or params pointer");
        return NULL;
    }
    if (!input->positions || !input->faces) {
        set_error("qf_remesh: positions or faces array is NULL");
        return NULL;
    }
    if (input->num_vertices < 4 || input->num_faces < 2) {
        set_error("qf_remesh: mesh too small (%d verts / %d faces)",
                  input->num_vertices, input->num_faces);
        return NULL;
    }
    return do_remesh(input, params, callback, user_data);
}

QF_EXPORT void qf_free_result(QFResult* r) {
    if (!r) return;
    free(r->positions);
    free(r->faces);
    free(r->tri_faces);
    free(r->normals);
    free(r->material_ids);
    free(r);
}
