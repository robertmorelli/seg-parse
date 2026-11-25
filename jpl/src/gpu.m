#import <Metal/Metal.h>
#include "gpu.h"

#include <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHUNK 256
#define GROUP 64
#define EMIT 0x200
#define RESET_AT 0x400
#define RESET_AFTER 0x800

typedef struct {
    uint32_t count;
    uint32_t offs[32];
    uint32_t sizes[32];
} Levels;

struct Gpu {
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipes;
    uint32_t states, maxk;
    size_t max_len, max_tok;

    id<MTLBuffer> src, step, img, nimg, imgpos, types, ctab, cmap;
    // Lexer scan: one summary array per level.
    int levels;
    id<MTLBuffer> sum_end[16], sum_cnt[16], sum_b0[16], inc_s[16], inc_c[16];
    id<MTLBuffer> tok_type, tok_start, tok_end, flags;

    id<MTLBuffer> kind, delta, eb, sdelta, sb, idelta, id_, match, sdepth, left, right, parent, perr, how, cflags, sepbits;
    id<MTLBuffer> t_open, t_term, t_keys;
    id<MTLBuffer> scan_tmp[8], zero;
    bool profile;
};

static double ms_now(void) {
    static mach_timebase_info_data_t tb;
    if (!tb.denom) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6;
}

static id<MTLBuffer> buf(Gpu *g, size_t bytes) {
    return [g->dev newBufferWithLength:bytes ? bytes : 16 options:MTLResourceStorageModeShared];
}

static Levels levels_for(uint32_t n) {
    Levels L = {0};
    uint32_t off = 0, size = n;
    for (;;) {
        L.offs[L.count] = off;
        L.sizes[L.count] = size;
        L.count++;
        off += (size + 7) & ~7u;
        if (size <= 1) break;
        size = (size + 7) / 8;
    }
    return L;
}

static size_t tree_len(size_t n) { return n + n / 7 + 256; }

Gpu *gpu_init(const Table *t, const char *metal_path, size_t max_len) {
    Gpu *g = calloc(1, sizeof *g);
    g->dev = MTLCreateSystemDefaultDevice();
    g->queue = [g->dev newCommandQueue];
    g->states = t->states;
    g->max_len = max_len;
    g->max_tok = max_len + 2;

    // Step table: next state with accept/ignore resolved, plus what happened.
    uint32_t S = t->states;
    uint16_t *step = malloc(S * 256 * 2);
    for (uint32_t s = 0; s < S; s++) {
        for (uint32_t b = 0; b < 256; b++) {
            uint16_t r = t->tabs[s * 256 + b], flags = 0;
            if (r == 2) {
                uint16_t r2 = t->tabs[1 * 256 + b];
                flags = EMIT;
                if (r2 == 3) {
                    flags |= RESET_AFTER;
                    r = 1;
                } else {
                    flags |= RESET_AT;
                    r = r2;
                }
            } else if (r == 3) {
                flags = RESET_AFTER;
                r = 1;
            }
            step[s * 256 + b] = r | flags;
        }
    }
    // Image of each byte's step function: the states a chunk can be in after
    // reading its first byte.
    uint8_t *imgpos = malloc(256 * S);
    memset(imgpos, 255, 256 * S);
    uint16_t img_tmp[256][64];
    uint8_t nimg[256];
    uint32_t maxk = 0;
    for (uint32_t b = 0; b < 256; b++) {
        nimg[b] = 0;
        for (uint32_t s = 0; s < S; s++) {
            uint16_t ns = step[s * 256 + b] & 0x1FF;
            if (imgpos[b * S + ns] == 255) {
                imgpos[b * S + ns] = nimg[b];
                img_tmp[b][nimg[b]++] = ns;
            }
        }
        if (nimg[b] > maxk) maxk = nimg[b];
    }
    g->maxk = maxk;
    uint16_t *img = calloc(256 * maxk, 2);
    for (uint32_t b = 0; b < 256; b++) memcpy(img + b * maxk, img_tmp[b], nimg[b] * 2);

    // Byte classes over the states below the first UTF-8 inner state.
    uint16_t nfast;
    memcpy(&nfast, (const uint8_t *)t->blob + 6, 2);
    uint8_t cmap[256];
    uint16_t *ctab = malloc((size_t)nfast * 256 * 2);
    uint32_t ncls = 0;
    for (uint32_t b = 0; b < 256; b++) {
        uint32_t c = 0;
        for (; c < ncls; c++) {
            bool same = true;
            for (uint32_t s = 0; s < nfast && same; s++) same = ctab[s * 256 + c] == step[s * 256 + b];
            if (same) break;
        }
        if (c == ncls) {
            for (uint32_t s = 0; s < nfast; s++) ctab[s * 256 + c] = step[s * 256 + b];
            ncls++;
        }
        cmap[b] = (uint8_t)c;
    }
    uint16_t *packed = malloc((size_t)nfast * ncls * 2);
    for (uint32_t s = 0; s < nfast; s++)
        for (uint32_t c = 0; c < ncls; c++) packed[s * ncls + c] = ctab[s * 256 + c];
    g->ctab = [g->dev newBufferWithBytes:packed length:(size_t)nfast * ncls * 2 options:MTLResourceStorageModeShared];
    g->cmap = [g->dev newBufferWithBytes:cmap length:256 options:MTLResourceStorageModeShared];
    free(ctab);
    free(packed);

    g->step = [g->dev newBufferWithBytes:step length:S * 256 * 2 options:MTLResourceStorageModeShared];
    g->img = [g->dev newBufferWithBytes:img length:256 * maxk * 2 options:MTLResourceStorageModeShared];
    g->nimg = [g->dev newBufferWithBytes:nimg length:256 options:MTLResourceStorageModeShared];
    g->imgpos = [g->dev newBufferWithBytes:imgpos length:256 * S options:MTLResourceStorageModeShared];
    g->types = [g->dev newBufferWithBytes:t->types length:S options:MTLResourceStorageModeShared];
    free(step);
    free(imgpos);
    free(img);

    NSError *err = nil;
    NSString *code = [NSString stringWithContentsOfFile:@(metal_path) encoding:NSUTF8StringEncoding error:&err];
    if (!code) {
        fprintf(stderr, "cannot read %s\n", metal_path);
        return NULL;
    }
    MTLCompileOptions *opts = [MTLCompileOptions new];
    opts.preprocessorMacros = @{@"MAXK" : @(maxk), @"NSTATES" : @(S), @"NFAST" : @(nfast), @"NCLS" : @(ncls)};
    opts.mathMode = MTLMathModeFast;
    id<MTLLibrary> lib = [g->dev newLibraryWithSource:code options:opts error:&err];
    if (!lib) {
        fprintf(stderr, "%s\n", err.localizedDescription.UTF8String);
        return NULL;
    }
    g->pipes = [NSMutableDictionary new];
    for (NSString *name in lib.functionNames) {
        id<MTLFunction> f = [lib newFunctionWithName:name];
        g->pipes[name] = [g->dev newComputePipelineStateWithFunction:f error:&err];
        if (!g->pipes[name]) {
            fprintf(stderr, "%s: %s\n", name.UTF8String, err.localizedDescription.UTF8String);
            return NULL;
        }
    }

    // Input, page-aligned so the GPU reads it without a copy.
    size_t src_bytes = (max_len + 1 + CHUNK + 16383) & ~(size_t)16383;
    void *mem = NULL;
    posix_memalign(&mem, 16384, src_bytes);
    memset(mem, 0, src_bytes);
    g->src = [g->dev newBufferWithBytesNoCopy:mem length:src_bytes options:MTLResourceStorageModeShared deallocator:nil];

    uint32_t n = (uint32_t)((max_len + CHUNK - 1) / CHUNK);
    for (g->levels = 0;; g->levels++) {
        int l = g->levels;
        g->sum_end[l] = buf(g, (size_t)n * maxk * 2);
        g->sum_cnt[l] = buf(g, (size_t)n * maxk * 4);
        g->sum_b0[l] = buf(g, n);
        g->inc_s[l] = buf(g, (size_t)n * 2);
        g->inc_c[l] = buf(g, (size_t)n * 4);
        if (n <= 1) break;
        n = (n + GROUP - 1) / GROUP;
    }
    g->levels++;

    size_t T = g->max_tok;
    g->tok_type = buf(g, T);
    g->tok_start = buf(g, (T + 2 * CHUNK) * 4);
    g->tok_end = buf(g, T * 4);
    g->flags = buf(g, 16);
    g->kind = buf(g, T);
    g->delta = buf(g, T * 4);
    g->eb = buf(g, T * 4);
    g->sdelta = buf(g, T * 4);
    g->sb = buf(g, T * 4);
    g->idelta = buf(g, T * 4 + 16);
    g->id_ = buf(g, T * 4);
    g->match = buf(g, T * 4);
    g->sdepth = buf(g, T * 4);
    g->left = buf(g, T * 4);
    g->right = buf(g, T * 4);
    g->parent = buf(g, T * 4);
    g->perr = buf(g, 16);
    g->how = buf(g, T);
    g->cflags = buf(g, T * 4);
    g->sepbits = buf(g, T * 4);
    g->t_open = buf(g, tree_len(T) * 4);
    g->t_term = buf(g, tree_len(T) * 4);
    g->t_keys = buf(g, tree_len(T) * 8);
    size_t m = T;
    for (int i = 0; i < 8; i++) {
        m = (m + 1023) / 1024 + 1;
        g->scan_tmp[i] = buf(g, m * 4 * 2);
    }
    g->zero = buf(g, 64);
    g->profile = getenv("JPL_PROFILE") != NULL;
    return g;
}

uint8_t *gpu_input(Gpu *g) { return g->src.contents; }

// ---- encoding helpers

typedef id<MTLComputeCommandEncoder> Enc;

static void dispatch(Gpu *g, Enc e, NSString *name, NSUInteger threads) {
    id<MTLComputePipelineState> p = g->pipes[name];
    [e setComputePipelineState:p];
    NSUInteger w = MIN((NSUInteger)256, p.maxTotalThreadsPerThreadgroup);
    if (threads == 0) return;
    [e dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
}

static void set_u32(Enc e, uint32_t v, int idx) { [e setBytes:&v length:4 atIndex:idx]; }

static void set_bufs(Enc e, NSArray<id<MTLBuffer>> *bufs) {
    for (NSUInteger i = 0; i < bufs.count; i++) [e setBuffer:bufs[i] offset:0 atIndex:i];
}

// out = exclusive (or inclusive) prefix sum of in.
static void scan(Gpu *g, Enc e, id<MTLBuffer> in, id<MTLBuffer> out, uint32_t n, bool inclusive, int depth) {
    uint32_t nb = (n + 1023) / 1024;
    id<MTLBuffer> offsets = g->zero;
    if (nb > 1) {
        id<MTLBuffer> sums = g->scan_tmp[depth];
        [e setComputePipelineState:g->pipes[@"scan_reduce"]];
        [e setBuffer:in offset:0 atIndex:0];
        set_u32(e, n, 1);
        [e setBuffer:sums offset:0 atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(nb, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        // Exclusive scan of the block sums, in place.
        scan(g, e, sums, sums, nb, false, depth + 1);
        offsets = sums;
    }
    [e setComputePipelineState:g->pipes[@"scan_apply"]];
    [e setBuffer:in offset:0 atIndex:0];
    set_u32(e, n, 1);
    [e setBuffer:offsets offset:0 atIndex:2];
    [e setBuffer:out offset:0 atIndex:3];
    set_u32(e, inclusive, 4);
    [e dispatchThreadgroups:MTLSizeMake(nb, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void build_tree(Gpu *g, Enc e, id<MTLBuffer> tree, const Levels *L, bool wide, bool packed) {
    for (uint32_t l = 0; l + 1 < L->count; l++) {
        [e setBuffer:tree offset:0 atIndex:0];
        set_u32(e, l, 1);
        [e setBytes:L length:sizeof *L atIndex:2];
        set_u32(e, packed, 3);
        dispatch(g, e, wide ? @"tree_build64" : @"tree_build32", L->sizes[l + 1]);
    }
}

// With JPL_PROFILE set, each stage runs in its own command buffer and its
// GPU time is printed.
static void stage(Gpu *g, id<MTLCommandBuffer> __strong *cb, Enc __strong *e, const char *name) {
    if (!g->profile) return;
    [*e endEncoding];
    [*cb commit];
    [*cb waitUntilCompleted];
    fprintf(stderr, "  %-14s %7.3f ms\n", name, ((*cb).GPUEndTime - (*cb).GPUStartTime) * 1e3);
    *cb = [g->queue commandBuffer];
    *e = [*cb computeCommandEncoder];
}

#define STAGE(name) stage(g, &cb, &e, name)

bool gpu_run(Gpu *g, size_t len, bool parse, GpuResult *out) {
    memset(out, 0, sizeof *out);
    if (len > g->max_len) return false;
    uint32_t len32 = (uint32_t)len;
    uint32_t nchunks = (uint32_t)((len + CHUNK - 1) / CHUNK);

    double t0 = ms_now();
    if (len == 0) {
        // No chunks to lex: the only token is EOF.
        ((uint8_t *)g->tok_type.contents)[0] = T_EOF;
        ((uint32_t *)g->tok_start.contents)[0] = 0;
        ((uint32_t *)g->tok_end.contents)[0] = 0;
        ((uint32_t *)g->flags.contents)[0] = 0;
        ((uint32_t *)g->flags.contents)[1] = 1;
    }
    id<MTLCommandBuffer> cb = [g->queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    if (len) [blit fillBuffer:g->flags range:NSMakeRange(0, 16) value:0];
    [blit endEncoding];

    Enc e = [cb computeCommandEncoder];
    set_bufs(e, @[ g->src ]);
    set_u32(e, len32, 1);
    set_u32(e, CHUNK, 2);
    set_u32(e, nchunks, 3);
    [e setBuffer:g->step offset:0 atIndex:4];
    [e setBuffer:g->img offset:0 atIndex:5];
    [e setBuffer:g->nimg offset:0 atIndex:6];
    [e setBuffer:g->sum_end[0] offset:0 atIndex:7];
    [e setBuffer:g->sum_cnt[0] offset:0 atIndex:8];
    [e setBuffer:g->sum_b0[0] offset:0 atIndex:9];
    [e setBuffer:g->ctab offset:0 atIndex:10];
    [e setBuffer:g->cmap offset:0 atIndex:11];
    [e setBuffer:g->tok_start offset:0 atIndex:12];
    dispatch(g, e, @"lex_summary", nchunks);
    STAGE("summary");

    uint32_t n_at[16];
    n_at[0] = nchunks;
    int top = 0;
    while (n_at[top] > 1) {
        uint32_t n_in = n_at[top], n_out = (n_in + GROUP - 1) / GROUP;
        [e setBuffer:g->sum_end[top] offset:0 atIndex:0];
        [e setBuffer:g->sum_cnt[top] offset:0 atIndex:1];
        [e setBuffer:g->sum_b0[top] offset:0 atIndex:2];
        set_u32(e, n_in, 3);
        set_u32(e, GROUP, 4);
        [e setBuffer:g->step offset:0 atIndex:5];
        [e setBuffer:g->nimg offset:0 atIndex:6];
        [e setBuffer:g->imgpos offset:0 atIndex:7];
        [e setBuffer:g->sum_end[top + 1] offset:0 atIndex:8];
        [e setBuffer:g->sum_cnt[top + 1] offset:0 atIndex:9];
        [e setBuffer:g->sum_b0[top + 1] offset:0 atIndex:10];
        dispatch(g, e, @"lex_reduce", (NSUInteger)n_out * g->maxk);
        n_at[++top] = n_out;
    }
    STAGE("reduce");
    // Down-sweep from the single top summary, which starts in the start state
    // with no tokens before it.
    for (int l = top; l >= 0; l--) {
        // Incoming of level l elements from level l+1 (or the virtual root).
        uint32_t n_in = n_at[l];
        uint32_t ng = l == top ? 1 : n_at[l + 1];
        [e setBuffer:g->sum_end[l] offset:0 atIndex:0];
        [e setBuffer:g->sum_cnt[l] offset:0 atIndex:1];
        [e setBuffer:g->sum_b0[l] offset:0 atIndex:2];
        set_u32(e, n_in, 3);
        set_u32(e, l == top ? n_in : GROUP, 4);
        [e setBuffer:g->step offset:0 atIndex:5];
        [e setBuffer:g->imgpos offset:0 atIndex:6];
        if (l == top) {
            static const uint16_t start_state = 1;
            static const uint32_t zero = 0;
            [e setBytes:&start_state length:2 atIndex:7];
            [e setBytes:&zero length:4 atIndex:8];
        } else {
            [e setBuffer:g->inc_s[l + 1] offset:0 atIndex:7];
            [e setBuffer:g->inc_c[l + 1] offset:0 atIndex:8];
        }
        [e setBuffer:g->inc_s[l] offset:0 atIndex:9];
        [e setBuffer:g->inc_c[l] offset:0 atIndex:10];
        set_u32(e, ng, 11);
        dispatch(g, e, @"lex_down", ng);
    }
    STAGE("down");

    set_bufs(e, @[ g->src ]);
    set_u32(e, len32, 1);
    set_u32(e, CHUNK, 2);
    set_u32(e, nchunks, 3);
    [e setBuffer:g->step offset:0 atIndex:4];
    [e setBuffer:g->types offset:0 atIndex:5];
    [e setBuffer:g->inc_s[0] offset:0 atIndex:6];
    [e setBuffer:g->inc_c[0] offset:0 atIndex:7];
    [e setBuffer:g->tok_type offset:0 atIndex:8];
    [e setBuffer:g->tok_start offset:0 atIndex:9];
    [e setBuffer:g->tok_end offset:0 atIndex:10];
    [e setBuffer:g->flags offset:0 atIndex:11];
    [e setBuffer:g->ctab offset:0 atIndex:12];
    [e setBuffer:g->cmap offset:0 atIndex:13];
    dispatch(g, e, @"lex_emit", nchunks);
    STAGE("emit");
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    double t1 = ms_now();

    uint32_t *flags = g->flags.contents;
    out->lex_ok = flags[0] == 0;
    out->toks.count = flags[1];
    out->toks.type = g->tok_type.contents;
    out->toks.start = g->tok_start.contents;
    out->toks.end = g->tok_end.contents;
    out->lex_ms = t1 - t0;
    if (!out->lex_ok || !parse) return out->lex_ok;

    uint32_t n = out->toks.count;
    Levels L = levels_for(n);
    cb = [g->queue commandBuffer];
    blit = [cb blitCommandEncoder];
    [blit fillBuffer:g->perr range:NSMakeRange(0, 4) value:0];
    [blit fillBuffer:g->perr range:NSMakeRange(4, 4) value:0xFF];
    [blit endEncoding];
    e = [cb computeCommandEncoder];

    set_bufs(e, @[ g->tok_type, g->tok_start, g->tok_end, g->src ]);
    set_u32(e, n, 4);
    [e setBuffer:g->kind offset:0 atIndex:5];
    [e setBuffer:g->delta offset:0 atIndex:6];
    [e setBuffer:g->match offset:0 atIndex:7];
    [e setBuffer:g->cflags offset:0 atIndex:8];
    [e setBuffer:g->sepbits offset:0 atIndex:9];
    dispatch(g, e, @"p_classify", n);
    STAGE("classify");

    scan(g, e, g->delta, g->eb, n, false, 0);
    STAGE("scan depth");

    set_bufs(e, @[ g->kind, g->eb ]);
    set_u32(e, n, 2);
    [e setBuffer:g->t_open offset:0 atIndex:3];
    dispatch(g, e, @"p_open_vals", n);
    build_tree(g, e, g->t_open, &L, false, false);
    STAGE("tree open");

    set_bufs(e, @[ g->kind, g->eb ]);
    set_u32(e, n, 2);
    [e setBuffer:g->t_open offset:0 atIndex:3];
    [e setBytes:&L length:sizeof L atIndex:4];
    [e setBuffer:g->match offset:0 atIndex:5];
    [e setBuffer:g->sdelta offset:0 atIndex:6];
    [e setBuffer:g->perr offset:0 atIndex:7];
    [e setBuffer:g->tok_type offset:0 atIndex:8];
    dispatch(g, e, @"p_match", n);
    STAGE("match");
    scan(g, e, g->sdelta, g->sb, n, false, 0);
    STAGE("scan sdepth");

    set_bufs(e, @[ g->kind, g->eb ]);
    set_u32(e, n, 2);
    [e setBuffer:g->t_term offset:0 atIndex:3];
    [e setBuffer:g->idelta offset:0 atIndex:4];
    [e setBuffer:g->match offset:0 atIndex:5];
    [e setBuffer:g->kind offset:0 atIndex:6];
    dispatch(g, e, @"p_term_vals", n);
    STAGE("term vals");
    build_tree(g, e, g->t_term, &L, false, true);
    STAGE("tree term");

    set_bufs(e, @[ g->kind, g->eb ]);
    set_u32(e, n, 2);
    [e setBuffer:g->t_term offset:0 atIndex:3];
    [e setBytes:&L length:sizeof L atIndex:4];
    [e setBuffer:g->idelta offset:0 atIndex:5];
    [e setBuffer:g->perr offset:0 atIndex:6];
    dispatch(g, e, @"p_implicit", n);
    STAGE("implicit");
    scan(g, e, g->idelta, g->id_, n, true, 0);
    STAGE("scan implicit");

    set_bufs(e, @[ g->kind, g->eb, g->sb, g->id_, g->match ]);
    set_u32(e, n, 5);
    [e setBuffer:g->t_keys offset:0 atIndex:6];
    [e setBuffer:g->sdepth offset:0 atIndex:7];
    dispatch(g, e, @"p_keys", n);
    STAGE("keys");
    build_tree(g, e, g->t_keys, &L, true, false);
    STAGE("tree keys");

    [e setBuffer:g->t_keys offset:0 atIndex:0];
    [e setBytes:&L length:sizeof L atIndex:1];
    set_u32(e, n, 2);
    [e setBuffer:g->left offset:0 atIndex:3];
    [e setBuffer:g->right offset:0 atIndex:4];
    dispatch(g, e, @"p_ansv", n);
    STAGE("ansv");

    set_bufs(e, @[ g->kind, g->t_keys, g->sdepth, g->left, g->right, g->match ]);
    set_u32(e, n, 6);
    [e setBuffer:g->parent offset:0 atIndex:7];
    [e setBuffer:g->how offset:0 atIndex:8];
    [e setBuffer:g->cflags offset:0 atIndex:9];
    dispatch(g, e, @"p_parent", n);
    STAGE("parent");

    set_bufs(e, @[ g->kind, g->tok_type, g->eb, g->left ]);
    set_u32(e, n, 4);
    [e setBuffer:g->sepbits offset:0 atIndex:5];
    [e setBuffer:g->perr offset:0 atIndex:6];
    dispatch(g, e, @"p_seps", n);
    STAGE("seps");
    set_bufs(e, @[ g->kind, g->tok_type, g->eb, g->parent, g->how, g->left, g->match, g->cflags, g->sepbits ]);
    set_u32(e, n, 9);
    [e setBuffer:g->perr offset:0 atIndex:10];
    dispatch(g, e, @"p_check_pairs", n);
    dispatch(g, e, @"p_check_nodes", n);
    STAGE("validate");
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    double t2 = ms_now();

    out->parent = g->parent.contents;
    out->parse_ok = ((uint32_t *)g->perr.contents)[0] == 0;
    out->error_at = ((uint32_t *)g->perr.contents)[1];
    out->parse_ms = t2 - t1;
    return out->parse_ok;
}
