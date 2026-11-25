// GPU lexer and parser for JPL.
//
// Lexer: the DFA table is resolved into a per-byte step function. Each chunk
// of bytes is summarized as "first byte + end state and token count for every
// state the first byte can lead to" (at most MAXK of them). Summaries compose,
// so a hierarchical scan gives every chunk its true incoming state, and each
// chunk then re-lexes itself and writes its tokens in place.
//
// Parser: every structural question is "nearest earlier/later element <= x",
// answered by walking an 8-ary min tree: bracket matching, where implicit
// groups (array/sum bodies, else branches, command lines) end, and the
// Cartesian tree of (depth, precedence, position) keys that is the AST. A
// segment's root attaches to its group's owner, found from its left neighbour.
// Validation then checks adjacent token pairs, separators, and node shapes, so
// the GPU rejects exactly what the CPU parser rejects.
#include <metal_stdlib>
using namespace metal;

// Set by the host: MAXK (largest per-byte image), NSTATES, NFAST, NCLS.
#define EMIT 0x200u
#define RESET_AT 0x400u
#define RESET_AFTER 0x800u
#define STATE_MASK 0x1FFu
#define S_START 1u
#define NONE 0xFFFFFFFFu
#define INF32 0xFFFFFFFFu
#define INF64 0xFFFFFFFFFFFFFFFFul

// ---------------------------------------------------------------- lexer

// States below NFAST (everything but partly-read UTF-8 letters) use a table
// compressed by byte class, kept in threadgroup memory.
static void load_fast(threadgroup ushort *ttab, threadgroup uchar *tcls, device const ushort *ctab,
                      device const uchar *cmap, uint lid, uint tgs) {
    for (uint i = lid; i < NFAST * NCLS; i += tgs) ttab[i] = ctab[i];
    for (uint i = lid; i < 256; i += tgs) tcls[i] = cmap[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

#define STEP(s, b) ((s) < NFAST ? uint(ttab[(s) * NCLS + tcls[b]]) : uint(step[(s) * 256 + (b)]))
#define BYTE(v, q) (((v)[(q) >> 2] >> (((q) & 3) * 8)) & 255u)

kernel void lex_summary(device const uchar *src [[buffer(0)]],
                        constant uint &len [[buffer(1)]],
                        constant uint &chunk [[buffer(2)]],
                        constant uint &nchunks [[buffer(3)]],
                        device const ushort *step [[buffer(4)]],
                        device const ushort *img [[buffer(5)]],
                        device const uchar *nimg [[buffer(6)]],
                        device ushort *out_end [[buffer(7)]],
                        device uint *out_cnt [[buffer(8)]],
                        device uchar *out_b0 [[buffer(9)]],
                        device const ushort *ctab [[buffer(10)]],
                        device const uchar *cmap [[buffer(11)]],
                        device uint4 *tok_start [[buffer(12)]],
                        uint c [[thread_position_in_grid]],
                        uint lid [[thread_position_in_threadgroup]],
                        uint tgs [[threads_per_threadgroup]]) {
    threadgroup ushort ttab[NFAST * NCLS];
    threadgroup uchar tcls[256];
    load_fast(ttab, tcls, ctab, cmap, lid, tgs);
    if (c >= nchunks) return;
    uint base = c * chunk;
    uint size = min(chunk, len - base);
    // Token i ends at or after byte i - 1, so clearing the start slots of this
    // chunk's byte range (and two past the end) clears every slot emit uses.
    uint clear_end = c + 1 == nchunks ? base + chunk + 4 : base + chunk;
    for (uint i = base / 4; i < clear_end / 4; i++) tok_start[i] = uint4(0);
    device const uint4 *blocks = (device const uint4 *)(src + base);
    uint4 v = blocks[0];
    uint b0 = v.x & 255u;
    uint K = nimg[b0];

    ushort st[MAXK];
    uint cnt[MAXK];
    uchar lead[MAXK];
    int off[MAXK];
    uchar act[MAXK];
    for (uint k = 0; k < K; k++) {
        st[k] = img[b0 * MAXK + k];
        cnt[k] = 0;
        lead[k] = k;
        off[k] = 0;
        act[k] = k;
    }
    uint A = K;

    uint at = 0;
    while (at < size && A > 1) {
        if (at > 0) v = blocks[at / 16];
        uint lo = at == 0 ? 1 : 0;
        uint hi = min(16u, size - at);
        for (uint a = 0; a < A; a++) {
            uint k = act[a];
            uint s = st[k];
            uint n = cnt[k];
            for (uint q = 0; q < 16; q++) {
                if (q < lo || q >= hi) continue;
                uint e = STEP(s, BYTE(v, q));
                n += (e >> 9) & 1u;
                s = e & STATE_MASK;
            }
            st[k] = s;
            cnt[k] = n;
        }
        at += 16;
        // Lanes that reached the same state behave identically from now on;
        // lanes in the error state stay there and need no more work.
        for (uint a = A; a-- > 0;) {
            uint k = act[a];
            bool drop = st[k] == 0 && A > 1;
            for (uint b = 0; b < a && !drop; b++) {
                uint into = act[b];
                if (st[into] != st[k]) continue;
                int d = int(cnt[k]) - int(cnt[into]);
                for (uint x = 0; x < K; x++) {
                    if (lead[x] == k) {
                        lead[x] = into;
                        off[x] += d;
                    }
                }
                drop = true;
            }
            if (drop) {
                act[a] = act[A - 1];
                A--;
            }
        }
    }
    if (at < size) {
        // One live lane: plain registers.
        uint k = act[0];
        uint s = st[k];
        uint n = cnt[k];
        for (; at < size; at += 16) {
            if (at > 0) v = blocks[at / 16];
            uint lo = at == 0 ? 1 : 0;
            uint hi = min(16u, size - at);
            for (uint q = 0; q < 16; q++) {
                if (q < lo || q >= hi) continue;
                uint e = STEP(s, BYTE(v, q));
                n += (e >> 9) & 1u;
                s = e & STATE_MASK;
            }
        }
        st[k] = s;
        cnt[k] = n;
    }

    for (uint k = 0; k < K; k++) {
        uint l = lead[k];
        out_end[c * MAXK + k] = st[l];
        out_cnt[c * MAXK + k] = uint(int(cnt[l]) + off[k]);
    }
    out_b0[c] = b0;
}

// Composes groups of `group` consecutive summaries into one.
kernel void lex_reduce(device const ushort *in_end [[buffer(0)]],
                       device const uint *in_cnt [[buffer(1)]],
                       device const uchar *in_b0 [[buffer(2)]],
                       constant uint &n_in [[buffer(3)]],
                       constant uint &group [[buffer(4)]],
                       device const ushort *step [[buffer(5)]],
                       device const uchar *nimg [[buffer(6)]],
                       device const uchar *imgpos [[buffer(7)]],
                       device ushort *out_end [[buffer(8)]],
                       device uint *out_cnt [[buffer(9)]],
                       device uchar *out_b0 [[buffer(10)]],
                       uint tid [[thread_position_in_grid]]) {
    uint g = tid / MAXK;
    uint k = tid % MAXK;
    uint first = g * group;
    if (first >= n_in) return;
    uchar b0 = in_b0[first];
    if (k == 0) out_b0[g] = b0;
    if (k >= nimg[b0]) return;
    uint s = in_end[first * MAXK + k];
    uint n = in_cnt[first * MAXK + k];
    uint last = min(first + group, n_in);
    for (uint i = first + 1; i < last; i++) {
        uint b = in_b0[i];
        uint e = step[s * 256 + b];
        n += (e >> 9) & 1u;
        uint idx = imgpos[b * NSTATES + (e & STATE_MASK)];
        s = in_end[i * MAXK + idx];
        n += in_cnt[i * MAXK + idx];
    }
    out_end[g * MAXK + k] = s;
    out_cnt[g * MAXK + k] = n;
}

// Given the incoming state of each group, finds the incoming state of each
// of its children.
kernel void lex_down(device const ushort *in_end [[buffer(0)]],
                     device const uint *in_cnt [[buffer(1)]],
                     device const uchar *in_b0 [[buffer(2)]],
                     constant uint &n_in [[buffer(3)]],
                     constant uint &group [[buffer(4)]],
                     device const ushort *step [[buffer(5)]],
                     device const uchar *imgpos [[buffer(6)]],
                     device const ushort *inc_s [[buffer(7)]],
                     device const uint *inc_c [[buffer(8)]],
                     device ushort *child_s [[buffer(9)]],
                     device uint *child_c [[buffer(10)]],
                     constant uint &n_groups [[buffer(11)]],
                     uint g [[thread_position_in_grid]]) {
    if (g >= n_groups) return;
    uint s = inc_s[g];
    uint n = inc_c[g];
    uint first = g * group;
    uint last = min(first + group, n_in);
    for (uint i = first; i < last; i++) {
        child_s[i] = s;
        child_c[i] = n;
        uint b = in_b0[i];
        uint e = step[s * 256 + b];
        n += (e >> 9) & 1u;
        uint idx = imgpos[b * NSTATES + (e & STATE_MASK)];
        s = in_end[i * MAXK + idx];
        n += in_cnt[i * MAXK + idx];
    }
}

kernel void lex_emit(device const uchar *src [[buffer(0)]],
                     constant uint &len [[buffer(1)]],
                     constant uint &chunk [[buffer(2)]],
                     constant uint &nchunks [[buffer(3)]],
                     device const ushort *step [[buffer(4)]],
                     device const uchar *types [[buffer(5)]],
                     device const ushort *inc_s [[buffer(6)]],
                     device const uint *inc_c [[buffer(7)]],
                     device uchar *tok_type [[buffer(8)]],
                     device atomic_uint *tok_start [[buffer(9)]],
                     device uint *tok_end [[buffer(10)]],
                     device atomic_uint *flags [[buffer(11)]],
                     device const ushort *ctab [[buffer(12)]],
                     device const uchar *cmap [[buffer(13)]],
                     uint c [[thread_position_in_grid]],
                     uint lid [[thread_position_in_threadgroup]],
                     uint tgs [[threads_per_threadgroup]]) {
    threadgroup ushort ttab[NFAST * NCLS];
    threadgroup uchar tcls[256];
    load_fast(ttab, tcls, ctab, cmap, lid, tgs);
    if (c >= nchunks) return;
    uint base = c * chunk;
    uint size = min(chunk, len - base);
    device const uint4 *blocks = (device const uint4 *)(src + base);
    uint s = inc_s[c];
    uint n = inc_c[c];
    bool have = false;
    uint ip = 0;
    for (uint at = 0; at < size; at += 16) {
        uint4 v = blocks[at / 16];
        uint hi = min(16u, size - at);
        for (uint q = 0; q < 16; q++) {
            if (q >= hi) break;
            uint j = base + at + q;
            uint e = STEP(s, BYTE(v, q));
            if (e & EMIT) {
                tok_type[n] = types[s];
                tok_end[n] = j;
                if (have) atomic_store_explicit(&tok_start[n], ip, memory_order_relaxed);
                n++;
            }
            if (e & (RESET_AT | RESET_AFTER)) {
                ip = (e & RESET_AT) ? j : j + 1;
                have = true;
            }
            s = e & STATE_MASK;
            if (s == 0) {
                atomic_store_explicit(&flags[0], 1u, memory_order_relaxed);
                return;
            }
        }
    }
    if (c + 1 == nchunks) {
        if (s != S_START) {
            uint t = types[s];
            if (t == 2) {
                atomic_store_explicit(&flags[0], 1u, memory_order_relaxed);
                return;
            }
            tok_type[n] = t;
            tok_end[n] = len;
            if (have) atomic_fetch_max_explicit(&tok_start[n], ip, memory_order_relaxed);
            n++;
        }
        tok_type[n] = 43;
        tok_end[n] = len;
        atomic_fetch_max_explicit(&tok_start[n], len, memory_order_relaxed);
        atomic_store_explicit(&flags[1], n + 1, memory_order_relaxed);
    } else if (have) {
        atomic_fetch_max_explicit(&tok_start[n], ip, memory_order_relaxed);
    }
}

// ---------------------------------------------------------------- scans

#define SCAN_TG 256
#define SCAN_PER 4
#define SCAN_BLOCK (SCAN_TG * SCAN_PER)

static int block_exclusive_scan(thread int *v, threadgroup int *tg_sums, uint lid, uint simd_lane, uint simd_id) {
    int local = 0;
    for (uint i = 0; i < SCAN_PER; i++) {
        int x = v[i];
        v[i] = local;
        local += x;
    }
    int pre = simd_prefix_exclusive_sum(local);
    if (simd_lane == 31) tg_sums[simd_id] = pre + local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        int t = simd_lane < (SCAN_TG / 32) ? tg_sums[simd_lane] : 0;
        int tp = simd_prefix_exclusive_sum(t);
        if (simd_lane < (SCAN_TG / 32)) tg_sums[simd_lane] = tp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int offset = tg_sums[simd_id] + pre;
    for (uint i = 0; i < SCAN_PER; i++) v[i] += offset;
    return 0;
}

kernel void scan_reduce(device const int *in [[buffer(0)]],
                        constant uint &n [[buffer(1)]],
                        device int *sums [[buffer(2)]],
                        uint gid [[threadgroup_position_in_grid]],
                        uint lid [[thread_position_in_threadgroup]],
                        uint simd_lane [[thread_index_in_simdgroup]],
                        uint simd_id [[simdgroup_index_in_threadgroup]]) {
    threadgroup int tg_sums[SCAN_TG / 32];
    uint base = gid * SCAN_BLOCK + lid * SCAN_PER;
    int local = 0;
    for (uint i = 0; i < SCAN_PER; i++) local += base + i < n ? in[base + i] : 0;
    int s = simd_sum(local);
    if (simd_lane == 0) tg_sums[simd_id] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        int t = 0;
        for (uint i = 0; i < SCAN_TG / 32; i++) t += tg_sums[i];
        sums[gid] = t;
    }
}

// out[i] = offset[block] + sum(in[block start .. i)) (+ in[i] if inclusive)
kernel void scan_apply(device const int *in [[buffer(0)]],
                       constant uint &n [[buffer(1)]],
                       device const int *offsets [[buffer(2)]],
                       device int *out [[buffer(3)]],
                       constant uint &inclusive [[buffer(4)]],
                       uint gid [[threadgroup_position_in_grid]],
                       uint lid [[thread_position_in_threadgroup]],
                       uint simd_lane [[thread_index_in_simdgroup]],
                       uint simd_id [[simdgroup_index_in_threadgroup]]) {
    threadgroup int tg_sums[SCAN_TG / 32];
    uint base = gid * SCAN_BLOCK + lid * SCAN_PER;
    int v[SCAN_PER];
    int orig[SCAN_PER];
    for (uint i = 0; i < SCAN_PER; i++) orig[i] = v[i] = base + i < n ? in[base + i] : 0;
    block_exclusive_scan(v, tg_sums, lid, simd_lane, simd_id);
    int off = offsets[gid];
    for (uint i = 0; i < SCAN_PER; i++) {
        if (base + i < n) out[base + i] = v[i] + off + (inclusive ? orig[i] : 0);
    }
}

// ---------------------------------------------------------------- segment trees
//
// Level 0 is the value array; level l+1 holds the min of pairs of level l.
// `offs` gives each level's offset into the tree buffer, `sizes` its length.

struct Levels {
    uint count;
    uint offs[32];
    uint sizes[32];
};

// 8-ary min trees: node i of level l+1 is the min of nodes 8i .. 8i+7 of
// level l, so a node's siblings share one cache line and a walk touches about
// log8(n) lines.
template <typename T>
static T tmin(T a, T b, bool packed) {
    if (!packed) return min(a, b);
    uint x = uint(a), y = uint(b);
    return T(min(x & 0xFFFFu, y & 0xFFFFu) | (min(x >> 16, y >> 16) << 16));
}

template <typename T>
static void build_level(device T *tree, uint lvl, constant Levels &L, uint i, T inf, bool packed) {
    if (i >= L.sizes[lvl + 1]) return;
    device const T *in = tree + L.offs[lvl];
    uint size = L.sizes[lvl];
    T v = inf;
    for (uint c = 0; c < 8; c++) {
        uint j = 8 * i + c;
        if (j < size) v = tmin(v, in[j], packed);
    }
    tree[L.offs[lvl + 1] + i] = v;
}

kernel void tree_build32(device uint *tree [[buffer(0)]],
                         constant uint &lvl [[buffer(1)]],
                         constant Levels &L [[buffer(2)]],
                         constant uint &packed [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    build_level<uint>(tree, lvl, L, i, INF32, packed != 0);
}

kernel void tree_build64(device ulong *tree [[buffer(0)]],
                         constant uint &lvl [[buffer(1)]],
                         constant Levels &L [[buffer(2)]],
                         constant uint &packed [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    build_level<ulong>(tree, lvl, L, i, INF64, false);
}

// Predicates for queries.
struct LE32 { uint x; bool ok(uint v) const { return v <= x; } };
struct LT64 { ulong x; bool ok(ulong v) const { return v < x; } };
struct TERM { // packed terminator depths: high half lines, low half expressions
    uint x;
    bool line;
    bool ok(uint v) const { return (line ? v >> 16 : v & 0xFFFFu) <= x; }
};

// Nearest j < k whose value satisfies p.
template <typename T, typename P>
static uint prev_q(device const T *tree, constant Levels &L, uint k, P p) {
    uint pos = k, l = 0;
    for (;;) {
        uint g = pos & ~7u;
        device const T *lv = tree + L.offs[l];
        for (uint j = pos; j-- > g;) {
            if (!p.ok(lv[j])) continue;
            uint node = j;
            while (l > 0) {
                l--;
                device const T *cv = tree + L.offs[l];
                uint c = min(8 * node + 8, L.sizes[l]);
                while (!p.ok(cv[--c])) {}
                node = c;
            }
            return node;
        }
        pos >>= 3;
        l++;
        if (l >= L.count) return NONE;
    }
}

// Nearest j > k whose value satisfies p.
template <typename T, typename P>
static uint next_q(device const T *tree, constant Levels &L, uint k, P p) {
    uint pos = k, l = 0;
    for (;;) {
        uint end = min((pos | 7u) + 1, L.sizes[l]);
        device const T *lv = tree + L.offs[l];
        for (uint j = pos + 1; j < end; j++) {
            if (!p.ok(lv[j])) continue;
            uint node = j;
            while (l > 0) {
                l--;
                device const T *cv = tree + L.offs[l];
                uint c = 8 * node;
                while (!p.ok(cv[c])) c++;
                node = c;
            }
            return node;
        }
        pos >>= 3;
        l++;
        if (l >= L.count) return NONE;
    }
}

// ---------------------------------------------------------------- parser

// Token types (jpl.h).
enum : uchar {
    T_ASSERT = 4, T_ARRAY, T_BOOL, T_ELSE, T_FALSE, T_FN, T_FLOAT, T_INT, T_IMAGE,
    T_IF, T_LET, T_PRINT, T_RETURN, T_READ, T_STRUCT, T_SHOW, T_SUM, T_TO,
    T_TIME, T_TRUE, T_THEN, T_VOID, T_WRITE,
    T_VARIABLE, T_COLON, T_RPAREN, T_LPAREN, T_RCURLY, T_RSQUARE, T_LSQUARE,
    T_LCURLY, T_NEWLINE, T_INTVAL, T_FLOATVAL, T_OP, T_COMMA, T_EQUALS,
    T_STRING, T_DOT, T_EOF,
};

// What a token is to the parser.
enum : uchar {
    K_LEAF,       // variables, literals, strings, base types
    K_CMD,        // command / statement keyword: opens a line group
    K_ARRAYSUM,   // array, sum: opens an expression group
    K_IF,         // opens if ... else
    K_ELSE,       // closes if ... else, opens the else branch
    K_THEN,       // separator inside if ... else
    K_OR, K_CMP, K_ADD, K_MUL,
    K_UNARY,
    K_DOT,
    K_LSQ_POST,   // index, array rank, argument dimensions
    K_LSQ_LIT,    // array literal
    K_LSQ_BIND,   // array / sum binding list
    K_LPAR_CALL,
    K_LPAR_FN,    // function parameter list
    K_LPAR_GROUP, // grouping: invisible
    K_LCUR_LIT,   // struct literal
    K_LCUR_BLOCK, // function / struct body
    K_CLOSE,      // ) ] }
    K_COLON_BIND, // x : T, i : N
    K_COLON_RET,  // ) : T
    K_COMMA,
    K_NEWLINE,    // also EOF
    K_EQUALS,
    K_IMAGE,
    K_TO,
    K_MINUS_AFTER_SQ, // - after ]: binary unless the ] ends a binding list
};

// Precedence classes for keys; 0 is punctuation.
enum : uint { P_PUNCT, P_CMD, P_BIND, P_OR, P_CMP, P_ADD, P_MUL, P_UNARY, P_POST, P_LEAF };

static bool operand_end(uchar t) {
    switch (t) {
    case T_VARIABLE: case T_INTVAL: case T_FLOATVAL: case T_TRUE: case T_FALSE:
    case T_VOID: case T_RPAREN: case T_RSQUARE: case T_RCURLY: case T_STRING:
    case T_INT: case T_BOOL: case T_FLOAT:
        return true;
    default:
        return false;
    }
}

kernel void p_classify(device const uchar *ty [[buffer(0)]],
                       device const uint *tstart [[buffer(1)]],
                       device const uint *tend [[buffer(2)]],
                       device const uchar *src [[buffer(3)]],
                       constant uint &n [[buffer(4)]],
                       device uchar *kind [[buffer(5)]],
                       device int *delta [[buffer(6)]],
                       device uint *match [[buffer(7)]],
                       device uint *cflags [[buffer(8)]],
                       device uint *sepbits [[buffer(9)]],
                       uint k [[thread_position_in_grid]]) {
    if (k >= n) return;
    match[k] = NONE;
    cflags[k] = 0;
    sepbits[k] = 0;
    uchar t = ty[k];
    uchar prev = k > 0 ? ty[k - 1] : T_NEWLINE;
    uchar k_;
    int d = 0;
    switch (t) {
    case T_LET: case T_ASSERT: case T_RETURN: case T_SHOW: case T_PRINT:
    case T_READ: case T_WRITE: case T_TIME: case T_FN: case T_STRUCT:
        k_ = K_CMD; break;
    case T_ARRAY: case T_SUM: k_ = K_ARRAYSUM; break;
    case T_IF: k_ = K_IF; d = 1; break;
    case T_ELSE: k_ = K_ELSE; d = -1; break;
    case T_THEN: k_ = K_THEN; break;
    case T_OP: {
        uchar c = src[tstart[k]];
        uint l = tend[k] - tstart[k];
        if (l == 1 && (c == '!' || (c == '-' && !operand_end(prev)))) k_ = K_UNARY;
        else if (c == '&' || c == '|') k_ = K_OR;
        else if (c == '<' || c == '>' || c == '=' || c == '!') k_ = K_CMP;
        else if (c == '-' && prev == T_RSQUARE) k_ = K_MINUS_AFTER_SQ;
        else if (c == '+' || c == '-') k_ = K_ADD;
        else k_ = K_MUL;
        break;
    }
    case T_DOT: k_ = K_DOT; break;
    case T_LSQUARE:
        d = 1;
        if (prev == T_ARRAY || prev == T_SUM) k_ = K_LSQ_BIND;
        else if (operand_end(prev) && prev != T_STRING) k_ = K_LSQ_POST;
        else k_ = K_LSQ_LIT;
        break;
    case T_LPAREN:
        d = 1;
        if (prev == T_VARIABLE) k_ = (k > 1 && ty[k - 2] == T_FN) ? K_LPAR_FN : K_LPAR_CALL;
        else k_ = K_LPAR_GROUP;
        break;
    case T_LCURLY:
        d = 1;
        k_ = (k + 1 < n && ty[k + 1] == T_NEWLINE) ? K_LCUR_BLOCK : K_LCUR_LIT;
        break;
    case T_RPAREN: case T_RSQUARE: case T_RCURLY: k_ = K_CLOSE; d = -1; break;
    case T_COLON: k_ = prev == T_RPAREN ? K_COLON_RET : K_COLON_BIND; break;
    case T_COMMA: k_ = K_COMMA; break;
    case T_NEWLINE: case T_EOF: k_ = K_NEWLINE; break;
    case T_EQUALS: k_ = K_EQUALS; break;
    case T_IMAGE: k_ = K_IMAGE; break;
    case T_TO: k_ = K_TO; break;
    default: k_ = K_LEAF; break;
    }
    kind[k] = k_;
    delta[k] = d;
}

static bool is_opener(uchar k) {
    return k == K_IF || (k >= K_LSQ_POST && k <= K_LCUR_BLOCK);
}

// Openers' interior depth, for matching closers.
kernel void p_open_vals(device const uchar *kind [[buffer(0)]],
                        device const int *eb [[buffer(1)]],
                        constant uint &n [[buffer(2)]],
                        device uint *tree [[buffer(3)]],
                        uint k [[thread_position_in_grid]]) {
    if (k >= n) return;
    tree[k] = is_opener(kind[k]) ? uint(eb[k] + 1) : INF32;
}

static bool brackets_match(uchar close_type, uchar open_kind) {
    switch (close_type) {
    case T_RPAREN: return open_kind == K_LPAR_CALL || open_kind == K_LPAR_FN || open_kind == K_LPAR_GROUP;
    case T_RSQUARE: return open_kind == K_LSQ_POST || open_kind == K_LSQ_LIT || open_kind == K_LSQ_BIND;
    case T_RCURLY: return open_kind == K_LCUR_LIT || open_kind == K_LCUR_BLOCK;
    default: return open_kind == K_IF; // else
    }
}

kernel void p_match(device const uchar *kind [[buffer(0)]],
                    device const int *eb [[buffer(1)]],
                    constant uint &n [[buffer(2)]],
                    device const uint *tree [[buffer(3)]],
                    constant Levels &L [[buffer(4)]],
                    device uint *match [[buffer(5)]],
                    device int *sdelta [[buffer(6)]],
                    device atomic_uint *err [[buffer(7)]],
                    device const uchar *ty [[buffer(8)]],
                    uint k [[thread_position_in_grid]]) {
    if (k >= n) return;
    uchar kk = kind[k];
    if (kk == K_LPAR_GROUP) {
        sdelta[k] = 0;
        return;
    }
    if (kk != K_CLOSE && kk != K_ELSE) {
        sdelta[k] = is_opener(kk) ? 1 : 0;
        return;
    }
    uint j = eb[k] < 1 ? NONE : prev_q(tree, L, k, LE32{uint(eb[k])});
    if (j == NONE || !brackets_match(ty[k], kind[j])) {
        atomic_store_explicit(&err[0], 1u, memory_order_relaxed);
        atomic_fetch_min_explicit(&err[1], (k << 4) | 9u, memory_order_relaxed);
        sdelta[k] = -1;
        return;
    }
    match[k] = j;
    match[j] = k;
    sdelta[k] = kind[j] == K_LPAR_GROUP ? 0 : -1;
}

static bool expr_term(uchar k) {
    return k == K_COMMA || k == K_THEN || k == K_NEWLINE || k == K_TO || k == K_CLOSE || k == K_ELSE;
}

// Terminator depths, packed: high half for lines (newline, EOF), low half
// for expressions (any separator or closer).
kernel void p_term_vals(device const uchar *kind [[buffer(0)]],
                        device const int *eb [[buffer(1)]],
                        constant uint &n [[buffer(2)]],
                        device uint *tterm [[buffer(3)]],
                        device int *idelta [[buffer(4)]],
                        device const uint *match [[buffer(5)]],
                        device uchar *kind_rw [[buffer(6)]],
                        uint k [[thread_position_in_grid]]) {
    if (k >= n) return;
    uchar kk = kind[k];
    // After the ] of a binding list an operand starts: - is unary, [ a literal.
    if (kk == K_MINUS_AFTER_SQ || kk == K_LSQ_POST) {
        uint m = k > 0 ? match[k - 1] : NONE;
        bool after_bind = k > 0 && kind[k - 1] == K_CLOSE && m < n && kind[m] == K_LSQ_BIND;
        if (kk == K_MINUS_AFTER_SQ) kind_rw[k] = after_bind ? K_UNARY : K_ADD;
        else if (after_bind) kind_rw[k] = K_LSQ_LIT;
    }
    uint d = min(uint(eb[k]), 0xFFFEu);
    uint line = kk == K_NEWLINE ? d : 0xFFFFu;
    uint expr = expr_term(kk) ? d : 0xFFFFu;
    tterm[k] = (line << 16) | expr;
    idelta[k] = 0;
}

kernel void p_implicit(device const uchar *kind [[buffer(0)]],
                       device const int *eb [[buffer(1)]],
                       constant uint &n [[buffer(2)]],
                       device const uint *tterm [[buffer(3)]],
                       constant Levels &L [[buffer(4)]],
                       device atomic_int *idelta [[buffer(5)]],
                       device atomic_uint *err [[buffer(6)]],
                       uint k [[thread_position_in_grid]]) {
    if (k >= n) return;
    uchar kk = kind[k];
    if (kk != K_CMD && kk != K_ARRAYSUM && kk != K_ELSE) return;
    int depth = eb[k] - (kk == K_ELSE ? 1 : 0);
    uint q = next_q(tterm, L, k, TERM{uint(max(depth, 0)), kk == K_CMD});
    if (q == NONE) {
        atomic_store_explicit(&err[0], 1u, memory_order_relaxed);
        atomic_fetch_min_explicit(&err[1], (k << 4) | 10u, memory_order_relaxed);
        return;
    }
    atomic_fetch_add_explicit(&idelta[k + 1], 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&idelta[q], -1, memory_order_relaxed);
}

static uint prec_of(uchar k) {
    switch (k) {
    case K_CMD: return P_CMD;
    case K_COLON_BIND: return P_BIND;
    case K_OR: return P_OR;
    case K_CMP: return P_CMP;
    case K_ADD: return P_ADD;
    case K_MUL: return P_MUL;
    case K_UNARY: return P_UNARY;
    case K_DOT: case K_LSQ_POST: case K_LPAR_CALL: case K_LCUR_LIT: return P_POST;
    case K_LEAF: case K_ARRAYSUM: case K_IF: case K_LSQ_LIT: return P_LEAF;
    default: return P_PUNCT;
    }
}

// Cartesian-tree keys and segment depth.
kernel void p_keys(device const uchar *kind [[buffer(0)]],
                   device const int *eb [[buffer(1)]],
                   device const int *sb [[buffer(2)]],
                   device const int *id [[buffer(3)]],
                   device const uint *match [[buffer(4)]],
                   constant uint &n [[buffer(5)]],
                   device ulong *keys [[buffer(6)]],
                   device uint *sdepth [[buffer(7)]],
                   uint k [[thread_position_in_grid]]) {
    if (k >= n) return;
    uchar kk = kind[k];
    uint d = uint(eb[k] + id[k]);
    bool invisible = kk == K_LPAR_GROUP || (kk == K_CLOSE && match[k] < n && kind[match[k]] == K_LPAR_GROUP);
    uint p = prec_of(kk);
    ulong key;
    if (invisible) key = INF64;
    else if (p == P_PUNCT) key = ulong(d) << 36;
    else {
        uint tie = (kk == K_UNARY) ? k : ~k;
        key = (ulong(d) << 36) | (ulong(p) << 32) | ulong(tie);
    }
    keys[k] = key;
    sdepth[k] = uint(sb[k] + id[k]);
}

static bool owns_itself(uchar k) {
    return k == K_IF || k == K_LSQ_POST || k == K_LSQ_LIT || k == K_LPAR_CALL || k == K_LCUR_LIT ||
           k == K_CMD || k == K_ARRAYSUM;
}

static bool is_punct_key(ulong key) { return key == INF64 || ((key >> 32) & 15ul) == P_PUNCT; }

// Nearest smaller keys: to the left for every visible token, to the right
// for nodes.
kernel void p_ansv(device const ulong *keys [[buffer(0)]],
                   constant Levels &L [[buffer(1)]],
                   constant uint &n [[buffer(2)]],
                   device uint *left [[buffer(3)]],
                   device uint *right [[buffer(4)]],
                   uint k [[thread_position_in_grid]]) {
    if (k >= n) return;
    ulong key = keys[k];
    if (key == INF64) return;
    left[k] = prev_q(keys, L, k, LT64{key});
    if (!is_punct_key(key)) right[k] = next_q(keys, L, k, LT64{key});
}

// Parent of each node: the larger of its nearest smaller neighbours within its
// segment. A segment's root attaches to the owner of its group, found from its
// left neighbour: that is the group's opener, or a separator whose own left
// neighbour is the opener (separators of one group share a key).
enum : uchar { HOW_CHILD, HOW_ATTACHED };

kernel void p_parent(device const uchar *kind [[buffer(0)]],
                     device const ulong *keys [[buffer(1)]],
                     device const uint *sdepth [[buffer(2)]],
                     device const uint *left [[buffer(3)]],
                     device const uint *right [[buffer(4)]],
                     device const uint *match [[buffer(5)]],
                     constant uint &n [[buffer(6)]],
                     device int *parent [[buffer(7)]],
                     device uchar *how [[buffer(8)]],
                     device atomic_uint *cflags [[buffer(9)]],
                     uint k [[thread_position_in_grid]]) {
    if (k >= n) return;
    ulong key = keys[k];
    if (is_punct_key(key)) {
        parent[k] = -2;
        return;
    }
    uint sd = sdepth[k];
    uint l = left[k];
    uint cand[2] = {l, right[k]};
    uint best = NONE;
    ulong best_key = 0;
    for (uint c = 0; c < 2; c++) {
        uint j = cand[c];
        if (j == NONE) continue;
        ulong kj = keys[j];
        if (is_punct_key(kj) || sdepth[j] != sd) continue;
        if (best == NONE || kj > best_key) {
            best = j;
            best_key = kj;
        }
    }
    if (best != NONE) {
        parent[k] = int(best);
        how[k] = HOW_CHILD;
        atomic_fetch_or_explicit(&cflags[best], k < best ? 1u : 2u, memory_order_relaxed);
        return;
    }
    int result = -1;
    uint j = sd == 0 ? NONE : l;
    for (uint hops = 0; j != NONE && hops < 4; hops++) {
        uchar kj = kind[j];
        if (owns_itself(kj)) {
            result = int(j);
            break;
        }
        if (kj == K_ELSE) {
            result = int(match[j]);
            break;
        }
        if (kj == K_LSQ_BIND) {
            result = int(j - 1);
            break;
        }
        j = left[j]; // separator, parameter list, or body: its opener's left neighbour
    }
    parent[k] = result;
    how[k] = HOW_ATTACHED;
}

// ---------------------------------------------------------------- validation
//
// The tree is only trusted for valid programs, so these checks reject exactly
// what the grammar rejects: every adjacent pair of tokens must be legal, each
// separator must belong to a group that allows it (at most once where the
// grammar says once), and every node must have the shape its syntactic
// category requires.

// err[0] is set on any error; err[1] keeps the smallest (token << 4 | reason).
static void fail(device atomic_uint *err, uint k, uint reason) {
    atomic_store_explicit(&err[0], 1u, memory_order_relaxed);
    atomic_fetch_min_explicit(&err[1], (k << 4) | reason, memory_order_relaxed);
}

static bool explicit_opener(uchar k) { return is_opener(k) && k != K_LPAR_GROUP; }

// Separators: where they must sit and how often.
kernel void p_seps(device const uchar *kind [[buffer(0)]],
                   device const uchar *ty [[buffer(1)]],
                   device const int *eb [[buffer(2)]],
                   device const uint *left [[buffer(3)]],
                   constant uint &n [[buffer(4)]],
                   device atomic_uint *sepbits [[buffer(5)]],
                   device atomic_uint *err [[buffer(6)]],
                   uint k [[thread_position_in_grid]]) {
    if (k >= n) return;
    uchar kk = kind[k];
    bool sep = kk == K_COMMA || kk == K_THEN || kk == K_EQUALS || kk == K_TO || kk == K_COLON_RET ||
               kk == K_LCUR_BLOCK || kk == K_IMAGE || (kk == K_NEWLINE && ty[k] == T_NEWLINE);
    if (!sep) return;
    uint o = left[k];
    if (o == NONE) {
        if (kk != K_NEWLINE || eb[k] != 0) fail(err, k, 1);
        return;
    }
    uchar ok_ = kind[o], ot = ty[o];
    // A separator sits directly in its group, not inside grouping parentheses.
    int inside = explicit_opener(ok_) ? eb[o] + 1 : (ok_ == K_CMD || ok_ == K_ARRAYSUM) ? eb[o] : -1;
    if (eb[k] != inside) {
        fail(err, k, 2);
        return;
    }
    uint bit = 0;
    bool ok = false;
    switch (kk) {
    case K_COMMA:
        ok = ok_ == K_LPAR_CALL || ok_ == K_LSQ_POST || ok_ == K_LSQ_LIT || ok_ == K_LCUR_LIT ||
             ok_ == K_LSQ_BIND || ok_ == K_LPAR_FN || ot == T_ASSERT;
        if (ot == T_ASSERT) bit = 1;
        break;
    case K_THEN: ok = ok_ == K_IF; bit = 1; break;
    case K_EQUALS: ok = ot == T_LET; bit = 1; break;
    case K_TO: ok = ot == T_READ || ot == T_WRITE; bit = 1; break;
    case K_IMAGE: ok = ot == T_READ || ot == T_WRITE; break;
    case K_COLON_RET: ok = ot == T_FN; bit = 1; break;
    case K_LCUR_BLOCK: ok = ot == T_FN || ot == T_STRUCT; bit = 2; break;
    default: ok = ok_ == K_LCUR_BLOCK; break; // newline inside a body
    }
    if (!ok) fail(err, k, 3);
    else if (bit && (atomic_fetch_or_explicit(&sepbits[o], bit, memory_order_relaxed) & bit)) fail(err, k, 4);
}

enum : uchar { C_BAD, C_CMD, C_STMT, C_NAME, C_STR, C_ARG, C_TYPE, C_EXPR, C_PBIND, C_FBIND, C_ABIND, NCAT };

struct Tree {
    device const uchar *kind;
    device const uchar *ty;
    device const int *parent;
    device const uchar *how;
    device const uint *left;
    device const uint *match;
    uint n;
};

// Category of x from a parent whose children's categories do not depend on
// the parent's own category. pk, pt: the parent's kind and type; l, lk, lt:
// x's left neighbour; bind_end: for an array/sum parent, the ] of its bindings.
static uchar fixed_cat(int pi, uint x, uchar pk, uchar pt, bool attached, uint l, uchar lk, uchar lt, uint bind_end) {
    if (pi == -1) return C_CMD;
    if (pi < 0) return C_BAD;
    uint p = uint(pi);
    switch (pk) {
    case K_CMD:
        if (!attached) return C_BAD;
        switch (pt) {
        case T_LET: return l == p ? C_ARG : lt == T_EQUALS ? C_EXPR : C_BAD;
        case T_ASSERT: return l == p ? C_EXPR : lt == T_COMMA ? C_STR : C_BAD;
        case T_RETURN: case T_SHOW: return l == p ? C_EXPR : C_BAD;
        case T_PRINT: return l == p ? C_STR : C_BAD;
        case T_READ: return lt == T_IMAGE ? C_STR : lt == T_TO ? C_ARG : C_BAD;
        case T_WRITE: return lt == T_IMAGE ? C_EXPR : lt == T_TO ? C_STR : C_BAD;
        case T_TIME: return l == p ? C_CMD : C_BAD;
        case T_STRUCT: return l == p ? C_NAME : lk == K_NEWLINE ? C_FBIND : C_BAD;
        default: // fn
            return l == p ? C_NAME : lk == K_COLON_RET ? C_TYPE : lk == K_NEWLINE ? C_STMT
                 : (lk == K_LPAR_FN || lk == K_COMMA) ? C_PBIND : C_BAD;
        }
    case K_IF: return attached ? C_EXPR : C_BAD;
    case K_ARRAYSUM: return !attached || bind_end == NONE ? C_BAD : x < bind_end ? C_ABIND : C_EXPR;
    case K_DOT: return attached ? C_BAD : x < p ? C_EXPR : C_NAME;
    case K_LPAR_CALL: case K_LCUR_LIT: return attached ? C_EXPR : x < p ? C_NAME : C_BAD;
    case K_LSQ_LIT: return attached ? C_EXPR : C_BAD;
    case K_OR: case K_CMP: case K_ADD: case K_MUL: case K_UNARY: return attached ? C_BAD : C_EXPR;
    default: return C_BAD;
    }
}

static uint bind_end_of(thread const Tree &t, int pi) {
    if (pi < 0 || uint(pi) + 1 >= t.n || t.kind[pi] != K_ARRAYSUM || t.kind[pi + 1] != K_LSQ_BIND) return NONE;
    return t.match[pi + 1];
}

static uchar fixed_cat_at(thread const Tree &t, uint x) {
    int pi = t.parent[x];
    uint p = pi < 0 ? 0 : uint(pi);
    uint l = t.left[x];
    bool lv = l < t.n;
    return fixed_cat(pi, x, t.kind[p], t.ty[p], t.how[x] == HOW_ATTACHED, l, lv ? t.kind[l] : K_LEAF,
                     lv ? t.ty[l] : 0, bind_end_of(t, pi));
}

// Maps from a parent's category to a child's, 4 bits per category.
#define CMAP(from, to) (ulong(to) << (4 * (from)))
constant ulong SQ_LEFT = CMAP(C_EXPR, C_EXPR) | CMAP(C_TYPE, C_TYPE) | CMAP(C_ARG, C_NAME);
constant ulong SQ_INSIDE = CMAP(C_EXPR, C_EXPR) | CMAP(C_ARG, C_NAME);
constant ulong BIND_LEFT = CMAP(C_PBIND, C_ARG) | CMAP(C_FBIND, C_NAME) | CMAP(C_ABIND, C_NAME);
constant ulong BIND_RIGHT = CMAP(C_PBIND, C_TYPE) | CMAP(C_FBIND, C_TYPE) | CMAP(C_ABIND, C_EXPR);

static uchar cmap_get(ulong m, uint c) { return uchar((m >> (4 * c)) & 15ul); }

static uchar cat_of(thread const Tree &t, uint k) {
    // Walk up through [ and : parents, whose children's categories depend on
    // their own, composing the maps on the way.
    ulong m = 0;
    for (uint c = 0; c < NCAT; c++) m |= CMAP(c, c);
    uint x = k;
    for (;;) {
        int pi = t.parent[x];
        if (pi < 0) break;
        uint p = uint(pi);
        uchar pk = t.kind[p];
        bool attached = t.how[x] == HOW_ATTACHED;
        ulong step;
        if (pk == K_LSQ_POST) step = attached ? SQ_INSIDE : x < p ? SQ_LEFT : 0;
        else if (pk == K_COLON_BIND) step = attached ? 0 : x < p ? BIND_LEFT : BIND_RIGHT;
        else break;
        ulong composed = 0;
        for (uint c = 0; c < NCAT; c++) composed |= CMAP(c, cmap_get(m, cmap_get(step, c)));
        m = composed;
        x = p;
    }
    return cmap_get(m, fixed_cat_at(t, x));
}

static bool shape_ok(uchar c, uchar kk, uchar tt, uint cf) {
    switch (c) {
    case C_CMD: return kk == K_CMD && tt != T_RETURN && cf == 0;
    case C_STMT: return (tt == T_LET || tt == T_ASSERT || tt == T_RETURN) && cf == 0;
    case C_NAME: return tt == T_VARIABLE && cf == 0;
    case C_STR: return tt == T_STRING && cf == 0;
    case C_ARG: return (tt == T_VARIABLE && cf == 0) || (kk == K_LSQ_POST && cf == 1);
    case C_TYPE:
        return ((tt == T_INT || tt == T_BOOL || tt == T_FLOAT || tt == T_VOID || tt == T_VARIABLE) && cf == 0) ||
               (kk == K_LSQ_POST && cf == 1);
    case C_PBIND: case C_FBIND: case C_ABIND: return kk == K_COLON_BIND && cf == 3;
    case C_EXPR:
        switch (kk) {
        case K_LEAF:
            return (tt == T_VARIABLE || tt == T_INTVAL || tt == T_FLOATVAL || tt == T_TRUE || tt == T_FALSE ||
                    tt == T_VOID) && cf == 0;
        case K_OR: case K_CMP: case K_ADD: case K_MUL: case K_DOT: return cf == 3;
        case K_UNARY: return cf == 2;
        case K_LSQ_POST: case K_LPAR_CALL: case K_LCUR_LIT: return cf == 1;
        case K_LSQ_LIT: case K_IF: case K_ARRAYSUM: return cf == 0;
        default: return false;
        }
    default: return false;
    }
}

static bool operand_start(uchar kk, uchar tt) {
    return tt == T_VARIABLE || tt == T_INTVAL || tt == T_FLOATVAL || tt == T_TRUE || tt == T_FALSE ||
           tt == T_VOID || kk == K_UNARY || kk == K_LPAR_GROUP || kk == K_LSQ_LIT || kk == K_IF ||
           kk == K_ARRAYSUM;
}

// What may follow a complete operand, argument, or type.
static bool after_operand(uchar kk, uchar tt, uchar prev_t) {
    switch (kk) {
    case K_OR: case K_CMP: case K_ADD: case K_MUL: case K_LSQ_POST: case K_DOT: case K_LPAR_CALL:
    case K_LPAR_FN: case K_LCUR_BLOCK: case K_COMMA: case K_CLOSE: case K_THEN: case K_ELSE: case K_TO:
    case K_COLON_BIND: case K_EQUALS:
        return true;
    case K_LCUR_LIT: return prev_t == T_VARIABLE;
    case K_NEWLINE: return tt == T_NEWLINE;
    default: return false;
    }
}

// The token before k must allow it.
static void check_pair(thread const Tree &t, device const int *eb, device atomic_uint *err, uint k) {
    device const uchar *kind = t.kind;
    device const uchar *ty = t.ty;
    device const uint *left = t.left;
    device const uint *match = t.match;
    uint n = t.n;
    uchar kk = kind[k], tt = ty[k];
    bool ok;
    if (k == 0) {
        ok = tt == T_NEWLINE || tt == T_EOF || (kk == K_CMD && tt != T_RETURN);
    } else {
        // Load everything any case below needs, the same for every lane.
        uint p = k - 1;
        uchar pk = kind[p], pt = ty[p], ppt = p > 0 ? ty[p - 1] : 0;
        uint mk = match[k], mp = match[p];
        uchar mpk = mp < n ? kind[mp] : K_LEAF;
        uint o = left[p];
        bool ov = o < n;
        uchar ok_ = ov ? kind[o] : K_LEAF, ot = ov ? ty[o] : 0;
        uint w = ov ? left[o] : NONE;
        uchar wt = w < n ? ty[w] : 0;
        bool closes_p = kk == K_CLOSE && mk == p;
        bool closes_o = kk == K_CLOSE && mk == o;
        bool start = operand_start(kk, tt);
        bool after = after_operand(kk, tt, pt);
        bool is_type = tt == T_INT || tt == T_BOOL || tt == T_FLOAT || tt == T_VOID || tt == T_VARIABLE;
        switch (pk) {
        case K_LEAF: ok = pt == T_STRING ? (tt == T_NEWLINE || tt == T_TO) : after; break;
        case K_CLOSE:
            ok = mpk == K_LPAR_FN ? kk == K_COLON_RET : mpk == K_LCUR_BLOCK ? tt == T_NEWLINE
               : mpk == K_LSQ_BIND ? start : after;
            break;
        case K_NEWLINE:
            if (pt == T_EOF) ok = false;
            else if (o == NONE) ok = tt == T_EOF || (kk == K_CMD && tt != T_RETURN);
            else if (wt == T_FN) ok = tt == T_LET || tt == T_ASSERT || tt == T_RETURN || closes_o;
            else if (wt == T_STRUCT) ok = tt == T_VARIABLE || closes_o;
            else ok = false;
            break;
        case K_CMD:
            switch (pt) {
            case T_LET: case T_STRUCT: case T_FN: ok = tt == T_VARIABLE; break;
            case T_ASSERT: case T_SHOW: case T_RETURN: ok = start; break;
            case T_PRINT: ok = tt == T_STRING; break;
            case T_READ: case T_WRITE: ok = tt == T_IMAGE; break;
            default: ok = kk == K_CMD && tt != T_RETURN; break; // time
            }
            break;
        case K_IMAGE: ok = ppt == T_READ ? tt == T_STRING : start; break;
        case K_TO: ok = ot == T_READ ? tt == T_VARIABLE : tt == T_STRING; break;
        case K_EQUALS: case K_THEN: case K_ELSE: case K_UNARY: case K_OR: case K_CMP: case K_ADD: case K_MUL:
        case K_LPAR_GROUP: case K_IF:
            ok = start;
            break;
        case K_COLON_BIND: ok = start || tt == T_INT || tt == T_BOOL || tt == T_FLOAT; break;
        case K_COLON_RET: ok = is_type; break;
        case K_DOT: ok = tt == T_VARIABLE; break;
        case K_LPAR_CALL: case K_LSQ_LIT: case K_LCUR_LIT: ok = start || closes_p; break;
        case K_LSQ_POST: ok = start || closes_p || kk == K_COMMA; break;
        case K_LSQ_BIND: case K_LPAR_FN: ok = tt == T_VARIABLE || closes_p; break;
        case K_LCUR_BLOCK: ok = tt == T_NEWLINE; break;
        case K_ARRAYSUM: ok = kk == K_LSQ_BIND; break;
        case K_COMMA:
            if (ok_ == K_LPAR_CALL || ok_ == K_LSQ_LIT || ok_ == K_LCUR_LIT) ok = start;
            else if (ok_ == K_LSQ_BIND || ok_ == K_LPAR_FN) ok = tt == T_VARIABLE;
            else if (ok_ == K_LSQ_POST) {
                uchar c = cat_of(t, o);
                ok = c == C_EXPR ? start : c == C_ARG ? tt == T_VARIABLE
                   : c == C_TYPE ? (kk == K_COMMA || closes_o) : false;
            } else ok = ot == T_ASSERT && tt == T_STRING;
            break;
        default: ok = false; break;
        }
        // Commas follow an operand, except inside an array rank.
        if (ok && kk == K_COMMA) {
            uint lo = left[k];
            bool rank = lo < n && kind[lo] == K_LSQ_POST && cat_of(t, lo) == C_TYPE;
            bool after_list_start = p == lo || pk == K_COMMA;
            ok = rank ? after_list_start : !after_list_start;
        }
        if (ok && kk == K_CLOSE && mk < n && kind[mk] == K_LCUR_BLOCK) ok = pt == T_NEWLINE;
        if (ok && kk == K_CMD) ok = pt == T_NEWLINE || pt == T_TIME;
        if (ok && kk == K_IMAGE) ok = pt == T_READ || pt == T_WRITE;
        if (ok && tt == T_EOF) ok = pt == T_NEWLINE;
    }
    if (ok && tt == T_EOF) ok = eb[k] == 0;
    if (!ok) fail(err, k, 5);
}

// Node k: the shape its category requires, and the separators it owns.
static void check_node(thread const Tree &t, device const uint *cflags, device const uint *sepbits,
                       device atomic_uint *err, uint k) {
    device const uchar *kind = t.kind;
    device const uchar *ty = t.ty;
    device const uint *left = t.left;
    device const uint *match = t.match;
    device const uchar *how = t.how;
    uint n = t.n;
    int pi = t.parent[k];
    if (pi == -2) return;
    uchar kk = kind[k], tt = ty[k];
    uint p = pi < 0 ? 0 : uint(pi);
    uchar pk = kind[p];
    uchar c;
    if (pi >= 0 && (pk == K_LSQ_POST || pk == K_COLON_BIND)) {
        c = cat_of(t, k);
    } else {
        uint l = left[k];
        bool lv = l < n;
        uint bend = NONE;
        if (pi >= 0 && pk == K_ARRAYSUM && p + 1 < n && kind[p + 1] == K_LSQ_BIND) bend = match[p + 1];
        c = fixed_cat(pi, k, pk, ty[p], how[k] == HOW_ATTACHED, l, lv ? kind[l] : K_LEAF, lv ? ty[l] : 0, bend);
    }
    bool ok = shape_ok(c, kk, tt, cflags[k]);
    uint bits = sepbits[k];
    if (kk == K_IF || tt == T_LET || tt == T_ASSERT || tt == T_READ || tt == T_WRITE) ok = ok && (bits & 1) != 0;
    else if (tt == T_FN) ok = ok && bits == 3;
    else if (tt == T_STRUCT) ok = ok && bits == 2;
    if (!ok) fail(err, k, 7);
}

// Two kernels: fused, they run slower.
kernel void p_check_pairs(device const uchar *kind [[buffer(0)]],
                    device const uchar *ty [[buffer(1)]],
                    device const int *eb [[buffer(2)]],
                    device const int *parent [[buffer(3)]],
                    device const uchar *how [[buffer(4)]],
                    device const uint *left [[buffer(5)]],
                    device const uint *match [[buffer(6)]],
                    device const uint *cflags [[buffer(7)]],
                    device const uint *sepbits [[buffer(8)]],
                    constant uint &n [[buffer(9)]],
                    device atomic_uint *err [[buffer(10)]],
                    uint k [[thread_position_in_grid]]) {
    if (k >= n) return;
    Tree t = {kind, ty, parent, how, left, match, n};
    check_pair(t, eb, err, k);
}

kernel void p_check_nodes(device const uchar *kind [[buffer(0)]],
                    device const uchar *ty [[buffer(1)]],
                    device const int *eb [[buffer(2)]],
                    device const int *parent [[buffer(3)]],
                    device const uchar *how [[buffer(4)]],
                    device const uint *left [[buffer(5)]],
                    device const uint *match [[buffer(6)]],
                    device const uint *cflags [[buffer(7)]],
                    device const uint *sepbits [[buffer(8)]],
                    constant uint &n [[buffer(9)]],
                    device atomic_uint *err [[buffer(10)]],
                    uint k [[thread_position_in_grid]]) {
    if (k >= n) return;
    Tree t = {kind, ty, parent, how, left, match, n};
    check_node(t, cflags, sepbits, err, k);
}
