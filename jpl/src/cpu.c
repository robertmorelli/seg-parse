// CPU reference: the dfabench table lexer and a recursive-descent parser for
// the JPL grammar (jpl_wasm/spec.md). The GPU pipeline must match both.
#include "jpl.h"

#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { S_ERROR = 0, S_START = 1, S_ACCEPT = 2, S_IGNORE = 3 };

bool table_load(Table *t, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *blob = malloc((size_t)size);
    bool ok = fread(blob, 1, (size_t)size, f) == (size_t)size;
    fclose(f);
    if (!ok) return false;
    uint16_t states;
    memcpy(&states, blob, 2);
    t->states = states;
    t->tabs = (const uint16_t *)(blob + 8);
    t->types = blob + 8 + (size_t)states * 512;
    t->blob = blob;
    return true;
}

bool cpu_lex(const Table *t, const uint8_t *src, size_t len, Tokens *out) {
    const uint16_t *tabs = t->tabs;
    uint32_t n = 0, i = 0, j = 0;
    uint16_t state = S_START;
    uint16_t next = tabs[S_START * 256 + src[0]];
    for (;;) {
        if (next == S_ERROR) {
            if (j == len) break;
            return false;
        } else if (next == S_ACCEPT) {
            out->type[n] = t->types[state];
            out->start[n] = i;
            out->end[n] = j;
            n++;
            i = j;
            next = tabs[S_START * 256 + src[j]];
        } else if (next == S_IGNORE) {
            i = ++j;
            next = tabs[S_START * 256 + src[j]];
        } else {
            state = next;
            j++;
            next = tabs[state * 256 + src[j]];
        }
    }
    if (i != j) {
        if (t->types[state] == T_INVALID) return false;
        out->type[n] = t->types[state];
        out->start[n] = i;
        out->end[n] = j;
        n++;
        i = j;
    }
    out->type[n] = T_EOF;
    out->start[n] = i;
    out->end[n] = j;
    out->count = n + 1;
    return true;
}

static uint64_t fnv(uint64_t h, uint32_t w) { return (h ^ w) * 0x100000001b3ull; }

uint64_t hash_tokens(const Tokens *toks) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint32_t k = 0; k < toks->count; k++) {
        h = fnv(h, toks->type[k]);
        h = fnv(h, toks->start[k]);
        h = fnv(h, toks->end[k]);
    }
    return h;
}

uint64_t hash_parents(const int32_t *parent, uint32_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint32_t k = 0; k < n; k++) h = fnv(h, (uint32_t)parent[k]);
    return h;
}

// ---- parser ----

typedef struct {
    const uint8_t *ty;
    const uint32_t *st;
    const uint32_t *en;
    const uint8_t *src;
    int32_t *par;
    uint32_t p;
    jmp_buf fail;
} P;

uint32_t cpu_parse_error_at;

static _Noreturn void fail(P *s) {
    cpu_parse_error_at = s->p;
    longjmp(s->fail, 1);
}

static int32_t expect(P *s, uint8_t t) {
    if (s->ty[s->p] != t) fail(s);
    return (int32_t)s->p++;
}

static bool at(P *s, uint8_t t) { return s->ty[s->p] == t; }

// Binary precedence of the OP token at p: 1 && ||, 2 comparisons, 3 + -, 4 * / %.
static int binary_prec(P *s) {
    if (s->ty[s->p] != T_OP) return 0;
    const uint8_t *c = s->src + s->st[s->p];
    uint32_t len = s->en[s->p] - s->st[s->p];
    switch (c[0]) {
    case '&': case '|': return 1;
    case '<': case '>': case '=': return 2;
    case '!': return len == 2 ? 2 : 0;
    case '+': case '-': return 3;
    default: return 4;
    }
}

static bool at_unary(P *s) {
    if (s->ty[s->p] != T_OP || s->en[s->p] - s->st[s->p] != 1) return false;
    uint8_t c = s->src[s->st[s->p]];
    return c == '-' || c == '!';
}

static int32_t expr(P *s);

static void expr_list(P *s, int32_t owner, uint8_t close) {
    if (at(s, close)) {
        s->p++;
        return;
    }
    for (;;) {
        s->par[expr(s)] = owner;
        if (!at(s, T_COMMA)) break;
        s->p++;
    }
    expect(s, close);
}

static int32_t primary(P *s) {
    int32_t n = (int32_t)s->p;
    switch (s->ty[s->p]) {
    case T_INTVAL: case T_FLOATVAL: case T_TRUE: case T_FALSE: case T_VOID:
        s->p++;
        return n;
    case T_VARIABLE:
        s->p++;
        if (at(s, T_LPAREN) || at(s, T_LCURLY)) {
            int32_t call = (int32_t)s->p++;
            s->par[n] = call;
            expr_list(s, call, s->ty[call] == T_LPAREN ? T_RPAREN : T_RCURLY);
            return call;
        }
        return n;
    case T_LSQUARE:
        s->p++;
        expr_list(s, n, T_RSQUARE);
        return n;
    case T_LPAREN: {
        s->p++;
        int32_t e = expr(s);
        expect(s, T_RPAREN);
        return e;
    }
    default:
        fail(s);
    }
}

static int32_t postfix(P *s) {
    if (at(s, T_IF) || at(s, T_ARRAY) || at(s, T_SUM)) return expr(s);
    int32_t x = primary(s);
    for (;;) {
        if (at(s, T_LSQUARE)) {
            int32_t n = (int32_t)s->p++;
            s->par[x] = n;
            expr_list(s, n, T_RSQUARE);
            x = n;
        } else if (at(s, T_DOT)) {
            int32_t n = (int32_t)s->p++;
            s->par[x] = n;
            s->par[expect(s, T_VARIABLE)] = n;
            x = n;
        } else {
            return x;
        }
    }
}

static int32_t unary(P *s) {
    if (!at_unary(s)) return postfix(s);
    int32_t n = (int32_t)s->p++;
    s->par[unary(s)] = n;
    return n;
}

static int32_t binary(P *s, int level) {
    if (level == 5) return unary(s);
    int32_t lhs = binary(s, level + 1);
    while (binary_prec(s) == level) {
        int32_t op = (int32_t)s->p++;
        s->par[lhs] = op;
        s->par[binary(s, level + 1)] = op;
        lhs = op;
    }
    return lhs;
}

static int32_t expr(P *s) {
    int32_t n = (int32_t)s->p;
    switch (s->ty[s->p]) {
    case T_IF:
        s->p++;
        s->par[expr(s)] = n;
        expect(s, T_THEN);
        s->par[expr(s)] = n;
        expect(s, T_ELSE);
        s->par[expr(s)] = n;
        return n;
    case T_ARRAY: case T_SUM:
        s->p++;
        expect(s, T_LSQUARE);
        if (!at(s, T_RSQUARE)) {
            for (;;) {
                int32_t v = expect(s, T_VARIABLE);
                int32_t colon = expect(s, T_COLON);
                s->par[v] = colon;
                s->par[expr(s)] = colon;
                s->par[colon] = n;
                if (!at(s, T_COMMA)) break;
                s->p++;
            }
        }
        expect(s, T_RSQUARE);
        s->par[expr(s)] = n;
        return n;
    default:
        return binary(s, 1);
    }
}

static int32_t type(P *s) {
    int32_t x = (int32_t)s->p;
    switch (s->ty[s->p]) {
    case T_INT: case T_BOOL: case T_FLOAT: case T_VOID: case T_VARIABLE:
        s->p++;
        break;
    default:
        fail(s);
    }
    while (at(s, T_LSQUARE)) {
        int32_t n = (int32_t)s->p++;
        s->par[x] = n;
        while (at(s, T_COMMA)) s->p++;
        expect(s, T_RSQUARE);
        x = n;
    }
    return x;
}

static int32_t argument(P *s) {
    int32_t v = expect(s, T_VARIABLE);
    if (!at(s, T_LSQUARE)) return v;
    int32_t n = (int32_t)s->p++;
    s->par[v] = n;
    if (!at(s, T_RSQUARE)) {
        for (;;) {
            s->par[expect(s, T_VARIABLE)] = n;
            if (!at(s, T_COMMA)) break;
            s->p++;
        }
    }
    expect(s, T_RSQUARE);
    return n;
}

static int32_t let_or_assert(P *s) {
    int32_t n = (int32_t)s->p++;
    if (s->ty[n] == T_LET) {
        s->par[argument(s)] = n;
        expect(s, T_EQUALS);
        s->par[expr(s)] = n;
    } else {
        s->par[expr(s)] = n;
        expect(s, T_COMMA);
        s->par[expect(s, T_STRING)] = n;
    }
    return n;
}

static int32_t stmt(P *s) {
    if (at(s, T_LET) || at(s, T_ASSERT)) return let_or_assert(s);
    int32_t n = expect(s, T_RETURN);
    s->par[expr(s)] = n;
    return n;
}

static int32_t cmd(P *s) {
    int32_t n = (int32_t)s->p;
    switch (s->ty[s->p]) {
    case T_LET: case T_ASSERT:
        return let_or_assert(s);
    case T_READ:
        s->p++;
        expect(s, T_IMAGE);
        s->par[expect(s, T_STRING)] = n;
        expect(s, T_TO);
        s->par[argument(s)] = n;
        return n;
    case T_WRITE:
        s->p++;
        expect(s, T_IMAGE);
        s->par[expr(s)] = n;
        expect(s, T_TO);
        s->par[expect(s, T_STRING)] = n;
        return n;
    case T_PRINT:
        s->p++;
        s->par[expect(s, T_STRING)] = n;
        return n;
    case T_SHOW:
        s->p++;
        s->par[expr(s)] = n;
        return n;
    case T_TIME:
        s->p++;
        s->par[cmd(s)] = n;
        return n;
    case T_STRUCT:
        s->p++;
        s->par[expect(s, T_VARIABLE)] = n;
        expect(s, T_LCURLY);
        expect(s, T_NEWLINE);
        while (!at(s, T_RCURLY)) {
            int32_t f = expect(s, T_VARIABLE);
            int32_t colon = expect(s, T_COLON);
            s->par[f] = colon;
            s->par[type(s)] = colon;
            s->par[colon] = n;
            expect(s, T_NEWLINE);
        }
        s->p++;
        return n;
    case T_FN:
        s->p++;
        s->par[expect(s, T_VARIABLE)] = n;
        expect(s, T_LPAREN);
        if (!at(s, T_RPAREN)) {
            for (;;) {
                int32_t a = argument(s);
                int32_t colon = expect(s, T_COLON);
                s->par[a] = colon;
                s->par[type(s)] = colon;
                s->par[colon] = n;
                if (!at(s, T_COMMA)) break;
                s->p++;
            }
        }
        expect(s, T_RPAREN);
        expect(s, T_COLON);
        s->par[type(s)] = n;
        expect(s, T_LCURLY);
        expect(s, T_NEWLINE);
        while (!at(s, T_RCURLY)) {
            s->par[stmt(s)] = n;
            expect(s, T_NEWLINE);
        }
        s->p++;
        return n;
    default:
        fail(s);
    }
}

bool cpu_parse(const uint8_t *src, const Tokens *toks, int32_t *parent) {
    P s = {.ty = toks->type, .st = toks->start, .en = toks->end, .src = src, .par = parent};
    for (uint32_t k = 0; k < toks->count; k++) parent[k] = NOT_NODE;
    if (setjmp(s.fail)) return false;
    if (at(&s, T_NEWLINE)) s.p++;
    while (!at(&s, T_EOF)) {
        s.par[cmd(&s)] = ROOT;
        expect(&s, T_NEWLINE);
    }
    return true;
}
