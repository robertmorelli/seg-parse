// Random JPL programs (the grammar of jpl_bench/fuzzbench) and token-level
// mutations of them, for checking the GPU parser against the CPU parser.
#include "fuzz.h"

#include <string.h>

typedef struct {
    const char *rhs;
    unsigned grow, base;
} Prod;

typedef struct {
    const char *name;
    const Prod *prods;
    int count;
} Rule;

#define RULE(name, ...) {name, (const Prod[]){__VA_ARGS__}, sizeof((const Prod[]){__VA_ARGS__}) / sizeof(Prod)}

static const Rule grammar[] = {
    RULE("S", {"", 0, 1}, {"$CMD$S", 1, 0}),
    RULE("CMD", {"let $LV = $E\n", 6, 6}, {"assert $E , $STR\n", 2, 2}, {"print $STR\n", 1, 1}, {"show $E\n", 3, 3},
         {"read image $STR to $ARG\n", 1, 1}, {"write image $E to $STR\n", 1, 1}, {"time $CMD", 1, 0},
         {"struct $VAR {\n$FIELDS}\n", 1, 1}, {"fn $VAR ( $PARAMS ) : $TYPE {\n$STMTS}\n", 3, 1}),
    RULE("FIELDS", {"", 1, 1}, {"$VAR : $TYPE\n$FIELDS", 2, 0}),
    RULE("STMTS", {"", 1, 1}, {"$STMT\n$STMTS", 3, 0}),
    RULE("STMT", {"let $LV = $E", 4, 4}, {"assert $E , $STR", 1, 1}, {"return $E", 2, 2}),
    RULE("PARAMS", {"", 1, 1}, {"$PLIST", 3, 0}),
    RULE("PLIST", {"$BINDING", 2, 1}, {"$BINDING , $PLIST", 1, 0}),
    RULE("BINDING", {"$ARG : $TYPE", 1, 1}),
    RULE("LV", {"$ARG", 1, 1}),
    RULE("ARG", {"$VAR", 3, 1}, {"$VAR [ $VARS ]", 1, 0}),
    RULE("VARS", {"$VAR", 2, 1}, {"$VAR , $VARS", 1, 0}),
    RULE("TYPE", {"int", 3, 3}, {"bool", 1, 1}, {"float", 2, 2}, {"void", 1, 1}, {"$VAR", 1, 1}, {"$TYPE [ ]", 1, 0},
         {"$TYPE [ , ]", 1, 0}),
    RULE("E", {"$OR", 60, 1}, {"if $E then $E else $E", 1, 0}, {"array [ $BINDS ] $E", 1, 0},
         {"sum [ $BINDS ] $E", 1, 0}),
    RULE("BINDS", {"$VAR : $E", 2, 1}, {"$VAR : $E , $BINDS", 1, 0}),
    RULE("OR", {"$CMP", 12, 1}, {"$OR && $CMP", 1, 0}, {"$OR || $CMP", 1, 0}),
    RULE("CMP", {"$ADD", 30, 1}, {"$CMP < $ADD", 1, 0}, {"$CMP > $ADD", 1, 0}, {"$CMP <= $ADD", 1, 0},
         {"$CMP >= $ADD", 1, 0}, {"$CMP == $ADD", 1, 0}, {"$CMP != $ADD", 1, 0}),
    RULE("ADD", {"$MUL", 10, 1}, {"$ADD + $MUL", 2, 0}, {"$ADD - $MUL", 1, 0}),
    RULE("MUL", {"$UN", 12, 1}, {"$MUL * $UN", 2, 0}, {"$MUL / $UN", 1, 0}, {"$MUL % $UN", 1, 0}),
    RULE("UN", {"$POST", 10, 1}, {"- $UN", 1, 0}, {"! $UN", 1, 0}),
    RULE("POST", {"$PRIM", 24, 1}, {"$POST [ $EXPRS ]", 2, 0}, {"$POST . $VAR", 1, 0}),
    RULE("PRIM", {"$INT", 12, 1}, {"$FLOAT", 6, 1}, {"true", 2, 1}, {"false", 2, 1}, {"void", 1, 1}, {"$VAR", 16, 1},
         {"$VAR { $EXPRS }", 1, 0}, {"[ $EXPRS ]", 1, 0}, {"( $E )", 1, 0}, {"$VAR ( $EXPRS )", 2, 0}),
    RULE("EXPRS", {"$E", 4, 1}, {"$E , $EXPRS", 1, 0}),
    RULE("INT", {"0", 1, 1}, {"1", 1, 1}, {"42", 1, 1}, {"65536", 1, 1}),
    RULE("FLOAT", {"0.5", 1, 1}, {"2.", 1, 1}, {".25", 1, 1}),
    RULE("VAR", {"a", 1, 1}, {"b", 1, 1}, {"x", 1, 1}, {"H", 1, 1}, {"img", 1, 1}, {"\xf0\x9d\x95\x92", 1, 1},
         {"rgba", 1, 1}),
    RULE("STR", {"\"\"", 1, 1}, {"\"a\"", 1, 1}, {"\"out.png\"", 1, 1}),
};

uint64_t fuzz_next(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

typedef struct {
    char *out;
    size_t len, cap, target;
    uint64_t *rng;
} Gen;

static void emit(Gen *g, const char *s, size_t n) {
    if (g->len + n >= g->cap) return;
    memcpy(g->out + g->len, s, n);
    g->len += n;
}

static void expand(Gen *g, const char *name, size_t name_len, int depth) {
    const Rule *r = NULL;
    for (size_t i = 0; i < sizeof grammar / sizeof grammar[0]; i++) {
        if (strlen(grammar[i].name) == name_len && !memcmp(grammar[i].name, name, name_len)) r = &grammar[i];
    }
    bool grow = g->len < g->target && depth < 40;
    unsigned total = 0;
    for (int i = 0; i < r->count; i++) total += grow ? r->prods[i].grow : r->prods[i].base;
    unsigned pick = (unsigned)(fuzz_next(g->rng) % total);
    int i = 0;
    while (pick >= (grow ? r->prods[i].grow : r->prods[i].base)) pick -= grow ? r->prods[i].grow : r->prods[i].base, i++;
    for (const char *c = r->prods[i].rhs; *c;) {
        if (*c == '$') {
            const char *e = c + 1;
            while (*e >= 'A' && *e <= 'Z') e++;
            expand(g, c + 1, (size_t)(e - c - 1), depth + 1);
            c = e;
        } else {
            emit(g, c, 1);
            c++;
        }
    }
}

size_t fuzz_program(uint64_t *rng, char *out, size_t cap, size_t target) {
    Gen g = {out, 0, cap, target, rng};
    expand(&g, "S", 1, 0);
    out[g.len] = 0;
    return g.len;
}

static const char *vocab[] = {
    "let", "=", "(", ")", "[", "]", "{", "}", ",", ":", ".", "+", "-", "*", "/", "!", "<", "==", "&&",
    "if", "then", "else", "array", "sum", "fn", "struct", "return", "show", "print", "read", "write",
    "image", "to", "time", "assert", "int", "float", "bool", "void", "true", "false", "x", "1", "2.0",
    "\"s\"", "\n",
};

size_t fuzz_mutate(uint64_t *rng, const uint8_t *src, const Tokens *t, char *out, size_t cap) {
    uint32_t n = t->count - 1; // without EOF
    int kind = (int)(fuzz_next(rng) % 5);
    uint32_t at = n ? (uint32_t)(fuzz_next(rng) % n) : 0;
    uint32_t other = n ? (uint32_t)(fuzz_next(rng) % n) : 0;
    const char *word = vocab[fuzz_next(rng) % (sizeof vocab / sizeof vocab[0])];
    size_t len = 0;
#define PUT(s, l)                                       \
    do {                                                \
        size_t l_ = (l);                                \
        if (len + l_ + 2 < cap) memcpy(out + len, s, l_); \
        len += l_;                                      \
    } while (0)
#define TOKEN(i) PUT(src + t->start[i], t->end[i] - t->start[i])
    for (uint32_t i = 0; i < n; i++) {
        uint32_t j = i;
        if (kind == 0 && i == at) continue;                 // delete
        if (kind == 1 && i == at) { // duplicate
            TOKEN(i);
            PUT(" ", 1);
        }
        if (kind == 2 && i == at) j = other;                // replace with another token
        if (kind == 2 && i == other) j = at;                // (swap)
        if (kind == 3 && i == at) { // insert
            PUT(word, strlen(word));
            PUT(" ", 1);
        }
        if (kind == 4 && i == at) {                         // replace with a word
            PUT(word, strlen(word));
            PUT(" ", 1);
            continue;
        }
        TOKEN(j);
        PUT(" ", 1);
    }
#undef TOKEN
#undef PUT
    if (len >= cap) len = cap - 1;
    out[len] = 0;
    return len;
}
