#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Token types, numbered as in jpl_bench/dfabench/zig/defs.zig.
enum {
    T_START, T_ACCEPT, T_INVALID, T_IGNORE,
    T_ASSERT, T_ARRAY, T_BOOL, T_ELSE, T_FALSE, T_FN, T_FLOAT, T_INT, T_IMAGE,
    T_IF, T_LET, T_PRINT, T_RETURN, T_READ, T_STRUCT, T_SHOW, T_SUM, T_TO,
    T_TIME, T_TRUE, T_THEN, T_VOID, T_WRITE,
    T_VARIABLE, T_COLON, T_RPAREN, T_LPAREN, T_RCURLY, T_RSQUARE, T_LSQUARE,
    T_LCURLY, T_NEWLINE, T_INTVAL, T_FLOATVAL, T_OP, T_COMMA, T_EQUALS,
    T_STRING, T_DOT, T_EOF,
};

// Parent value for tokens that are punctuation rather than AST nodes.
#define NOT_NODE (-2)
#define ROOT (-1)

typedef struct {
    uint16_t states;
    const uint16_t *tabs; // tabs[state * 256 + byte]
    const uint8_t *types; // token type a state ends
    void *blob;
} Table;

typedef struct {
    uint32_t count;
    uint8_t *type;
    uint32_t *start;
    uint32_t *end;
} Tokens;

bool table_load(Table *t, const char *path);

// src must have a 0 byte at src[len].
bool cpu_lex(const Table *t, const uint8_t *src, size_t len, Tokens *out);

// Fills parent[i] for each token: the index of its AST parent, ROOT for
// top-level commands, NOT_NODE for punctuation. Returns false on a syntax
// error.
bool cpu_parse(const uint8_t *src, const Tokens *toks, int32_t *parent);
extern uint32_t cpu_parse_error_at; // token index of the last syntax error

uint64_t hash_tokens(const Tokens *toks);
uint64_t hash_parents(const int32_t *parent, uint32_t n);
