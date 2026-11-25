#pragma once
#include "jpl.h"

typedef struct Gpu Gpu;

typedef struct {
    Tokens toks;     // points into GPU-shared memory
    int32_t *parent; // points into GPU-shared memory
    bool lex_ok, parse_ok;
    uint32_t error_at; // smallest failing token << 4 | reason code
    double lex_ms, parse_ms; // wall time per phase, including submission
} GpuResult;

// Builds the step tables from the DFA and compiles the kernels.
Gpu *gpu_init(const Table *t, const char *metal_path, size_t max_len);

// Input buffer the GPU reads in place; src[len] must be 0.
uint8_t *gpu_input(Gpu *g);

bool gpu_run(Gpu *g, size_t len, bool parse, GpuResult *out);
