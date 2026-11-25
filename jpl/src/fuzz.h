#pragma once
#include "jpl.h"

uint64_t fuzz_next(uint64_t *state);

// A random program of roughly `target` bytes.
size_t fuzz_program(uint64_t *rng, char *out, size_t cap, size_t target);

// src with one random token-level change (delete, duplicate, swap, insert,
// replace); tokens are rejoined with spaces.
size_t fuzz_mutate(uint64_t *rng, const uint8_t *src, const Tokens *t, char *out, size_t cap);
