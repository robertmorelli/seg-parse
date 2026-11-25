// jplparse FILE [REPS]      lex and parse FILE on the CPU and the GPU, check
//                           that both agree, and report the best times
// jplparse --fuzz N [SEED]  compare both on N random and mutated programs
#import <Foundation/Foundation.h>
#include "fuzz.h"
#include "gpu.h"

#include <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double ms_now(void) {
    static mach_timebase_info_data_t tb;
    if (!tb.denom) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6;
}

static double min_d(double a, double b) { return a < b ? a : b; }

typedef struct {
    Table table;
    Gpu *gpu;
    Tokens cpu;
    int32_t *cpu_parent;
} Ctx;

static void dump_around(const uint8_t *src, const Tokens *t, const int32_t *a, const int32_t *b, uint32_t at) {
    for (uint32_t q = at > 8 ? at - 8 : 0; q < at + 8 && q < t->count; q++) {
        fprintf(stderr, "%c %6u t%-2d %-12.*s cpu %6d gpu %6d\n", q == at ? '>' : ' ', q, t->type[q],
                (int)MIN(12u, t->end[q] - t->start[q]), src + t->start[q], a[q], b ? b[q] : 0);
    }
}

// Runs both on the input already in the GPU buffer. Returns false on any
// disagreement.
static bool compare(Ctx *c, size_t len, bool verbose, bool *cpu_ok) {
    uint8_t *src = gpu_input(c->gpu);
    bool lex_ok = cpu_lex(&c->table, src, len, &c->cpu);
    bool parse_ok = lex_ok && cpu_parse(src, &c->cpu, c->cpu_parent);
    *cpu_ok = parse_ok;
    GpuResult r;
    gpu_run(c->gpu, len, true, &r);
    if (r.lex_ok != lex_ok) {
        if (verbose) fprintf(stderr, "lex: cpu %s, gpu %s\n", lex_ok ? "ok" : "error", r.lex_ok ? "ok" : "error");
        return false;
    }
    if (!lex_ok) return true;
    if (r.toks.count != c->cpu.count || hash_tokens(&r.toks) != hash_tokens(&c->cpu)) {
        if (verbose) fprintf(stderr, "tokens differ\n");
        return false;
    }
    if (r.parse_ok != parse_ok) {
        if (verbose) {
            fprintf(stderr, "parse: cpu %s, gpu %s\n", parse_ok ? "ok" : "error", r.parse_ok ? "ok" : "error");
            uint32_t at = r.parse_ok ? cpu_parse_error_at : r.error_at >> 4;
            if (!r.parse_ok) fprintf(stderr, "gpu error at token %u, reason %u\n", at, r.error_at & 15);
            else fprintf(stderr, "cpu error at token %u\n", at);
            dump_around(src, &c->cpu, c->cpu_parent, r.parent, at);
        }
        return false;
    }
    if (!parse_ok) return true;
    for (uint32_t i = 0; i < c->cpu.count; i++) {
        if (c->cpu_parent[i] != r.parent[i]) {
            if (verbose) {
                fprintf(stderr, "trees differ at token %u\n", i);
                dump_around(src, &c->cpu, c->cpu_parent, r.parent, i);
            }
            return false;
        }
    }
    return true;
}

static int fuzz(Ctx *c, long count, uint64_t seed) {
    size_t cap = 1 << 16;
    char *prog = malloc(cap);
    uint8_t *src = gpu_input(c->gpu);
    long rejected = 0;
    for (long i = 0; i < count; i++) {
        uint64_t rng = seed * 1000003 + (uint64_t)i;
        size_t len = fuzz_program(&rng, prog, cap, 20 + fuzz_next(&rng) % 400);
        int mutations = (int)(fuzz_next(&rng) % 4); // 0 to 3
        for (int m = 0; m < mutations; m++) {
            memcpy(src, prog, len + 1);
            if (!cpu_lex(&c->table, src, len, &c->cpu)) break;
            len = fuzz_mutate(&rng, src, &c->cpu, prog, cap);
        }
        memcpy(src, prog, len + 1);
        bool ok;
        if (!compare(c, len, true, &ok)) {
            fprintf(stderr, "mismatch on case %ld (seed %llu):\n----\n%s----\n", i, (unsigned long long)seed, prog);
            FILE *f = fopen("fuzz_fail.jpl", "wb");
            fwrite(prog, 1, len, f);
            fclose(f);
            return 1;
        }
        rejected += !ok;
    }
    printf("%ld programs agree (%ld rejected by both)\n", count, rejected);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s FILE [REPS] | --fuzz N [SEED]\n", argv[0]);
        return 2;
    }
    NSString *root = [[@(argv[0]) stringByDeletingLastPathComponent] stringByDeletingLastPathComponent];
    Ctx c;
    if (!table_load(&c.table, [root stringByAppendingPathComponent:@"jpl_utf8.bin"].UTF8String)) {
        fprintf(stderr, "cannot load jpl_utf8.bin\n");
        return 1;
    }
    const char *metal = [root stringByAppendingPathComponent:@"src/jpl.metal"].UTF8String;
    bool fuzzing = !strcmp(argv[1], "--fuzz");

    size_t len = 1 << 16;
    FILE *f = NULL;
    if (!fuzzing) {
        f = fopen(argv[1], "rb");
        if (!f) {
            perror(argv[1]);
            return 1;
        }
        fseek(f, 0, SEEK_END);
        len = (size_t)ftell(f);
        fseek(f, 0, SEEK_SET);
    }
    c.gpu = gpu_init(&c.table, metal, len);
    if (!c.gpu) return 1;
    c.cpu = (Tokens){0, malloc(len + 2), malloc(4 * (len + 2)), malloc(4 * (len + 2))};
    c.cpu_parent = malloc(4 * (len + 2));
    if (fuzzing) return fuzz(&c, argc > 2 ? atol(argv[2]) : 10000, argc > 3 ? strtoull(argv[3], NULL, 10) : 1);

    uint8_t *src = gpu_input(c.gpu);
    if (fread(src, 1, len, f) != len) return 1;
    fclose(f);
    src[len] = 0;
    int reps = argc > 2 ? atoi(argv[2]) : 10;

    double cpu_lex_ms = 1e9, cpu_parse_ms = 1e9;
    bool lex_ok = false, parse_ok = false;
    for (int r = 0; r < reps; r++) {
        double a = ms_now();
        lex_ok = cpu_lex(&c.table, src, len, &c.cpu);
        double b = ms_now();
        parse_ok = lex_ok && cpu_parse(src, &c.cpu, c.cpu_parent);
        double d = ms_now();
        cpu_lex_ms = min_d(cpu_lex_ms, b - a);
        cpu_parse_ms = min_d(cpu_parse_ms, d - b);
    }
    double gpu_lex = 1e9, gpu_parse = 1e9, gpu_total = 1e9;
    GpuResult r;
    for (int i = 0; i < reps + 1; i++) {
        gpu_run(c.gpu, len, true, &r);
        if (i == 0) continue; // warm-up
        gpu_lex = min_d(gpu_lex, r.lex_ms);
        gpu_parse = min_d(gpu_parse, r.parse_ms);
        gpu_total = min_d(gpu_total, r.lex_ms + r.parse_ms);
    }
    const char *status = !lex_ok ? "lex error" : parse_ok ? "ok" : "syntax error";
    printf("%s: %zu bytes, %u tokens, %s\n", argv[1], len, c.cpu.count, status);
    printf("cpu  lex %7.2f ms   parse %7.2f ms   total %7.2f ms\n", cpu_lex_ms, cpu_parse_ms, cpu_lex_ms + cpu_parse_ms);
    printf("gpu  lex %7.2f ms   parse %7.2f ms   total %7.2f ms\n", gpu_lex, gpu_parse, gpu_total);
    bool ok;
    bool same = compare(&c, len, true, &ok);
    printf("cpu and gpu %s\n", same ? "agree" : "DISAGREE");
    return same ? 0 : 1;
}
