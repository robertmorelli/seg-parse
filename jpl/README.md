# jpl

A GPU (Metal) lexer and parser for JPL (the language in `jpl_wasm/spec.md`) with Unicode identifiers, built on the same idea as the parent directory: the parse tree falls out of nearest-smaller-value queries over a min tree.

Both stages run in parallel over the whole input. Nothing walks the file in order.

- **Lexer.** It reads any DFA table in the `jpl_bench` format (`jpl_utf8.bin` is copied from there). A byte's step function never reaches more than 18 states, so each 256-byte chunk is summarized as its end state and token count for each of them. The summaries compose, a scan gives every chunk its real starting state, and each chunk then writes its own tokens.
- **Parser.** Each token gets a key (nesting depth, precedence, position), and the AST is the Cartesian tree of those keys. Bracket matching, where `if`/`array`/`sum` bodies and command lines end, and each node's parent are all "nearest element ≤ x" queries. Validation checks adjacent tokens, separators and node shapes. A recursive-descent parser in `src/cpu.c` serves as the reference.

## Results

`t2_utf8.jpl` from `jpl_bench`: 12.9 MB, 5.76M tokens. Apple M1 Pro, best of 20 runs, GPU times are wall clock including submission.

| | lex | parse | total |
|---|---:|---:|---:|
| GPU (this) | 6.9 ms | 15.3 ms | **22.4 ms** |
| CPU, hand-written NEON lexer (`jpl_bench`) | 34.2 ms | | |
| CPU, scalar Zig lexer (`jpl_bench`) | 42.7 ms | | |
| CPU, `src/cpu.c` | 52.2 ms | 38.4 ms | 90.6 ms |

The GPU finishes lexing and parsing before the fastest CPU lexer finishes lexing.

## Testing

Every run checks the GPU's tokens and parse tree against the CPU reference, and the tokens match `jpl_bench`'s hashes. `--fuzz` generates programs from the `jpl_bench` fuzzer grammar, mutates some of them, and checks that both sides accept the same programs and build the same trees. About 185,000 programs agree so far.

## Usage

```
make
bin/jplparse ../../jpl_bench/inputs/t2_utf8.jpl 20   # time both, check they agree
bin/jplparse --fuzz 20000                            # random and mutated programs
JPL_PROFILE=1 bin/jplparse FILE 5                    # GPU time per stage
```
