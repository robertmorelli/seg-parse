# seg-parse

Parsing with range-minimum queries instead of recursive descent.

Give every operator a key: nesting depth first, then precedence, then position.
The root of any subexpression is the operator with the smallest key in its
range, so the whole tree comes from asking a segment tree for minimums. Each
question is independent, which makes the approach parallel.

| | |
|---|---|
| [`src/`](src/) | The idea on arithmetic expressions, in Zig. |
| [`jpl/`](jpl/) | The idea on a real language: a Metal GPU lexer and parser for JPL that lexes and parses 12.9 MB in 22 ms, before the fastest CPU lexer finishes lexing. |

## Expressions

Requires Zig 0.16.

```
$ zig build run -- --expr "v-(v+v)*v"
expr: v-(v+v)*v
tree:
        /-- v [h=0]
    /-- * [h=2]
    |   |   /-- v [h=0]
    |   \-- + [h=1]
    |       \-- v [h=0]
- [h=3]
    \-- v [h=0]
```

```
zig build test
zig build run -- --expr "<expr>"     # v, + - * /, parentheses
zig build run -- --seed 42           # random expression
zig build run -Doptimize=ReleaseFast -- --bench 10000000
```

## JPL

Requires macOS with Metal. See [`jpl/README.md`](jpl/README.md).

```
cd jpl && make
bin/jplparse FILE 20
```
