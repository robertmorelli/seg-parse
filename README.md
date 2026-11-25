# seg-parse

Parses arithmetic expressions by range-minimum queries instead of recursive descent.

Parentheses are stripped into a depth on each operator. The root of any
subexpression is then the operator with the lowest (depth, precedence), rightmost
on ties, so the tree is built by repeatedly asking a segment tree for that
minimum and splitting the range around it.

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

## Usage

Requires Zig 0.16.

```
zig build test
zig build run -- --expr "<expr>"     # v, + - * /, parentheses
zig build run -- --seed 42           # random expression
zig build run -Doptimize=ReleaseFast -- --bench 10000000
```

## JPL on the GPU

[`jpl/`](jpl/) applies the same idea to a real language: a Metal lexer and parser for JPL that lexes and parses 12.9 MB in 22 ms, faster than the fastest CPU lexer alone.
