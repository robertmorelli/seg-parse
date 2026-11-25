const std = @import("std");

const Allocator = std.mem.Allocator;

/// Small random expression, shaped like a hand-written one.
pub fn genExprFromSeed(alloc: Allocator, seed: u64) ![]u8 {
    var prng = std.Random.DefaultPrng.init(seed);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    try genExpr(alloc, &out, prng.random(), 0);
    return out.toOwnedSlice(alloc);
}

fn genExpr(alloc: Allocator, out: *std.ArrayList(u8), r: std.Random, depth: u8) !void {
    if (depth > 8 or r.intRangeAtMost(u8, 0, 9) == 0)
        return out.append(alloc, 'v');
    const choice = r.intRangeAtMost(u8, 0, 10);
    if (choice < 8) {
        try genExpr(alloc, out, r, depth + 1);
        try out.append(alloc, "+-*/"[r.intRangeAtMost(u8, 0, 3)]);
        try genExpr(alloc, out, r, depth + 1);
    } else {
        try out.append(alloc, '(');
        try genExpr(alloc, out, r, depth + 1);
        try out.append(alloc, ')');
    }
}

/// Random expression with exactly `operands` values, for benchmarking.
pub fn genLargeExpr(alloc: Allocator, seed: u64, operands: usize) ![]u8 {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();

    var out = try std.ArrayList(u8).initCapacity(alloc, operands * 4);
    errdefer out.deinit(alloc);

    var open: usize = 0;
    for (0..operands) |i| {
        while (r.intRangeAtMost(u8, 0, 7) == 0) {
            try out.append(alloc, '(');
            open += 1;
        }
        try out.append(alloc, 'v');
        while (open > 0 and r.intRangeAtMost(u8, 0, 5) == 0) {
            try out.append(alloc, ')');
            open -= 1;
        }
        if (i + 1 < operands) try out.append(alloc, "+-*/"[r.intRangeAtMost(u8, 0, 3)]);
    }
    try out.appendNTimes(alloc, ')', open);

    return out.toOwnedSlice(alloc);
}
