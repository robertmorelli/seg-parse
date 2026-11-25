const std = @import("std");

const Allocator = std.mem.Allocator;

pub fn genExprFromSeed(alloc: Allocator, seed_opt: ?u64) ![]u8 {
    const seed: u64 = seed_opt orelse @intCast(std.time.nanoTimestamp());
    var prng = std.Random.DefaultPrng.init(seed);
    return genExpr(alloc, prng.random(), 0);
}

fn genExpr(alloc: Allocator, r: std.Random, depth: u8) ![]u8 {
    if (depth > 20 or r.intRangeAtMost(u8, 0, 9) == 0)
        return std.fmt.allocPrint(alloc, "v", .{});
    const choice = r.intRangeAtMost(u8, 0, 10);
    if (choice < 8) {
        const left = try genExpr(alloc, r, depth + 1);
        defer alloc.free(left);
        const right = try genExpr(alloc, r, depth + 1);
        defer alloc.free(right);
        const op = switch (r.intRangeAtMost(u8, 0, 3)) {
            0 => "+",
            1 => "-",
            2 => "*",
            else => "/",
        };
        return std.fmt.allocPrint(alloc, "{s}{s}{s}", .{ left, op, right });
    } else {
        const inner = try genExpr(alloc, r, depth + 1);
        defer alloc.free(inner);
        return std.fmt.allocPrint(alloc, "({s})", .{inner});
    }
}
