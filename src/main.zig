const std = @import("std");
const Io = std.Io;
const expr = @import("expr.zig");
const parser = @import("parser.zig");
const viz = @import("viz.zig");

const Allocator = std.mem.Allocator;

const Options = struct {
    expr: ?[]const u8 = null,
    seed: ?u64 = null,
    bench: ?usize = null,
    help: bool = false,
};

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer: Io.File.Writer = .initStreaming(.stdout(), io, &stdout_buffer);
    const out = &stdout_writer.interface;
    defer out.flush() catch {};

    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const opts = parseArgs(args) catch |err| {
        try printUsage(out);
        return err;
    };

    if (opts.help) return printUsage(out);

    const seed = opts.seed orelse @as(u64, @truncate(@as(u96, @bitCast(Io.Clock.real.now(io).nanoseconds))));

    if (opts.bench) |operands| return bench(alloc, io, out, seed, operands);

    const expression = if (opts.expr) |value| try alloc.dupe(u8, value) else try expr.genExprFromSeed(alloc, seed);
    defer alloc.free(expression);

    var tree = try parser.parseToTreeRmq(alloc, expression);
    defer tree.deinit(alloc);

    try out.print("expr: {s}\n", .{expression});
    try out.writeAll("tree:\n");
    try viz.writeAscii(alloc, out, &tree);
}

fn bench(alloc: Allocator, io: Io, out: *Io.Writer, seed: u64, operands: usize) !void {
    const expression = try expr.genLargeExpr(alloc, seed, operands);
    defer alloc.free(expression);

    var best: u64 = std.math.maxInt(u64);
    var height: u32 = 0;
    for (0..5) |_| {
        const start = Io.Clock.awake.now(io);
        var tree = try parser.parseToTreeRmq(alloc, expression);
        const elapsed: u64 = @intCast(start.durationTo(Io.Clock.awake.now(io)).nanoseconds);
        height = tree.nodes[tree.root].height;
        tree.deinit(alloc);
        best = @min(best, elapsed);
    }

    const ns_per_op = @as(f64, @floatFromInt(best)) / @as(f64, @floatFromInt(operands));
    try out.print("{d} operands, {d} bytes, tree height {d}\n", .{ operands, expression.len, height });
    try out.print("best of 5: {d:.2} ms ({d:.1} ns/operand)\n", .{ @as(f64, @floatFromInt(best)) / 1e6, ns_per_op });
}

fn parseArgs(args: []const [:0]const u8) !Options {
    var opts = Options{};

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--expr")) {
            i += 1;
            if (i >= args.len) return error.MissingExpr;
            opts.expr = args[i];
        } else if (std.mem.eql(u8, arg, "--seed")) {
            i += 1;
            if (i >= args.len) return error.MissingSeed;
            opts.seed = try std.fmt.parseInt(u64, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--bench")) {
            i += 1;
            if (i >= args.len) return error.MissingBench;
            opts.bench = try std.fmt.parseInt(usize, args[i], 10);
            if (opts.bench.? == 0) return error.MissingBench;
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            opts.help = true;
        } else {
            return error.UnknownArg;
        }
    }

    return opts;
}

fn printUsage(out: *Io.Writer) !void {
    try out.writeAll(
        "Usage: seg_parse [--expr <expr>] [--seed <u64>] [--bench <n>]\n" ++
            "  --expr <expr>  Provide an expression (uses v, +, -, *, /, parentheses).\n" ++
            "  --seed <u64>   Seed for random expression generation.\n" ++
            "  --bench <n>    Time parsing a random expression with n operands.\n" ++
            "  --help         Show this help.\n" ++
            "\n" ++
            "Heights are measured as edge-count from node to deepest leaf (leaf height = 0).\n" ++
            "ASCII output is inorder (right subtree printed above, left below).\n",
    );
}

test {
    _ = parser;
}
