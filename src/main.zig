const std = @import("std");
const expr = @import("expr.zig");
const parser = @import("parser.zig");
const viz = @import("viz.zig");

const Allocator = std.mem.Allocator;

const Options = struct {
    expr: ?[]const u8 = null,
    seed: ?u64 = null,
    help: bool = false,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const args = try std.process.argsAlloc(alloc);
    defer std.process.argsFree(alloc, args);

    var opts = parseArgs(alloc, args) catch |err| {
        try printUsage();
        return err;
    };
    defer if (opts.expr) |value| alloc.free(value);

    if (opts.help) {
        try printUsage();
        return;
    }

    var expr_owned = false;
    const expression = if (opts.expr) |value| value else blk: {
        expr_owned = true;
        break :blk try expr.genExprFromSeed(alloc, opts.seed);
    };
    defer if (expr_owned) alloc.free(expression);

    const tree = try parser.parseToTreeRmq(alloc, expression);
    defer {
        tree.deinit(alloc);
        alloc.destroy(tree);
    }

    _ = tree.computeHeights();

    var buffered = std.io.bufferedWriter(std.io.getStdOut().writer());
    const out = buffered.writer();

    try out.print("expr: {s}\n", .{expression});
    try out.writeAll("tree:\n");
    try viz.writeAscii(alloc, out, tree);

    try buffered.flush();
}

fn parseArgs(alloc: Allocator, args: [][]const u8) !Options {
    var opts = Options{};
    errdefer if (opts.expr) |value| alloc.free(value);

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--expr")) {
            i += 1;
            if (i >= args.len) return error.MissingExpr;
            opts.expr = try alloc.dupe(u8, args[i]);
        } else if (std.mem.eql(u8, arg, "--seed")) {
            i += 1;
            if (i >= args.len) return error.MissingSeed;
            opts.seed = try std.fmt.parseInt(u64, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            opts.help = true;
        } else {
            return error.UnknownArg;
        }
    }

    return opts;
}

fn printUsage() !void {
    var buffered = std.io.bufferedWriter(std.io.getStdOut().writer());
    const out = buffered.writer();
    try out.writeAll(
        "Usage: seg_parse [--expr <expr>] [--seed <u64>]\n" ++
        "  --expr <expr>  Provide an expression (uses v, +, -, *, /, parentheses).\n" ++
        "  --seed <u64>   Seed for random expression generation.\n" ++
        "  --help         Show this help.\n" ++
        "\n" ++
        "Heights are measured as edge-count from node to deepest leaf (leaf height = 0).\n" ++
        "ASCII output is inorder (right subtree printed above, left below).\n",
    );
    try buffered.flush();
}
