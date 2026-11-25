const std = @import("std");
const ast = @import("ast.zig");
const rmq = @import("rmq.zig");

const Allocator = std.mem.Allocator;

pub fn parseToTreeRmq(alloc: Allocator, input: []const u8) !ast.Tree {
    const tokens = try tokenize(alloc, input);
    defer alloc.free(tokens);

    var rmq_tree = try rmq.SegmentTree.init(alloc, tokens);
    defer rmq_tree.deinit();

    return build(alloc, tokens, &rmq_tree);
}

fn precedence(op: u8) u8 {
    return switch (op) {
        '+', '-' => 1,
        '*', '/' => 2,
        else => 0,
    };
}

/// Strips parentheses into per-operator depths and checks the grammar, so
/// every range the builder visits is a well-formed subexpression.
pub fn tokenize(alloc: Allocator, input: []const u8) ![]rmq.Token {
    var tokens = try std.ArrayList(rmq.Token).initCapacity(alloc, input.len);
    errdefer tokens.deinit(alloc);

    var depth: u16 = 0;
    var want_operand = true;
    for (input) |c| {
        switch (c) {
            ' ', '\t', '\n', '\r' => continue,
            '(' => {
                if (!want_operand) return error.MissingOperator;
                if (depth == std.math.maxInt(u16)) return error.MaxParenDepth;
                depth += 1;
            },
            ')' => {
                if (want_operand) return error.MissingOperand;
                if (depth == 0) return error.UnbalancedParen;
                depth -= 1;
            },
            'v' => {
                if (!want_operand) return error.MissingOperator;
                tokens.appendAssumeCapacity(.{ .kind = .value });
                want_operand = false;
            },
            '+', '-', '*', '/' => {
                if (want_operand) return error.MissingOperand;
                tokens.appendAssumeCapacity(.{ .kind = .op, .op = c, .depth = depth, .prec = precedence(c) });
                want_operand = true;
            },
            else => return error.InvalidToken,
        }
    }

    if (tokens.items.len == 0) return error.EmptyExpression;
    if (want_operand) return error.MissingOperand;
    if (depth != 0) return error.UnbalancedParen;

    return tokens.toOwnedSlice(alloc);
}

const Range = struct { l: u32, r: u32, at: u32 };

/// Splits each range on its best operator. Iterative so long operator chains
/// cannot overflow the stack.
fn build(alloc: Allocator, tokens: []const rmq.Token, rmq_tree: *const rmq.SegmentTree) !ast.Tree {
    const n: u32 = @intCast(tokens.len);

    const nodes = try alloc.alloc(ast.Node, n);
    errdefer alloc.free(nodes);

    // Operators in the order they were split; children come after parents.
    const order = try alloc.alloc(u32, n / 2);
    defer alloc.free(order);
    var visited: usize = 0;

    var stack = try std.ArrayList(Range).initCapacity(alloc, n / 2 + 1);
    defer stack.deinit(alloc);

    const root = visit(tokens, rmq_tree, nodes, &stack, 0, n - 1);
    while (stack.pop()) |range| {
        order[visited] = range.at;
        visited += 1;
        nodes[range.at] = .{
            .op = tokens[range.at].op,
            .left = visit(tokens, rmq_tree, nodes, &stack, range.l, range.at - 1),
            .right = visit(tokens, rmq_tree, nodes, &stack, range.at + 1, range.r),
        };
    }

    var k = visited;
    while (k > 0) {
        k -= 1;
        const node = &nodes[order[k]];
        node.height = @max(nodes[node.left].height, nodes[node.right].height) + 1;
    }

    return .{ .nodes = nodes, .root = root };
}

/// Returns the node index for [l, r]: leaves are filled in directly, operator
/// ranges are queued so their children get built later.
fn visit(
    tokens: []const rmq.Token,
    rmq_tree: *const rmq.SegmentTree,
    nodes: []ast.Node,
    stack: *std.ArrayList(Range),
    l: u32,
    r: u32,
) u32 {
    if (l == r) {
        std.debug.assert(tokens[l].kind == .value);
        nodes[l] = .{ .op = 'v' };
        return l;
    }
    const at: u32 = @intCast(rmq_tree.query(l, r).?);
    // Each operator is pushed exactly once, so the stack never outgrows n / 2 + 1.
    stack.appendAssumeCapacity(.{ .l = l, .r = r, .at = at });
    return at;
}

fn writeInfix(tree: *const ast.Tree, i: u32, out: *std.Io.Writer) !void {
    const node = tree.nodes[i];
    if (node.left == ast.none) return out.writeByte('v');
    try out.writeByte('(');
    try writeInfix(tree, node.left, out);
    try out.writeByte(node.op);
    try writeInfix(tree, node.right, out);
    try out.writeByte(')');
}

fn expectParse(input: []const u8, expected: []const u8) !void {
    const alloc = std.testing.allocator;
    var tree = try parseToTreeRmq(alloc, input);
    defer tree.deinit(alloc);

    var out: std.Io.Writer.Allocating = .init(alloc);
    defer out.deinit();
    try writeInfix(&tree, tree.root, &out.writer);
    try std.testing.expectEqualStrings(expected, out.written());
}

test "precedence and associativity" {
    try expectParse("v", "v");
    try expectParse("v+v*v", "(v+(v*v))");
    try expectParse("v*v+v", "((v*v)+v)");
    try expectParse("v-v-v", "((v-v)-v)");
    try expectParse("v/v/v", "((v/v)/v)");
    try expectParse("v-v+v*v/v", "((v-v)+((v*v)/v))");
}

test "parentheses" {
    try expectParse("(v+v)*v", "((v+v)*v)");
    try expectParse("v-(v-v)", "(v-(v-v))");
    try expectParse("((v))", "v");
    try expectParse("(v*(v+v))/v", "((v*(v+v))/v)");
    try expectParse(" v + ( v ) ", "(v+v)");
}

test "heights" {
    const alloc = std.testing.allocator;
    var tree = try parseToTreeRmq(alloc, "v+v*v");
    defer tree.deinit(alloc);
    try std.testing.expectEqual(@as(u32, 2), tree.nodes[tree.root].height);
}

test "rejects malformed input" {
    const alloc = std.testing.allocator;
    const cases = [_]struct { []const u8, anyerror }{
        .{ "", error.EmptyExpression },
        .{ "()", error.MissingOperand },
        .{ "v+", error.MissingOperand },
        .{ "+v", error.MissingOperand },
        .{ "vv", error.MissingOperator },
        .{ "v(v)", error.MissingOperator },
        .{ "(v+)v", error.MissingOperand },
        .{ "(v", error.UnbalancedParen },
        .{ "v)", error.UnbalancedParen },
        .{ "v%v", error.InvalidToken },
    };
    for (cases) |case| {
        try std.testing.expectError(case[1], parseToTreeRmq(alloc, case[0]));
    }
}

test "long chain does not overflow the stack" {
    const alloc = std.testing.allocator;
    const n = 200_000;
    const input = try alloc.alloc(u8, 2 * n - 1);
    defer alloc.free(input);
    for (input, 0..) |*c, i| c.* = if (i % 2 == 0) 'v' else '-';

    var tree = try parseToTreeRmq(alloc, input);
    defer tree.deinit(alloc);
    try std.testing.expectEqual(@as(u32, n - 1), tree.nodes[tree.root].height);
}

/// Reference precedence-climbing parser used to cross-check the RMQ parser.
const Reference = struct {
    src: []const u8,
    pos: usize = 0,

    fn peek(self: *Reference) u8 {
        return if (self.pos < self.src.len) self.src[self.pos] else 0;
    }

    fn primary(self: *Reference, buf: *std.ArrayList(u8), alloc: Allocator) Allocator.Error!void {
        if (self.peek() == '(') {
            self.pos += 1;
            try self.expr(buf, alloc, 1);
            self.pos += 1;
        } else {
            self.pos += 1;
            try buf.append(alloc, 'v');
        }
    }

    fn expr(self: *Reference, buf: *std.ArrayList(u8), alloc: Allocator, min_prec: u8) Allocator.Error!void {
        var lhs: std.ArrayList(u8) = .empty;
        defer lhs.deinit(alloc);
        try self.primary(&lhs, alloc);
        while (true) {
            const op = self.peek();
            const p = precedence(op);
            if (p == 0 or p < min_prec) break;
            self.pos += 1;
            var rhs: std.ArrayList(u8) = .empty;
            defer rhs.deinit(alloc);
            try self.expr(&rhs, alloc, p + 1);
            var combined: std.ArrayList(u8) = .empty;
            try combined.append(alloc, '(');
            try combined.appendSlice(alloc, lhs.items);
            try combined.append(alloc, op);
            try combined.appendSlice(alloc, rhs.items);
            try combined.append(alloc, ')');
            lhs.deinit(alloc);
            lhs = combined;
        }
        try buf.appendSlice(alloc, lhs.items);
    }
};

test "matches reference parser on random expressions" {
    const alloc = std.testing.allocator;
    const gen = @import("expr.zig");

    var seed: u64 = 0;
    while (seed < 500) : (seed += 1) {
        const input = if (seed % 2 == 0)
            try gen.genExprFromSeed(alloc, seed)
        else
            try gen.genLargeExpr(alloc, seed, 1 + seed);
        defer alloc.free(input);

        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();
        var expected: std.ArrayList(u8) = .empty;
        var ref = Reference{ .src = input };
        try ref.expr(&expected, arena.allocator(), 1);

        try expectParse(input, expected.items);
    }
}
