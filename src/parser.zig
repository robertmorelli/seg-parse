const std = @import("std");
const ast = @import("ast.zig");
const rmq = @import("rmq.zig");

const Allocator = std.mem.Allocator;

pub fn parseToTreeRmq(alloc: Allocator, input: []const u8) !*ast.Node {
    const tokens = try tokenize(alloc, input);
    defer alloc.free(tokens);

    var rmq_tree = try rmq.SegmentTree.init(alloc, tokens);
    defer rmq_tree.deinit();

    return parseRange(alloc, tokens, &rmq_tree, 0, tokens.len - 1);
}

fn precedence(op: u8) u8 {
    return switch (op) {
        '+', '-' => 1,
        '*', '/' => 2,
        else => 0,
    };
}

fn tokenize(alloc: Allocator, input: []const u8) ![]rmq.Token {
    var tokens = try std.ArrayList(rmq.Token).initCapacity(alloc, input.len);
    errdefer tokens.deinit(alloc);

    var depth: u16 = 0;
    for (input) |c| {
        switch (c) {
            ' ', '\t', '\n', '\r' => continue,
            '(' => {
                if (depth == std.math.maxInt(u16)) return error.MaxParenDepth;
                depth += 1;
            },
            ')' => {
                if (depth == 0) return error.UnbalancedParen;
                depth -= 1;
            },
            'v' => try tokens.append(alloc, .{ .kind = .value }),
            '+', '-', '*', '/' => try tokens.append(alloc, .{ .kind = .op, .op = c, .depth = depth, .prec = precedence(c) }),
            else => return error.InvalidToken,
        }
    }

    if (depth != 0) return error.UnbalancedParen;
    if (tokens.items.len == 0) return error.EmptyExpression;

    return tokens.toOwnedSlice(alloc);
}

fn parseRange(
    alloc: Allocator,
    tokens: []const rmq.Token,
    rmq_tree: *rmq.SegmentTree,
    l: usize,
    r: usize,
) !*ast.Node {
    if (l > r) return error.MissingOperand;

    const op_index = rmq_tree.query(l, r) orelse {
        if (l == r and tokens[l].kind == .value) {
            return try newLeaf(alloc);
        }
        return error.MissingOperator;
    };

    if (op_index == l or op_index == r) return error.MissingOperand;

    const left = try parseRange(alloc, tokens, rmq_tree, l, op_index - 1);
    errdefer {
        left.deinit(alloc);
        alloc.destroy(left);
    }

    const right = try parseRange(alloc, tokens, rmq_tree, op_index + 1, r);
    errdefer {
        right.deinit(alloc);
        alloc.destroy(right);
    }

    return newNode(alloc, tokens[op_index].op, left, right);
}

fn newLeaf(alloc: Allocator) !*ast.Node {
    const node = try alloc.create(ast.Node);
    node.* = .{ .op = 'v', .left = null, .right = null, .height = 0 };
    return node;
}

fn newNode(alloc: Allocator, op: u8, left: *ast.Node, right: *ast.Node) !*ast.Node {
    const node = try alloc.create(ast.Node);
    node.* = .{ .op = op, .left = left, .right = right, .height = 0 };
    return node;
}
