const std = @import("std");

pub const TokenKind = enum {
    value,
    op,
};

pub const Token = struct {
    kind: TokenKind,
    op: u8 = 0,
    depth: u16 = 0,
    prec: u8 = 0,
};

pub fn betterIndex(tokens: []const Token, left: ?usize, right: ?usize) ?usize {
    if (left == null) return right;
    if (right == null) return left;

    const l_idx = left.?;
    const r_idx = right.?;
    const l_tok = tokens[l_idx];
    const r_tok = tokens[r_idx];

    if (l_tok.depth != r_tok.depth) {
        return if (l_tok.depth < r_tok.depth) left else right;
    }
    if (l_tok.prec != r_tok.prec) {
        return if (l_tok.prec < r_tok.prec) left else right;
    }

    return if (l_idx > r_idx) left else right;
}

pub const SegmentTree = struct {
    alloc: std.mem.Allocator,
    tokens: []const Token,
    tree: []?usize,
    size: usize,

    pub fn init(alloc: std.mem.Allocator, tokens: []const Token) !SegmentTree {
        var size: usize = 1;
        while (size < tokens.len) size <<= 1;
        const tree = try alloc.alloc(?usize, size * 2);
        @memset(tree, null);

        for (tokens, 0..) |tok, i| {
            if (tok.kind == .op) {
                tree[size + i] = i;
            }
        }

        var i: usize = size - 1;
        while (i > 0) : (i -= 1) {
            tree[i] = betterIndex(tokens, tree[i * 2], tree[i * 2 + 1]);
        }

        return .{
            .alloc = alloc,
            .tokens = tokens,
            .tree = tree,
            .size = size,
        };
    }

    pub fn deinit(self: *SegmentTree) void {
        self.alloc.free(self.tree);
    }

    pub fn query(self: *SegmentTree, l: usize, r: usize) ?usize {
        var left = l + self.size;
        var right = r + self.size;
        var best_left: ?usize = null;
        var best_right: ?usize = null;

        while (left <= right) {
            if ((left & 1) == 1) {
                best_left = betterIndex(self.tokens, best_left, self.tree[left]);
                left += 1;
            }
            if ((right & 1) == 0) {
                best_right = betterIndex(self.tokens, self.tree[right], best_right);
                right -= 1;
            }
            left >>= 1;
            right >>= 1;
        }

        return betterIndex(self.tokens, best_left, best_right);
    }
};
