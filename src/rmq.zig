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

const none: u64 = std.math.maxInt(u64);

/// Packs the split priority of an operator into one integer so the segment
/// tree is a plain min over u64: shallowest paren depth first, then lowest
/// precedence, then rightmost index (left associativity).
fn key(tok: Token, i: usize) u64 {
    if (tok.kind != .op) return none;
    const inv_index: u64 = std.math.maxInt(u32) - @as(u32, @intCast(i));
    return (@as(u64, tok.depth) << 40) | (@as(u64, tok.prec) << 32) | inv_index;
}

fn indexOf(k: u64) usize {
    return std.math.maxInt(u32) - @as(u32, @truncate(k));
}

pub const SegmentTree = struct {
    alloc: std.mem.Allocator,
    tree: []u64,
    size: usize,

    pub fn init(alloc: std.mem.Allocator, tokens: []const Token) !SegmentTree {
        if (tokens.len >= std.math.maxInt(u32)) return error.TooManyTokens;

        const size = tokens.len;
        const tree = try alloc.alloc(u64, size * 2);

        for (tokens, 0..) |tok, i| tree[size + i] = key(tok, i);

        var i: usize = size - 1;
        while (i > 0) : (i -= 1) {
            tree[i] = @min(tree[i * 2], tree[i * 2 + 1]);
        }

        return .{ .alloc = alloc, .tree = tree, .size = size };
    }

    pub fn deinit(self: *SegmentTree) void {
        self.alloc.free(self.tree);
    }

    /// Index of the operator to split [l, r] on, or null if the range has none.
    pub fn query(self: *const SegmentTree, l: usize, r: usize) ?usize {
        var left = l + self.size;
        var right = r + self.size + 1;
        var best = none;

        while (left < right) {
            if ((left & 1) == 1) {
                best = @min(best, self.tree[left]);
                left += 1;
            }
            if ((right & 1) == 1) {
                right -= 1;
                best = @min(best, self.tree[right]);
            }
            left >>= 1;
            right >>= 1;
        }

        return if (best == none) null else indexOf(best);
    }
};
