const std = @import("std");

pub const none = std.math.maxInt(u32);

/// Nodes live in one flat array indexed by token position: an operator token
/// becomes an inner node, a value token becomes a leaf.
pub const Node = struct {
    op: u8,
    left: u32 = none,
    right: u32 = none,
    height: u32 = 0,
};

pub const Tree = struct {
    nodes: []Node,
    root: u32,

    pub fn deinit(self: *Tree, alloc: std.mem.Allocator) void {
        alloc.free(self.nodes);
    }

    pub fn get(self: *const Tree, i: u32) ?*const Node {
        return if (i == none) null else &self.nodes[i];
    }
};
