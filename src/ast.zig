const std = @import("std");

pub const Node = struct {
    op: u8,
    left: ?*Node,
    right: ?*Node,
    height: u16 = 0,

    pub fn deinit(self: *Node, alloc: std.mem.Allocator) void {
        if (self.left) |l| {
            l.deinit(alloc);
            alloc.destroy(l);
        }
        if (self.right) |r| {
            r.deinit(alloc);
            alloc.destroy(r);
        }
    }

    pub fn computeHeights(self: *Node) u16 {
        var left_h: u16 = 0;
        var right_h: u16 = 0;

        if (self.left) |l| {
            left_h = l.computeHeights();
        }
        if (self.right) |r| {
            right_h = r.computeHeights();
        }

        const max_child = if (left_h > right_h) left_h else right_h;
        self.height = if (self.left == null and self.right == null) 0 else max_child + 1;
        return self.height;
    }
};
