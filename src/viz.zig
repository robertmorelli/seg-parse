const std = @import("std");
const ast = @import("ast.zig");

const Side = enum {
    root,
    left,
    right,
};

pub fn writeAscii(alloc: std.mem.Allocator, writer: anytype, node: *const ast.Node) !void {
    var prefix = std.ArrayList(u8).init(alloc);
    defer prefix.deinit();

    try writeInorder(writer, node, &prefix, .root);
}

fn writeInorder(writer: anytype, node: *const ast.Node, prefix: *std.ArrayList(u8), side: Side) !void {
    if (node.right) |right| {
        const segment = switch (side) {
            .left => "|   ",
            .right, .root => "    ",
        };
        try prefix.appendSlice(segment);
        try writeInorder(writer, right, prefix, .right);
        prefix.shrinkRetainingCapacity(prefix.items.len - segment.len);
    }

    try writer.writeAll(prefix.items);
    switch (side) {
        .root => {},
        .left => try writer.writeAll("\\-- "),
        .right => try writer.writeAll("/-- "),
    }
    try writer.print("{c} [h={d}]\n", .{ node.op, node.height });

    if (node.left) |left| {
        const segment = switch (side) {
            .left => "    ",
            .right, .root => "|   ",
        };
        try prefix.appendSlice(segment);
        try writeInorder(writer, left, prefix, .left);
        prefix.shrinkRetainingCapacity(prefix.items.len - segment.len);
    }
}
