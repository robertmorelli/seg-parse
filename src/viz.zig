const std = @import("std");
const ast = @import("ast.zig");

const Side = enum {
    root,
    left,
    right,
};

pub fn writeAscii(alloc: std.mem.Allocator, writer: *std.Io.Writer, tree: *const ast.Tree) !void {
    var prefix: std.ArrayList(u8) = .empty;
    defer prefix.deinit(alloc);

    try writeInorder(alloc, writer, tree, tree.root, &prefix, .root);
}

fn writeInorder(
    alloc: std.mem.Allocator,
    writer: *std.Io.Writer,
    tree: *const ast.Tree,
    index: u32,
    prefix: *std.ArrayList(u8),
    side: Side,
) !void {
    const node = tree.nodes[index];

    if (node.right != ast.none) {
        const segment = switch (side) {
            .left => "|   ",
            .right, .root => "    ",
        };
        try prefix.appendSlice(alloc, segment);
        try writeInorder(alloc, writer, tree, node.right, prefix, .right);
        prefix.shrinkRetainingCapacity(prefix.items.len - segment.len);
    }

    try writer.writeAll(prefix.items);
    switch (side) {
        .root => {},
        .left => try writer.writeAll("\\-- "),
        .right => try writer.writeAll("/-- "),
    }
    try writer.print("{c} [h={d}]\n", .{ node.op, node.height });

    if (node.left != ast.none) {
        const segment = switch (side) {
            .left, .root => "    ",
            .right => "|   ",
        };
        try prefix.appendSlice(alloc, segment);
        try writeInorder(alloc, writer, tree, node.left, prefix, .left);
        prefix.shrinkRetainingCapacity(prefix.items.len - segment.len);
    }
}
