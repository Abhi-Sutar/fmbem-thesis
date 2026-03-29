import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ==========================================
# 1. Boundary & QuadTree Logic
# ==========================================

class Node:
    def __init__(self, x, y, w, h, depth, index_in_parent=0, parent=None):
        self.x, self.y = x, y
        self.w, self.h = w, h
        self.depth = depth
        self.index_in_parent = index_in_parent
        self.parent = parent
        self.point_data = [] 
        self.children = []
        self.is_leaf = True
        self.plot_x = 0
        self.plot_y = 0
        
    @property
    def cx(self):
        return self.x + self.w / 2.0
        
    @property
    def cy(self):
        return self.y + self.h / 2.0

class QuadTree:
    def __init__(self, capacity, max_depth):
        self.capacity = capacity
        self.max_depth = max_depth
        self.nodes = []

    def build(self, points, x, y, w, h, depth=0, index_in_parent=0, parent=None):
        node = Node(x, y, w, h, depth, index_in_parent, parent)
        self.nodes.append(node)
        
        for p in points:
            if x <= p[1] < x + w and y <= p[2] < y + h:
                node.point_data.append(p)
                
        if len(node.point_data) > self.capacity and depth < self.max_depth:
            node.is_leaf = False
            hw, hh = w / 2, h / 2
            node.children.append(self.build(node.point_data, x, y, hw, hh, depth + 1, 0, node))
            node.children.append(self.build(node.point_data, x + hw, y, hw, hh, depth + 1, 1, node))
            node.children.append(self.build(node.point_data, x, y + hh, hw, hh, depth + 1, 2, node))
            node.children.append(self.build(node.point_data, x + hw, y + hh, hw, hh, depth + 1, 3, node))
            
        return node

def generate_boundary(n_points=35):
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    r = 0.8 + 0.2 * np.sin(2 * t) + 0.1 * np.cos(3 * t)
    x = r * np.cos(t)
    y = r * np.sin(t) * 0.7 
    x = (x - x.min()) / (x.max() - x.min()) * 0.7 + 0.15
    y = (y - y.min()) / (y.max() - y.min()) * 0.7 + 0.15
    ids = np.arange(1, n_points + 1)
    return np.column_stack((ids, x, y))

# ==========================================
# 2. Base Plotting Functions
# ==========================================

def has_points(node):
    if node.is_leaf: return len(node.point_data) > 0
    return any(has_points(c) for c in node.children)

def calculate_tree_layout(node, x_offset=0):
    node.plot_y = -node.depth * 1.5 
    active_children = [c for c in node.children if has_points(c)]
    
    if not active_children:
        node.plot_x = x_offset + 0.5
        return x_offset + 1.2 
        
    current_x = x_offset
    for c in active_children:
        current_x = calculate_tree_layout(c, current_x)
        
    node.plot_x = (active_children[0].plot_x + active_children[-1].plot_x) / 2.0
    return current_x

def plot_spatial_tree(ax, tree, boundary, show_reference=False, show_ids=True):
    bx = np.append(boundary[:, 1], boundary[0, 1])
    by = np.append(boundary[:, 2], boundary[0, 2])
    
    ax.plot(bx, by, 'k-', lw=1.5, zorder=1, clip_on=True)
    ax.scatter(boundary[:, 1], boundary[:, 2], c='k', s=15, zorder=2, clip_on=True)
    
    for i in range(len(boundary)):
        p1 = boundary[i]
        p2 = boundary[(i+1)%len(boundary)]
        mx, my = (p1[1]+p2[1])/2, (p1[2]+p2[2])/2
        dx, dy = p2[1]-p1[1], p2[2]-p1[2]
        length = np.hypot(dx, dy)
        nx, ny = -dy/length * 0.015, dx/length * 0.015
        ax.plot([mx-nx, mx+nx], [my-ny, my+ny], 'k-', lw=1, zorder=1, clip_on=True)
            
    if show_ids:
        for p in boundary:
            pid, px, py = p
            ax.text(px + 0.012, py + 0.012, str(int(pid)), fontsize=8, zorder=3, clip_on=True)
            
    for node in tree.nodes:
        if node.is_leaf and len(node.point_data) > 0:
            rect = Rectangle((node.x, node.y), node.w, node.h, fill=True, color='lightgrey', ec='k', lw=0.5, zorder=0, clip_on=True)
        else:
            rect = Rectangle((node.x, node.y), node.w, node.h, fill=False, ec='k', lw=0.5, clip_on=True)
        ax.add_patch(rect)
        
    ax.set_aspect('equal')
    ax.axis('off')
        
    if show_reference:
        ins_ax = ax.inset_axes([1.05, 0.05, 0.15, 0.15])
        ins_ax.set_xticks([])
        ins_ax.set_yticks([])
        ins_ax.set_xlim(0, 2)
        ins_ax.set_ylim(0, 2)
        ins_ax.plot([1, 1], [0, 2], 'k-', lw=1)
        ins_ax.plot([0, 2], [1, 1], 'k-', lw=1)
        ins_ax.text(0.5, 0.5, '0', ha='center', va='center', fontsize=12)
        ins_ax.text(1.5, 0.5, '1', ha='center', va='center', fontsize=12)
        ins_ax.text(0.5, 1.5, '2', ha='center', va='center', fontsize=12)
        ins_ax.text(1.5, 1.5, '3', ha='center', va='center', fontsize=12)
        for spine in ins_ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)

def plot_logical_tree(ax, root_node, tree):
    ax.axis('off')
    total_width = calculate_tree_layout(root_node, x_offset=0)
    
    def draw_node(node):
        active_children = [c for c in node.children if has_points(c)]
        for c in active_children:
            ax.plot([node.plot_x, c.plot_x], [node.plot_y - 0.25, c.plot_y + 0.25], 'k-', lw=1.0, zorder=1)
            draw_node(c)
            
        is_colored = node.is_leaf and len(node.point_data) > 0
        fc = 'lightgrey' if is_colored else 'white'
        rect = Rectangle((node.plot_x - 0.3, node.plot_y - 0.3), 0.6, 0.6, fill=True, facecolor=fc, ec='k', lw=1.2, zorder=2)
        ax.add_patch(rect)
        ax.text(node.plot_x, node.plot_y, str(node.index_in_parent), ha='center', va='center', fontsize=12, zorder=3)
        
        if node.is_leaf and len(node.point_data) > 0:
            pids = [str(int(p[0])) for p in node.point_data]
            id_str = ",".join(pids)
            ax.plot([node.plot_x, node.plot_x], [node.plot_y - 0.3, node.plot_y - 0.7], 'k-', lw=1.0, zorder=1)
            ax.text(node.plot_x, node.plot_y - 0.7, id_str, ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle="ellipse,pad=0.2", fc="white", ec="black", lw=1.0), zorder=3)

    draw_node(root_node)
    max_depth_reached = max(n.depth for n in tree.nodes if has_points(n))
    axis_x = total_width + 1.0  
    
    ax.text(axis_x, 1.0, "Cell level:", ha='center', va='center', fontsize=14)
    ax.plot([axis_x, axis_x], [0, -max_depth_reached * 1.5 - 0.7], 'k--', lw=1.2, zorder=0)
    
    for d in range(max_depth_reached + 1):
        ax.text(axis_x, -d * 1.5, str(d), ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="none"))
                
    element_y = -(max_depth_reached + 1) * 1.5
    ax.plot([axis_x, axis_x], [-max_depth_reached * 1.5, element_y], 'k--', lw=1.2, zorder=0)
    ax.text(axis_x, element_y, "element", ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle="ellipse,pad=0.3", fc="white", ec="black", lw=1.2))

    ax.set_xlim(-1, axis_x + 2)
    ax.set_ylim(element_y - 1, 2)

# ==========================================
# 3. FMM Specific Plotting Logic
# ==========================================

def draw_arrow(ax, start, end, color, ls, lw=1.5, zorder=4):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=color, ls=ls, lw=lw, 
                                mutation_scale=15, clip_on=True),
                # annotation_clip=False prevents the arrow from disappearing 
                # when the target (xy) is outside the zoomed axes view
                annotation_clip=False, 
                zorder=zorder, clip_on=True)

def label_z_points(ax, parent_node, label):
    pts = []
    for c in parent_node.children:
        if c.is_leaf: pts.extend(c.point_data)
    if pts:
        pts.sort(key=lambda p: p[2], reverse=True)
        ax.text(pts[0][1] - 0.02, pts[0][2] + 0.01, f"${label}$", fontsize=22, zorder=6, ha='right', clip_on=True)

def get_leaf_by_id(node, target_id):
    """Recursively search for the leaf node containing a specific point ID."""
    if node.is_leaf:
        if any(p[0] == target_id for p in node.point_data): return node
        return None
    for child in node.children:
        res = get_leaf_by_id(child, target_id)
        if res: return res
    return None

def get_ancestor_at_depth(node, depth):
    curr = node
    while curr and curr.depth > depth:
        curr = curr.parent
    return curr

def setup_fmm_nodes(tree):
    # 1. Target node (keep as leftmost)
    leaves = [n for n in tree.nodes if n.is_leaf and n.point_data]
    leaves.sort(key=lambda n: n.x)
    t_parent = get_ancestor_at_depth(leaves[0], 3)
    
    # 2. Source nodes based on specific element IDs
    s_parents = []
    for point_id in [8, 6, 3]: # Order determines M2L mapping
        leaf = get_leaf_by_id(tree.nodes[0], point_id)
        if leaf:
            parent = get_ancestor_at_depth(leaf, 3)
            # Avoid duplicates if two IDs share the same depth=3 parent
            if parent not in s_parents:
                s_parents.append(parent)
                
    return t_parent, s_parents

def plot_fmm_interactions(ax, t_parent, s_parents):
    # Source Side (Iterate over all requested source parents)
    for sp in s_parents:
        for child in sp.children:
            if child.is_leaf and child.point_data:
                for p in child.point_data:
                    draw_arrow(ax, (p[1], p[2]), (child.cx, child.cy), 'red', 'solid', lw=2)
                draw_arrow(ax, (child.cx, child.cy), (sp.cx, sp.cy), 'mediumblue', 'dashed', lw=1.5)
                ax.plot(child.cx, child.cy, '^', color='red', markersize=5, zorder=5, clip_on=True)
        ax.plot(sp.cx, sp.cy, 's', color='mediumblue', markersize=8, zorder=5, clip_on=True)

    # Target Side
    for child in t_parent.children:
        if child.is_leaf and child.point_data:
            draw_arrow(ax, (t_parent.cx, t_parent.cy), (child.cx, child.cy), 'saddlebrown', 'dotted', lw=2)
            for p in child.point_data:
                draw_arrow(ax, (child.cx, child.cy), (p[1], p[2]), 'deeppink', 'dashdot', lw=2)
            ax.plot(child.cx, child.cy, '^', color='red', markersize=5, zorder=5, clip_on=True)
    ax.plot(t_parent.cx, t_parent.cy, 's', color='mediumblue', markersize=8, zorder=5, clip_on=True)

    # Green: M2L Translations (From each source parent DIRECTLY to the single target parent)
    for sp in s_parents:
        draw_arrow(ax, (sp.cx, sp.cy), (t_parent.cx, t_parent.cy), 'forestgreen', 'dashdot', lw=1.5)

def plot_main_fmm(ax, tree, boundary, t_parent, s_parents):
    plot_spatial_tree(ax, tree, boundary, show_ids=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plot_fmm_interactions(ax, t_parent, s_parents)
    
    # Label z_0 and z
    label_z_points(ax, t_parent, 'z_0')
    # Use the parent of element 3 as the definitive 'z' label location
    s_parent_3 = s_parents[-1] if len(s_parents) > 0 else None
    if s_parent_3:
        label_z_points(ax, s_parent_3, 'z')
    
    pad = 0.03
    # Draw dashed box for Target
    rect_t = Rectangle((t_parent.x - pad, t_parent.y - pad), t_parent.w + 2*pad, t_parent.h + 2*pad, 
                       fill=False, ec='black', ls='--', lw=1, zorder=6)
    ax.add_patch(rect_t)
    
    # ONLY draw dashed box for the Source parent containing Element 3
    if s_parent_3:
        rect_s = Rectangle((s_parent_3.x - pad, s_parent_3.y - pad), s_parent_3.w + 2*pad, s_parent_3.h + 2*pad, 
                           fill=False, ec='black', ls='--', lw=1, zorder=6)
        ax.add_patch(rect_s)

def plot_zoom_fmm(ax, tree, boundary, t_parent, s_parents, focus_node):
    plot_spatial_tree(ax, tree, boundary, show_ids=False)
    plot_fmm_interactions(ax, t_parent, s_parents)
    
    label = 'z_0' if focus_node == t_parent else 'z'
    label_z_points(ax, focus_node, label)
    
    pad = 0.03
    ax.set_xlim(focus_node.x - pad, focus_node.x + focus_node.w + pad)
    ax.set_ylim(focus_node.y - pad, focus_node.y + focus_node.h + pad)
    
    ax.axis('on')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linestyle('--')
        spine.set_linewidth(2)
        spine.set_zorder(10) 

# ==========================================
# 4. Execution & Saving
# ==========================================

if __name__ == "__main__":
    boundary_pts = generate_boundary(n_points=35)
    qtree = QuadTree(capacity=1, max_depth=6) 
    root = qtree.build(boundary_pts, 0.0, 0.0, 1.0, 1.0)
    
    t_parent, s_parents = setup_fmm_nodes(qtree)

    fig1, ax1 = plt.subplots(figsize=(8, 8))
    plot_spatial_tree(ax1, qtree, boundary_pts, show_reference=True, show_ids=True)
    fig1.savefig("diagram_1_spatial.svg", format="svg", bbox_inches='tight')

    fig2, ax2 = plt.subplots(figsize=(16, 8)) 
    plot_logical_tree(ax2, root, qtree)
    fig2.savefig("diagram_2_logical.svg", format="svg", bbox_inches='tight')

    fig3, ax3 = plt.subplots(figsize=(8, 8))
    plot_main_fmm(ax3, qtree, boundary_pts, t_parent, s_parents)
    fig3.savefig("diagram_3_fmm_main.svg", format="svg", bbox_inches='tight')
    
    fig4, ax4 = plt.subplots(figsize=(4, 4))
    plot_zoom_fmm(ax4, qtree, boundary_pts, t_parent, s_parents, focus_node=t_parent)
    fig4.savefig("diagram_4_fmm_target_zoom.svg", format="svg", bbox_inches='tight')
    
    fig5, ax5 = plt.subplots(figsize=(4, 4))
    # Pass the specific parent box for element 3 as the focus_node
    plot_zoom_fmm(ax5, qtree, boundary_pts, t_parent, s_parents, focus_node=s_parents[-1])
    fig5.savefig("diagram_5_fmm_source_zoom.svg", format="svg", bbox_inches='tight')

    print("Successfully generated and saved 5 individual SVG files.")