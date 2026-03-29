import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# ==========================================
# 0. CONFIGURATION SETTINGS
# ==========================================

# --- Geometry & QuadTree Settings ---
NUM_ELEMENTS = 30               # Total number of boundary elements
MAX_ELEMENTS_PER_CELL = 1       # QuadTree capacity (1 guarantees unique leaves for FMM)
TREE_MAX_DEPTH = 6              # Maximum subdivision depth allowed for the tree

# --- FMM Interaction Targets ---
TARGET_ID = 18                  # Element ID to act as the Target (z_0) 
SOURCE_IDS = [9, 5, 2]         # List of Element IDs to act as Sources (z)

# --- Tree Level Settings ---
# Note: Root is 0, smaller numbers = larger boxes
M2L_DEPTH = 2                   # Tree level where M2L translations occur (Grandparent nodes)

# --- View / Zoom Settings ---
SOURCE_ZOOM_ID = 3              # Which Source ID to center the right-hand zoom box around


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

def generate_boundary(n_points):
    """Generates a parametric kidney shape based on true geometric coordinates"""
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    r = 1.0 + 0.5 * np.sin(1 * t - 3*np.pi/8) + 0.2 * np.cos(2 * t- 3*np.pi/8)
    x = r * np.cos(t)
    y = r * np.sin(t)
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
    
    root_node = tree.nodes[0]
    root_w = root_node.w
    
    ax.plot(bx, by, 'k-', lw=1.5, zorder=1, clip_on=True)
    ax.scatter(boundary[:, 1], boundary[:, 2], c='k', s=15, zorder=2, clip_on=True)
    
    for i in range(len(boundary)):
        p1 = boundary[i]
        p2 = boundary[(i+1)%len(boundary)]
        mx, my = (p1[1]+p2[1])/2, (p1[2]+p2[2])/2
        dx, dy = p2[1]-p1[1], p2[2]-p1[2]
        length = np.hypot(dx, dy)
        nx, ny = -dy/length * (root_w * 0.015), dx/length * (root_w * 0.015)
        ax.plot([mx-nx, mx+nx], [my-ny, my+ny], 'k-', lw=1, zorder=1, clip_on=True)
            
    if show_ids:
        for p in boundary:
            pid, px, py = p
            ax.text(px + root_w*0.012, py + root_w*0.012, str(int(pid)), fontsize=8, zorder=3, clip_on=True)
            
    for node in tree.nodes:
        if node.is_leaf and len(node.point_data) > 0:
            rect = Rectangle((node.x, node.y), node.w, node.h, fill=True, color='lightgrey', ec='k', lw=0.5, zorder=0, clip_on=True)
        else:
            rect = Rectangle((node.x, node.y), node.w, node.h, fill=False, ec='k', lw=0.5, clip_on=True)
        ax.add_patch(rect)
        
    ax.set_aspect('equal')
    ax.set_xlim(root_node.x, root_node.x + root_node.w)
    ax.set_ylim(root_node.y, root_node.y + root_node.h)
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
                annotation_clip=False, 
                zorder=zorder, clip_on=True)

def label_z_points(ax, parent_node, label, root_w):
    pts = []
    def collect_pts(node):
        if node.is_leaf: pts.extend(node.point_data)
        else:
            for c in node.children: collect_pts(c)
    if parent_node:
        collect_pts(parent_node)
    
    if pts:
        pts.sort(key=lambda p: p[2], reverse=True)
        ax.text(pts[0][1] - root_w*0.02, pts[0][2] + root_w*0.01, f"${label}$", fontsize=22, zorder=6, ha='right', clip_on=True)

def get_leaf_by_id(node, target_id):
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

def get_all_leaves(node):
    if node.is_leaf:
        return [node] if node.point_data else []
    leaves = []
    for c in node.children:
        leaves.extend(get_all_leaves(c))
    return leaves

def setup_fmm_nodes(tree, target_id, source_ids, m2l_depth):
    t_leaf = get_leaf_by_id(tree.nodes[0], target_id)
    
    if not t_leaf or t_leaf.depth <= m2l_depth: 
        deep_leaves = [n for n in tree.nodes if n.is_leaf and n.point_data and n.depth > m2l_depth]
        if deep_leaves:
            deep_leaves.sort(key=lambda n: n.x)
            t_leaf = deep_leaves[0]
        else:
            leaves = [n for n in tree.nodes if n.is_leaf and n.point_data]
            leaves.sort(key=lambda n: n.x)
            t_leaf = leaves[0]
            
    t_m2l_node = get_ancestor_at_depth(t_leaf, m2l_depth)
    
    s_m2l_nodes = []
    for point_id in source_ids: 
        leaf = get_leaf_by_id(tree.nodes[0], point_id)
        if leaf:
            m2l_node = get_ancestor_at_depth(leaf, m2l_depth)
            if m2l_node and m2l_node not in s_m2l_nodes: s_m2l_nodes.append(m2l_node)
                
    return t_leaf, t_m2l_node, s_m2l_nodes

def plot_fmm_interactions(ax, t_leaf, t_m2l_node, s_m2l_nodes):
    
    def upward_pass(node, stop_node):
        """Recursively traverses upward, starting from the deepest leaves"""
        if not has_points(node): return
        
        if node.is_leaf:
            # Multipole Expansions (Leaf Points -> Leaf Center)
            for p in node.point_data:
                draw_arrow(ax, (p[1], p[2]), (node.cx, node.cy), 'red', 'solid', lw=2)
            ax.plot(node.cx, node.cy, '^', color='red', markersize=5, zorder=5, clip_on=True)
        else:
            for child in node.children:
                upward_pass(child, stop_node)
        
        # M2M Translation (Node Center -> Parent Center)
        if node != stop_node and node.parent:
            draw_arrow(ax, (node.cx, node.cy), (node.parent.cx, node.parent.cy), 'mediumblue', 'dashed', lw=1.5)
            if not node.is_leaf:
                ax.plot(node.cx, node.cy, 's', color='mediumblue', markersize=6, zorder=5, clip_on=True)

    def targeted_downward_pass(node, target_leaf):
        """Recursively traverses downward, strictly following the path to the target leaf"""
        if node == target_leaf:
            # Local Expansions (Leaf Center -> Leaf Points)
            for p in node.point_data:
                draw_arrow(ax, (node.cx, node.cy), (p[1], p[2]), 'deeppink', 'dashdot', lw=2)
            ax.plot(node.cx, node.cy, '^', color='red', markersize=5, zorder=5, clip_on=True)
            return

        for child in node.children:
            # Check if target_leaf is in this child's subtree
            curr = target_leaf
            is_in_child = False
            while curr:
                if curr == child:
                    is_in_child = True
                    break
                curr = curr.parent
            
            if is_in_child:
                # L2L Translation (Node Center -> Child Center)
                draw_arrow(ax, (node.cx, node.cy), (child.cx, child.cy), 'saddlebrown', 'dotted', lw=2)
                if child != target_leaf:
                    ax.plot(child.cx, child.cy, 's', color='mediumblue', markersize=6, zorder=5, clip_on=True)
                
                # Recurse only down the correct branch
                targeted_downward_pass(child, target_leaf)
                break

    # ---- Execution ----
    
    # Source Side (Upward Passes)
    for sm2l in s_m2l_nodes:
        upward_pass(sm2l, sm2l)
        ax.plot(sm2l.cx, sm2l.cy, 'D', color='mediumblue', markersize=8, zorder=5, clip_on=True)

    # Target Side (Targeted Downward Pass)
    if t_m2l_node:
        ax.plot(t_m2l_node.cx, t_m2l_node.cy, 'D', color='mediumblue', markersize=8, zorder=5, clip_on=True)
        targeted_downward_pass(t_m2l_node, t_leaf)

    # Translations (M2L - Green)
    if t_m2l_node:
        for sm2l in s_m2l_nodes:
            draw_arrow(ax, (sm2l.cx, sm2l.cy), (t_m2l_node.cx, t_m2l_node.cy), 'forestgreen', 'dashdot', lw=1.5)

def add_fmm_legend(ax):
    legend_elements = [
        Line2D([0], [0], color='red', lw=1.5, ls='solid', marker='>', markersize=8, label='Multipole expansion'),
        Line2D([0], [0], color='mediumblue', lw=1.5, ls='dashed', marker='>', markersize=8, label='M2M translation'),
        Line2D([0], [0], color='forestgreen', lw=1.5, ls='dashdot', marker='>', markersize=8, label='M2L translation'),
        Line2D([0], [0], color='saddlebrown', lw=1.5, ls='dotted', marker='>', markersize=8, label='L2L translation'),
        Line2D([0], [0], color='deeppink', lw=1.5, ls='dashdot', marker='>', markersize=8, label='Local expansion'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='mediumblue', markersize=8, label='Center of parent cells'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=8, label='Center of leaves')
    ]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=11)

def plot_main_fmm(ax, tree, boundary, t_leaf, t_m2l_node, s_m2l_nodes, s_zoom_focus):
    plot_spatial_tree(ax, tree, boundary, show_ids=False)
    
    plot_fmm_interactions(ax, t_leaf, t_m2l_node, s_m2l_nodes)
    
    root_w = tree.nodes[0].w
    if t_m2l_node: label_z_points(ax, t_m2l_node, 'z_0', root_w)
    if s_zoom_focus: label_z_points(ax, s_zoom_focus, 'z', root_w)
    
    pad = t_m2l_node.w * 0.05 if t_m2l_node else 0.05
    if t_m2l_node:
        rect_t = Rectangle((t_m2l_node.x - pad, t_m2l_node.y - pad), t_m2l_node.w + 2*pad, t_m2l_node.h + 2*pad, 
                           fill=False, ec='black', ls='--', lw=1, zorder=6)
        ax.add_patch(rect_t)
    
    if s_zoom_focus:
        rect_s = Rectangle((s_zoom_focus.x - pad, s_zoom_focus.y - pad), s_zoom_focus.w + 2*pad, s_zoom_focus.h + 2*pad, 
                           fill=False, ec='black', ls='--', lw=1, zorder=6)
        ax.add_patch(rect_s)
        
    add_fmm_legend(ax)

def plot_zoom_fmm(ax, tree, boundary, t_leaf, t_m2l_node, s_m2l_nodes, focus_node):
    plot_spatial_tree(ax, tree, boundary, show_ids=False)
    plot_fmm_interactions(ax, t_leaf, t_m2l_node, s_m2l_nodes)
    
    root_w = tree.nodes[0].w
    label = 'z_0' if focus_node == t_m2l_node else 'z'
    label_z_points(ax, focus_node, label, root_w)
    
    pad = focus_node.w * 0.15
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
    boundary_pts = generate_boundary(n_points=NUM_ELEMENTS)
    
    # ---------------------------------------------------------
    # C++ Equivalent: Dynamic Minimum Bounding Box with 2% Padding
    # ---------------------------------------------------------
    min_x, min_y = np.min(boundary_pts[:, 1]), np.min(boundary_pts[:, 2])
    max_x, max_y = np.max(boundary_pts[:, 1]), np.max(boundary_pts[:, 2])
    
    max_len = max(max_x - min_x, max_y - min_y)
    max_len *= 1.02 # 2% padding
    
    center_x, center_y = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
    
    root_x = center_x - max_len / 2.0
    root_y = center_y - max_len / 2.0
    root_w = max_len
    root_h = max_len
    # ---------------------------------------------------------
    
    qtree = QuadTree(capacity=MAX_ELEMENTS_PER_CELL, max_depth=TREE_MAX_DEPTH) 
    root = qtree.build(boundary_pts, root_x, root_y, root_w, root_h)
    
    t_leaf, t_m2l_node, s_m2l_nodes = setup_fmm_nodes(
        qtree, TARGET_ID, SOURCE_IDS, M2L_DEPTH
    )
    
    leaf_zoom = get_leaf_by_id(root, SOURCE_ZOOM_ID)
    s_zoom_focus = get_ancestor_at_depth(leaf_zoom, M2L_DEPTH) if leaf_zoom else None

    # Diagram Generation
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    plot_spatial_tree(ax1, qtree, boundary_pts, show_reference=True, show_ids=True)
    fig1.savefig("diagram_1_spatial.svg", format="svg", bbox_inches='tight')
    fig1.savefig("diagram_1_spatial.pdf", format="pdf", bbox_inches='tight')

    fig2, ax2 = plt.subplots(figsize=(16, 8)) 
    plot_logical_tree(ax2, root, qtree)
    fig2.savefig("diagram_2_logical.svg", format="svg", bbox_inches='tight')
    fig2.savefig("diagram_2_logical.pdf", format="pdf", bbox_inches='tight')

    fig3, ax3 = plt.subplots(figsize=(10, 8))
    plot_main_fmm(ax3, qtree, boundary_pts, t_leaf, t_m2l_node, s_m2l_nodes, s_zoom_focus)
    fig3.savefig("diagram_3_fmm_main.svg", format="svg", bbox_inches='tight')
    fig3.savefig("diagram_3_fmm_main.pdf", format="pdf", bbox_inches='tight')
    
    fig4, ax4 = plt.subplots(figsize=(4, 4))
    if t_m2l_node:
        plot_zoom_fmm(ax4, qtree, boundary_pts, t_leaf, t_m2l_node, s_m2l_nodes, focus_node=t_m2l_node)
        fig4.savefig("diagram_4_fmm_target_zoom.svg", format="svg", bbox_inches='tight')
        fig4.savefig("diagram_4_fmm_target_zoom.pdf", format="pdf", bbox_inches='tight')
    
    fig5, ax5 = plt.subplots(figsize=(4, 4))
    if s_zoom_focus:
        plot_zoom_fmm(ax5, qtree, boundary_pts, t_leaf, t_m2l_node, s_m2l_nodes, focus_node=s_zoom_focus)
        fig5.savefig("diagram_5_fmm_source_zoom.svg", format="svg", bbox_inches='tight')
        fig5.savefig("diagram_5_fmm_source_zoom.pdf", format="pdf", bbox_inches='tight')

    print("Successfully generated and saved tight-bounded SVG and PDF files.")