"""3D Brillouin-zone, IBZ, and spin-flip figures."""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from matplotlib.transforms import Bbox
from matplotlib.colors import to_rgb
from matplotlib.text import Annotation, Text

from .plotting_common import (
    IBZ_FACE_COLORS,
    IBZ_UP_EXTRA_SECTOR_COLORS,
    _get_bz_path_style,
    _math_label,
    _print_saved_paths,
    _save_figure,
    generated_plain_path_segments,
    grouped_point_labels,
    label_aliases,
)
from .symmetry import (
    _axis_bz_exit,
    _classify_spin_down_ops,
    _classify_spinflip_op,
    _doubled_ibz_extra_flags,
    _hp1_four_sector_label_groups,
    _format_miller,
    _mirror_plane_bz_polygon,
    _perp_unit,
    _reduce_int_vector,
    _rotation_sense,
    _seekpath_label_to_internal,
)
from .geometry import (
    _bz_kz_plane_outline,
    _get_ibz_frame_edges,
    _mapped_spin_hulls,
    _points_on_kz_plane,
    _IN_PLANE_AXES,
    find_bz_exit,
)

class _Arrow3D(FancyArrowPatch):
    """FancyArrowPatch whose endpoints are projected from 3D coordinates."""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


class _ReciprocalAxisLabel(Annotation):
    """Keep upright text beyond a projected arrow tip with a gap in points."""

    def __init__(self, text, start, tip, gap=7.0, **kwargs):
        super().__init__(
            text, xy=(0, 0), xytext=(0, 0), xycoords='data',
            textcoords='offset points', ha='center', va='center',
            annotation_clip=False, **kwargs,
        )
        self._start3d = np.asarray(start, dtype=float)
        self._tip3d = np.asarray(tip, dtype=float)
        self._gap_points = gap
        self.set_clip_on(False)

    def update_positions(self, renderer):
        # Annotation updates on every draw, including camera motion and exports.
        points = np.vstack([self._start3d, self._tip3d])
        x, y, _ = proj3d.proj_transform(*points.T, self.axes.get_proj())
        projected = np.column_stack([x, y])
        self.xy = tuple(projected[1])
        screen = self.axes.transData.transform(projected)
        direction = screen[1] - screen[0]
        length = np.linalg.norm(direction)
        # Looking directly down an axis leaves no visible outward direction.
        direction = direction / length if length > 1e-6 else np.array([0., 1.])

        super().update_positions(renderer)
        bounds = Text.get_window_extent(self, renderer)
        # Put the entire text box beyond the tip, not just its anchor point.
        half_extent = 0.5 * (
            abs(direction[0]) * bounds.width + abs(direction[1]) * bounds.height
        )
        distance = half_extent / renderer.points_to_pixels(1.0) + self._gap_points
        self.set_position(tuple(direction * distance))
        super().update_positions(renderer)


def _draw_ibz_faces_by_sector(ax, hull_pts, hull_simplices, hull_labels,
                              main_color, extra_color,
                              main_alpha=0.20, extra_alpha=0.14):
    """Draw ordinary and project-added IBZ sectors with different fills."""
    points = np.array(hull_pts)
    labels = list(hull_labels) if hull_labels is not None else None
    extra_flags = (
        _doubled_ibz_extra_flags(labels) if labels is not None
        else [False] * len(points)
    )
    sector_groups = (
        _hp1_four_sector_label_groups(labels) if labels is not None else None
    )
    if sector_groups is not None:
        colors = (main_color,) + IBZ_UP_EXTRA_SECTOR_COLORS
        for indices, color in zip(sector_groups, colors):
            sector_points = points[indices]
            sector_hull = ConvexHull(sector_points)
            faces = [
                [sector_points[index] for index in simplex]
                for simplex in sector_hull.simplices
            ]
            ax.add_collection3d(Poly3DCollection(
                faces, facecolor=color, alpha=main_alpha, edgecolor='none'))
        return
    if len(extra_flags) != len(points) or not any(extra_flags):
        faces = [[points[s] for s in simplex] for simplex in hull_simplices]
        ax.add_collection3d(Poly3DCollection(
            faces, facecolor=main_color, alpha=main_alpha, edgecolor='none'))
        return

    full_faces = [[points[s] for s in simplex] for simplex in hull_simplices]
    ax.add_collection3d(Poly3DCollection(
        full_faces, facecolor=extra_color, alpha=extra_alpha, edgecolor='none'))

    original_points = np.array([
        point for point, is_extra in zip(points, extra_flags)
        if not is_extra
    ])
    if len(original_points) < 4:
        return
    try:
        original_hull = ConvexHull(original_points)
    except Exception:
        return

    original_faces = [
        [original_points[s] for s in simplex]
        for simplex in original_hull.simplices
    ]
    ax.add_collection3d(Poly3DCollection(
        original_faces, facecolor=main_color,
        alpha=main_alpha, edgecolor='none'))


def _get_view_direction(ax):
    """Get the unit vector pointing from the scene toward the camera."""
    elev = np.radians(ax.elev)
    azim = np.radians(ax.azim)
    # Camera direction in data coordinates
    view = np.array([
        np.cos(elev) * np.cos(azim),
        np.cos(elev) * np.sin(azim),
        np.sin(elev),
    ])
    return view


def _classify_bz_edges(bz_loops, view_dir):
    """Classify BZ edges as front or back for the given view direction.

    An edge is front-facing if any adjacent face points toward the viewer;
    otherwise, it is back-facing.
    """
    # Map each BZ edge to the normals of its adjacent faces.
    from collections import defaultdict
    edge_normals = defaultdict(list)

    for loop in bz_loops:
        pts = loop[:-1]  # remove closing duplicate
        center = np.mean(pts, axis=0)
        # Face normal (cross product of two edge vectors)
        if len(pts) >= 3:
            n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
            # Orient outward (away from origin)
            if np.dot(n, center) < 0:
                n = -n
            n = n / (np.linalg.norm(n) + 1e-15)
        else:
            n = np.array([0., 0., 0.])
        for i in range(len(pts)):
            p1 = tuple(np.round(pts[i], 8))
            p2 = tuple(np.round(pts[(i+1) % len(pts)], 8))
            edge_key = (min(p1, p2), max(p1, p2))
            edge_normals[edge_key].append(n)

    front_edges = []
    back_edges = []

    for edge_key, normals in edge_normals.items():
        is_front = any(np.dot(n, view_dir) > 1e-6 for n in normals)
        seg = np.array([list(edge_key[0]), list(edge_key[1])])
        if is_front:
            front_edges.append(seg)
        else:
            back_edges.append(seg)

    return front_edges, back_edges


def draw_bz_edges(ax, bz_loops, dashed_back=False):
    """Draw BZ edges, optionally dashing those behind the visible faces."""
    if dashed_back:
        view_dir = _get_view_direction(ax)
        front, back = _classify_bz_edges(bz_loops, view_dir)
        for seg in back:
            ax.plot(seg[:,0], seg[:,1], seg[:,2],
                    c='black', ls='--', lw=1.0, alpha=0.4)
        for seg in front:
            ax.plot(seg[:,0], seg[:,1], seg[:,2],
                    c='black', ls='-', lw=1.5, alpha=0.7)
    else:
        for loop in bz_loops:
            ax.plot(loop[:,0], loop[:,1], loop[:,2],
                    c='black', ls='-', lw=1.5, alpha=0.6)


def setup_3d_ax(title, bz_loops, b_matrix, bz_center,
                elev=14, azim=20, dashed_back=False):
    """Create 3D axes with the BZ frame and reciprocal-vector arrows."""
    b1, b2, b3 = b_matrix[0], b_matrix[1], b_matrix[2]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=elev, azim=azim)
    draw_bz_edges(ax, bz_loops, dashed_back=dashed_back)
    vec_labels = [r'$\mathbf{b}_1$', r'$\mathbf{b}_2$', r'$\mathbf{b}_3$']
    for i, vec in enumerate([b1, b2, b3]):
        t_exit = find_bz_exit(vec, b_matrix)
        t_exit = min(t_exit, 1.0)
        exit_pt = vec * t_exit
        # Dotted line inside BZ
        ax.plot([0,exit_pt[0]], [0,exit_pt[1]], [0,exit_pt[2]],
                color='black', ls=':', lw=1.5, alpha=0.6, zorder=100)
        # Arrow outside BZ: FancyArrowPatch projected into 3D
        outside = vec - exit_pt
        arrow_frac = 0.45
        tip = exit_pt + outside * arrow_frac
        arrow = _Arrow3D(
            [exit_pt[0], tip[0]], [exit_pt[1], tip[1]], [exit_pt[2], tip[2]],
            arrowstyle='->', mutation_scale=28,
            color='black', lw=2.5, shrinkA=0, shrinkB=0, zorder=100,
        )
        ax.add_artist(arrow)
        ax.add_artist(_ReciprocalAxisLabel(
            vec_labels[i], exit_pt, tip,
            color='black', fontsize=24, fontweight='bold', zorder=101,
        ))
    # Use independent axis ranges so short BZ dimensions retain their scale.
    all_pts = np.vstack([np.array(loop) for loop in bz_loops])
    ranges = np.ptp(all_pts, axis=0)  # [dx, dy, dz]
    pad = 0.25  # fractional padding around BZ
    for i, (set_lim, r) in enumerate(zip(
            [ax.set_xlim, ax.set_ylim, ax.set_zlim], ranges)):
        half = r / 2 * (1 + pad)
        set_lim(bz_center[i] - half, bz_center[i] + half)
    ax.set_box_aspect(ranges / ranges.max())
    ax.set_axis_off()
    ax.set_title(title, fontsize=20)
    return fig, ax


def attach_camera_angle_display(fig, ax):
    """Show TOML-ready camera angles on an interactive 3D figure."""
    def _view_text():
        return (
            f"view_elev = {ax.elev:.2f}\n"
            f"view_azim = {ax.azim:.2f}"
        )

    label = ax.text2D(
        0.02,
        0.02,
        _view_text(),
        transform=ax.transAxes,
        fontsize=10,
        family="monospace",
        va="bottom",
        ha="left",
        zorder=1000,
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.85},
    )

    def _update_view_text(event):
        if event.canvas is fig.canvas:
            label.set_text(_view_text())
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_release_event", _update_view_text)
    return label


def plot_ibz(ax, kpoints_cart, kpath, hull, centroid_cart,
             hull_pts=None, lattice_type=None, hull_labels=None):
    """Draw Figure 1's IBZ, path, high-symmetry points, and centroid."""
    points_list = list(kpoints_cart.values())
    # Draw IBZ faces (skipped only when no hull could be built)
    if hull is not None:
        face_points = np.array(hull_pts) if hull_pts is not None else np.array(points_list)
        _draw_ibz_faces_by_sector(
            ax, face_points, hull.simplices, hull_labels,
            IBZ_FACE_COLORS["up_main"], IBZ_FACE_COLORS["up_extra"])
        for p1, p2 in _get_ibz_frame_edges(face_points, hull.simplices, hull_labels):
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    c='darkred', lw=1.8, alpha=0.85, zorder=10)
    for k1, k2 in kpath:
        if k1 in kpoints_cart and k2 in kpoints_cart:
            p1, p2 = kpoints_cart[k1], kpoints_cart[k2]
            style = _get_bz_path_style(lattice_type, k1, k2)
            ax.plot([p1[0],p2[0]], [p1[1],p2[1]], [p1[2],p2[2]],
                    c=style["color"], ls=style["ls"], lw=style["lw"],
                    alpha=style["alpha"], zorder=50)
    ibz_center = np.mean(points_list, axis=0)
    ibz_span = np.max(np.ptp(np.array(points_list), axis=0))
    label_distance = ibz_span * 0.1
    path_labels = {label for segment in kpath for label in segment}
    for label, coords in grouped_point_labels(kpoints_cart, path_labels):
        ax.scatter(coords[0], coords[1], coords[2],
                   c='red', s=80, zorder=110, edgecolors='darkred', linewidths=0.5)
        direction = coords - ibz_center
        norm_dir = np.linalg.norm(direction)
        label_displacement = (
            direction / norm_dir * label_distance
            if norm_dir > 1e-8 else np.array([0, 0, label_distance])
        )
        ax.text(coords[0] + label_displacement[0],
                coords[1] + label_displacement[1],
                coords[2] + label_displacement[2],
                _math_label(label),
                fontsize=22, color='black',
                zorder=111, ha='center', va='center')
    if hull is not None:
        ax.scatter(*centroid_cart, c='gold', marker='*', s=400,
                   edgecolors='k', zorder=112, label="Vol. Centroid")
        ax.legend(loc='upper right')


def _screen_xy(ax, pt3):
    """Project a 3D point to the axes' 2D projected (screen) coordinates."""
    x, y, _ = proj3d.proj_transform(pt3[0], pt3[1], pt3[2], ax.get_proj())
    return np.array([x, y])


def _best_operation_label_position(ax, positions, avoid_pts):
    """Return the operation-label position farthest from the projected avoid points."""
    positions = np.asarray(positions, dtype=float)
    if avoid_pts is None or len(avoid_pts) == 0:
        return positions[0]
    avoid_screen = np.array([_screen_xy(ax, p) for p in avoid_pts])
    best, best_score = positions[0], -np.inf
    for position in positions:
        d = float(np.min(
            np.linalg.norm(avoid_screen - _screen_xy(ax, position), axis=1)
        ))
        if d > best_score:
            best_score, best = d, position
    return best


def _is_operation_colour(colour):
    """True for the green rotation-axis colour used by the operation glyph."""
    try:
        rgb = np.asarray(to_rgb(colour), dtype=float)
    except (ValueError, TypeError):
        return False
    target = np.asarray(to_rgb(
        os.environ.get('ALTERSEEK_OP_AXIS_COLOR', '#00c853')
    ), dtype=float)
    return bool(np.allclose(rgb, target, atol=0.02))


def _drawn_points_for_label_layout(ax, samples_per_segment=16):
    """Return screen points representing content that saved labels should not cover."""
    drawn_points = []
    drawn_part_ids = []
    part_id = 0

    def to_display(point3d):
        x, y = _screen_xy(ax, point3d)
        return ax.transData.transform((x, y))

    def sample_line_segment(start, end):
        nonlocal part_id
        start, end = np.asarray(start, float), np.asarray(end, float)
        part_id += 1
        for fraction in np.linspace(0.0, 1.0, samples_per_segment):
            drawn_points.append(to_display(start + (end - start) * fraction))
            drawn_part_ids.append(part_id)

    curved_glyphs = []
    for line in ax.lines:
        try:
            xs, ys, zs = line.get_data_3d()
        except AttributeError:
            continue
        verts = np.column_stack([xs, ys, zs])
        if _is_operation_colour(line.get_color()) and len(verts) >= 5:
            screen = np.array([to_display(vertex) for vertex in verts])
            span = np.linalg.norm(screen[-1] - screen[0])
            middle = screen[len(screen) // 2]
            bulge = np.linalg.norm(middle - 0.5 * (screen[0] + screen[-1]))
            # The rotation arc bulges away from its chord; the straight axis line does not, and treating it as a filled shape would cover most of the figure.
            if span > 0 and bulge > 0.25 * span:
                curved_glyphs.append(screen)
        for first, second in zip(verts[:-1], verts[1:]):
            sample_line_segment(first, second)
    for child in ax.get_children():
        if not isinstance(child, FancyArrowPatch):
            continue
        verts3d = getattr(child, '_verts3d', None)
        if verts3d is None:
            continue
        xs, ys, zs = verts3d
        sample_line_segment(
            (xs[0], ys[0], zs[0]),
            (xs[-1], ys[-1], zs[-1]),
        )
    for collection in ax.collections:
        offsets = getattr(collection, '_offsets3d', None)
        if offsets is None:
            continue
        xs, ys, zs = offsets
        for point in np.column_stack([xs, ys, zs]):
            part_id += 1
            drawn_points.append(to_display(point))
            drawn_part_ids.append(part_id)
    # Treat the rotation arc's screen-space bounding box as occupied so a label is not placed inside the arc.
    for screen in curved_glyphs:
        low, high = screen.min(axis=0), screen.max(axis=0)
        part_id += 1
        for x in np.linspace(low[0], high[0], 8):
            for y in np.linspace(low[1], high[1], 8):
                drawn_points.append(np.array([x, y]))
                drawn_part_ids.append(part_id)
    if not drawn_points:
        return np.empty((0, 2)), np.empty(0, dtype=int)
    return (
        np.asarray(drawn_points, dtype=float),
        np.asarray(drawn_part_ids, dtype=int),
    )


def _relayout_labels_for_save(fig, ax, max_shift=2.0, rounds=2):
    """Move labels away from overlaps in a fixed saved view.

    Interactive labels retain their 3D positions because their screen positions
    change as the camera moves. Reciprocal-axis labels keep their draw-time
    arrow-tip spacing and act as fixed obstacles for the movable labels.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    placed = []
    fixed_boxes = []
    for text in list(ax.texts):
        if not text.get_visible() or not text.get_text().strip():
            continue
        if isinstance(text, _ReciprocalAxisLabel):
            # Keep draw-time arrow clearance; other labels must avoid these.
            fixed_boxes.append(text.get_window_extent(renderer))
            continue
        position_3d = getattr(text, 'get_position_3d', None)
        if position_3d is not None:
            x, y = _screen_xy(ax, position_3d())
            original_position = ax.transData.transform((x, y))
        else:
            original_position = text.get_transform().transform(text.get_position())
        # Replace each 3D label with a 2D copy so it can be repositioned in screen space.
        clone = fig.text(
            *fig.transFigure.inverted().transform(original_position), text.get_text(),
            color=text.get_color(), fontsize=text.get_fontsize(),
            fontweight=text.get_fontweight(), ha=text.get_ha(),
            va=text.get_va(), zorder=200,
        )
        text.set_visible(False)
        placed.append((clone, np.asarray(original_position, dtype=float)))
    if not placed:
        return

    avoid_points, drawn_part_ids = _drawn_points_for_label_layout(ax)
    boxes = [clone.get_window_extent(renderer) for clone, _ in placed]
    step = float(np.median([box.height for box in boxes])) * 0.75
    if not np.isfinite(step) or step <= 0:
        return

    def box_for(index, offset):
        clone, original_position = placed[index]
        clone.set_position(
            fig.transFigure.inverted().transform(original_position + offset)
        )
        return clone.get_window_extent(renderer)

    def penalty_for(index, box, offset):
        penalty = 0.0
        for other, other_box in enumerate(boxes):
            if other == index:
                continue
            overlap = Bbox.intersection(box, other_box)
            if overlap is not None:
                penalty += overlap.width * overlap.height
        for fixed_box in fixed_boxes:
            overlap = Bbox.intersection(box, fixed_box)
            if overlap is not None:
                penalty += overlap.width * overlap.height
        if len(avoid_points):
            hits = (
                (avoid_points[:, 0] > box.x0) & (avoid_points[:, 0] < box.x1)
                & (avoid_points[:, 1] > box.y0) & (avoid_points[:, 1] < box.y1)
            )
            crossings = len(np.unique(drawn_part_ids[hits]))
            penalty += 1.2 * step * step * float(crossings)
        # Penalize larger shifts quadratically to keep each label near its point.
        shift = float(np.linalg.norm(offset)) / step
        penalty += 0.9 * step * step * shift * shift
        # Slightly prefer upward shifts, where point labels are conventionally placed.
        return penalty - 0.25 * step * float(offset[1])

    directions = np.array([
        [np.cos(angle), np.sin(angle)]
        for angle in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    ])
    candidates = [np.zeros(2)] + [
        direction * radius * step
        for radius in (0.5, 1.0, 1.5, 2.0)
        for direction in directions
        if radius <= max_shift
    ]
    chosen = [np.zeros(2)] * len(placed)
    for _round in range(rounds):
        for index in range(len(placed)):
            best_offset, best_penalty = chosen[index], None
            for offset in candidates:
                penalty = penalty_for(index, box_for(index, offset), offset)
                if best_penalty is None or penalty < best_penalty - 1e-9:
                    best_penalty, best_offset = penalty, offset
            chosen[index] = best_offset
            boxes[index] = box_for(index, best_offset)


def _mirror_label_positions(poly, bz_radius):
    """Return candidate label positions just beyond the mirror-plane polygon boundary."""
    poly = np.asarray(poly, dtype=float)
    c0 = poly.mean(axis=0)
    margin = bz_radius * 0.18
    lift = np.array([0.0, 0.0, 1.0]) * bz_radius * 0.05
    n = len(poly)
    cands = []
    for j in range(n):
        for p in (poly[j], 0.5 * (poly[j] + poly[(j + 1) % n])):
            o = p - c0
            no = np.linalg.norm(o)
            o = o / no if no > 1e-9 else np.array([0.0, 0.0, 1.0])
            cands.append(p + o * margin + lift)
    return np.array(cands)


def _draw_op_visual(ax, R_cart, bz_loops, bz_radius, b_matrix,
                    avoid_pts=None, elev=None, azim=None):
    """Draw the selected spin-flip operation in the 3D BZ.

    A mirror is shown as a shaded plane. A proper rotation or rotoinversion is
    shown as a colored axis and curved arrow. Identity and inversion are not drawn.

    Mirror labels use the plane normal in the reciprocal basis. The view angles
    orient the curved arrow relative to the camera.
    """
    op = _classify_spinflip_op(R_cart)
    op_type = op['type']

    if op_type == 'mirror' and op['axis'] is not None:
        poly = _mirror_plane_bz_polygon(op['axis'], bz_loops)
        if poly is not None and len(poly) >= 3:
            poly = np.asarray(poly, dtype=float)
            ax.add_collection3d(Poly3DCollection(
                [poly],
                alpha=0.30,
                facecolor='#606060',
                edgecolor='#303030',
                linewidth=0.8,
                zorder=5,
            ))

            # Label the mirror-plane normal in reduced reciprocal-basis indices, using grey to match the plane and screen projection so it does not rotate or stack in saved views.
            MIRROR_COLOR = os.environ.get('ALTERSEEK_OP_MIRROR_COLOR', '#606060')
            n = np.asarray(op['axis'], dtype=float)
            idx = _reduce_int_vector(n @ np.linalg.inv(b_matrix))
            m_label = _format_miller('m', idx)
            # Place the label at the boundary position farthest from existing screen-space points so it does not land on a BZ-boundary point such as K.
            positions = _mirror_label_positions(poly, bz_radius)
            label_pt = _best_operation_label_position(ax, positions, avoid_pts)
            mlx, mly, _ = proj3d.proj_transform(*label_pt, ax.get_proj())
            m_artist = ax.text2D(
                mlx, mly, m_label, transform=ax.transData,
                fontsize=28, fontweight='bold', color=MIRROR_COLOR, zorder=120,
                ha='center', va='center',
            )

            def _update_mirror_label():
                lx, ly, _ = proj3d.proj_transform(*label_pt, ax.get_proj())
                m_artist.set_position((lx, ly))

            ax._alterseek_operation_visual_updater = _update_mirror_label

    elif op_type in ('rotation', 'rotoreflection') and op['axis'] is not None:
        axis = np.array(op['axis'], dtype=float)
        # Orient tip toward upper hemisphere for consistent appearance
        if axis[2] < -1e-6 or (abs(axis[2]) < 1e-6 and axis[1] < -1e-6):
            axis = -axis

        # Use a distinct rotation-axis color against the red/navy IBZ and black BZ frame, with ALTERSEEK_OP_AXIS_COLOR available as an override.
        AXIS_COLOR = os.environ.get('ALTERSEEK_OP_AXIS_COLOR', '#00c853')
        OP_ZORDER = 200

        # Extend the axis beyond its positive and negative BZ exits so it remains visibly longer than the BZ even for flat hexagonal cells with c* much smaller than a*.
        bz_exit     = _axis_bz_exit( axis, bz_loops)
        bz_exit_neg = _axis_bz_exit(-axis, bz_loops)
        extension   = max(bz_radius * 0.35, bz_exit * 0.18)
        tip  =  axis * (bz_exit     + extension)
        base = -axis * (bz_exit_neg + extension)

        # Draw the lower half thin and dashed so the axis visibly passes through Gamma without overpowering the figure.
        ax.plot(
            [base[0], 0.0], [base[1], 0.0], [base[2], 0.0],
            color=AXIS_COLOR, lw=1.6, ls='--', alpha=0.55,
            zorder=OP_ZORDER,
        )
        # Draw a solid arrowhead at the positive end of the axis.
        head_len = bz_radius * 0.18
        shaft_pt = tip - axis * head_len
        # Upper shaft stops at the arrow base; the final segment is the arrow.
        ax.plot(
            [0.0, shaft_pt[0]], [0.0, shaft_pt[1]], [0.0, shaft_pt[2]],
            color=AXIS_COLOR, lw=2.8, ls='-', alpha=0.95,
            zorder=OP_ZORDER,
        )
        axis_arrow = ax.annotate(
            '', xy=_screen_xy(ax, tip), xytext=_screen_xy(ax, shaft_pt),
            xycoords=ax.transData, textcoords=ax.transData,
            arrowprops=dict(
                arrowstyle='-|>', mutation_scale=26,
                color=AXIS_COLOR, lw=2.8, shrinkA=0, shrinkB=0,
            ),
            zorder=OP_ZORDER + 1, annotation_clip=False,
        )
        axis_arrow.set_clip_on(False)
        if axis_arrow.arrow_patch is not None:
            axis_arrow.arrow_patch.set_clip_on(False)

        # Draw the rotation arrow as a 3D ring perpendicular to the axis so it rotates with the cell, with the 60-degree gap behind the starting view and the arrowhead in front.
        order = op['order'] or 2
        # Use International/Bilbao labels rather than Schoenflies notation: a proper Cn is shown as n, while S3, S4, and S6 become 6-bar, 4-bar, and 3-bar.
        # S1 and S2 are the separately handled mirror and inversion cases and do not reach this branch.
        if op_type == 'rotation':
            display_order = order
            improper = False
        else:
            display_order = {3: 6, 4: 4, 6: 3}.get(order, order)
            improper = True
        # Show n+ or n- by measuring rotation sense about the drawn upper-hemisphere axis; order two needs no sign.
        sense = _rotation_sense(R_cart, axis)
        sgn = '' if (order < 3 or sense == 0) else ('+' if sense > 0 else '-')
        # The arc's increasing angle is counterclockwise about the positive axis in the right-handed basis, so reverse it for an n- operation.
        dir_sign = -1 if sense < 0 else 1
        idx = _reduce_int_vector(np.asarray(axis) @ np.linalg.inv(np.asarray(b_matrix)))
        axis_sub = "".join(rf"\bar{{{abs(i)}}}" if i < 0 else f"{i}" for i in idx)
        digit = rf"\bar{{{display_order}}}" if improper else f"{display_order}"
        order_str = (rf"$\mathbf{{{digit}^{{{sgn}}}_{{{axis_sub}}}}}$" if sgn
                     else rf"$\mathbf{{{digit}_{{{axis_sub}}}}}$")
        u = _perp_unit(axis)
        v = np.cross(axis, u)
        v = v / np.linalg.norm(v)

        arc_center = tip - axis * (bz_radius * 0.05)
        r_arc = bz_radius * 0.20
        span  = np.radians(300.0)

        # Angle of the ring point nearest the camera (the front) for this view.
        elev = ax.elev if elev is None else elev
        azim = ax.azim if azim is None else azim
        el0, az0 = np.radians(elev), np.radians(azim)
        view0 = np.array([np.cos(el0) * np.cos(az0),
                          np.cos(el0) * np.sin(az0), np.sin(el0)])
        cu, cv = float(view0 @ u), float(view0 @ v)
        phi_front = np.arctan2(cv, cu) if (abs(cu) + abs(cv)) > 1e-9 else 0.0

        # Center the 300-degree arc on the front so its 60-degree gap lies behind, and match the sweep direction to the rotation sense.
        theta = np.linspace(phi_front - dir_sign * span / 2,
                             phi_front + dir_sign * span / 2, 121)
        arc_pts = (arc_center[None, :]
                   + r_arc * (np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v))
        arc_head_start = -10
        arc_line = ax.plot(arc_pts[:arc_head_start, 0],
                           arc_pts[:arc_head_start, 1],
                           arc_pts[:arc_head_start, 2],
                           color=AXIS_COLOR, lw=3.0, alpha=0.95,
                           zorder=OP_ZORDER)[0]

        # Draw only the final arc segment as an arrow and stop the plain arc at its base so the stroke does not pass through the arrowhead.
        arc_arrow = ax.annotate(
            '', xy=_screen_xy(ax, arc_pts[-1]),
            xytext=_screen_xy(ax, arc_pts[arc_head_start]),
            xycoords=ax.transData, textcoords=ax.transData,
            arrowprops=dict(
                arrowstyle='->', mutation_scale=22,
                color=AXIS_COLOR, lw=3.0, shrinkA=0, shrinkB=0,
            ),
            zorder=OP_ZORDER + 1, annotation_clip=False,
        )
        arc_arrow.set_clip_on(False)
        if arc_arrow.arrow_patch is not None:
            arc_arrow.arrow_patch.set_clip_on(False)

        # Place vertical-axis labels directly beyond the tip, while lifting in-plane or tilted-axis labels modestly in +z so they clear b1/b2 without floating far above the figure.
        if abs(axis[2]) > 0.9:
            # Scale the label gap by the axis-direction BZ extent rather than the wider in-plane radius so it stays near the arrowhead on flat cells, with a small radius floor for clearance.
            label_pt = tip + axis * max(bz_exit * 0.30, bz_radius * 0.05)
        else:
            label_pt = tip + axis * bz_radius * 0.10 + np.array([0.0, 0.0, 1.0]) * bz_radius * 0.16
        label_x, label_y, _ = proj3d.proj_transform(*label_pt, ax.get_proj())
        label_artist = ax.text2D(
            label_x, label_y, order_str, transform=ax.transData,
            fontsize=28, fontweight='bold', color=AXIS_COLOR,
            zorder=OP_ZORDER + 2,
            ha='center', va='center',
        )

        def _update_camera_facing_arc():
            el_now, az_now = np.radians(ax.elev), np.radians(ax.azim)
            view_now = np.array([
                np.cos(el_now) * np.cos(az_now),
                np.cos(el_now) * np.sin(az_now),
                np.sin(el_now),
            ])
            cu_now, cv_now = float(view_now @ u), float(view_now @ v)
            phi_now = (
                np.arctan2(cv_now, cu_now)
                if (abs(cu_now) + abs(cv_now)) > 1e-9 else 0.0
            )
            theta_now = np.linspace(phi_now - dir_sign * span / 2,
                                     phi_now + dir_sign * span / 2, 121)
            arc_pts_now = (
                arc_center[None, :]
                + r_arc * (
                    np.cos(theta_now)[:, None] * u
                    + np.sin(theta_now)[:, None] * v
                )
            )
            arc_line.set_data_3d(
                arc_pts_now[:arc_head_start, 0],
                arc_pts_now[:arc_head_start, 1],
                arc_pts_now[:arc_head_start, 2],
            )
            axis_arrow.xy = _screen_xy(ax, tip)
            axis_arrow.set_position(_screen_xy(ax, shaft_pt))
            arc_arrow.xy = _screen_xy(ax, arc_pts_now[-1])
            arc_arrow.set_position(_screen_xy(ax, arc_pts_now[arc_head_start]))
            label_x_now, label_y_now, _ = proj3d.proj_transform(
                *label_pt, ax.get_proj()
            )
            label_artist.set_position((label_x_now, label_y_now))

        ax._alterseek_operation_visual_updater = _update_camera_facing_arc


def _connect_operation_visual_view_follow(fig, ax):
    """Update the operation visual when the interactive camera moves.

    Rotation arcs remain on the visible side, while projected arrows and labels
    remain aligned with their 3D positions.
    """
    updater = getattr(ax, '_alterseek_operation_visual_updater', None)
    if updater is None:
        return
    state = {'view': (ax.elev, ax.azim)}

    def _on_move(event):
        cur = (ax.elev, ax.azim)
        if cur != state['view']:
            state['view'] = cur
            updater()
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', _on_move)


def plot_spin_flip_figure(b_matrix, bz_loops, bz_center,
                          kpoints_data, ibz_kpoints_frac,
                          hull_pts, hull_simplices,
                          centroid_frac, R,
                          output_path, path_sequence, elev=14, azim=20,
                          show_plot=True, block=True, R_cart=None,
                          defer_show=False, save_pdf=False):
    """Plot the 3D spin-flip IBZs and generated path connections.

    Shows the 3D BZ with:
      - Salmon shading   : spin-up IBZ (original)
      - Blue shading     : spin-down IBZ (R-mapped)
      - Red solid lines  : spin-up segments of the generated path (non-primed / k)
      - Blue solid lines : spin-down segments of the generated path (primed / k')
      - Gold star / label: k  (IBZ centroid) with dashed lines to original high-sym points
      - Blue circle      : k' (spin-flip partner) with dashed lines to mapped high-sym points
      - Operation visual : mirror plane or rotation/rotoinversion axis and arrow

    path_sequence : list returned by KPathBuilder.insert_general_kpoints().
        Each entry is [kx, ky, kz, label] (fractional coords) or None (segment break).
        It controls the generated high-symmetry segments and k/k' connections.
    """
    if path_sequence is None or len(path_sequence) == 0:
        raise ValueError("Figure 2 requires a nonempty generated path_sequence")

    b1, b2, b3 = b_matrix[0], b_matrix[1], b_matrix[2]
    b_T = b_matrix.T
    b_T_inv = np.linalg.inv(b_T)   # maps Cartesian k -> fractional
    R_inv_T = np.linalg.inv(R).T
    if R_cart is None:
        R_cart = b_T @ R_inv_T @ b_T_inv
    else:
        R_cart = np.array(R_cart, dtype=float)

    # k and k' in Cartesian
    k_frac = np.array(centroid_frac[:3])
    k_cart = k_frac[0] * b1 + k_frac[1] * b2 + k_frac[2] * b3
    kp_cart = R_cart @ k_cart

    # Original IBZ high-sym points (from lattice_kpoints, not KPOINTS file)
    ibz_orig = {}
    for lbl, frac in ibz_kpoints_frac.items():
        if not lbl.startswith('_'):
            f = np.array(frac)
            ibz_orig[lbl] = f[0] * b1 + f[1] * b2 + f[2] * b3
    if not ibz_orig and kpoints_data:
        # If the IBZ label map is unavailable, derive its unprimed high-symmetry points from KPOINTS; k and k' are handled separately.
        for row in kpoints_data:
            if len(row) < 4:
                continue
            lbl = str(row[3])
            if lbl in ('k', "k'") or lbl.endswith("'"):
                continue
            if lbl not in ibz_orig:
                ibz_orig[lbl] = row[0] * b1 + row[1] * b2 + row[2] * b3

    # Apply R^{-T} to the original high-symmetry points to obtain the spin-down IBZ.
    # Keep only noncoincident mapped points for labels to avoid stacked labels such as Gamma and Gamma'.
    # Keep every mapped point for dashed lines so k' can still connect to a self-mapped point without a separate primed label.
    ibz_mapped       = {}
    ibz_mapped_lines = {}
    for lbl, frac in ibz_kpoints_frac.items():
        if lbl.startswith('_'):
            continue
        f = np.array(frac)
        pt0 = f[0] * b1 + f[1] * b2 + f[2] * b3
        pt = R_cart @ pt0
        ibz_mapped_lines[lbl + "'"] = pt
        coincident = any(
            np.linalg.norm(pt - orig_pt) < 1e-6
            for orig_pt in ibz_orig.values()
        )
        if not coincident:
            ibz_mapped[lbl + "'"] = pt

    # Map the Figure 1 hull vertices to the spin-down IBZ in Cartesian coordinates.
    mapped_hull_pts = None
    if hull_pts is not None:
        mapped_hull_pts = (R_cart @ np.array(hull_pts).T).T
    hull_labels = (
        list(ibz_kpoints_frac.keys())
        if hull_pts is not None and len(ibz_kpoints_frac) == len(hull_pts)
        else None
    )

    bz_radius = np.max(np.linalg.norm(np.vstack(bz_loops), axis=1))

    def _draw(ax):
        # Draw the operation visual first; the mirror plane stays below the IBZ, while rotation and rotoinversion axes use a high z-order so they remain visible.
        # Pass existing high-symmetry points and reciprocal-axis tips so the operation label can avoid them in screen space.
        avoid = list(ibz_orig.values()) + list(ibz_mapped.values())
        avoid += [b1 * 0.5, b2 * 0.5, b3 * 0.5]
        avoid_pts = np.array(avoid) if avoid else None
        _draw_op_visual(ax, R_cart, bz_loops, bz_radius, b_matrix,
                        avoid_pts=avoid_pts, elev=ax.elev, azim=ax.azim)

        # Spin-up IBZ: use the same curated HPKOT/project hull as Figure 1.
        up_pts, up_simplices = hull_pts, hull_simplices
        if up_pts is not None and up_simplices is not None:
            up_pts = np.array(up_pts)
            _draw_ibz_faces_by_sector(
                ax, up_pts, up_simplices, hull_labels,
                IBZ_FACE_COLORS["up_main"], IBZ_FACE_COLORS["up_extra"])
            for p1, p2 in _get_ibz_frame_edges(up_pts, up_simplices, hull_labels):
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                        c='darkred', lw=1.8, alpha=0.85, zorder=10)

        # Spin-down IBZ: the same Figure 1 hull mapped by the chosen spin-flip operation.
        down_pts, down_simplices = mapped_hull_pts, hull_simplices
        if down_pts is not None and down_simplices is not None:
            down_pts = np.array(down_pts)
            _draw_ibz_faces_by_sector(
                ax, down_pts, down_simplices, hull_labels,
                IBZ_FACE_COLORS["down_main"], IBZ_FACE_COLORS["down_extra"])
            for p1, p2 in _get_ibz_frame_edges(down_pts, down_simplices, hull_labels):
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                        c='navy', lw=1.8, alpha=0.85, zorder=10)

        # Use path_sequence for the connection order, but draw each label at its Figure 1 coordinate so Figures 1 and 2 use the same plotting basis.
        def _path_point(label):
            if label in ('k', "k'"):
                return None
            for alias in label_aliases(label):
                is_prime = alias.endswith("'")
                raw_base = alias.rstrip("'")
                base = _seekpath_label_to_internal(raw_base)
                if is_prime:
                    pt = ibz_mapped_lines.get(raw_base + "'")
                    if pt is None:
                        pt = ibz_mapped_lines.get(base + "'")
                else:
                    pt = ibz_orig.get(raw_base)
                    if pt is None:
                        pt = ibz_orig.get(base)
                if pt is not None:
                    return pt
            return None

        for A, B, spin_side in generated_plain_path_segments(path_sequence):
            pa_c = _path_point(A[3])
            pb_c = _path_point(B[3])
            if pa_c is None or pb_c is None:
                continue
            col = 'navy' if spin_side == 'down' else 'red'
            ax.plot([pa_c[0], pb_c[0]], [pa_c[1], pb_c[1]], [pa_c[2], pb_c[2]],
                    c=col, lw=4.0, alpha=0.9, zorder=50)

        # Use the center of both IBZs to place each label outward from the plotted points.
        all_ibz_points = np.array(list(ibz_orig.values()) + list(ibz_mapped.values()))
        combined_center = (
            np.mean(all_ibz_points, axis=0)
            if len(all_ibz_points) else np.zeros(3)
        )
        # Set the label-to-point distance from the original IBZ size so a distant mapped IBZ does not move labels too far from their points.
        original_ibz_points = (
            np.array(list(ibz_orig.values())) if ibz_orig else all_ibz_points
        )
        original_ibz_span = (
            max(np.max(np.ptp(original_ibz_points, axis=0)), 1e-8)
            if len(original_ibz_points) else 1.0
        )
        label_distance = original_ibz_span * 0.1
        sequence_labels = {
            alias
            for point in path_sequence
            if point is not None
            for alias in label_aliases(point[3])
        }

        def _label_pts(pts_dict, color, edgecolor):
            for lbl, hpt in grouped_point_labels(pts_dict, sequence_labels):
                ax.scatter(*hpt, c=color, s=60, zorder=110,
                           edgecolors=edgecolor, linewidths=0.5)
                direction = hpt - combined_center
                nd = np.linalg.norm(direction)
                label_displacement = (
                    direction / nd * label_distance
                    if nd > 1e-8 else np.array([0, 0, label_distance])
                )
                ax.text(*(hpt + label_displacement), _math_label(lbl), fontsize=22, color=edgecolor,
                        zorder=111, ha='center', va='center')

        _label_pts(ibz_orig,   color='salmon',          edgecolor='darkred')
        _label_pts(ibz_mapped, color='cornflowerblue',  edgecolor='navy')

        # Draw the dashed k-to-high-symmetry path connections.
        for hpt in ibz_orig.values():
            ax.plot([hpt[0], k_cart[0]], [hpt[1], k_cart[1]], [hpt[2], k_cart[2]],
                    c='deepskyblue', lw=2.0, ls='--', alpha=0.75, zorder=40)

        # Draw the dashed k'-to-mapped-high-symmetry path connections, including self-mapped points.
        for hpt in ibz_mapped_lines.values():
            ax.plot([hpt[0], kp_cart[0]], [hpt[1], kp_cart[1]], [hpt[2], kp_cart[2]],
                    c='deepskyblue', lw=2.0, ls='--', alpha=0.75, zorder=40)

        # k and k' markers
        ax.scatter(*k_cart, c='gold', s=300, marker='*',
                   edgecolors='k', linewidths=0.8, zorder=120, label=r'$k$')
        ax.scatter(*kp_cart, c='cornflowerblue', s=150, marker='o',
                   edgecolors='k', linewidths=0.8, zorder=120, label=r"$k'$")
        ax.legend(loc='upper right', fontsize=18)

    fig, ax = setup_3d_ax("Spin-flip path connections",
                          bz_loops, b_matrix, bz_center,
                          elev=elev, azim=azim, dashed_back=False)
    if show_plot:
        attach_camera_angle_display(fig, ax)
    _draw(ax)
    # Keep the operation visual aligned with the camera in the interactive window.
    if show_plot:
        _connect_operation_visual_view_follow(fig, ax)
    plt.tight_layout()

    display_fig = fig if show_plot and defer_show else None
    if display_fig is not None:
        def _save_after_show(fig=fig, ax=ax):
            save_figure = None
            try:
                save_figure, save_ax = setup_3d_ax(
                    "Spin-flip path connections",
                    bz_loops, b_matrix, bz_center,
                    elev=ax.elev, azim=ax.azim,
                    dashed_back=True,
                )
                _draw(save_ax)
                plt.tight_layout()
                _relayout_labels_for_save(save_figure, save_ax)
                saved_paths = _save_figure(
                    save_figure,
                    output_path,
                    extra_formats=("pdf",) if save_pdf else (),
                    dpi=300,
                    bbox_inches='tight',
                )
                view = (ax.elev, ax.azim)
                fig._alterseek_camera_angle = (saved_paths[0],) + view
                _print_saved_paths(saved_paths, view=view)
            finally:
                if save_figure is not None:
                    plt.close(save_figure)
                plt.close(fig)
        display_fig._alterseek_save_after_show = _save_after_show
        return display_fig

    if show_plot and not defer_show:
        plt.show(block=block)
        if block:
            # Blocking mode: window already closed, capture adjusted view
            elev, azim = ax.elev, ax.azim
            plt.close(fig)
        # In nonblocking mode, leave the window open beside the next figure and save the original elevation and azimuth rather than interactive adjustments.
    elif not show_plot:
        plt.close(fig)

    # Re-render with dashed back edges for the saved output.
    fig_save, ax_save = setup_3d_ax("Spin-flip path connections",
                                    bz_loops, b_matrix, bz_center,
                                    elev=elev, azim=azim, dashed_back=True)
    _draw(ax_save)
    plt.tight_layout()
    _relayout_labels_for_save(fig_save, ax_save)
    saved_paths = _save_figure(
        fig_save,
        output_path,
        extra_formats=("pdf",) if save_pdf else (),
        dpi=300,
        bbox_inches='tight',
    )
    plt.close(fig_save)
    _print_saved_paths(saved_paths, view=(elev, azim))
    return display_fig


def plot_general_path_figure(
    b_matrix, bz_loops, bz_center, kpoints_data, ibz_kpoints_frac,
    hull_pts, hull_simplices, centroid_frac, output_path, path_sequence,
    elev=14, azim=20, show_plot=True, block=True, defer_show=False,
    save_pdf=False,
):
    """Plot the red-only general-k path for a 3D non-altermagnet."""
    if path_sequence is None or len(path_sequence) == 0:
        raise ValueError(
            "General-path Figure 2 requires a nonempty generated path_sequence"
        )

    b1, b2, b3 = np.asarray(b_matrix, dtype=float)
    centroid_frac = np.asarray(centroid_frac[:3], dtype=float)
    centroid_cart = centroid_frac[0] * b1 + centroid_frac[1] * b2 + centroid_frac[2] * b3

    ibz_points = {}
    for label, frac in ibz_kpoints_frac.items():
        if str(label).startswith('_'):
            continue
        frac = np.asarray(frac, dtype=float)
        ibz_points[label] = frac[0] * b1 + frac[1] * b2 + frac[2] * b3
    if not ibz_points and kpoints_data:
        for row in kpoints_data:
            if len(row) < 4 or row[3] == 'k':
                continue
            label = str(row[3])
            if label not in ibz_points:
                ibz_points[label] = row[0] * b1 + row[1] * b2 + row[2] * b3

    hull_labels = (
        list(ibz_kpoints_frac.keys())
        if hull_pts is not None and len(ibz_kpoints_frac) == len(hull_pts)
        else None
    )

    def _path_point(label):
        if label == 'k':
            return None
        for alias in label_aliases(label):
            base = _seekpath_label_to_internal(alias)
            point = ibz_points.get(alias)
            if point is None:
                point = ibz_points.get(base)
            if point is not None:
                return point
        return None

    sequence_labels = {
        alias
        for point in path_sequence
        if point is not None and point[3] not in ('k', "k'")
        for alias in label_aliases(point[3])
    }

    def _draw(ax):
        if hull_pts is not None and hull_simplices is not None:
            points = np.asarray(hull_pts, dtype=float)
            _draw_ibz_faces_by_sector(
                ax, points, hull_simplices, hull_labels,
                IBZ_FACE_COLORS["up_main"], IBZ_FACE_COLORS["up_extra"],
            )
            for first, second in _get_ibz_frame_edges(
                points, hull_simplices, hull_labels
            ):
                ax.plot(
                    [first[0], second[0]], [first[1], second[1]],
                    [first[2], second[2]], c='darkred', lw=1.8,
                    alpha=0.85, zorder=10,
                )

        for first, second, spin_side in generated_plain_path_segments(
            path_sequence
        ):
            if spin_side != 'up':
                continue
            first_cart = _path_point(first[3])
            second_cart = _path_point(second[3])
            if first_cart is None or second_cart is None:
                continue
            ax.plot(
                [first_cart[0], second_cart[0]],
                [first_cart[1], second_cart[1]],
                [first_cart[2], second_cart[2]],
                c='red', lw=4.0, alpha=0.9, zorder=50,
            )

        if ibz_points:
            point_array = np.asarray(list(ibz_points.values()), dtype=float)
            center = np.mean(point_array, axis=0)
            span = max(float(np.max(np.ptp(point_array, axis=0))), 1e-8)
            label_distance = 0.1 * span
            for label, point in grouped_point_labels(
                ibz_points, sequence_labels
            ):
                ax.scatter(
                    *point, c='salmon', s=60, zorder=110,
                    edgecolors='darkred', linewidths=0.5,
                )
                direction = point - center
                norm = np.linalg.norm(direction)
                displacement = (
                    direction / norm * label_distance
                    if norm > 1e-8 else np.array([0, 0, label_distance])
                )
                ax.text(
                    *(point + displacement), _math_label(label),
                    fontsize=22, color='darkred', zorder=111,
                    ha='center', va='center',
                )

        for point in ibz_points.values():
            ax.plot(
                [point[0], centroid_cart[0]],
                [point[1], centroid_cart[1]],
                [point[2], centroid_cart[2]],
                c='deepskyblue', lw=2.0, ls='--', alpha=0.75, zorder=40,
            )
        ax.scatter(
            *centroid_cart, c='gold', s=300, marker='*', edgecolors='k',
            linewidths=0.8, zorder=120, label=r'$k$',
        )
        ax.legend(loc='upper right', fontsize=18)

    title = "General-k path connections"
    fig, ax = setup_3d_ax(
        title, bz_loops, b_matrix, bz_center,
        elev=elev, azim=azim, dashed_back=False,
    )
    if show_plot:
        attach_camera_angle_display(fig, ax)
    _draw(ax)
    plt.tight_layout()

    display_fig = fig if show_plot and defer_show else None
    if display_fig is not None:
        def _save_after_show(fig=fig, ax=ax):
            save_figure = None
            try:
                save_figure, save_ax = setup_3d_ax(
                    title, bz_loops, b_matrix, bz_center,
                    elev=ax.elev, azim=ax.azim, dashed_back=True,
                )
                _draw(save_ax)
                plt.tight_layout()
                _relayout_labels_for_save(save_figure, save_ax)
                saved_paths = _save_figure(
                    save_figure, output_path,
                    extra_formats=("pdf",) if save_pdf else (),
                    dpi=300, bbox_inches='tight',
                )
                view = (ax.elev, ax.azim)
                fig._alterseek_camera_angle = (saved_paths[0],) + view
                _print_saved_paths(saved_paths, view=view)
            finally:
                if save_figure is not None:
                    plt.close(save_figure)
                plt.close(fig)
        display_fig._alterseek_save_after_show = _save_after_show
        return display_fig

    if show_plot and not defer_show:
        plt.show(block=block)
        if block:
            elev, azim = ax.elev, ax.azim
            plt.close(fig)
    elif not show_plot:
        plt.close(fig)

    fig_save, ax_save = setup_3d_ax(
        title, bz_loops, b_matrix, bz_center,
        elev=elev, azim=azim, dashed_back=True,
    )
    _draw(ax_save)
    plt.tight_layout()
    _relayout_labels_for_save(fig_save, ax_save)
    saved_paths = _save_figure(
        fig_save, output_path,
        extra_formats=("pdf",) if save_pdf else (),
        dpi=300, bbox_inches='tight',
    )
    plt.close(fig_save)
    _print_saved_paths(saved_paths, view=(elev, azim))
    return display_fig


def plot_spin_bz_figure(b_matrix, bz_loops, bz_center,
                        unique_ops, centroid_cart,
                        hull_pts, hull_simplices,
                        R, output_path,
                        flip_ops_frac=None,
                        preserve_ops_frac=None,
                        elev=14, azim=20, show_plot=True,
                        defer_show=False, z0=0.0, cut_axis=2,
                        show_helper_plane=True,
                        hull_labels=None, save_pdf=False):
    """
    Show Figure 1's IBZ hull mapped by spin-preserving/spin-flipping operations.

    The outer BZ remains the structural BZ. Each colored copy is exactly the
    Figure 1 hull; reduced spin symmetry can therefore leave part of the
    structural BZ uncolored.
      - RED   (salmon)         : spin-up  regions
      - BLUE  (cornflowerblue) : spin-down regions
    """
    if hull_pts is None or hull_simplices is None or not len(unique_ops):
        print("[Note] Skipping spin-BZ figure (no hull or symmetry ops available).")
        return

    hull_pts = np.array(hull_pts)
    centroid_cart = np.array(centroid_cart)
    hull_simplices_arr = np.array(hull_simplices)

    mapped_spin_hulls = None
    if (
        preserve_ops_frac is not None and len(preserve_ops_frac)
        and flip_ops_frac is not None and len(flip_ops_frac)
    ):
        mapped_spin_hulls = _mapped_spin_hulls(
            b_matrix, hull_pts, hull_simplices_arr,
            preserve_ops_frac, flip_ops_frac
        )
        if mapped_spin_hulls is None:
            print("[Warning] Skipping spin-BZ figure: inconsistent spin-operation classes.")
            return
    else:
        # Without separate spin-preserving operation data, classify the available operations directly.
        spin_down_mask = _classify_spin_down_ops(
            b_matrix, unique_ops, flip_ops_frac
        )

    def _draw(ax):
        if show_helper_plane:
            draw_cut_plane_boundary(ax, bz_loops, z0=z0, axis=cut_axis)

        if mapped_spin_hulls is not None:
            cells_to_draw = mapped_spin_hulls
        else:
            cells_to_draw = [
                ((g @ hull_pts.T).T, hull_simplices_arr, bool(spin_down_mask[i]))
                for i, g in enumerate(unique_ops)
            ]
        for cell_pts, cell_simplices, is_down in cells_to_draw:
            alpha = 0.2
            main_key = "down_main" if is_down else "up_main"
            extra_key = "down_extra" if is_down else "up_extra"
            _draw_ibz_faces_by_sector(
                ax, cell_pts, cell_simplices, hull_labels,
                IBZ_FACE_COLORS[main_key], IBZ_FACE_COLORS[extra_key],
                main_alpha=alpha, extra_alpha=0.10)
        # Gold star at the original IBZ centroid
        ax.scatter(*centroid_cart, c='gold', s=350, marker='*',
                   edgecolors='k', linewidths=0.8, zorder=200,
                   label=r'$k$ (IBZ centroid)', depthshade=False)
        ax.legend(loc='upper right', fontsize=10)

    fig, ax = setup_3d_ax("Spin-up (red) / Spin-down (blue) BZ",
                          bz_loops, b_matrix, bz_center,
                          elev=elev, azim=azim, dashed_back=False)
    if show_plot:
        attach_camera_angle_display(fig, ax)
    _draw(ax)
    plt.tight_layout()

    display_fig = fig if show_plot and defer_show else None
    if display_fig is not None:
        def _save_after_show(fig=fig, ax=ax):
            save_figure = None
            try:
                save_figure, save_ax = setup_3d_ax(
                    "Spin-up (red) / Spin-down (blue) BZ",
                    bz_loops, b_matrix, bz_center,
                    elev=ax.elev, azim=ax.azim,
                    dashed_back=True,
                )
                _draw(save_ax)
                plt.tight_layout()
                saved_paths = _save_figure(
                    save_figure,
                    output_path,
                    extra_formats=("pdf",) if save_pdf else (),
                    dpi=300,
                    bbox_inches='tight',
                )
                view = (ax.elev, ax.azim)
                fig._alterseek_camera_angle = (saved_paths[0],) + view
                _print_saved_paths(saved_paths, view=view)
            finally:
                if save_figure is not None:
                    plt.close(save_figure)
                plt.close(fig)
        display_fig._alterseek_save_after_show = _save_after_show
        return display_fig

    if show_plot and not defer_show:
        plt.show()
        elev, azim = ax.elev, ax.azim
        plt.close(fig)
    elif not show_plot:
        plt.close(fig)

    # Re-render with dashed back edges for the saved output.
    fig, ax = setup_3d_ax("Spin-up (red) / Spin-down (blue) BZ",
                          bz_loops, b_matrix, bz_center,
                          elev=elev, azim=azim, dashed_back=True)
    _draw(ax)
    plt.tight_layout()
    saved_paths = _save_figure(
        fig,
        output_path,
        extra_formats=("pdf",) if save_pdf else (),
        dpi=300,
        bbox_inches='tight',
    )
    plt.close(fig)
    _print_saved_paths(saved_paths, view=(elev, azim))
    return display_fig


def draw_cut_plane_boundary(ax, bz_loops, z0=0.0, axis=2):
    """Draw the boundary where the k[axis] = z0 plane intersects the BZ."""
    outline = _bz_kz_plane_outline(bz_loops, z0=z0, axis=axis)
    if outline is not None:
        i0, i1 = _IN_PLANE_AXES[axis]
        outline3d = np.empty((len(outline), 3))
        outline3d[:, i0] = outline[:, 0]
        outline3d[:, i1] = outline[:, 1]
        outline3d[:, axis] = z0
        closed_outline = np.vstack([outline3d, outline3d[0]])
        ax.plot(closed_outline[:, 0], closed_outline[:, 1], closed_outline[:, 2],
                color='#3f5268', lw=3.0, ls='--', alpha=0.95, zorder=90)


def draw_cut_plane_axis_key(ax, outline, axis, color='#202020'):
    """Draw a small corner key naming the two Cartesian axes spanning the cut plane."""
    names = ('x', 'y', 'z')
    i0, i1 = _IN_PLANE_AXES[axis]
    lo = outline.min(axis=0)
    span = float(np.max(outline.max(axis=0) - lo))
    arm = 0.16 * span
    # Sit well below the section: the projected b arrows radiate from Gamma and their labels reach past the BZ outline on every side.
    x0 = lo[0] - 0.02 * span
    y0 = lo[1] - 0.42 * span
    for dx, dy, name, ha, va in ((arm, 0.0, names[i0], 'left', 'center'),
                                 (0.0, arm, names[i1], 'center', 'bottom')):
        ax.annotate('', xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2.0,
                                    mutation_scale=22, shrinkA=0, shrinkB=0),
                    annotation_clip=False)
        # fontweight does not reach mathtext, so the bold comes from \mathbf, matching the b-arrow and high-symmetry labels.
        ax.text(x0 + dx * 1.12, y0 + dy * 1.12, rf'$\mathbf{{k_{name}}}$',
                fontsize=20, ha=ha, va=va, color=color, clip_on=False)
    # Annotations do not extend the data limits, so include the key's extent with invisible points.
    ax.plot([x0, x0 + 1.45 * arm, x0], [y0, y0, y0 + 1.45 * arm], alpha=0.0)


def draw_projected_reciprocal_axes(ax, b_matrix, bz_loops, z0=0.0, axis=2):
    """Project b1, b2, b3 onto the k[axis]=z0 plane and draw the projected in-plane arrows."""
    i0, i1 = _IN_PLANE_AXES[axis]
    all_pts = np.vstack([np.array(loop, dtype=float) for loop in bz_loops])
    span_xy = np.ptp(all_pts[:, [i0, i1]], axis=0)
    span = max(float(np.max(span_xy)), 1e-8)
    target = 0.60 * span
    origin = np.array([0.0, 0.0])
    colors = ['#202020', '#202020', '#202020']
    labels = [r'$\mathbf{b}_1$', r'$\mathbf{b}_2$', r'$\mathbf{b}_3$']
    outline = _bz_kz_plane_outline(bz_loops, z0=z0, axis=axis)

    def _ray_outline_exit(unit_vec):
        if outline is None or len(outline) < 3:
            return None
        t_hits = []
        closed = np.vstack([outline, outline[0]])
        for p1, p2 in zip(closed[:-1], closed[1:]):
            edge = p2 - p1
            mat = np.column_stack([unit_vec, -edge])
            det = np.linalg.det(mat)
            if abs(det) < 1e-12:
                continue
            t, u = np.linalg.solve(mat, p1 - origin)
            if t > 1e-10 and -1e-10 <= u <= 1.0 + 1e-10:
                t_hits.append(float(t))
        return min(t_hits) if t_hits else None

    for vec, label, color in zip(np.array(b_matrix, dtype=float), labels, colors):
        projected = np.array([vec[i0], vec[i1]], dtype=float)
        length = float(np.linalg.norm(projected))
        if length < 1e-10:
            ax.scatter(origin[0], origin[1], s=36, c=color, zorder=220)
            ax.text(origin[0] + 0.025 * span, origin[1] + 0.025 * span,
                    label, fontsize=32, fontweight='bold',
                    ha='left', va='bottom', color=color, zorder=221,
                    clip_on=False)
            continue

        unit = projected / length
        exit_t = _ray_outline_exit(unit)
        exit_len = exit_t if exit_t is not None else target * 0.65
        exit_pt = origin + unit * min(exit_len, target * 0.86)
        end = origin + unit * max(target, exit_len * 1.16)
        ax.plot([origin[0], exit_pt[0]], [origin[1], exit_pt[1]],
                color=color, ls=':', lw=1.5, alpha=0.6, zorder=219,
                clip_on=False)
        ann = ax.annotate(
            '',
            xy=end, xytext=exit_pt,
            arrowprops=dict(arrowstyle='->', color=color, lw=2.2,
                            mutation_scale=28, shrinkA=0, shrinkB=0),
            zorder=220,
            annotation_clip=False,
        )
        ann.set_clip_on(False)
        if ann.arrow_patch is not None:
            ann.arrow_patch.set_clip_on(False)
        offset = projected / length * (0.04 * span)
        ax.text(end[0] + offset[0], end[1] + offset[1], label,
                fontsize=32, fontweight='bold', ha='center', va='center',
                color=color, zorder=221, clip_on=False)


# Expand the top-view limits so the BZ appears at the same size as in the 3D figures, which use wider margins.
_TOP_VIEW_VIEW_SCALE = 1.36


def plot_spin_bz_top_view_figure(b_matrix, bz_loops,
                                 unique_ops, centroid_cart,
                                 hull_pts, hull_simplices,
                                 R, output_path,
                                 flip_ops_frac=None,
                                 preserve_ops_frac=None,
                                 show_plot=True, defer_show=False,
                                 z0=0.0, cut_axis=2, show_title=False,
                                 show_projected_axes=True,
                                 show_legend=False,
                                 hull_labels=None, save_pdf=False):
    """Draw a darker top view of the Figure 3 spin-BZ section at k[cut_axis] = z0."""
    if hull_pts is None or hull_simplices is None or not len(unique_ops):
        print("[Note] Skipping spin-BZ top-view figure (no hull or symmetry ops available).")
        return

    hull_pts = np.array(hull_pts, dtype=float)
    hull_simplices_arr = np.array(hull_simplices, dtype=int)
    centroid_cart = np.array(centroid_cart, dtype=float)
    mapped_spin_hulls = None
    if (
        preserve_ops_frac is not None and len(preserve_ops_frac)
        and flip_ops_frac is not None and len(flip_ops_frac)
    ):
        mapped_spin_hulls = _mapped_spin_hulls(
            b_matrix, hull_pts, hull_simplices_arr,
            preserve_ops_frac, flip_ops_frac
        )
        if mapped_spin_hulls is None:
            print("[Warning] Skipping spin-BZ top-view figure: inconsistent spin-operation classes.")
            return
    else:
        spin_down_mask = _classify_spin_down_ops(
            b_matrix, unique_ops, flip_ops_frac)

    # Move the spin-up and spin-down section slightly off the requested plane, toward the centroid when possible, so a sector-boundary cut has unambiguous colors.
    z_span = np.ptp(np.vstack(bz_loops)[:, cut_axis])
    z_eps = max(float(z_span) * 1e-6, 1e-8)
    side = np.sign(centroid_cart[cut_axis] - z0)
    if abs(side) < 1e-12:
        side = 1.0
    section_z = z0 + side * z_eps

    fig, ax = plt.subplots(figsize=(9, 9))
    up_labeled = False
    down_labeled = False

    # Only project-added letter suffixes identify enlarged sectors; ordinary
    # HPKOT _2 labels alone do not.
    extra_flags = (
        _doubled_ibz_extra_flags(hull_labels) if hull_labels is not None
        else None
    )
    sector_groups = (
        _hp1_four_sector_label_groups(hull_labels)
        if hull_labels is not None else None
    )

    if mapped_spin_hulls is not None:
        cells_to_draw = mapped_spin_hulls
    else:
        cells_to_draw = [
            ((g @ hull_pts.T).T, hull_simplices_arr, bool(spin_down_mask[i]))
            for i, g in enumerate(unique_ops)
        ]
    for cell_pts, cell_simplices, is_down in cells_to_draw:
        poly = _points_on_kz_plane(cell_pts, cell_simplices, z0=section_z,
                                   axis=cut_axis)
        if poly is None:
            continue

        has_extra = extra_flags is not None and any(extra_flags)
        if is_down:
            color = '#1f4e9e'
            extra_color = IBZ_FACE_COLORS["down_extra"]
            label = 'spin-down' if not down_labeled else None
            down_labeled = True
        else:
            color = '#b22222'
            extra_color = IBZ_FACE_COLORS["up_extra"]
            label = 'spin-up' if not up_labeled else None
            up_labeled = True

        closed = np.vstack([poly, poly[0]])
        if sector_groups is not None and not is_down:
            sector_colors = (color,) + IBZ_UP_EXTRA_SECTOR_COLORS
            for group_index, (indices, sector_color) in enumerate(zip(
                sector_groups, sector_colors
            )):
                sector_pts = cell_pts[indices]
                try:
                    sector_hull = ConvexHull(sector_pts)
                    sector_poly = _points_on_kz_plane(
                        sector_pts, sector_hull.simplices,
                        z0=section_z, axis=cut_axis,
                    )
                except Exception:
                    sector_poly = None
                if sector_poly is not None:
                    ax.fill(
                        sector_poly[:, 0], sector_poly[:, 1],
                        facecolor=sector_color, alpha=0.68,
                        edgecolor='none',
                        label=label if group_index == 0 else None,
                    )
        elif has_extra:
            ax.fill(poly[:, 0], poly[:, 1], facecolor=extra_color, alpha=0.46,
                    edgecolor='none', label=label)
            original_pts = np.array([
                point for point, is_extra in zip(cell_pts, extra_flags)
                if not is_extra
            ])
            if len(original_pts) >= 4:
                try:
                    original_hull = ConvexHull(original_pts)
                    original_poly = _points_on_kz_plane(
                        original_pts, original_hull.simplices, z0=section_z,
                        axis=cut_axis)
                except Exception:
                    original_poly = None
                if original_poly is not None:
                    ax.fill(original_poly[:, 0], original_poly[:, 1],
                            facecolor=color, alpha=0.68, edgecolor='none')
        else:
            ax.fill(poly[:, 0], poly[:, 1], facecolor=color, alpha=0.68,
                    edgecolor='none', label=label)
        ax.plot(closed[:, 0], closed[:, 1], color=color, lw=0.9, alpha=0.95)

    outline = _bz_kz_plane_outline(bz_loops, z0=z0, axis=cut_axis)
    if outline is not None:
        closed = np.vstack([outline, outline[0]])
        ax.plot(closed[:, 0], closed[:, 1], color='black', lw=2.0, label='BZ boundary')
        if cut_axis != 2:
            # Add an in-plane coordinate key when the cut is not the usual constant-kz plane.
            draw_cut_plane_axis_key(ax, outline, cut_axis)

    if show_projected_axes and abs(z0) < 1e-8:
        # Draw reciprocal-axis arrows only for a cut through Gamma because their displayed origin is fixed at Gamma.
        draw_projected_reciprocal_axes(ax, b_matrix, bz_loops, z0=z0,
                                       axis=cut_axis)

    ax.set_aspect('equal', adjustable='box')
    # Expand the square plot limits so the BZ appears at the same size as in the 3D figures.
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    center = (0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi))
    half = 0.5 * max(x_hi - x_lo, y_hi - y_lo) * _TOP_VIEW_VIEW_SCALE
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    if show_title:
        cut_label = ('x', 'y', 'z')[cut_axis]
        cut_value = f'{float(z0):.6g}'
        ax.set_title('Spin-up / Spin-down BZ top view '
                     f'($k_{cut_label} = {cut_value}$)', fontsize=18)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    if show_legend:
        ax.legend(loc='upper right', fontsize=12)
    fig.tight_layout()

    display_fig = fig if show_plot and defer_show else None
    if display_fig is not None:
        def _save_after_show(fig=fig):
            try:
                saved_paths = _save_figure(
                    fig,
                    output_path,
                    extra_formats=("pdf",) if save_pdf else (),
                    dpi=300,
                    bbox_inches='tight',
                )
                _print_saved_paths(saved_paths)
            finally:
                plt.close(fig)
        display_fig._alterseek_save_after_show = _save_after_show
        return display_fig

    if show_plot and not defer_show:
        plt.show()
    saved_paths = _save_figure(
        fig,
        output_path,
        extra_formats=("pdf",) if save_pdf else (),
        dpi=300,
        bbox_inches='tight',
    )
    _print_saved_paths(saved_paths)
    plt.close(fig)
    return display_fig
