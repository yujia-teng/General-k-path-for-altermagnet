"""KPathBuilder: build the general-k path around the IBZ-centroid
general k-point, drive the interactive Step 0-5 workflow, and write the
VASP, QE, and ABINIT path files.
"""
import os
import warnings
from typing import List, Optional
import numpy as np

import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

from .version import __version__
from .find_sf_operations import SpinSymmetryError, run as find_sf_run
from .compute_centroid_3d import run as compute_centroid
from .symmetry import (no_altermagnetism_reason,
                       laue_group_from_spacegroup_number,
                       point_group_from_spacegroup_number,
                       is_valid_2d_spin_flip_cartesian,
                       is_trivial_2d_spin_flip_cartesian,
                       keeps_2d_plane_cartesian,
                       slab_plane_normal_cartesian,
                       describe_spinflip_op)
from .plotting_3d import (plot_general_path_figure,
                          plot_spin_flip_figure,
                          plot_spin_bz_figure,
                          plot_spin_bz_top_view_figure)
from .mode2d.plotting import (plot_2d_figures,
                              plot_2d_general_path_figure)
from .plotting_common import (
    alterseek_plot_style,
    combine_point_labels,
    label_aliases,
    POINT_COINCIDENCE_ATOL,
    prime_point_label,
    write_camera_angle_file,
)
from .run_log import read_answer
from .submitted_cell_analysis import (
    prepare_submitted_cell_analysis,
)
from .atomic_write import (
    _atomic_write_text,
    _atomic_open_text,
)
from .io import (
    write_bandplot_lattice_config,
    write_qe_bandplot_config,
    write_abinit_bandplot_config,
)


STEP0_VERBOSE_SUMMARY = False

INPUT_CONFIG_FILE = "alterseek_input.toml"

_INPUT_CONFIG_KEYS = {
    "structure", "spin_axis", "moments", "flip_option",
    "output_code", "view_elev", "view_azim", "save_pdf", "symprec",
    "vacuum_axis",
}

_VACUUM_AXIS_INDEX = {"a": 0, "b": 1, "c": 2}


def _fmt_coord(value):
    """Format a k-point coordinate, collapsing signed zero to plain zero.

    -0.0 and 0.0 are the same k-point, but they render differently and so make
    otherwise identical KPOINTS files compare unequal. Which one comes out
    depends on sign carried through the basis conversion -- that is, on which
    cell the path was built in -- not on the physics.
    """
    text = f"{value:.10f}"
    return text[1:] if text.startswith("-") and float(text) == 0.0 else text


# Fix the label width so the submitted and reference-cell fields align.
_CELL_LABEL_WIDTH = 30

# Keep diagnostics under OUTPUT_DIR; calculation inputs and plot configs stay in the working directory.
OUTPUT_DIR = "alterseek_output"



def _figure_basename(struct_file):
    """Name figures after the submitted structure."""
    if not struct_file:
        return None
    return os.path.splitext(os.path.basename(struct_file))[0]


@alterseek_plot_style
def _display_and_save_figures(display_figures):
    """Display deferred figures, save each independently, and always close it.

    Figure display and persistence are optional after KPOINTS has been written.
    One failed window or save must neither change that successful result nor
    prevent later figures from being saved and released.
    """
    print('Displaying generated figure(s)...')
    try:
        plt.show()
    except Exception as exc:
        print(
            "[Warning] Could not display/save generated figures: "
            f"{exc}"
        )

    for fig in display_figures:
        try:
            save_after_show = getattr(
                fig, '_alterseek_save_after_show', None
            )
            if save_after_show is not None:
                save_after_show()
        except Exception as exc:
            print(
                "[Warning] Could not display/save generated figures: "
                f"{exc}"
            )
        finally:
            try:
                plt.close(fig)
            except Exception as exc:
                print(f"[Warning] Could not close generated figure: {exc}")

    # Each window captures its own angle, so the record is written only once
    # every figure has been closed and saved.
    camera_angles = [
        angle for angle in (
            getattr(fig, '_alterseek_camera_angle', None)
            for fig in display_figures
        )
        if angle is not None
    ]
    try:
        camera_angle_path = write_camera_angle_file(camera_angles)
    except Exception as exc:
        print(f"[Warning] Could not write the camera-angle record: {exc}")
    else:
        if camera_angle_path is not None:
            print(f"Saved: {camera_angle_path}")


def _cell_suffix(sites, lattice_tag):
    """Trailing size/lattice tag describing the cell on that line."""
    parts = []
    if sites:
        parts.append(f"{sites} atoms")
    if lattice_tag and lattice_tag != 'unknown':
        parts.append(lattice_tag)
    return f"[{', '.join(parts)}]" if parts else ""


def _print_cell_rows(rows, note=None, note_after_index=0, group_kind="SG"):
    """Print cell-summary fields in columns sized to the supplied rows."""
    group_w = max(len(row[1]) for row in rows)
    pg_w = max(len(row[2]) for row in rows)
    laue_w = max((len(row[3]) for row in rows if row[3]), default=0)
    for index, (label, group, pg, laue, suffix) in enumerate(rows):
        laue_field = f"  Laue {laue:<{laue_w}}" if laue else ""
        line = (f"{label:<{_CELL_LABEL_WIDTH}}"
                f"{group_kind} {group:<{group_w}}  PG {pg:<{pg_w}}"
                f"{laue_field}  {suffix}")
        print(line.rstrip())
        if index == note_after_index and note:
            print(f"{'':<{_CELL_LABEL_WIDTH}}{note}")


def _print_2d_structure_summary(analysis_preparation, centroid_result):
    """Print the layer-group analogue of the 3D cell summary."""
    summary = analysis_preparation["layer_cell_summary"]
    rows = [
        ("Input cell:", summary["input_cell"]),
        (
            "Nonmagnetic primitive cell:",
            summary["nonmagnetic_primitive_cell"],
        ),
    ]
    if summary["magnetic_primitive_cell"] is not None:
        rows.append((
            "Magnetic primitive cell:",
            summary["magnetic_primitive_cell"],
        ))
    _print_cell_rows(
        [
            (
                label,
                symmetry["label"],
                symmetry["point_group"],
                symmetry["laue_group"],
                _cell_suffix(symmetry["sites"], None),
            )
            for label, symmetry in rows
        ],
        group_kind="LG",
    )
    lattice_type = str(centroid_result.get("sc_type", "Unknown")).replace(
        "_", " "
    )
    print(f"2D lattice: {lattice_type}")


def _g0_symmetry(sf_result, sites=None):
    """Return symmetry metadata for FindSpinGroup's reported spatial subgroup G0.

    Detecting symmetry from atomic coordinates alone could instead recover the
    higher-symmetry nonmagnetic parent structure.
    """
    number = sf_result.get('g0_number')
    laue_group = laue_group_from_spacegroup_number(number)
    if laue_group is None:
        return None
    return {
        'label': f"{sf_result.get('g0_symbol')} ({int(number)})",
        'spacegroup_number': int(number),
        'point_group': point_group_from_spacegroup_number(number),
        'laue_group': laue_group,
        'sites': sites,
    }


def _altermagnetism_gate(sf_result, working_cell_symmetry=None):
    """Return why the working-cell symmetry forbids altermagnetism, or None.

    Fall back to the submitted-cell point group when no working-cell symmetry
    is available.
    """
    if working_cell_symmetry is not None:
        return no_altermagnetism_reason(
            spacegroup=working_cell_symmetry['spacegroup_number'])
    return no_altermagnetism_reason(sf_result.get('point_group'))


def _validate_input_config(config):
    unknown = sorted(set(config) - _INPUT_CONFIG_KEYS)
    if unknown:
        raise ValueError(
            f"unknown setting{'s' if len(unknown) != 1 else ''}: {', '.join(unknown)}"
        )

    for key in ("structure", "spin_axis", "moments", "output_code"):
        if key in config and not isinstance(config[key], str):
            raise ValueError(f"{key} must be a TOML string")

    if "output_code" in config:
        code = config["output_code"].strip().lower()
        if code not in {"vasp", "qe", "abinit"}:
            raise ValueError("output_code must be \"vasp\", \"qe\", or \"abinit\"")

    if "flip_option" in config:
        choice = config["flip_option"]
        if isinstance(choice, bool) or not isinstance(choice, int) or choice < 1:
            raise ValueError("flip_option must be a positive integer")

    has_elev = "view_elev" in config
    has_azim = "view_azim" in config
    if has_elev != has_azim:
        raise ValueError("view_elev and view_azim must be supplied together")
    for key in ("view_elev", "view_azim"):
        if key in config and (isinstance(config[key], bool) or
                              not isinstance(config[key], (int, float))):
            raise ValueError(f"{key} must be a number")

    if "save_pdf" in config and not isinstance(config["save_pdf"], bool):
        raise ValueError("save_pdf must be true or false")

    if "vacuum_axis" in config:
        axis = config["vacuum_axis"]
        if not isinstance(axis, str) or axis.strip().lower() not in _VACUUM_AXIS_INDEX:
            raise ValueError("vacuum_axis must be \"a\", \"b\", or \"c\"")

    if "symprec" in config:
        value = config["symprec"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("symprec must be a number")
        if not np.isfinite(value):
            raise ValueError("symprec must be finite")
        if not value > 0:
            raise ValueError("symprec must be positive")
    return config


def _read_input_config(path=INPUT_CONFIG_FILE):
    """Read and validate a per-run TOML config, or return an empty dict if absent."""
    if not os.path.exists(path):
        return {}
    try:
        import tomllib
        with open(path, "rb") as f:
            config = tomllib.load(f)
        return _validate_input_config(config)
    except Exception as exc:
        raise ValueError(f"Failed to read {path}: {exc}") from exc


class KPathBuilder:
    """Build the general-k path and write it for VASP, QE, or ABINIT."""

    def __init__(self, mode_2d: bool = False, input_vacuum_axis: int = None):
        self.kpoints_data = []
        self.header_lines = []
        self.extra_general_points = []
        self.butterfly_kpoints_data = None
        self.butterfly_extra_general_points = None
        self.kpoints_basis_matrix = None
        self.output_basis_matrix = None
        self.kpoints_basis_rotation = None
        self.plane_normal_cartesian = None
        self.mode_2d = mode_2d
        self._vacuum_axis_from_cli = input_vacuum_axis is not None
        self.input_vacuum_axis = (
            2 if input_vacuum_axis is None else input_vacuum_axis
        )

    def _configure_2d_plane(self, centroid_result, submitted_lattice=None):
        """Record the physical slab plane independently of any cell basis."""
        if not self.mode_2d:
            self.plane_normal_cartesian = None
            return
        if submitted_lattice is None:
            b_input = np.asarray(centroid_result["b_matrix_input"], dtype=float)
            submitted_lattice = 2 * np.pi * np.linalg.inv(b_input).T
        normal = slab_plane_normal_cartesian(
            submitted_lattice,
            self.input_vacuum_axis,
        )
        self.plane_normal_cartesian = normal
        centroid_result["plane_normal_cartesian"] = normal.copy()

    def _is_valid_2d_operation(self, operation, centroid_result):
        """Classify an operation in its source basis against the physical plane."""
        if self.plane_normal_cartesian is None:
            raise RuntimeError("Physical 2D plane has not been configured")
        operation_basis = np.asarray(
            centroid_result["b_matrix_input"], dtype=float
        )
        return is_valid_2d_spin_flip_cartesian(
            operation,
            operation_basis,
            self.plane_normal_cartesian,
        )

    def _forces_2d_degeneracy(self, operation, centroid_result):
        """Return whether a spin flip enforces degeneracy throughout the 2D plane.

        A plane-preserving C_2z T or U m_z operation acts as -I or +I in the
        plane and enforces spin degeneracy at every in-plane k, ruling out
        altermagnetic spin splitting.
        """
        if self.plane_normal_cartesian is None:
            raise RuntimeError("Physical 2D plane has not been configured")
        operation_basis = np.asarray(
            centroid_result["b_matrix_input"], dtype=float
        )
        return (
            keeps_2d_plane_cartesian(
                operation, operation_basis, self.plane_normal_cartesian
            )
            and is_trivial_2d_spin_flip_cartesian(
                operation, operation_basis, self.plane_normal_cartesian
            )
        )

    @staticmethod
    def _display_label(label: str) -> str:
        return KPathBuilder._kpoints_label(label)

    @staticmethod
    def _kpoints_label(label: str) -> str:
        """Normalize point labels for generated path files."""
        safe_aliases = []
        for alias in label_aliases(label):
            safe_aliases.append(
                'GAMMA'
                if alias.strip().upper() == 'GAMMA' or alias == '\u0393'
                else alias
            )
        return combine_point_labels(*safe_aliases)

    @classmethod
    def _format_path(cls, path_segments) -> str:
        parts = []
        prev_end = None
        for seg_start, seg_end in path_segments:
            start = cls._kpoints_label(seg_start)
            end = cls._kpoints_label(seg_end)
            if seg_start != prev_end:
                parts.append(f"| {start}-{end}" if prev_end else f"{start}-{end}")
            else:
                parts.append(end)
            prev_end = seg_end
        return "-".join(parts).replace("-| ", " | ")

    @staticmethod
    def _format_matrix(op: np.ndarray) -> str:
        rows = []
        for row in op:
            row_str = " ".join(
                f"{int(x): 3d}" if float(x) % 1 == 0 else f"{x: .2f}"
                for x in row
            )
            rows.append(f"    [ {row_str} ]")
        return "\n".join(rows)

    @staticmethod
    def _count_written_segments(kpoints) -> int:
        count = 0
        for i in range(len(kpoints) - 1):
            start_point = kpoints[i]
            end_point = kpoints[i + 1]
            if start_point is None or end_point is None:
                continue
            if start_point[3] == end_point[3]:
                continue
            if {start_point[3], end_point[3]} == {"k", "k'"}:
                continue
            if np.allclose(
                    start_point[:3], end_point[:3],
                    atol=POINT_COINCIDENCE_ATOL, rtol=0.0):
                continue
            count += 1
        return count
    
    def _kpoint_for_output_basis(self, point: List) -> List:
        """Convert an internal k-point to the submitted cell's reciprocal basis."""
        if self.kpoints_basis_matrix is None or self.output_basis_matrix is None:
            return point

        try:
            k_frac = np.array(point[:3], dtype=float)
            b_kpoints = np.array(self.kpoints_basis_matrix, dtype=float)
            if self.kpoints_basis_rotation is not None:
                b_kpoints = b_kpoints @ np.array(self.kpoints_basis_rotation, dtype=float)
            b_output = np.array(self.output_basis_matrix, dtype=float)
            k_out = k_frac @ b_kpoints @ np.linalg.inv(b_output)
        except Exception as exc:
            raise RuntimeError(
                f"Output-basis conversion failed for k-point '{point[3]}': {exc}. "
                "Refusing to write unconverted coordinates into KPOINTS."
            ) from exc
        if self.mode_2d:
            if self.plane_normal_cartesian is None:
                raise RuntimeError(
                    "Physical 2D plane was not configured before KPOINTS output."
                )
            normal = np.asarray(self.plane_normal_cartesian, dtype=float)
            normal /= np.linalg.norm(normal)
            k_cart = k_out @ b_output
            normal_component = float(k_cart @ normal)
            plane_tolerance = 1e-7 * max(1.0, float(np.linalg.norm(k_cart)))
            if abs(normal_component) > plane_tolerance:
                raise RuntimeError(
                    f"2D k-point '{point[3]}' lies outside the physical slab "
                    f"plane ({normal_component:.3g} 1/Angstrom)."
                )
            k_cart = k_cart - normal_component * normal
            k_out = k_cart @ np.linalg.inv(b_output)
        return [k_out[0], k_out[1], k_out[2], point[3]]

    def load_flip_operations(self, filename: str = None) -> List[np.ndarray]:
        """Reads pre-calculated rotation matrices from file"""
        if filename is None:
            filename = os.path.join(OUTPUT_DIR, "spin_flip_operations.txt")
        matrices = []
        unique = []
        current_matrix = []
        try:
            with open(filename, 'r', encoding='utf-8-sig') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("Operation"):
                    continue
                if "|" in line:
                    line = line.split("|", 1)[0].strip()
                
                parts = line.split()
                if len(parts) == 3:
                    current_matrix.append([float(x) for x in parts])
                
                if len(current_matrix) == 3:
                    mat = np.array(current_matrix)
                    if not any(np.allclose(mat, ex, atol=1e-8) for ex in unique):
                        unique.append(mat)
                        matrices.append(mat)
                    current_matrix = []
            return matrices
        except FileNotFoundError:
            return []

    def load_preserve_operations(self, filename: str = None) -> List[np.ndarray]:
        """Reads pre-calculated spin-preserve rotation matrices from file."""
        if filename is None:
            filename = os.path.join(OUTPUT_DIR, "spin_preserve_operations.txt")
        return self.load_flip_operations(filename)

    def transform_kpoint(self, kpoint: List[float], transformation_matrix: np.ndarray) -> List[float]:
        """Apply transformation matrix to k-point: R^(-1)^T * k"""
        k = np.array(kpoint[:3])  # Take only x, y, z coordinates
        # Calculate R^(-1)^T
        R_inv_T = np.linalg.inv(transformation_matrix).T
        k_transformed = R_inv_T @ k
        return k_transformed.tolist()
    
    def insert_general_kpoints(self,
                               general_kpoint: List[float],
                               transformation_matrix: np.ndarray,
                               extra_general_points: Optional[List[List]] = None,
                               *,
                               path_points: Optional[List[List]] = None,
                               report: bool = True) -> List[List]:
        """Build the alternating plain/butterfly form of a high-symmetry path.

        A full butterfly for A-B is A-k | k'-A'-B'-k' | k-B, with primed
        points obtained using R^(-T). None marks path discontinuities.
        """
        source_path = self.kpoints_data if path_points is None else path_points
        if not source_path:
            print("Error: No k-points data loaded. Please read KPOINTS file first.")
            return []

        k_prime = self.transform_kpoint(general_kpoint, transformation_matrix)
        kpt = general_kpoint
        kp  = k_prime

        def coords_eq(p, q, tol=POINT_COINCIDENCE_ATOL):
            return abs(p[0]-q[0]) < tol and abs(p[1]-q[1]) < tol and abs(p[2]-q[2]) < tol

        def pt_key(p):
            return (round(p[0], 6), round(p[1], 6), round(p[2], 6))

        def is_gamma(p):
            aliases = label_aliases(p[3])
            return bool(aliases) and all(
                label.strip().upper() == 'GAMMA' or label == '\u0393'
                for label in aliases
            )

        def get_prime(p):
            """Return primed version of p (Gamma stays unprimed)."""
            if is_gamma(p):
                return p.copy()
            tc = self.transform_kpoint(p, transformation_matrix)
            return [tc[0], tc[1], tc[2], prime_point_label(p[3])]

        # --- Phase 1: group flat kpoints_data into segment pairs ---
        raw = self._combine_coincident_path_labels(source_path)
        seg_pairs = [(raw[i], raw[i+1]) for i in range(0, len(raw) - 1, 2)]
        seg_pairs = [
            (start, end) for start, end in seg_pairs
            if not self._general_k_covers_segment(general_kpoint, start, end)
        ]
        if not seg_pairs:
            print("Error: Every path segment already runs through general k.")
            return []

        # --- Phase 2: build connected chains ---
        chains = []
        current_chain = [seg_pairs[0][0], seg_pairs[0][1]]
        for sp_start, sp_end in seg_pairs[1:]:
            if coords_eq(current_chain[-1], sp_start):
                current_chain.append(sp_end)
            else:
                chains.append(current_chain)
                current_chain = [sp_start, sp_end]
        chains.append(current_chain)

        # --- Phase 3: apply the paper's alternating plain/butterfly construction ---
        # For the first chain A-B-C-..., keep A-B plain, then butterfly B-C:
        # A-B-k | k'-B'-C'-k' | k-C.
        # Even parity keeps the original segment; odd parity applies a butterfly.
        # The first chain starts even; each later disconnected chain starts odd.
        # Do not butterfly an endpoint twice; GAMMA maps to itself and stays unprimed.
        path_sequence = []

        def emit_plain(A, B):
            path_sequence.append(A.copy())
            path_sequence.append(B.copy())

        def emit_butterfly(A, B):
            path_sequence.append(A.copy())
            path_sequence.append([kpt[0], kpt[1], kpt[2], "k"])
            path_sequence.append([kp[0],  kp[1],  kp[2],  "k'"])
            path_sequence.append(get_prime(A))
            path_sequence.append(get_prime(B))
            # If B was already butterflied, omit the repeated B'-k' | k-B closure.
            if pt_key(B) not in butterflied:
                path_sequence.append([kp[0],  kp[1],  kp[2],  "k'"])
                path_sequence.append([kpt[0], kpt[1], kpt[2], "k"])
                path_sequence.append(B.copy())

        butterflied = set()  # pt_key of points that have received butterfly treatment

        for ci, chain in enumerate(chains):
            # Merge consecutive coincident vertices while preserving all labels.
            unique = []
            for pt in chain:
                if unique and coords_eq(unique[-1], pt):
                    unique[-1][3] = combine_point_labels(unique[-1][3], pt[3])
                else:
                    unique.append(pt.copy())

            # Skip chains that collapse to a single reciprocal-space point.
            if len(unique) < 2:
                labels = [p[3] for p in chain]
                if report:
                    print(
                        f"  [Note] Part {ci+1} "
                        f"({' - '.join(labels)}) skipped: "
                        "endpoints coincide in coordinates"
                    )
                continue

            if ci > 0:
                path_sequence.append(None)

            start_parity = 0 if ci == 0 else 1

            parity = start_parity  # tracks alternation independently of s
            for s, (A, B) in enumerate(zip(unique, unique[1:])):
                A_key = pt_key(A)
                B_key = pt_key(B)
                is_last = (s == len(unique) - 2)
                # If A was already butterflied, continue from A or A' without repeating its A-k | k'-A' connections.
                if A_key in butterflied:
                    A_start = A
                    if path_sequence:
                        prev_pt = path_sequence[-1]
                        A_prime = get_prime(A)
                        if (prev_pt is not None and prev_pt[3] == A_prime[3]
                                and A_prime[3] != A[3]):
                            A_start = A_prime
                    emit_plain(A_start, B)
                    # Give an untreated chain endpoint B the partial butterfly B-k | k'-B'.
                    if B_key not in butterflied and is_last:
                        path_sequence.append(B.copy())
                        path_sequence.append([kpt[0], kpt[1], kpt[2], "k"])
                        path_sequence.append([kp[0],  kp[1],  kp[2],  "k'"])
                        path_sequence.append(get_prime(B))
                        butterflied.add(B_key)
                    # Keep odd parity so the next segment with an unbutterflied start receives the butterfly.
                    if parity % 2 == 0:
                        parity += 1
                elif parity % 2 == 1:
                    emit_butterfly(A, B)
                    butterflied.add(A_key)
                    butterflied.add(B_key)
                    parity += 1
                else:
                    emit_plain(A, B)
                    parity += 1

            # If the chain ends at an unbutterflied point B, add B-k | k'-B'.
            last_key = pt_key(unique[-1])
            if last_key not in butterflied:
                last_pt = unique[-1]
                path_sequence.append(last_pt.copy())
                path_sequence.append([kpt[0], kpt[1], kpt[2], "k"])
                path_sequence.append([kp[0],  kp[1],  kp[2],  "k'"])
                path_sequence.append(get_prime(last_pt))
                butterflied.add(last_key)

        if path_sequence and path_sequence[-1] is None:
            path_sequence.pop()

        # Append P-k | k'-P' for extra doubled-IBZ vertices outside the path.
        extra_general_points = extra_general_points or []
        for pt in extra_general_points:
            if path_sequence:
                path_sequence.append(None)
            path_sequence.append(pt.copy())
            path_sequence.append([kpt[0], kpt[1], kpt[2], "k"])
            path_sequence.append([kp[0],  kp[1],  kp[2],  "k'"])
            path_sequence.append(get_prime(pt))

        # Build a compact path preview; None and k/k' transitions mark breaks.
        tokens = []
        prev = None
        i = 0
        while i < len(path_sequence) - 1:
            cur = path_sequence[i]
            nxt = path_sequence[i + 1]
            if cur is None or nxt is None:
                if tokens and tokens[-1] != '|':
                    tokens.append('|')
                i += 1
                continue
            if (cur[3] == "k" and nxt[3] == "k'") or \
               (cur[3] == "k'" and nxt[3] == "k"):
                if tokens and tokens[-1] != '|':
                    tokens.append('|')
                i += 1
                continue
            if cur[3] == nxt[3]:
                i += 1
                continue
            if prev != cur[3]:
                tokens.append(cur[3])
            tokens.append(nxt[3])
            prev = nxt[3]
            i += 1
        display_segments = []
        current_segment = []
        for t in tokens:
            if t == '|':
                if current_segment:
                    display_segments.append('-'.join(self._display_label(x) for x in current_segment))
                    current_segment = []
            else:
                current_segment.append(t)
        if current_segment:
            display_segments.append('-'.join(self._display_label(x) for x in current_segment))

        if len(display_segments) > 6:
            display = " | ".join(display_segments[:3] + ["..."] + display_segments[-3:])
        else:
            display = " | ".join(display_segments)
        if report:
            print(f"Generated path: {display}")

        generated_segments = self._count_written_segments(path_sequence)
        generated_points = sum(1 for pt in path_sequence if pt is not None)
        if report:
            print(f"Full path: {len(seg_pairs)} original segments -> "
                  f"{generated_segments} generated segments, "
                  f"{generated_points} k-points")
        if report and extra_general_points:
            labels = ", ".join(
                self._display_label(pt[3]) for pt in extra_general_points
            )
            print(f"Added general-k connections: {labels}")

        return path_sequence

    def build_ordinary_path_with_general_k(
            self, kpoint: List[float],
            extra_general_points: Optional[List[List]] = None) -> List[List]:
        """Keep the ordinary path and append A-k-B connections through general k."""
        kpt = [kpoint[0], kpoint[1], kpoint[2], "k"]
        raw = self._combine_coincident_path_labels(self.kpoints_data)
        seg_pairs = [(raw[i], raw[i + 1]) for i in range(0, len(raw) - 1, 2)]
        seg_pairs = [
            (start, end) for start, end in seg_pairs
            if not self._general_k_covers_segment(kpoint, start, end)
        ]
        path_sequence = []
        for idx, (start, end) in enumerate(seg_pairs):
            if idx:
                path_sequence.append(None)
            path_sequence.append(start.copy())
            path_sequence.append(end.copy())

        seen = set()
        general_points = []
        for pt in self.kpoints_data + (extra_general_points or []):
            if pt is None:
                continue
            key = (
                round(float(pt[0]), 8),
                round(float(pt[1]), 8),
                round(float(pt[2]), 8),
                str(pt[3]),
            )
            if key in seen:
                continue
            seen.add(key)
            general_points.append(pt)

        for idx in range(0, len(general_points), 2):
            if path_sequence:
                path_sequence.append(None)
            path_sequence.append(general_points[idx].copy())
            path_sequence.append(kpt.copy())
            if idx + 1 < len(general_points):
                path_sequence.append(general_points[idx + 1].copy())

        pair_count = len(general_points) // 2
        leftover = len(general_points) % 2
        print(
            f"Kept ordinary path and added {pair_count} A-k-B connection segments"
            f"{' plus 1 A-k tail' if leftover else ''}."
        )
        return path_sequence
    
    @staticmethod
    def _general_k_covers_segment(kpoint, start, end, tol=1e-8):
        """Whether general k sits between the endpoints and so is drawn twice.

        The ordinary segment is then the duplicate and is dropped.
        """
        first = np.asarray(start[:3], dtype=float)
        second = np.asarray(end[:3], dtype=float)
        along = second - first
        length_squared = float(np.dot(along, along))
        if length_squared <= tol:
            return False
        offset = np.asarray(kpoint[:3], dtype=float) - first
        position = float(np.dot(offset, along)) / length_squared
        if not tol < position < 1.0 - tol:
            return False
        return bool(np.allclose(offset, position * along, atol=1e-6, rtol=0.0))

    @staticmethod
    def _combine_coincident_path_labels(points):
        """Return copies with coincident path labels combined, e.g. M/Z_0.

        Combining before zero-length segments are removed preserves every label
        on retained occurrences of the same point.
        """
        groups = []
        assignments = []
        copied = []
        for point in points:
            current = point.copy()
            copied.append(current)
            for index, group in enumerate(groups):
                if np.allclose(
                        current[:3], group["coords"],
                        atol=POINT_COINCIDENCE_ATOL, rtol=0.0):
                    group["label"] = combine_point_labels(
                        group["label"], current[3]
                    )
                    assignments.append(index)
                    break
            else:
                groups.append({
                    "coords": np.asarray(current[:3], dtype=float),
                    "label": str(current[3]),
                })
                assignments.append(len(groups) - 1)

        for point, group_index in zip(copied, assignments):
            point[3] = groups[group_index]["label"]
        return copied

    @staticmethod
    def _merge_consecutive_coincident_path_points(new_kpoints):
        """Merge consecutive equal-coordinate path points and retain their names."""
        merged = []
        for index, point in enumerate(new_kpoints):
            if point is None:
                merged.append(None)
                continue
            current = point.copy()
            if merged and merged[-1] is not None:
                previous, _previous_index = merged[-1]
                helper_gap = {previous[3], current[3]} == {"k", "k'"}
                if (not helper_gap and np.allclose(
                        previous[:3], current[:3],
                        atol=POINT_COINCIDENCE_ATOL, rtol=0.0)):
                    previous[3] = combine_point_labels(previous[3], current[3])
                    merged[-1] = (previous, index)
                    continue
            merged.append((current, index))
        return merged

    def _prepare_output_segments(self, new_kpoints: List[List]):
        """Prepare writable segments in the submitted cell's reciprocal basis.

        Merge consecutive coincident points, treat None and k-k' as path breaks,
        and omit zero-length segments. Return
        (start, end, break_before, start_index, raw_end_label) tuples.
        """
        path_points = self._merge_consecutive_coincident_path_points(new_kpoints)
        pairs = []
        forced_break = False
        i = 0
        while i < len(path_points) - 1:
            start_item = path_points[i]
            end_item = path_points[i + 1]
            if start_item is None or end_item is None:
                i += 1
                continue
            sp, original_index = start_item
            ep, _end_original_index = end_item
            if (sp[3] == "k" and ep[3] == "k'") or (sp[3] == "k'" and ep[3] == "k"):
                forced_break = True
                i += 1
                continue
            if sp[3] == ep[3]:
                i += 1
                continue
            sp_out = self._kpoint_for_output_basis(sp)
            ep_out = self._kpoint_for_output_basis(ep)
            if np.allclose(
                    sp_out[:3], ep_out[:3],
                    atol=POINT_COINCIDENCE_ATOL, rtol=0.0):
                i += 1
                continue
            break_before = forced_break
            if pairs:
                previous_end = pairs[-1][1]
                break_before = break_before or not np.allclose(
                    previous_end[:3], sp_out[:3], atol=1e-10, rtol=0.0
                )
            pairs.append((sp_out, ep_out, break_before, original_index, ep[3]))
            forced_break = False
            i += 1
        return pairs

    def _general_kpoint_output_basis(self, general_kpoint) -> Optional[List[float]]:
        """Return k in the submitted cell's reciprocal basis, or None if unavailable."""
        if general_kpoint is None:
            return None
        if self.kpoints_basis_matrix is None or self.output_basis_matrix is None:
            return None
        try:
            out = self._kpoint_for_output_basis([*general_kpoint[:3], "k"])
        except Exception:
            return None
        return [float(out[0]), float(out[1]), float(out[2])]

    def write_kpoints_file_vasp(self, new_kpoints: List[List], output_file: str = "KPOINTS_alter",
                                transformation_matrix: Optional[np.ndarray] = None,
                                transformation_label: Optional[str] = None,
                                operation_basis_label: Optional[str] = None):
        """Write the path in VASP line-mode KPOINTS format."""
        segments = self._prepare_output_segments(new_kpoints)
        if not segments:
            print("Error writing file: path contains no writable segments.")
            return False

        if transformation_matrix is not None:
            flat_matrix = np.array(transformation_matrix).flatten()
            matrix_str = " ".join(f"{x:.8f}" for x in flat_matrix)
            label = f" ({transformation_label})" if transformation_label else ""
            # Record the spin-flip matrix in its source basis, which may differ from the submitted cell.
            basis_name = operation_basis_label or "operation-source structure"
            first_line = (
                f"Selected spin-flip operation{label} in {basis_name} "
                f"real-space fractional basis: {matrix_str}\n"
            )
        else:
            first_line = f"{self.header_lines[0]}\n"

        lines = [
            first_line,
            "   30\n",
            f"{self.header_lines[2]}\n",
            f"{self.header_lines[3]}\n",
        ]

        for start_point_out, end_point_out, _break_before, i, end_raw_label in segments:
            start_label = self._kpoints_label(start_point_out[3])
            end_label = self._kpoints_label(end_point_out[3])
            lines.append(
                f"   {_fmt_coord(start_point_out[0])}   {_fmt_coord(start_point_out[1])}   "
                f"{_fmt_coord(start_point_out[2])}     {start_label}\n"
            )
            lines.append(
                f"   {_fmt_coord(end_point_out[0])}   {_fmt_coord(end_point_out[1])}   "
                f"{_fmt_coord(end_point_out[2])}     {end_label}\n"
            )
            if end_raw_label == "k" or i < len(new_kpoints) - 2:
                lines.append("\n")

        try:
            _atomic_write_text(output_file, "".join(lines))
        except (OSError, UnicodeError) as exc:
            print(f"Error writing file: {exc}")
            return False

        print(f"Modified KPOINTS file written to: {output_file}")
        return True

    def write_kpoints_file_qe(self, new_kpoints: List[List],
                              output_file: str = "KPOINTS_alter_qe",
                              transformation_matrix: Optional[np.ndarray] = None,
                              transformation_label: Optional[str] = None,
                               ninterp: int = 30,
                               operation_basis_label: Optional[str] = None):
        """Write the path in QE K_POINTS crystal_b format."""
        segments = self._prepare_output_segments(new_kpoints)
        if not segments:
            print("Error writing QE KPOINTS: path contains no writable segments.")
            return False

        # List each path point once. Chain endpoints use 1; all other points use ninterp.
        path_points = []
        for idx, (sp_out, ep_out, break_before, _i, _end_raw_label) in enumerate(segments):
            if idx == 0:
                path_points.append((sp_out, ninterp))
            elif break_before:
                previous_point, _ = path_points[-1]
                path_points[-1] = (previous_point, 1)
                path_points.append((sp_out, ninterp))
            path_points.append((ep_out, ninterp))
        final_point, _ = path_points[-1]
        path_points[-1] = (final_point, 1)

        lines = ["K_POINTS crystal_b\n", f"  {len(path_points)}\n"]
        for pt_out, ni in path_points:
            lbl = self._kpoints_label(pt_out[3])
            lines.append(
                f"  {_fmt_coord(pt_out[0])}  {_fmt_coord(pt_out[1])}  "
                f"{_fmt_coord(pt_out[2])}  {ni:3d}  ! {lbl}\n"
            )
        if transformation_matrix is not None:
            flat = np.array(transformation_matrix).flatten()
            mat_str = " ".join(f"{x:.8f}" for x in flat)
            lbl = f" ({transformation_label})" if transformation_label else ""
            # Record the spin-flip matrix in its source basis, which may differ from the submitted cell.
            basis_name = operation_basis_label or "operation-source structure"
            lines.append(
                f"! Spin-flip operation{lbl} in {basis_name} "
                f"real-space fractional basis: {mat_str}\n"
            )

        try:
            _atomic_write_text(output_file, "".join(lines))
        except (OSError, UnicodeError) as exc:
            print(f"Error writing QE KPOINTS: {exc}")
            return False

        print(f"QE KPOINTS written to: {output_file}")
        return True

    def write_kpoints_file_abinit(self, new_kpoints: List[List],
                                  output_file: str = "KPOINTS_alter_abinit",
                                  transformation_matrix: Optional[np.ndarray] = None,
                                  transformation_label: Optional[str] = None,
                                  ndivk: int = 30,
                                  operation_basis_label: Optional[str] = None):
        """Write the k-path as an ABINIT kptopt/kptbounds/ndivk block."""
        segments = self._prepare_output_segments(new_kpoints)
        if not segments:
            print("Error writing ABINIT KPOINTS: path contains no writable segments.")
            return False

        # List each path point once. Chain endpoints use 1; ndivk stores each following segment's division count separately from kptbounds.
        path_points = []
        for idx, (sp_out, ep_out, break_before, _i, _end_raw_label) in enumerate(segments):
            if idx == 0:
                path_points.append((sp_out, ndivk))
            elif break_before:
                previous_point, _ = path_points[-1]
                path_points[-1] = (previous_point, 1)
                path_points.append((sp_out, ndivk))
            path_points.append((ep_out, ndivk))
        final_point, _ = path_points[-1]
        path_points[-1] = (final_point, 1)

        n_segments = len(path_points) - 1
        segment_divisions = [ni for _, ni in path_points[:-1]]

        lines = [
            f"kptopt   -{n_segments}\n",
            "ndivk    " + " ".join(str(ni) for ni in segment_divisions) + "\n",
            "kptbounds\n",
        ]
        for pt_out, _ni in path_points:
            lbl = self._kpoints_label(pt_out[3])
            lines.append(
                f"   {_fmt_coord(pt_out[0])}   {_fmt_coord(pt_out[1])}   "
                f"{_fmt_coord(pt_out[2])}   # {lbl}\n"
            )
        if transformation_matrix is not None:
            flat = np.array(transformation_matrix).flatten()
            mat_str = " ".join(f"{x:.8f}" for x in flat)
            lbl = f" ({transformation_label})" if transformation_label else ""
            # Record the spin-flip matrix in its source basis, which may differ from the submitted cell.
            basis_name = operation_basis_label or "operation-source structure"
            lines.append(
                f"# Spin-flip operation{lbl} in {basis_name} "
                f"real-space fractional basis: {mat_str}\n"
            )

        try:
            _atomic_write_text(output_file, "".join(lines))
        except (OSError, UnicodeError) as exc:
            print(f"Error writing ABINIT KPOINTS: {exc}")
            return False

        print(f"ABINIT KPOINTS written to: {output_file}")
        return True

    @alterseek_plot_style
    def _generate_spin_figures(self, centroid_result, struct_file, general_kpoint,
                               R_for_kpts, R_cart_for_plot, flip_ops_for_plot,
                               preserve_ops_for_plot, new_kpoints, display_figures,
                               save_pdf=False):
        """Generate 2D or 3D spin-analysis figures for the selected operation and add them to display_figures."""
        if centroid_result is not None and self.mode_2d:
            # In 2D mode, render top-down figures instead of the 3D BZ plate.
            basename = (os.path.splitext(os.path.basename(struct_file))[0]
                        if struct_file else 'output')
            if 'b_matrix' in centroid_result:
                try:
                    return plot_2d_figures(
                        centroid_result, general_kpoint, R_for_kpts,
                        basename, output_dir=OUTPUT_DIR,
                        flip_ops_for_plot=(flip_ops_for_plot
                                           if flip_ops_for_plot else None),
                        save_pdf=save_pdf,
                        deferred_figures=display_figures,
                        path_sequence=new_kpoints,
                    )
                except Exception as _e:
                    print(f"[Warning] Could not generate 2D figures: {_e}")
            return []
        elif centroid_result is not None:
            if centroid_result.get('bz_loops') is None:
                print(
                    "[Warning] Spin figures were skipped because optional "
                    "BZ figure geometry is unavailable."
                )
                return
            self._validate_standardized_spin_map(
                centroid_result,
                R_for_kpts,
                R_cart_for_plot,
            )
            basename = (os.path.splitext(os.path.basename(struct_file))[0]
                        if struct_file else 'output')
            sc_type = centroid_result.get('sc_type', 'BZ')
            flip_kwargs = dict(
                flip_ops_frac=flip_ops_for_plot if flip_ops_for_plot else None,
                preserve_ops_frac=preserve_ops_for_plot if preserve_ops_for_plot else None,
            )
            view_kwargs = dict(
                bz_center=centroid_result.get('bz_center'),
                elev=centroid_result.get('elev', 14),
                azim=centroid_result.get('azim', 20),
            )
            figure_specs = []
            if 'b_matrix' in centroid_result:
                figure_specs.append((
                    plot_spin_flip_figure, 'spin-flip',
                    os.path.join(OUTPUT_DIR, f'{basename}_spinflip_{sc_type}.png'),
                    dict(
                        kpoints_data=self.kpoints_data,
                        ibz_kpoints_frac=centroid_result.get('ibz_kpoints_frac', {}),
                        centroid_frac=general_kpoint,
                        R_cart=R_cart_for_plot,
                        block=False,
                        path_sequence=new_kpoints,
                        **view_kwargs,
                    ),
                ))
            if 'unique_ops' in centroid_result:
                spin_bz_kwargs = dict(
                    unique_ops=centroid_result['unique_ops'],
                    centroid_cart=centroid_result.get('centroid_cart'),
                    hull_labels=centroid_result.get('hull_labels'),
                    **flip_kwargs,
                )
                top_view_z0 = 0.0
                top_view_axis = 2
                if sc_type in ('mP1', 'mC1', 'mC2', 'mC3'):
                    # Cut perpendicular to the monoclinic twofold axis and sample just off ky=0 to reveal the bulk d-wave pattern.
                    top_view_axis = 1
                elif (sc_type in ('hR1', 'hR2') or sc_type.startswith('c')):
                    # Measure the actual Cartesian kz extent from the BZ boundary because the primitive reciprocal vectors need not align with z.
                    # Move off kz=0 to reveal the bulk g- or i-wave pattern, using z0=0.5*z_max for rhombohedral and 0.25*z_max for cubic.
                    # Cubic uses 0.25*z_max because 0.5*z_max can coincide with a BZ boundary plane.
                    bz_pts = np.vstack(centroid_result['bz_loops'])
                    z_max = float(np.abs(bz_pts[:, 2]).max())
                    z_frac = 0.5 if sc_type in ('hR1', 'hR2') else 0.25
                    top_view_z0 = z_frac * z_max
                figure_specs.append((
                    plot_spin_bz_figure, 'spin-BZ',
                    os.path.join(OUTPUT_DIR, f'{basename}_spinbz_{sc_type}.png'),
                    dict(z0=top_view_z0, cut_axis=top_view_axis,
                         **spin_bz_kwargs, **view_kwargs),
                ))
                figure_specs.append((
                    plot_spin_bz_top_view_figure, 'spin-BZ top-view',
                    os.path.join(OUTPUT_DIR, f'{basename}_spinbz_top_{sc_type}.png'),
                    dict(z0=top_view_z0, cut_axis=top_view_axis,
                         **spin_bz_kwargs),
                ))
            for plot_fn, fig_name, fig_path, extra_kwargs in figure_specs:
                try:
                    fig = plot_fn(
                        b_matrix=centroid_result['b_matrix'],
                        bz_loops=centroid_result['bz_loops'],
                        hull_pts=centroid_result.get('hull_pts'),
                        hull_simplices=centroid_result.get('hull_simplices'),
                        R=R_for_kpts,
                        output_path=fig_path,
                        show_plot=True,
                        defer_show=True,
                        save_pdf=save_pdf,
                        **extra_kwargs,
                    )
                    if fig is not None:
                        display_figures.append(fig)
                except Exception as _e:
                    print(f"[Warning] Could not generate {fig_name} figure: {_e}")

    @alterseek_plot_style
    def _generate_non_altermagnetic_figure(
        self, centroid_result, struct_file, general_kpoint, path_sequence,
        display_figures, save_pdf=False,
    ):
        """Add the red-only general-k figure for a non-altermagnet."""
        # The 2D centroid module writes no diagnostics, so on a slab run
        # nothing has created the output directory yet.
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        basename = (
            os.path.splitext(os.path.basename(struct_file))[0]
            if struct_file else 'output'
        )
        if self.mode_2d:
            plot_2d_general_path_figure(
                centroid_result,
                general_kpoint,
                basename,
                output_dir=OUTPUT_DIR,
                path_sequence=path_sequence,
                save_pdf=save_pdf,
                deferred_figures=display_figures,
            )
            return
        if centroid_result.get('bz_loops') is None:
            return
        sc_type = centroid_result.get('sc_type', 'BZ')
        figure = plot_general_path_figure(
            b_matrix=centroid_result['b_matrix'],
            bz_loops=centroid_result['bz_loops'],
            bz_center=centroid_result.get('bz_center'),
            kpoints_data=self.kpoints_data,
            ibz_kpoints_frac=centroid_result.get('ibz_kpoints_frac', {}),
            hull_pts=centroid_result.get('hull_pts'),
            hull_simplices=centroid_result.get('hull_simplices'),
            centroid_frac=general_kpoint,
            output_path=os.path.join(
                OUTPUT_DIR, f'{basename}_generalpath_{sc_type}.png'
            ),
            path_sequence=path_sequence,
            elev=centroid_result.get('elev', 14),
            azim=centroid_result.get('azim', 20),
            show_plot=True,
            defer_show=True,
            save_pdf=save_pdf,
        )
        if figure is not None:
            display_figures.append(figure)

    @staticmethod
    def _validate_standardized_spin_map(
        centroid_result,
        operation,
        cartesian_operation=None,
    ):
        """Require the mapped IBZ to remain inside the standardized BZ."""
        hull_points = centroid_result.get("hull_pts")
        bz_loops = centroid_result.get("bz_loops")
        if hull_points is None or bz_loops is None:
            return

        b_matrix = np.asarray(centroid_result["b_matrix"], dtype=float)
        if cartesian_operation is None:
            b_transpose = b_matrix.T
            cartesian_operation = (
                b_transpose
                @ np.linalg.inv(np.asarray(operation, dtype=float)).T
                @ np.linalg.inv(b_transpose)
            )
        else:
            cartesian_operation = np.asarray(
                cartesian_operation, dtype=float
            )
        mapped_points = (
            cartesian_operation @ np.asarray(hull_points, dtype=float).T
        ).T
        bz_points = np.vstack(bz_loops)
        bz_hull = ConvexHull(bz_points)
        signed_distances = (
            mapped_points @ bz_hull.equations[:, :-1].T
            + bz_hull.equations[:, -1]
        )
        scale = max(
            1.0,
            float(np.max(np.linalg.norm(bz_points, axis=1))),
        )
        if float(np.max(signed_distances)) > 1e-7 * scale:
            raise RuntimeError(
                "The selected spin operation maps the standardized IBZ "
                "outside the standardized first BZ. This indicates a "
                "reciprocal-basis mismatch."
            )

    def _select_spin_flip_operation(
        self,
        flip_ops,
        centroid_result,
        preset_choice=None,
        operation_basis_label="operation-source structure",
    ):
        """Select a detected spin-flip operation and return its matrix and option label."""
        def _op_name(op_input):
            try:
                if centroid_result is not None and 'b_matrix' in centroid_result:
                    b_mat = np.array(centroid_result['b_matrix'], dtype=float)
                    b_in = np.array(centroid_result.get(
                        'b_matrix_input', centroid_result.get('b_matrix_conv', b_mat)),
                        dtype=float)
                    b_prim = b_mat @ np.array(
                        centroid_result.get('seekpath_rotation_matrix', np.eye(3)),
                        dtype=float)
                    R_arr = np.array(op_input, dtype=float)
                    R_cart_in = b_in.T @ np.linalg.inv(R_arr).T @ np.linalg.inv(b_in.T)
                    R_prim_inv_T = np.linalg.inv(b_prim.T) @ R_cart_in @ b_prim.T
                    R_prim = np.linalg.inv(R_prim_inv_T.T)
                    R_cart = b_mat.T @ np.linalg.inv(R_prim).T @ np.linalg.inv(b_mat.T)
                    return describe_spinflip_op(R_cart, b_mat)
                # Without a standardized basis, the operation type and order remain correct but its axis is basis-dependent and must be omitted.
                return describe_spinflip_op(np.array(op_input, dtype=float), None)
            except Exception:
                return ""

        R = None
        selected_transformation_label = None
        if flip_ops:
            print(f"Found {len(flip_ops)} spin-flip operations R.")
            _names_available = (centroid_result is not None
                                and 'b_matrix' in centroid_result)
            if _names_available:
                print(
                    f"  Note: R is in the {operation_basis_label} "
                    "fractional basis;"
                )
                print("  rotation axis/mirror plane indices are in the reciprocal (b1,b2,b3) basis.")
            print("Default R: Option 1")
            _preset_pending = preset_choice is not None
            while R is None:
                print("Press [Enter] to use default, type a number, or 'list' to show matrices: ", end='', flush=True)
                if _preset_pending:
                    choice = str(preset_choice).strip().lower()
                    _preset_pending = False
                    print(choice)
                else:
                    choice = read_answer().strip().lower()

                if not choice:
                    R = flip_ops[0]
                    selected_transformation_label = "Option 1"
                    _nm = _op_name(flip_ops[0])
                    print("Selected: Option 1" + (f"  ({_nm})" if _nm else ""))
                elif choice == 'list':
                    for i, op in enumerate(flip_ops):
                        _nm = _op_name(op)
                        print(f"\n  Option {i+1}:" + (f"  {_nm}" if _nm else ""))
                        print(self._format_matrix(op))
                    print()
                else:
                    try:
                        idx = int(choice) - 1
                        if 0 <= idx < len(flip_ops):
                            R = flip_ops[idx]
                            selected_transformation_label = f"Option {idx+1}"
                            _nm = _op_name(flip_ops[idx])
                            print(f"Selected: Option {idx+1}"
                                  + (f"  ({_nm})" if _nm else ""))
                        else:
                            print(f"Please choose 1-{len(flip_ops)} or 'list'.")
                    except ValueError:
                        print(f"Please choose 1-{len(flip_ops)} or 'list'.")
        return R, selected_transformation_label

    def _convert_operation_to_primitive_basis(
        self,
        R,
        flip_ops,
        preserve_ops,
        centroid_result,
        operation_basis_label,
        output_flip_ops=None,
        output_preserve_ops=None,
    ):
        """Express spin operations in SeeK-path's standardized primitive basis.

        FindSpinGroup reports real-space fractional matrices in the submitted basis.
        Convert their reciprocal-space actions through Cartesian coordinates so R^(-T) preserves the same physical k mapping after the basis change.
        """
        #   R_cart_k       = b_input.T @ inv(R_input).T @ inv(b_input.T)
        #   R_prim^{-T}    = inv(b_prim.T) @ R_cart_k @ b_prim.T
        R_cart_for_plot = None
        flip_ops_for_plot = flip_ops
        preserve_ops_for_plot = preserve_ops
        output_flip_ops = flip_ops if output_flip_ops is None else output_flip_ops
        output_preserve_ops = (
            preserve_ops if output_preserve_ops is None else output_preserve_ops
        )
        if centroid_result is not None and 'b_matrix' in centroid_result:
            _b_input = np.array(centroid_result.get('b_matrix_input',
                                                    centroid_result['b_matrix_conv']), dtype=float)
            _b_prim = np.array(centroid_result['b_matrix'], dtype=float) @ np.array(
                centroid_result.get('seekpath_rotation_matrix', np.eye(3)),
                dtype=float,
            )
            def _convert_input_frac_R_to_prim(_R):
                _R_arr = np.array(_R, dtype=float)
                _R_cart = _b_input.T @ np.linalg.inv(_R_arr).T @ np.linalg.inv(_b_input.T)
                _R_prim_inv_T = np.linalg.inv(_b_prim.T) @ _R_cart @ _b_prim.T
                return np.linalg.inv(_R_prim_inv_T.T), _R_cart

            R_for_kpts, _ = _convert_input_frac_R_to_prim(R)
            # Figure 2 draws the HPKOT hull in SeeK-path's standardized Cartesian frame, so let it reconstruct the operation there rather than reuse the input structure's orientation, which notably differs for MCIF input.
            R_cart_for_plot = None
            if flip_ops:
                flip_ops_for_plot = [
                    _convert_input_frac_R_to_prim(op)[0] for op in flip_ops
                ]
            if preserve_ops:
                preserve_ops_for_plot = [
                    _convert_input_frac_R_to_prim(op)[0] for op in preserve_ops
                ]
            output_flip_ops_standardized = [
                _convert_input_frac_R_to_prim(op)[0] for op in output_flip_ops
            ]
            output_preserve_ops_standardized = [
                _convert_input_frac_R_to_prim(op)[0] for op in output_preserve_ops
            ]
            def _annotate_ops_with_standardized_basis(filename, input_ops, standardized_ops, label):
                try:
                    with _atomic_open_text(filename) as f:
                        f.write(
                            f"# Left basis: {operation_basis_label} real-space "
                            "fractional basis (a1, a2, a3).\n"
                        )
                        f.write(
                            "# Right basis: SeeK-path standardized primitive "
                            "real-space fractional basis (a1, a2, a3).\n"
                        )
                        f.write(
                            "# k mapping: k' = R^(-T) k (mod G) in each "
                            "corresponding reciprocal basis (b1, b2, b3).\n"
                        )
                        f.write(f"# Found {len(input_ops)} {label} point operations\n")
                        for i, (input_op, std_op) in enumerate(zip(input_ops, standardized_ops), 1):
                            f.write(f"Operation_{i}\n")
                            input_int = np.rint(np.array(input_op, dtype=float)).astype(int)
                            std_int = np.rint(np.array(std_op, dtype=float)).astype(int)
                            for input_row, std_row in zip(input_int, std_int):
                                left = " ".join(str(int(x)) for x in input_row)
                                right = " ".join(str(int(x)) for x in std_row)
                                f.write(f"{left}    |    {right}\n")
                            f.write("\n")
                except Exception as exc:
                    print(f"[Warning] Could not write {filename}: {exc}")

            operation_basis_changed = (
                not np.allclose(R, R_for_kpts)
                or any(
                    not np.allclose(input_op, standard_op)
                    for input_op, standard_op in zip(
                        flip_ops, flip_ops_for_plot
                    )
                )
                or any(
                    not np.allclose(input_op, standard_op)
                    for input_op, standard_op in zip(
                        preserve_ops, preserve_ops_for_plot
                    )
                )
            )
            if operation_basis_changed:
                _annotate_ops_with_standardized_basis(
                    os.path.join(OUTPUT_DIR, "spin_flip_operations.txt"),
                    output_flip_ops,
                    output_flip_ops_standardized,
                    "spin-flipping",
                )
                _annotate_ops_with_standardized_basis(
                    os.path.join(OUTPUT_DIR, "spin_preserve_operations.txt"),
                    output_preserve_ops,
                    output_preserve_ops_standardized,
                    "spin-preserving",
                )
        else:
            R_for_kpts = R
        return R_for_kpts, R_cart_for_plot, flip_ops_for_plot, preserve_ops_for_plot

    def interactive_build(self):
        """Run the interactive session, from the structure to the written k-path.

        Steps 0-5: spin symmetry, high-symmetry path, general k point,
        spin-flip operation, path construction, and output.
        """
        BOLD  = "\033[1m"
        RESET = "\033[0m"
        print(f"=== AlterSeeK-Path {__version__} ===")

        try:
            input_config = _read_input_config()
        except ValueError as exc:
            print(f"[Error] {exc}")
            return False
        view_elev = float(input_config['view_elev']) if 'view_elev' in input_config else None
        view_azim = float(input_config['view_azim']) if 'view_azim' in input_config else None
        symprec = float(input_config['symprec']) if 'symprec' in input_config else None
        save_pdf = bool(input_config.get('save_pdf', False))
        # The command line wins when it was given explicitly; otherwise the toml setting applies, matching how the other settings resolve.
        if 'vacuum_axis' in input_config and not self._vacuum_axis_from_cli:
            self.input_vacuum_axis = _VACUUM_AXIS_INDEX[
                str(input_config['vacuum_axis']).strip().lower()
            ]

        def _ask(prompt_text, key):
            print(prompt_text, end='', flush=True)
            if key in input_config:
                val = str(input_config[key]).strip()
                print(val)
                return val
            return read_answer().strip()

        def _choose_output_code():
            while True:
                choice = _ask("Output code ([vasp]/qe/abinit): ", "output_code").lower() or "vasp"
                if choice in {"vasp", "qe", "abinit"}:
                    return choice
                print("Invalid output code. Enter 'vasp', 'qe', 'abinit', or press Enter for VASP.")

        # --- Step 0: Compute spin-flip operations from structure ---
        print(f"\n{BOLD}>>> Step 0: Spin symmetry{RESET}")
        if self.mode_2d:
            print(
                "[2D mode] input vacuum axis is set to "
                f"'{'abc'[self.input_vacuum_axis]}'"
            )
        struct_file = _ask(
            "Enter structure file (default: POSCAR, supports .vasp/.cif/.mcif): ",
            "structure",
        )
        if not struct_file: struct_file = "POSCAR"

        # Track operation-log ownership separately because a magnetic non-altermagnet can write a log with zero spin-flip operations.
        _step0_wrote_operation_log = False
        # None = Step 0 not run; True = file freshly written; False = ran but no flip ops found
        _step0_wrote_flip_file = None
        standard_path_reason = None
        standard_path_reason_reported = False
        centroid_result = None
        centroid_struct_file = struct_file
        centroid_seekpath_type_numbers = None
        operation_basis_label = (
            f"submitted structure '{os.path.basename(struct_file)}'"
        )
        submitted_lattice_for_2d = None
        display_figures = []
        self.extra_general_points = []
        analysis_preparation = None

        def _save_and_finish(path_points, R_matrix, R_label):
            """Step 5: choose the output code, write the path file (plus
            its band-plot config), and show any deferred figures."""
            print(f"\n{BOLD}>>> Step 5: Save{RESET}")
            code_choice = _choose_output_code()
            if code_choice == "qe":
                write_ok = self.write_kpoints_file_qe(
                    path_points, "KPOINTS_alter_qe", R_matrix, R_label,
                    operation_basis_label=operation_basis_label,
                )
                if write_ok:
                    write_qe_bandplot_config()
            elif code_choice == "abinit":
                write_ok = self.write_kpoints_file_abinit(
                    path_points, "KPOINTS_alter_abinit", R_matrix, R_label,
                    operation_basis_label=operation_basis_label,
                )
                if write_ok:
                    write_abinit_bandplot_config(struct_file)
            else:
                write_ok = self.write_kpoints_file_vasp(
                    path_points, "KPOINTS_alter", R_matrix, R_label,
                    operation_basis_label=operation_basis_label,
                )
                if write_ok and centroid_result is not None:
                    write_bandplot_lattice_config(
                        centroid_result.get('sc_type')
                        if centroid_result.get('path_source_2d')
                        else centroid_result.get(
                            'lattice_key', centroid_result.get('sc_type')
                        )
                    )
            if not write_ok:
                print("[Error] KPOINTS output was not written.")
                return False
            print("\nDone.")
            if display_figures:
                _display_and_save_figures(display_figures)
            return True

        if not os.path.exists(struct_file):
            print(f"[Error] Structure file '{struct_file}' not found. Aborting "
                  "(not falling back to a possibly stale spin_flip_operations.txt).")
            return False
        else:
            is_mcif = struct_file.lower().endswith('.mcif')
            spin_axis_cart = None
            sf_result = None
            if is_mcif:
                print("Detected .mcif file --magnetic moments will be read from file.")
                moments_str = ""
            else:
                spin_axis_cart = _ask(
                    "Spin axis in Cartesian coordinates (default: 0 0 1): ", "spin_axis"
                )
                moments_str = _ask(
                    "Magnetic moments along this axis (atom order, trailing atoms auto-fill to 0): ",
                    "moments",
                )
            if not is_mcif and not moments_str:
                standard_path_reason = "No magnetic moments entered."
                standard_path_reason_reported = True
                _step0_wrote_flip_file = False
            else:
                try:
                    sf_result = find_sf_run(
                        struct_file,
                        moments_str,
                        verbose=False,
                        spin_axis_cart=spin_axis_cart,
                        symprec=symprec,
                        output_dir=OUTPUT_DIR,
                    )
                    _step0_wrote_operation_log = True
                except SpinSymmetryError as e:
                    print(f"[Error] Spin-symmetry analysis failed: {e} Aborting.")
                    return False
            try:
                analysis_preparation = prepare_submitted_cell_analysis(
                    struct_file,
                    moments_str=moments_str,
                    spin_axis_cart=spin_axis_cart,
                    output_dir=OUTPUT_DIR,
                    symprec=(1e-3 if symprec is None else symprec),
                    write_magnetic_diagnostic=sf_result is not None,
                    input_vacuum_axis=(
                        self.input_vacuum_axis if self.mode_2d else None
                    ),
                )
                submitted_lattice_for_2d = analysis_preparation[
                    "submitted_lattice"
                ]
            except Exception as exc:
                print(
                    "[Error] Submitted-cell analysis helper construction "
                    f"failed: {exc} Aborting."
                )
                return False

            # This is either the result dictionary from the magnetic branch or None because the no-moments branch never ran spin analysis.
            if sf_result is not None:
                working_cell_symmetry = _g0_symmetry(
                    sf_result,
                    sites=analysis_preparation.get("magnetic_primitive_sites"),
                )
                try:
                    centroid_result = compute_centroid(
                        centroid_struct_file, output_dir=OUTPUT_DIR, show_plot=True,
                        defer_show=True, verbose=False,
                        seekpath_type_numbers=centroid_seekpath_type_numbers,
                        mode_2d=self.mode_2d,
                        input_vacuum_axis=self.input_vacuum_axis,
                        view_elev=view_elev, view_azim=view_azim, symprec=symprec,
                        figure_basename=_figure_basename(struct_file),
                        save_pdf=save_pdf,
                        analysis_cell=analysis_preparation["analysis_cell"],
                        analysis_has_markers=analysis_preparation[
                            "analysis_has_markers"
                        ],
                    )
                except Exception as e:
                    print(
                        "[Error] IBZ centroid construction failed: "
                        f"{e} Aborting."
                    )
                    return False
                centroid_result["b_matrix_output"] = centroid_result[
                    "b_matrix_input"
                ]
                centroid_result["b_matrix_submitted"] = centroid_result[
                    "b_matrix_input"
                ]
                display_figures.extend(centroid_result.get('display_figures', []))
                _step0_wrote_flip_file = sf_result.get('spin_flip_operations', 0) > 0
                gate_laue_group = (working_cell_symmetry or sf_result).get('laue_group')
                laue_no_altermag = _altermagnetism_gate(sf_result, working_cell_symmetry)
                spin_split_diagnostic = sf_result.get('spin_split_diagnostic', '')

                parent_recovery = None
                lattice_tag = None
                if centroid_result is not None:
                    parent_recovery = centroid_result.get("mcif_parent_recovery")
                    lattice_tag = centroid_result.get(
                        'sc_type', centroid_result.get('seekpath_bravais', 'unknown'))
                print(f"\nInput structure: {sf_result['structure_file']}, "
                      f"{sf_result['num_atoms']} atoms")
                if not self.mode_2d and analysis_preparation.get(
                    "uses_conventional_supercell_bz", False
                ):
                    primitive_count = analysis_preparation["summary"][
                        "submitted_to_primitive_volume_index"
                    ]
                    print(
                        "Conventional/supercell detected: input contains "
                        f"{primitive_count} magnetic primitive cells"
                    )
                input_cell_symmetry = analysis_preparation.get(
                    "input_cell_symmetry"
                ) or analysis_preparation.get(
                    "physical_symmetry"
                ) or analysis_preparation["analysis_symmetry"]
                cell_rows = [(
                    'Input cell:',
                    f"{input_cell_symmetry['symbol']} "
                    f"({input_cell_symmetry['number']})",
                    input_cell_symmetry['point_group'],
                    laue_group_from_spacegroup_number(
                        input_cell_symmetry['number']
                    ) or "Unknown",
                    _cell_suffix(
                        sf_result.get('num_atoms'),
                        None if self.mode_2d
                        else input_cell_symmetry.get('seekpath_bravais'),
                    ),
                ), (
                    'Nonmagnetic primitive cell:',
                    sf_result['space_group'],
                    sf_result['point_group'],
                    sf_result['laue_group'],
                    _cell_suffix(
                        sf_result.get('nonmagnetic_sites'),
                        None if self.mode_2d
                        else sf_result.get('nonmagnetic_lattice'),
                    ),
                )]
                reported_g0_symmetry = working_cell_symmetry
                if reported_g0_symmetry is not None:
                    cell_rows.append((
                        'Magnetic primitive cell:',
                        reported_g0_symmetry['label'],
                        reported_g0_symmetry['point_group'],
                        reported_g0_symmetry['laue_group'],
                        _cell_suffix(
                            reported_g0_symmetry.get('sites'),
                            None if self.mode_2d
                            else analysis_preparation.get(
                                'magnetic_primitive_lattice_tag'
                            ),
                        ),
                    ))
                recovery_note = None
                if parent_recovery:
                    recovery_note = (
                        "recovered from the input cell at symprec="
                        f"{parent_recovery['symprec']:g} (index {parent_recovery['index']})")
                if self.mode_2d:
                    _print_2d_structure_summary(
                        analysis_preparation, centroid_result
                    )
                else:
                    _print_cell_rows(
                        cell_rows,
                        note=recovery_note,
                        note_after_index=1,
                    )
                if not self.mode_2d and analysis_preparation.get(
                    "uses_conventional_supercell_bz", False
                ):
                    print(
                        f"{'Conventional/supercell BZ:':<{_CELL_LABEL_WIDTH}}"
                        f"{lattice_tag}"
                    )
                print(f"Phase: {sf_result['magnetic_phase']}")
                print(f"Oriented SSG: {sf_result['ssg_index']}")
                print(f"SSG Symbol (Chen-Liu): {sf_result['ssg_symbol']}")
                print(f"MSG without SOC: {sf_result['magnetic_space_group_without_soc']}")

                if laue_no_altermag:
                    laue = laue_no_altermag.get('laue_group', gate_laue_group)
                    standard_path_reason = f"Laue group {laue}: no altermagnetism."
                    print(f"{BOLD}[Note] {standard_path_reason}{RESET} Default path will be written.")
                    standard_path_reason_reported = True
                else:
                    print("Spin operations: "
                          f"{sf_result['actual_spin_flip_point_operations']} flip, "
                          f"{sf_result['actual_spin_preserve_point_operations']} preserve")
                    if spin_split_diagnostic:
                        standard_path_reason = spin_split_diagnostic
                        print(f"{BOLD}[Note] {standard_path_reason}{RESET} Default path will be written.")
                        standard_path_reason_reported = True

                if STEP0_VERBOSE_SUMMARY:
                    print(f"Magnetic SG: {sf_result['magnetic_space_group']}")
                    print(f"G0: {sf_result['g0_symbol']} ({sf_result['g0_number']}), "
                          f"L0: {sf_result['l0_symbol']} ({sf_result['l0_number']}), "
                          f"EMPG: {sf_result['empg']}")
                    if sf_result.get('findspingroup_warning'):
                        print(f"FindSpinGroup warning: {sf_result['findspingroup_warning']}")
                    print(f"Spin axis: {sf_result['spin_group']}")
                    unique_ops = sf_result.get('unique_point_operations',
                                               sf_result['total_operations'])
                    print(f"Space-group operations: {sf_result['total_operations']} total")
                    print(f"Point operations: {unique_ops} unique")
                    print("Inversion-extended k operations: "
                          f"{sf_result['extended_spin_flip_point_operations']} spin-flip, "
                          f"{sf_result['extended_spin_preserve_point_operations']} spin-preserving "
                          f"({sf_result['extended_spin_flip_operations']} + "
                          f"{sf_result['extended_spin_preserve_operations']} with translations)")
                    print(f"Saved: {', '.join(sf_result['saved_files'])}")

        # A path file cannot replace centroid analysis because the same analysis also establishes the reciprocal-basis mappings required for correct output.
        if centroid_result is None:
            try:
                centroid_result = compute_centroid(
                    centroid_struct_file, output_dir=OUTPUT_DIR, show_plot=True,
                    defer_show=True, verbose=False,
                    seekpath_type_numbers=centroid_seekpath_type_numbers,
                    mode_2d=self.mode_2d,
                    input_vacuum_axis=self.input_vacuum_axis,
                    view_elev=view_elev, view_azim=view_azim, symprec=symprec,
                    figure_basename=_figure_basename(struct_file),
                    save_pdf=save_pdf,
                    analysis_cell=analysis_preparation["analysis_cell"],
                    analysis_has_markers=analysis_preparation[
                        "analysis_has_markers"
                    ],
                )
            except Exception as exc:
                print(
                    "[Error] IBZ centroid construction failed: "
                    f"{exc} Aborting."
                )
                return False
            centroid_result["b_matrix_output"] = centroid_result[
                "b_matrix_input"
            ]
            centroid_result["b_matrix_submitted"] = centroid_result[
                "b_matrix_input"
            ]
            display_figures.extend(centroid_result.get('display_figures', []))
            lattice_tag = centroid_result.get(
                'sc_type', centroid_result.get('seekpath_bravais', 'unknown')
            )
            print(
                f"\nInput structure: {os.path.basename(struct_file)}, "
                f"{analysis_preparation['submitted_sites']} atoms"
            )
            if not self.mode_2d and analysis_preparation.get(
                "uses_conventional_supercell_bz", False
            ):
                primitive_count = analysis_preparation["summary"][
                    "submitted_to_primitive_volume_index"
                ]
                print(
                    "Conventional/supercell detected: input contains "
                    f"{primitive_count} nonmagnetic primitive cells"
                )
            input_cell_symmetry = analysis_preparation["input_cell_symmetry"]
            primitive_symmetry = analysis_preparation[
                "nonmagnetic_primitive_symmetry"
            ]
            structural_rows = [(
                'Input cell:',
                f"{input_cell_symmetry['symbol']} "
                f"({input_cell_symmetry['number']})",
                input_cell_symmetry['point_group'],
                laue_group_from_spacegroup_number(
                    input_cell_symmetry['number']
                ) or "Unknown",
                _cell_suffix(
                    analysis_preparation['submitted_sites'],
                    None if self.mode_2d
                    else input_cell_symmetry.get('seekpath_bravais'),
                ),
            ), (
                'Nonmagnetic primitive cell:',
                f"{primitive_symmetry['symbol']} "
                f"({primitive_symmetry['number']})",
                primitive_symmetry['point_group'],
                laue_group_from_spacegroup_number(
                    primitive_symmetry['number']
                ) or "Unknown",
                _cell_suffix(
                    primitive_symmetry['sites'],
                    None if self.mode_2d
                    else primitive_symmetry.get('seekpath_bravais'),
                ),
            )]
            if self.mode_2d:
                _print_2d_structure_summary(
                    analysis_preparation, centroid_result
                )
            else:
                _print_cell_rows(structural_rows)
            if not self.mode_2d and analysis_preparation.get(
                "uses_conventional_supercell_bz", False
            ):
                print(
                    f"{'Conventional/supercell BZ:':<{_CELL_LABEL_WIDTH}}"
                    f"{lattice_tag}"
                )
            print(
                f"{BOLD}[Note] {standard_path_reason}{RESET} Ordinary "
                "structural path will be written."
            )
        if self.mode_2d:
            try:
                self._configure_2d_plane(
                    centroid_result,
                    submitted_lattice=submitted_lattice_for_2d,
                )
            except Exception as exc:
                print(
                    "[Error] Could not establish the physical 2D slab "
                    f"plane: {exc} Aborting."
                )
                return False

        # --- Step 1: Read the high-symmetry path from the centroid analysis ---
        print(f"\n{BOLD}>>> Step 1: High-symmetry k-path{RESET}")
        sp_path   = centroid_result['sp_path']
        sp_coords = centroid_result['sp_point_coords']
        displayed_path = centroid_result.get(
            'band_kpath',
            centroid_result.get('ibz_kpath', sp_path)
        )
        print(f"Path: {self._format_path(displayed_path)}")
        # Build kpoints_data in Figure 1's HPKOT/SeeK-path convention.
        # Curated closure vertices stay in the centroid hull, while path-only labels such as H_2 remain available without entering that hull.
        self.kpoints_data = []
        sc_type_auto = centroid_result.get('sc_type', '')
        if (
            ('band_kpath' in centroid_result and 'band_kpoints_frac' in centroid_result)
            or ('ibz_kpath' in centroid_result and 'ibz_kpoints_frac' in centroid_result)
        ):
            path_source = (
                f"2D {sc_type_auto}"
                if centroid_result.get('path_source_2d')
                else f"HPKOT {sc_type_auto}"
            )
            self.header_lines = [
                f'K-Path generated by AlterSeeK-Path ({path_source})',
                '20', 'Line-Mode', 'Reciprocal'
            ]
            self.kpoints_basis_matrix = np.array(
                centroid_result['b_matrix'], dtype=float
            )
            self.kpoints_basis_rotation = np.array(
                centroid_result.get(
                    'seekpath_rotation_matrix', np.eye(3)
                ),
                dtype=float,
            )
            self.output_basis_matrix = np.array(
                centroid_result.get(
                    'b_matrix_output',
                    centroid_result.get('b_matrix_input', centroid_result['b_matrix']),
                ),
                dtype=float,
            )
            # Prefer the selected band path so the prompt, Figure 1, and KPOINTS remain consistent.
            auto_path = centroid_result.get(
                'band_kpath',
                centroid_result['ibz_kpath']
            )
            ibz_coords = centroid_result.get(
                'band_kpoints_frac',
                centroid_result.get(
                    'path_kpoints_frac',
                    centroid_result['ibz_kpoints_frac']
                )
            )
            for seg_start, seg_end in auto_path:
                for label in (seg_start, seg_end):
                    coords = ibz_coords[label]
                    self.kpoints_data.append([coords[0], coords[1], coords[2], label])
            extra_vertices = centroid_result.get('extra_general_vertices', [])
            self.extra_general_points = []
            for label in extra_vertices:
                if label in ibz_coords:
                    coords = ibz_coords[label]
                    self.extra_general_points.append([coords[0], coords[1], coords[2], label])
            if self.mode_2d and centroid_result.get('path_source_2d'):
                print(
                    f"Using 2D {sc_type_auto} path "
                    f"({len(auto_path)} segments, {len(self.kpoints_data)} k-points)"
                )
            else:
                print(
                    f"Using HPKOT {sc_type_auto} path "
                    f"({len(auto_path)} segments, {len(self.kpoints_data)} k-points)"
                )
            if self.extra_general_points:
                labels = ", ".join(str(pt[3]) for pt in self.extra_general_points)
                print(f"Extra doubled-IBZ general-k: {labels}")

            butterfly_path = centroid_result.get('butterfly_kpath')
            butterfly_extra = centroid_result.get('butterfly_extra_vertices')
            self.butterfly_kpoints_data = None
            self.butterfly_extra_general_points = None
            if butterfly_path is not None:
                self.butterfly_kpoints_data = []
                for seg_start, seg_end in butterfly_path:
                    for label in (seg_start, seg_end):
                        coords = ibz_coords[label]
                        self.butterfly_kpoints_data.append([
                            coords[0], coords[1], coords[2], label
                        ])
                self.butterfly_extra_general_points = []
                for label in butterfly_extra or []:
                    coords = ibz_coords[label]
                    self.butterfly_extra_general_points.append([
                        coords[0], coords[1], coords[2], label
                    ])
                print(
                    "Using 2D 4/m specific path "
                    f"{self._format_path(butterfly_path)}"
                )
        else:
            self.header_lines = ['K-Path generated by AlterSeeK-Path (seekpath)', '20', 'Line-Mode', 'Reciprocal']
            self.kpoints_basis_matrix = np.array(
                centroid_result['b_matrix'], dtype=float
            )
            self.kpoints_basis_rotation = np.array(
                centroid_result.get(
                    'seekpath_rotation_matrix', np.eye(3)
                ),
                dtype=float,
            )
            self.output_basis_matrix = np.array(
                centroid_result.get(
                    'b_matrix_output',
                    centroid_result.get('b_matrix_input', centroid_result['b_matrix']),
                ),
                dtype=float,
            )
            for seg_start, seg_end in sp_path:
                for label in (seg_start, seg_end):
                    coords = sp_coords[label]
                    self.kpoints_data.append([coords[0], coords[1], coords[2], label])
            print(f"Using auto-generated path ({len(sp_path)} segments, {len(self.kpoints_data)} k-points)")

        # Laue groups -1, -3, and m-3 have no one-dimensional, nonidentical inversion-even irrep, so they cannot support altermagnetic splitting.
        # Use the ordinary IBZ path without butterfly insertion.
        no_altermag = None
        if centroid_result is not None:
            no_altermag = centroid_result.get('no_altermagnetism')
            if no_altermag is None:
                no_altermag = no_altermagnetism_reason(
                    centroid_result.get('point_group'),
                    centroid_result.get('spacegroup'),
                )
        if standard_path_reason is None and no_altermag:
            laue = no_altermag.get('laue_group', 'unknown')
            standard_path_reason = f"Laue group {laue}: no altermagnetism."

        # --- Step 2: Use the automatically computed IBZ centroid ---
        print(f"\n{BOLD}>>> Step 2: General k-point{RESET}")
        if centroid_result is None or centroid_result.get('centroid_frac') is None:
            print("[Error] IBZ centroid is unavailable. Aborting.")
            return False
        c = centroid_result['centroid_frac']
        general_kpoint = [c[0], c[1], c[2]]
        # Append the centroid only to a log owned by this run to avoid creating a centroid-only file or modifying a stale log.
        out_k = self._general_kpoint_output_basis(general_kpoint)
        path_source_2d = bool(centroid_result.get('path_source_2d'))
        if not path_source_2d:
            print(
                "IBZ centroid (standardized basis): "
                f"[{c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f}]"
            )
        input_cell_k = general_kpoint if path_source_2d else out_k
        if input_cell_k is not None:
            print(
                "IBZ centroid (input-cell basis): "
                f"[{input_cell_k[0]:.6f}, {input_cell_k[1]:.6f}, "
                f"{input_cell_k[2]:.6f}]"
            )
        if _step0_wrote_operation_log:
            try:
                with open(os.path.join(OUTPUT_DIR, "spin_operations.txt"),
                          "a", encoding="utf-8", newline="\n") as f:
                    if not path_source_2d:
                        f.write(
                            "\nGeneral k-point (IBZ centroid, standardized basis): "
                            f"[{general_kpoint[0]:.6f}, "
                            f"{general_kpoint[1]:.6f}, "
                            f"{general_kpoint[2]:.6f}]\n"
                        )
                    else:
                        f.write("\n")
                    if input_cell_k is not None:
                        f.write(
                            "General k-point (IBZ centroid, input-cell basis): "
                            f"[{input_cell_k[0]:.6f}, {input_cell_k[1]:.6f}, "
                            f"{input_cell_k[2]:.6f}]\n"
                        )
            except OSError as exc:
                print("[Warning] Could not append the general k-point to "
                      f"spin_operations.txt: {exc}")

        def _write_ordinary_path_and_stop(reason, reason_reported):
            if not reason_reported:
                print(f"\n{BOLD}[Note] {reason}{RESET}")
            print("Writing ordinary k-path with general-k.")
            standard_general_path = self.build_ordinary_path_with_general_k(
                general_kpoint,
                self.extra_general_points,
            )
            # Every route to this exit is a non-altermagnet: no moments,
            # Laue -1/-3/m-3, PT, Ut, no spin-flip point operation, or
            # the 2D in-plane +-I rule.  The helper picks the 2D or 3D
            # plate.
            try:
                self._generate_non_altermagnetic_figure(
                    centroid_result, struct_file, general_kpoint,
                    standard_general_path, display_figures,
                    save_pdf=save_pdf,
                )
            except Exception as exc:
                print(
                    "[Warning] Could not generate general-path figure: "
                    f"{exc}"
                )
            return _save_and_finish(standard_general_path, None, None)

        if standard_path_reason:
            return _write_ordinary_path_and_stop(
                standard_path_reason, standard_path_reason_reported
            )

        # --- Step 3: Select a detected spin-flip operation ---
        print(f"\n{BOLD}>>> Step 3: Spin-flip operation{RESET}")
        if _step0_wrote_flip_file is False:
            # Step 0 ran but found no spin-flip operations for this structure, so do not load a stale file from a previous run.
            print("[Note] Step 0 found no spin-flip operations --skipping any existing spin_flip_operations.txt.")
            flip_ops = []
        else:
            flip_ops = self.load_flip_operations()
        preserve_ops = self.load_preserve_operations()
        output_flip_ops = list(flip_ops)
        output_preserve_ops = list(preserve_ops)

        # Add inversion partners because inversion changes only the spatial operation, not whether the spin operation flips or preserves spin.
        def _inversion_extended(ops):
            expanded = list(ops)
            for op in ops:
                neg_op = -np.array(op, dtype=float)
                if not any(np.allclose(neg_op, ex, atol=1e-8) for ex in expanded):
                    expanded.append(neg_op)
            return expanded

        flip_ops = _inversion_extended(flip_ops)
        preserve_ops = _inversion_extended(preserve_ops)

        # Test in Cartesian reciprocal space because the magnetic-cell fractional basis may have reordered axes.
        # An in-plane action of +I or -I rules out 2D altermagnetic spin splitting; keep only other nontrivial plane-preserving flips.
        _flip_ops_emptied_2d = False
        if self.mode_2d and flip_ops:
            try:
                degeneracy_forcing_ops = [
                    op for op in flip_ops
                    if self._forces_2d_degeneracy(op, centroid_result)
                ]
                valid_flip_ops = [
                    op for op in flip_ops
                    if self._is_valid_2d_operation(op, centroid_result)
                ]
            except Exception as exc:
                print(f"[Error] 2D spin-operation filtering failed: {exc}")
                return False
            if degeneracy_forcing_ops:
                print("[2D mode] C_2z T / U m_z symmetry detected, "
                      "not a 2D altermagnet.")
                flip_ops = []
                _flip_ops_emptied_2d = True
            elif valid_flip_ops:
                flip_ops = valid_flip_ops
            else:
                print("[2D mode] No spin-flip operation, "
                      "not a 2D altermagnet.")
                flip_ops = []
                _flip_ops_emptied_2d = True

        if not flip_ops:
            if _flip_ops_emptied_2d:
                return _write_ordinary_path_and_stop(
                    "No in-plane spin-flip point operation available: not a "
                    "2D altermagnet.",
                    True,
                )
            # In 3D, reaching this point without flip operations indicates missing or inconsistent operation output, so abort instead of writing an ordinary path.
            print(
                "[Error] Spin-symmetry analysis reached the general-k path, "
                "but no detected spin-flip point operation is available. "
                "The symmetry result or operation output is inconsistent. "
                "Aborting."
            )
            return False

        preset_flip_choice = input_config.get('flip_option')
        if preset_flip_choice is not None and preset_flip_choice > len(flip_ops):
            print(
                f"[Error] flip_option={preset_flip_choice} is out of range; "
                f"available choices are 1-{len(flip_ops)}."
            )
            return False

        # Classify operation names in Cartesian space because source-basis matrix axes may differ from the axes shown in the figures.
        R, selected_transformation_label = self._select_spin_flip_operation(
            flip_ops,
            centroid_result,
            preset_choice=preset_flip_choice,
            operation_basis_label=operation_basis_label,
        )
        if self.mode_2d:
            try:
                selected_is_valid_2d = self._is_valid_2d_operation(
                    R, centroid_result
                )
            except Exception as exc:
                print(f"[Error] Selected 2D spin operation could not be checked: {exc}")
                return False
            if not selected_is_valid_2d:
                print(
                    "[Error] Selected spin-flip operation does not preserve the "
                    "physical slab plane or is trivial within it. Aborting."
                )
                return False

        # --- Step 4: Process k-points ---
        print(f"\n{BOLD}>>> Step 4: Build general-k path{RESET}")

        (R_for_kpts, R_cart_for_plot, flip_ops_for_plot,
         preserve_ops_for_plot) = self._convert_operation_to_primitive_basis(
            R,
            flip_ops,
            preserve_ops,
            centroid_result,
            operation_basis_label,
            output_flip_ops,
            output_preserve_ops,
        )
        R_for_output = R_for_kpts

        # Report the partner that will actually be written to KPOINTS.
        k_prime = self.transform_kpoint(general_kpoint, R_for_output)
        print(f"k' = [{k_prime[0]:.4f}, {k_prime[1]:.4f}, {k_prime[2]:.4f}]")

        butterfly_path_points = (
            self.butterfly_kpoints_data
            if self.butterfly_kpoints_data is not None
            else self.kpoints_data
        )
        butterfly_extra_points = (
            self.butterfly_extra_general_points
            if self.butterfly_extra_general_points is not None
            else self.extra_general_points
        )
        figure_kpoints = self.insert_general_kpoints(
            general_kpoint,
            R_for_kpts,
            butterfly_extra_points,
            path_points=butterfly_path_points,
        )
        if not figure_kpoints:
            print("[Error] Failed to build a nonempty general-k path.")
            return False
        output_kpoints = figure_kpoints

        try:
            self._generate_spin_figures(
                centroid_result, struct_file, general_kpoint, R_for_kpts,
                R_cart_for_plot, flip_ops_for_plot, preserve_ops_for_plot,
                figure_kpoints, display_figures, save_pdf=save_pdf)
        except Exception as exc:
            print(f"[Warning] Could not generate spin figures: {exc}")

        # --- Step 5: Save modified file ---
        return _save_and_finish(
            output_kpoints, R, selected_transformation_label
        )
