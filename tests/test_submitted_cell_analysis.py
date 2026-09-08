"""The submitted translation lattice defines the conventional/supercell BZ."""

import itertools
from pathlib import Path

import numpy as np
import spglib
from ase import Atoms
from ase.build import bulk, make_supercell
from ase.io import read, write
from scipy.spatial import ConvexHull

import alterseek.submitted_cell_analysis as submitted_cell_analysis
from alterseek.compute_centroid_3d import run as compute_centroid
from alterseek.submitted_cell_analysis import prepare_submitted_cell_analysis
from alterseek.submitted_cell_analysis import _build_nonprimitive_bz_marker_cell


REFERENCES = Path(__file__).parent / "references"


def test_magnetic_primitive_tag_uses_g0_despite_tetragonal_metric(tmp_path):
    preparation = prepare_submitted_cell_analysis(
        str(REFERENCES / "La2NiO4.mcif"),
        output_dir=str(tmp_path),
        write_magnetic_diagnostic=True,
    )

    lengths = np.linalg.norm(preparation["magnetic_primitive_lattice"], axis=1)
    assert np.isclose(lengths[0], lengths[1])
    assert preparation["magnetic_primitive_sites"] == 28
    assert preparation["physical_symmetry"]["number"] == 56
    assert preparation["magnetic_primitive_lattice_tag"] == "oP1"
    assert preparation["input_cell_symmetry"]["seekpath_bravais"] == "oP1"
    assert preparation["bz_helper_symmetry"]["seekpath_bravais"] == "oP1"


def _assert_ibz_volume_matches_k_group(result):
    bz_volume = abs(float(np.linalg.det(result["b_matrix"])))
    assert np.isclose(
        result["ibz_volume"] * result["n_symmetry_ops"],
        bz_volume,
        rtol=1e-7,
        atol=1e-10,
    )


def _signed_permutation_operations():
    operations = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            rotation = np.zeros((3, 3), dtype=int)
            for row, column in enumerate(permutation):
                rotation[row, column] = signs[row]
            operations.append(rotation)
    return operations


def _orthorhombic_operations():
    return [
        np.diag(signs).astype(int)
        for signs in itertools.product((-1, 1), repeat=3)
    ]


def _change_operation_basis(rotation, source_lattice, target_lattice):
    cartesian = source_lattice.T @ rotation @ np.linalg.inv(source_lattice.T)
    transformed = (
        np.linalg.inv(target_lattice.T) @ cartesian @ target_lattice.T
    )
    return np.rint(transformed).astype(int)


def test_primitive_input_is_not_reduced():
    lattice = np.diag([4.0, 5.0, 6.0])
    result = _build_nonprimitive_bz_marker_cell(
        lattice,
        real_type_numbers=[26],
        space_operations=_orthorhombic_operations(),
    )

    assert np.allclose(result["cell"][0], lattice)
    assert result["volume_original_wrt_prim"] == 1.0
    assert result["intended_point_operation_count"] == 8
    assert result["detected_point_operation_count"] == 8


def test_true_222_supercell_keeps_its_native_bz():
    lattice = np.diag([8.0, 8.0, 8.0])
    result = _build_nonprimitive_bz_marker_cell(
        lattice,
        real_type_numbers=[26] * 8,
        space_operations=_signed_permutation_operations(),
    )

    assert np.allclose(result["cell"][0], lattice)
    assert result["volume_original_wrt_prim"] == 1.0
    assert result["seekpath_bravais"] == "cP2"
    assert {"GAMMA", "X", "M", "R"}.issubset(result["point_coords"])
    expected_bz_volume = (2 * np.pi) ** 3 / abs(np.linalg.det(lattice))
    assert np.isclose(result["submitted_bz_volume"], expected_bz_volume)


def test_cubic_221_supercell_has_tetragonal_bz_helper():
    result = _build_nonprimitive_bz_marker_cell(
        np.diag([8.0, 8.0, 4.0]),
        real_type_numbers=[26],
        space_operations=_signed_permutation_operations(),
    )

    assert result["analysis_spacegroup_symbol"] == "P4/mmm"
    assert result["seekpath_bravais"].startswith("tP")
    assert result["intended_point_operation_count"] == 16
    assert result["source_space_operation_count"] == 48
    assert result["compatible_space_operation_count"] == 16
    assert result["volume_original_wrt_prim"] == 1.0


def test_cubic_321_supercell_has_orthorhombic_bz_helper():
    result = _build_nonprimitive_bz_marker_cell(
        np.diag([12.0, 8.0, 4.0]),
        real_type_numbers=[26],
        space_operations=_signed_permutation_operations(),
    )

    assert result["analysis_spacegroup_symbol"] == "Pmmm"
    assert result["seekpath_bravais"] == "oP1"
    assert result["intended_point_operation_count"] == 8
    assert result["compatible_space_operation_count"] == 8


def test_fd3m_uses_closed_pure_rotation_helper_in_conventional_cube():
    fd3m = spglib.get_symmetry_from_database(525)
    physical_operations = [
        {"real_rotation": rotation, "translation": translation}
        for rotation, translation in zip(
            fd3m["rotations"], fd3m["translations"]
        )
    ]
    result = _build_nonprimitive_bz_marker_cell(
        np.eye(3) * 8.0,
        real_type_numbers=[14],
        space_operations=physical_operations,
    )

    assert len(physical_operations) == 192
    assert result["source_space_operation_count"] == 192
    assert result["intended_point_operation_count"] == 48
    assert result["analysis_spacegroup_symbol"] == "Pm-3m"
    assert result["seekpath_bravais"] == "cP2"
    assert result["volume_original_wrt_prim"] == 1.0


def test_conventional_diamond_keeps_fd3m_but_uses_cP_bz(tmp_path):
    structure = bulk("Si", "diamond", a=5.43, cubic=True)
    path = tmp_path / "Si_diamond_conventional.vasp"
    write(path, structure, format="vasp", direct=True, vasp5=True)

    preparation = prepare_submitted_cell_analysis(str(path), symprec=1e-5)
    result = compute_centroid(
        str(path),
        output_dir=str(tmp_path),
        show_plot=False,
        verbose=False,
        analysis_cell=preparation["analysis_cell"],
        analysis_has_markers=preparation["analysis_has_markers"],
        symprec=1e-5,
    )

    assert preparation["physical_symmetry"]["number"] == 227
    assert preparation["physical_symmetry"]["symbol"] == "Fd-3m"
    assert preparation["bz_helper_symmetry"]["number"] == 221
    assert preparation["bz_helper_symmetry"]["symbol"] == "Pm-3m"
    assert preparation["bz_helper_symmetry"]["seekpath_bravais"] == "cP2"
    assert preparation["summary"]["physical_space_operations"] == 192
    assert preparation["summary"]["intended_point_operations"] == 48
    assert preparation["summary"]["volume_original_wrt_prim"] == 1.0
    assert result["sc_type"] == "cP2"
    _assert_ibz_volume_matches_k_group(result)


def test_determinant_one_basis_change_keeps_submitted_lattice():
    orthorhombic = np.diag([4.0, 5.0, 6.0])
    transform = np.array([
        [0, -1, 0],
        [1, 1, 0],
        [0, 0, 1],
    ])
    submitted = transform @ orthorhombic
    submitted_operations = [
        _change_operation_basis(rotation, orthorhombic, submitted)
        for rotation in _orthorhombic_operations()
    ]

    result = _build_nonprimitive_bz_marker_cell(
        submitted,
        real_type_numbers=[64],
        space_operations=submitted_operations,
    )

    assert round(np.linalg.det(transform)) == 1
    assert np.allclose(result["cell"][0], submitted)
    assert result["volume_original_wrt_prim"] == 1.0
    assert result["seekpath_bravais"].startswith("o")


def test_artificial_type_is_positive_and_distinct_from_real_types():
    result = _build_nonprimitive_bz_marker_cell(
        np.array([
            [4.0, 0.0, 0.0],
            [0.3, 5.0, 0.0],
            [0.2, 0.4, 6.0],
        ]),
        real_type_numbers=[1, 2, 3, 118],
        space_operations=[np.eye(3, dtype=int)],
    )

    all_types = set(result["cell"][2])
    assert all_types == set(result["marker_types"])
    assert all_types.isdisjoint({1, 2, 3, 118})
    assert min(result["marker_types"]) > 0


def test_nonsymmorphic_translation_is_removed_only_from_bz_helper():
    fourfold = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ])
    screw_operations = [
        {
            "real_rotation": np.linalg.matrix_power(fourfold, power),
            "translation": [0.0, 0.0, power / 4],
        }
        for power in range(4)
    ]

    result = _build_nonprimitive_bz_marker_cell(
        np.diag([4.0, 4.0, 7.0]),
        real_type_numbers=[26] * 4,
        space_operations=screw_operations,
    )
    dataset = spglib.get_symmetry_dataset(result["cell"])

    assert dataset.number == 75
    assert dataset.international == "P4"
    assert result["intended_space_operation_count"] == 4
    assert result["detected_space_operation_count"] == 4
    assert result["source_space_operation_count"] == 4
    assert all(
        np.allclose(translation, 0.0)
        for translation in dataset.translations
    )
    assert result["volume_original_wrt_prim"] == 1.0


def test_anisotropic_supercell_keeps_its_full_submitted_volume():
    lattice = np.diag([8.0, 15.0, 6.0])
    result = _build_nonprimitive_bz_marker_cell(
        lattice,
        real_type_numbers=[26],
        space_operations=_orthorhombic_operations(),
    )

    assert result["seekpath_bravais"] == "oP1"
    assert result["volume_original_wrt_prim"] == 1.0
    assert np.isclose(
        result["submitted_bz_volume"],
        (2 * np.pi) ** 3 / np.linalg.det(lattice),
    )


def test_q_nonzero_enlarged_cell_is_not_returned_to_a_parent_period():
    lattice = np.diag([12.0, 4.0, 4.0])
    result = _build_nonprimitive_bz_marker_cell(
        lattice,
        real_type_numbers=[25, 25, 25],
        space_operations=_orthorhombic_operations(),
    )

    assert np.allclose(result["cell"][0], lattice)
    assert result["volume_original_wrt_prim"] == 1.0
    assert np.isclose(
        result["submitted_bz_volume"],
        (2 * np.pi) ** 3 / (12.0 * 4.0 * 4.0),
    )


def test_rhombohedral_primitive_and_hexagonal_settings_use_own_lattices():
    a_hex = np.array([
        [5.0, 0.0, 0.0],
        [-2.5, 2.5 * np.sqrt(3.0), 0.0],
        [0.0, 0.0, 13.0],
    ])
    to_rhombohedral = np.array([
        [2 / 3, 1 / 3, 1 / 3],
        [-1 / 3, 1 / 3, 1 / 3],
        [-1 / 3, -2 / 3, 1 / 3],
    ])
    a_rhombohedral = to_rhombohedral @ a_hex
    hex_positions = np.array([
        [0.0, 0.0, 0.0],
        [2 / 3, 1 / 3, 1 / 3],
        [1 / 3, 2 / 3, 2 / 3],
    ])

    def structural_operations(lattice, positions):
        symmetry = spglib.get_symmetry(
            (lattice, positions, [26] * len(positions)),
            symprec=1e-5,
        )
        return [
            {
                "real_rotation": rotation,
                "translation": translation,
            }
            for rotation, translation in zip(
                symmetry["rotations"], symmetry["translations"]
            )
        ]

    primitive = _build_nonprimitive_bz_marker_cell(
        a_rhombohedral,
        real_type_numbers=[26],
        space_operations=structural_operations(
            a_rhombohedral, np.zeros((1, 3))
        ),
        symprec=1e-5,
    )
    conventional = _build_nonprimitive_bz_marker_cell(
        a_hex,
        real_type_numbers=[26, 26, 26],
        space_operations=structural_operations(a_hex, hex_positions),
        symprec=1e-5,
    )

    assert np.isclose(np.linalg.det(a_hex) / np.linalg.det(a_rhombohedral), 3)
    assert np.isclose(primitive["volume_original_wrt_prim"], 1.0)
    assert np.isclose(conventional["volume_original_wrt_prim"], 1.0)
    assert conventional["seekpath_bravais"].startswith("hP")
    assert np.isclose(
        primitive["submitted_bz_volume"],
        3 * conventional["submitted_bz_volume"],
    )


def test_no_moments_supercell_uses_the_same_submitted_cell_helper(tmp_path):
    positions = list(itertools.product((0.0, 0.5), repeat=3))
    structure = Atoms(
        symbols=["Fe"] * 8,
        scaled_positions=positions,
        cell=np.eye(3) * 8.0,
        pbc=True,
    )
    path = tmp_path / "POSCAR"
    write(path, structure, format="vasp", direct=True, vasp5=True)

    preparation = prepare_submitted_cell_analysis(str(path))
    result = compute_centroid(
        str(path),
        output_dir=str(tmp_path),
        show_plot=False,
        verbose=False,
        analysis_cell=preparation["analysis_cell"],
        analysis_has_markers=preparation["analysis_has_markers"],
    )

    assert result["sc_type"] == "cP2"
    assert np.allclose(result["b_matrix_input"], np.eye(3) * np.pi / 4)
    analysis_types = preparation["analysis_cell"][2]
    assert 26 not in analysis_types
    assert set(analysis_types) == set(
        preparation["summary"]["marker_types"]
    )
    assert not (tmp_path / "POSCAR_seekpath_standard.vasp").exists()
    assert (tmp_path / "POSCAR_seekpath_basis_mapping.txt").exists()
    assert not (tmp_path / "POSCAR_magnetic_primitive.mcif").exists()


def test_no_moments_cubic_221_uses_tetragonal_submitted_cell_bz(tmp_path):
    structure = Atoms(
        symbols=["Fe"] * 4,
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.5, 0.5, 0.0],
        ],
        cell=np.diag([8.0, 8.0, 4.0]),
        pbc=True,
    )
    path = tmp_path / "POSCAR_221"
    write(path, structure, format="vasp", direct=True, vasp5=True)

    preparation = prepare_submitted_cell_analysis(str(path))
    result = compute_centroid(
        str(path),
        output_dir=str(tmp_path),
        show_plot=False,
        verbose=False,
        analysis_cell=preparation["analysis_cell"],
        analysis_has_markers=preparation["analysis_has_markers"],
    )

    assert preparation["physical_symmetry"]["number"] == 221
    assert preparation["physical_symmetry"]["point_group"] == "m-3m"
    assert preparation["bz_helper_symmetry"]["number"] == 123
    assert preparation["bz_helper_symmetry"]["symbol"] == "P4/mmm"
    assert preparation["bz_helper_symmetry"]["seekpath_bravais"].startswith(
        "tP"
    )
    assert preparation["summary"]["physical_space_operations"] == 64
    assert preparation["summary"]["intended_point_operations"] == 16
    assert preparation["summary"]["volume_original_wrt_prim"] == 1.0
    assert result["sc_type"].startswith("tP")
    _assert_ibz_volume_matches_k_group(result)
    assert np.allclose(
        result["b_matrix_input"],
        np.diag([np.pi / 4, np.pi / 4, np.pi / 2]),
    )


def test_magnetic_cubic_221_uses_compatible_tetragonal_point_group(tmp_path):
    structure = Atoms(
        symbols=["Fe"] * 4,
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.5, 0.5, 0.0],
        ],
        cell=np.diag([8.0, 8.0, 4.0]),
        pbc=True,
    )
    path = tmp_path / "POSCAR_221_magnetic"
    write(path, structure, format="vasp", direct=True, vasp5=True)

    preparation = prepare_submitted_cell_analysis(
        str(path),
        moments_str="1 -1 -1 1",
        spin_axis_cart="0 0 1",
    )

    assert preparation["physical_symmetry"]["number"] == 123
    assert preparation["physical_symmetry"]["symbol"] == "P4/mmm"
    assert preparation["bz_helper_symmetry"]["number"] == 123
    assert preparation["bz_helper_symmetry"]["seekpath_bravais"].startswith(
        "tP"
    )
    assert preparation["summary"]["intended_point_operations"] == 16
    assert preparation["summary"]["volume_original_wrt_prim"] == 1.0


def test_no_moments_221_uses_pure_rotation_hP_helper():
    preparation = prepare_submitted_cell_analysis(
        str(REFERENCES / "SUPERCELL_221.vasp")
    )

    assert preparation["analysis_symmetry"]["number"] == 183
    assert preparation["analysis_symmetry"]["symbol"] == "P6mm"
    assert preparation["analysis_symmetry"]["seekpath_bravais"] == "hP2"
    assert preparation["summary"]["intended_space_operations"] == 12
    assert preparation["summary"]["detected_space_operations"] == 12
    assert preparation["summary"]["physical_space_operations"] == 48
    assert preparation["summary"]["physical_operation_set_verified"] is True
    assert np.isclose(
        preparation["summary"]["volume_original_wrt_prim"], 1.0
    )


def test_magnetic_211_keeps_full_seitz_helper_for_basis_change(tmp_path):
    preparation = prepare_submitted_cell_analysis(
        str(REFERENCES / "SUPERCELL_211.vasp"),
        moments_str="1 -1 1 -1",
        spin_axis_cart="0 0 1",
    )

    assert preparation["uses_conventional_supercell_bz"] is False
    assert preparation["summary"][
        "submitted_to_primitive_volume_index"
    ] == 1
    assert preparation["analysis_symmetry"]["number"] == 36
    assert preparation["analysis_symmetry"]["symbol"] == "Cmc2_1"
    assert preparation["analysis_symmetry"]["seekpath_bravais"] == "oC1"
    assert preparation["summary"]["intended_space_operations"] == 8
    assert preparation["summary"]["detected_space_operations"] == 8
    assert preparation["summary"]["physical_space_operations"] == 8
    assert preparation["input_cell_symmetry"]["number"] == 36
    assert preparation["input_cell_symmetry"]["symbol"] == "Cmc2_1"
    assert preparation["input_cell_symmetry"][
        "seekpath_bravais"
    ] == "oC1"
    assert np.isclose(
        preparation["summary"]["volume_original_wrt_prim"], 2.0
    )

    result = compute_centroid(
        str(REFERENCES / "SUPERCELL_211.vasp"),
        output_dir=str(tmp_path),
        show_plot=False,
        verbose=False,
        analysis_cell=preparation["analysis_cell"],
        analysis_has_markers=preparation["analysis_has_markers"],
    )
    assert result["sc_type"] == "oC1"
    _assert_ibz_volume_matches_k_group(result)

    bz_hull = ConvexHull(np.vstack(result["bz_loops"]))
    ibz_cart = np.array(result["hull_pts"])
    signed_distances = (
        ibz_cart @ bz_hull.equations[:, :-1].T
        + bz_hull.equations[:, -1]
    )
    assert np.all(signed_distances <= 1e-8)


def test_primitive_magnetic_input_keeps_native_ops_when_hall_is_ambiguous(
    monkeypatch,
):
    def ambiguous_hall(*_args, **_kwargs):
        raise RuntimeError(
            "Multiple inequivalent standard Hall settings match the "
            "submitted G0 operation representatives: [30, 31, 32]."
        )

    monkeypatch.setattr(
        submitted_cell_analysis,
        "_complete_g0_operations_in_submitted_basis",
        ambiguous_hall,
    )
    preparation = prepare_submitted_cell_analysis(
        str(REFERENCES / "SUPERCELL_211.vasp"),
        moments_str="1 -1 1 -1",
        spin_axis_cart="0 0 1",
    )

    assert preparation["uses_conventional_supercell_bz"] is False
    assert preparation["summary"]["physical_operation_set_verified"] is True
    assert preparation["analysis_symmetry"]["number"] == 36
    assert preparation["analysis_symmetry"]["seekpath_bravais"] == "oC1"
    marker_types = preparation["summary"]["marker_types"]
    assert len(marker_types) >= 2
    assert len(marker_types) == len(set(marker_types))


def test_bifeo3_rhombohedral_and_hexagonal_settings_use_distinct_bz_helpers(
    tmp_path,
):
    primitive_path = REFERENCES / "BiFeO3_R3c_primitive.vasp"
    primitive = read(primitive_path)
    primitive.set_initial_magnetic_moments([4.0, -4.0] + [0.0] * 8)
    transform = np.array([
        [1, -1, 0],
        [0, 1, -1],
        [1, 1, 1],
    ])
    conventional = make_supercell(
        primitive, transform, order="cell-major", wrap=True
    )
    species_order = {"Fe": 0, "Bi": 1, "O": 2}
    conventional = conventional[sorted(
        range(len(conventional)),
        key=lambda index: species_order[conventional[index].symbol],
    )]
    conventional_path = tmp_path / "BiFeO3_hexagonal.vasp"
    write(
        conventional_path,
        conventional,
        format="vasp",
        direct=True,
        vasp5=True,
    )
    conventional_moments = " ".join(
        f"{moment:g}"
        for moment in conventional.get_initial_magnetic_moments()
    )

    primitive_result = prepare_submitted_cell_analysis(
        str(primitive_path),
        moments_str="4 -4 8*0",
        spin_axis_cart="0 0 1",
    )
    conventional_result = prepare_submitted_cell_analysis(
        str(conventional_path),
        moments_str=conventional_moments,
        spin_axis_cart="0 0 1",
    )
    nonmagnetic_conventional = prepare_submitted_cell_analysis(
        str(conventional_path)
    )

    assert primitive_result["uses_conventional_supercell_bz"] is False
    assert primitive_result["summary"][
        "submitted_to_primitive_volume_index"
    ] == 1
    assert primitive_result["analysis_symmetry"]["number"] == 161
    assert primitive_result["analysis_symmetry"]["symbol"] == "R3c"
    assert primitive_result["analysis_symmetry"]["seekpath_bravais"] == "hR1"
    assert primitive_result["summary"]["intended_space_operations"] == 6
    assert primitive_result["summary"]["detected_space_operations"] == 6
    assert np.isclose(
        primitive_result["summary"]["volume_original_wrt_prim"], 1.0
    )

    assert conventional_result["analysis_symmetry"]["number"] == 156
    assert conventional_result["analysis_symmetry"]["symbol"] == "P3m1"
    assert conventional_result["analysis_symmetry"][
        "seekpath_bravais"
    ] == "hP2"
    assert conventional_result["summary"]["intended_space_operations"] == 6
    assert conventional_result["summary"]["detected_space_operations"] == 6
    assert conventional_result["summary"]["physical_space_operations"] == 18
    assert conventional_result["input_cell_symmetry"]["number"] == 161
    assert conventional_result["input_cell_symmetry"][
        "seekpath_bravais"
    ] == "hR1"
    assert conventional_result["uses_conventional_supercell_bz"] is True
    assert conventional_result["summary"][
        "submitted_to_primitive_volume_index"
    ] == 3
    assert np.isclose(
        conventional_result["summary"]["volume_original_wrt_prim"], 1.0
    )
    assert nonmagnetic_conventional["analysis_symmetry"]["number"] == 156
    assert nonmagnetic_conventional["summary"][
        "intended_space_operations"
    ] == 6
    assert nonmagnetic_conventional["summary"][
        "detected_space_operations"
    ] == 6
    assert nonmagnetic_conventional["summary"][
        "physical_space_operations"
    ] == 18
    assert np.isclose(
        nonmagnetic_conventional["summary"]["volume_original_wrt_prim"],
        1.0,
    )
