"""Prepare marker cells and symmetry data so SeeK-path analyzes the submitted cell."""

import os
import warnings

import numpy as np
import seekpath
import spglib
from findspingroup import find_spin_group_acc_primitive_from_data

from .io import (
    _dedupe_frac_positions,
    _group_poscar_sites,
    _load_magnetic_input_data,
    _min_periodic_cart_distance,
    _write_magnetic_mcif,
)
from .symmetry import laue_group_from_point_group


# Generic fractional seeds for the marker orbits; the trailing 1e-8 keeps a seed off special positions.
_MARKER_SEEDS = [
    np.array([0.11000000, 0.12000000, 0.15000001]),
    np.array([0.13000000, 0.17000000, 0.23000001]),
    np.array([0.07100000, 0.19300000, 0.31700001]),
    np.array([0.21100000, 0.13700000, 0.29300001]),
]

# A lone orbit can gain unintended symmetry (one seed under {E, C2z} also
# gains inversion), so each set uses two or three orbits of distinct types,
# tried in order until spglib recovers exactly the operations passed in.
_MARKER_SEED_SETS = (
    (_MARKER_SEEDS[0], _MARKER_SEEDS[1]),
    (_MARKER_SEEDS[0], _MARKER_SEEDS[2]),
    (_MARKER_SEEDS[1], _MARKER_SEEDS[3]),
    (_MARKER_SEEDS[0], _MARKER_SEEDS[1], _MARKER_SEEDS[2]),
)


def _wrapped_translation(translation, tol=1e-7):
    wrapped = np.mod(np.asarray(translation, dtype=float), 1.0)
    wrapped[np.isclose(wrapped, 0.0, atol=tol, rtol=0.0)] = 0.0
    wrapped[np.isclose(wrapped, 1.0, atol=tol, rtol=0.0)] = 0.0
    return wrapped


def _space_operation_key(rotation, translation, tol=1e-7):
    decimals = max(0, int(np.ceil(-np.log10(tol))))
    return (
        tuple(np.rint(rotation).astype(int).ravel()),
        tuple(np.round(_wrapped_translation(translation, tol), decimals)),
    )


def _validated_space_operations(space_operations, tol=1e-7):
    """Return distinct ``(R, t)`` operations with integer unimodular rotations."""
    operations = []
    keys = set()
    for operation in space_operations:
        if isinstance(operation, dict):
            value = operation.get("real_rotation")
            translation = operation.get("translation", np.zeros(3))
        else:
            value = operation
            translation = np.zeros(3)
        rotation = np.asarray(value, dtype=float)
        translation = np.asarray(translation, dtype=float)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            continue
        if translation.shape != (3,) or not np.all(np.isfinite(translation)):
            continue
        rounded = np.rint(rotation).astype(int)
        if not np.allclose(rotation, rounded, atol=tol):
            continue
        if not np.isclose(abs(np.linalg.det(rounded)), 1.0, atol=tol):
            continue
        wrapped = _wrapped_translation(translation, tol)
        key = _space_operation_key(rounded, wrapped, tol)
        if key not in keys:
            keys.add(key)
            operations.append({
                "real_rotation": rounded,
                "translation": wrapped,
            })
    if not operations:
        raise RuntimeError(
            "No spatial symmetry operation is compatible with the submitted "
            "cell lattice."
        )
    return operations


def _point_operation_keys(rotations):
    return {
        tuple(np.asarray(rotation, dtype=int).ravel())
        for rotation in rotations
    }


def _space_operation_keys(rotations, translations, tol=1e-7):
    return {
        _space_operation_key(rotation, translation, tol)
        for rotation, translation in zip(rotations, translations)
    }


def _cartesian_translation_offset(lattice, first, second):
    """Return the minimum-image Cartesian distance between two translations."""
    difference = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    difference = difference - np.rint(difference)
    return float(np.linalg.norm(difference @ np.asarray(lattice, dtype=float)))


def _match_space_operations(lattice, intended, detected, symprec):
    """Check the marker cell against the operations it was built from.

    ``intended`` is the operation set the markers encode, ``detected`` is what
    spglib found in the finished cell. Returns the intended operations spglib
    missed and the extra ones it found by accident; both empty means the cell
    reproduces the intended set. Translations match within symprec because
    spglib refits them to the coordinates it is given.
    """
    tolerance = max(float(symprec), 1e-8)
    remaining = [
        (np.asarray(rotation, dtype=float), np.asarray(translation, dtype=float))
        for rotation, translation in detected
    ]
    missing = []
    for rotation, translation in intended:
        key = tuple(np.rint(rotation).astype(int).ravel())
        matched = None
        for index, (other_rotation, other_translation) in enumerate(remaining):
            if tuple(np.rint(other_rotation).astype(int).ravel()) != key:
                continue
            if _cartesian_translation_offset(
                lattice, translation, other_translation
            ) <= tolerance:
                matched = index
                break
        if matched is None:
            missing.append(
                (
                    np.asarray(rotation, dtype=float),
                    np.asarray(translation, dtype=float),
                )
            )
        else:
            remaining.pop(matched)
    return missing, remaining


def _format_space_operation(rotation, translation):
    rows = np.rint(rotation).astype(int).reshape(3, 3)
    matrix = "; ".join(" ".join(f"{value:d}" for value in row) for row in rows)
    shift = ", ".join(
        f"{value:.6f}" for value in _wrapped_translation(translation)
    )
    return f"[{matrix} | {shift}]"


def _space_operation_mismatch_report(missing, unexpected, intended_count):
    """Describe which Seitz operations failed to pair, not just how many."""
    parts = [
        "helper does not reproduce the intended Seitz set of "
        f"{intended_count} operations"
    ]
    if missing:
        listed = ", ".join(
            _format_space_operation(rotation, translation)
            for rotation, translation in missing[:2]
        )
        suffix = ", ..." if len(missing) > 2 else ""
        parts.append(f"missing {len(missing)}: {listed}{suffix}")
    if unexpected:
        listed = ", ".join(
            _format_space_operation(rotation, translation)
            for rotation, translation in unexpected[:2]
        )
        suffix = ", ..." if len(unexpected) > 2 else ""
        parts.append(f"unexpected {len(unexpected)}: {listed}{suffix}")
    return "; ".join(parts)


def _point_operations_preserving_submitted_lattice(
    lattice, space_operations, tol=1e-7
):
    """Return the point operations compatible with the submitted cell.

    Keep the distinct rotations that preserve the lengths and angles of the
    submitted lattice vectors, and verify that they form a closed point group.
    The caller uses these rotations as ``(R, 0)`` only for the conventional or
    supercell BZ helper. Removing additional pure translations, such as centering
    translations, prevents SeeK-path from reducing the submitted translation
    lattice to the physical primitive lattice.
    """
    lattice = np.asarray(lattice, dtype=float)
    lattice_dot_products = lattice @ lattice.T
    candidates = _validated_space_operations(space_operations, tol=tol)
    compatible = []
    dot_product_tolerance = tol * max(
        1.0, float(np.max(np.abs(lattice_dot_products)))
    )
    for operation in candidates:
        rotation = np.asarray(operation["real_rotation"], dtype=int)
        if np.allclose(
            rotation.T @ lattice_dot_products @ rotation,
            lattice_dot_products,
            atol=dot_product_tolerance,
            rtol=0.0,
        ):
            compatible.append(operation)
    if not compatible:
        raise RuntimeError(
            "No point operation preserves the submitted lattice-vector lengths and angles."
        )
    rotations = {}
    for operation in compatible:
        rotation = np.asarray(operation["real_rotation"], dtype=int)
        rotations.setdefault(tuple(rotation.ravel()), rotation)

    identity_key = tuple(np.eye(3, dtype=int).ravel())
    if identity_key not in rotations:
        raise RuntimeError(
            "Submitted-cell compatible point operations do not contain the identity."
        )

    keys = set(rotations)
    for left in rotations.values():
        for right in rotations.values():
            product_key = tuple((left @ right).ravel())
            if product_key not in keys:
                raise RuntimeError(
                    "Submitted-cell compatible rotations do not form a closed point group."
                )
    return list(rotations.values()), compatible, candidates


def _database_operations_in_input_basis(
    hall_number,
    input_to_standard,
    origin_shift,
    *,
    tol=1e-6,
):
    """Express one complete standard Hall operation set in the submitted basis."""
    input_to_standard = np.asarray(input_to_standard, dtype=float)
    origin_shift = np.asarray(origin_shift, dtype=float)
    if (
        input_to_standard.shape != (3, 3)
        or origin_shift.shape != (3,)
        or not np.all(np.isfinite(input_to_standard))
        or not np.all(np.isfinite(origin_shift))
        or abs(np.linalg.det(input_to_standard)) < 1e-12
    ):
        raise RuntimeError("Invalid input-to-standard setting transformation.")

    database = spglib.get_symmetry_from_database(int(hall_number))
    if database is None:
        raise RuntimeError(
            f"spglib has no operation set for Hall number {hall_number}."
        )
    standard_to_input = np.linalg.inv(input_to_standard)
    transformed = []
    for rotation_standard, translation_standard in zip(
        database["rotations"], database["translations"]
    ):
        rotation_standard = np.asarray(rotation_standard, dtype=float)
        translation_standard = np.asarray(translation_standard, dtype=float)
        rotation_input = (
            standard_to_input
            @ rotation_standard
            @ input_to_standard
        )
        if not np.allclose(
            rotation_input, np.rint(rotation_input), atol=tol, rtol=0.0
        ):
            raise RuntimeError(
                f"Hall {hall_number} produces a nonintegral input rotation."
            )
        translation_input = standard_to_input @ (
            rotation_standard @ origin_shift
            + translation_standard
            - origin_shift
        )
        transformed.append({
            "real_rotation": rotation_input,
            "translation": translation_input,
        })
    return _validated_space_operations(transformed, tol=tol)


def _magnetic_primitive_nssg_operations(result):
    """Return FindSpinGroup's nontrivial SSG operations in the magnetic-primitive setting."""
    views = (
        result.get("operation_views", {})
        .get("magnetic_primitive_cartesian", {})
        .get("views", {})
    )
    nssg = views.get("nssg", {})
    operations = nssg.get("ops", []) if isinstance(nssg, dict) else []
    if not operations:
        raise RuntimeError(
            "FindSpinGroup did not return magnetic-primitive nssg operations."
        )
    return operations


def _g0_representatives_in_submitted_basis(result):
    """Express FindSpinGroup's G0 representatives in the submitted basis."""
    transform = result.get("T_input_to_acc_primitive")
    if not transform or len(transform) != 2:
        raise RuntimeError(
            "FindSpinGroup did not return the input-to-magnetic-primitive setting transformation."
        )
    input_to_primitive = np.asarray(transform[0], dtype=float)
    origin_shift = np.asarray(transform[1], dtype=float)
    if input_to_primitive.shape != (3, 3) or origin_shift.shape != (3,):
        raise RuntimeError(
            "FindSpinGroup returned an invalid magnetic-primitive setting transformation."
        )
    primitive_to_input = np.linalg.inv(input_to_primitive)
    operations = []
    for operation in _magnetic_primitive_nssg_operations(result):
        rotation_primitive = np.asarray(
            operation["real_rotation"], dtype=float
        )
        translation_primitive = np.asarray(
            operation.get("translation", np.zeros(3)), dtype=float
        )
        rotation_input = (
            primitive_to_input
            @ rotation_primitive
            @ input_to_primitive
        )
        translation_input = primitive_to_input @ (
            rotation_primitive @ origin_shift
            + translation_primitive
            - origin_shift
        )
        operations.append({
            "real_rotation": rotation_input,
            "translation": translation_input,
        })
    return _validated_space_operations(operations)


def _g0_spacegroup_number(result):
    details = result.get("identify_index_details")
    number = details.get("G0_id") if isinstance(details, dict) else None
    try:
        number = int(number)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "FindSpinGroup did not return a valid G0 space-group number."
        ) from exc
    if not 1 <= number <= 230:
        raise RuntimeError(
            f"FindSpinGroup returned invalid G0 space-group number {number}."
        )
    return number


def _input_to_g0_standard_transform(result):
    """Return the transformation from submitted to G0 standard coordinates."""
    transforms = []
    for key in ("T_input_to_acc_primitive", "T_acc_primitive_to_G0std"):
        transform = result.get(key)
        if not transform or len(transform) != 2:
            raise RuntimeError(
                f"FindSpinGroup did not return the required {key} transform."
            )
        matrix = np.asarray(transform[0], dtype=float)
        shift = np.asarray(transform[1], dtype=float)
        if (
            matrix.shape != (3, 3)
            or shift.shape != (3,)
            or not np.all(np.isfinite(matrix))
            or not np.all(np.isfinite(shift))
            or abs(np.linalg.det(matrix)) < 1e-12
        ):
            raise RuntimeError(
                f"FindSpinGroup returned an invalid {key} transform."
            )
        transforms.append((matrix, shift))

    (input_to_primitive, input_shift), (
        primitive_to_standard,
        primitive_shift,
    ) = transforms
    return (
        primitive_to_standard @ input_to_primitive,
        primitive_to_standard @ input_shift + primitive_shift,
    )


def _complete_g0_operations_in_submitted_basis(result, tol=1e-6):
    """Return all G0 space-group operations in the submitted basis.

    FindSpinGroup reports the operations for its magnetic primitive cell, so
    the list may lack centering translations needed for a submitted conventional
    cell. Use that list to identify the matching Hall setting, retrieve the
    complete operation set from spglib, and transform it to the submitted basis.
    """
    g0_number = _g0_spacegroup_number(result)
    representatives = _g0_representatives_in_submitted_basis(result)
    representative_keys = _space_operation_keys(
        [operation["real_rotation"] for operation in representatives],
        [operation["translation"] for operation in representatives],
        tol=tol,
    )
    input_to_standard, origin_shift = _input_to_g0_standard_transform(result)
    distinct_candidates = {}
    failures = []
    for hall_number in range(1, 531):
        spacegroup_type = spglib.get_spacegroup_type(hall_number)
        if (
            spacegroup_type is None
            or int(spacegroup_type.number) != g0_number
        ):
            continue
        try:
            operations = _database_operations_in_input_basis(
                hall_number,
                input_to_standard,
                origin_shift,
                tol=tol,
            )
        except RuntimeError as exc:
            failures.append(str(exc))
            continue
        operation_keys = _space_operation_keys(
            [operation["real_rotation"] for operation in operations],
            [operation["translation"] for operation in operations],
            tol=tol,
        )
        if not representative_keys.issubset(operation_keys):
            failures.append(
                f"Hall {hall_number}: does not contain the transformed "
                "magnetic-primitive representatives"
            )
            continue
        distinct_candidates.setdefault(
            frozenset(operation_keys), (hall_number, operations)
        )

    if not distinct_candidates:
        raise RuntimeError(
            "Could not reconstruct the complete G0 operation set in the "
            f"submitted basis for space group {g0_number}. "
            f"{' ; '.join(failures)}"
        )
    if len(distinct_candidates) != 1:
        hall_numbers = sorted(
            hall_number
            for hall_number, _operations in distinct_candidates.values()
        )
        raise RuntimeError(
            "Multiple inequivalent standard Hall settings match the "
            f"submitted G0 operation representatives: {hall_numbers}."
        )
    return next(iter(distinct_candidates.values()))[1]


def _moment_colored_types(elements, moments, tol=0.02):
    """Assign one spglib type to each distinct element-and-moment color."""
    colors = []
    types = []
    for element, moment in zip(elements, np.asarray(moments, dtype=float)):
        for index, (other_element, other_moment) in enumerate(colors):
            if element == other_element and np.allclose(
                moment, other_moment, atol=tol, rtol=0.0
            ):
                types.append(index + 1)
                break
        else:
            colors.append((element, np.asarray(moment, dtype=float).copy()))
            types.append(len(colors))
    return types


def _layer_group_record(cell, vacuum_axis, symprec):
    """Return layer-group labels and the layer-primitive site count."""
    dataset = spglib.get_symmetry_layerdataset(
        cell,
        aperiodic_dir=int(vacuum_axis),
        symprec=float(symprec),
    )
    if dataset is None:
        raise RuntimeError("Could not determine the physical layer group.")
    point_group = str(dataset.pointgroup)
    laue_group = laue_group_from_point_group(point_group)
    if laue_group is None:
        raise RuntimeError(
            f"Could not determine the Laue group for layer point group {point_group}."
        )
    return {
        "label": f"{dataset.international} ({int(dataset.number)})",
        "point_group": point_group,
        "laue_group": laue_group,
        "sites": len(set(
            int(value) for value in dataset.mapping_to_primitive
        )),
    }


def _layer_cell_summary(lattice, positions, elements, moments, vacuum_axis, symprec):
    """Build the 2D input/nonmagnetic/magnetic cell-summary records."""
    from ase.data import atomic_numbers

    nonmagnetic = _layer_group_record(
        (lattice, positions, [atomic_numbers[str(value)] for value in elements]),
        vacuum_axis, symprec,
    )
    magnetic = None
    if np.any(np.linalg.norm(np.asarray(moments, dtype=float), axis=1) > 1e-10):
        magnetic = _layer_group_record(
            (lattice, positions, _moment_colored_types(elements, moments)),
            vacuum_axis, symprec,
        )

    return {
        "input_cell": {**(magnetic or nonmagnetic), "sites": len(elements)},
        "nonmagnetic_primitive_cell": nonmagnetic,
        "magnetic_primitive_cell": magnetic,
    }


def _physical_symmetry_from_operations(
    lattice,
    operations,
    expected_number,
    symprec,
):
    """Identify and validate the complete physical G0 operation set."""
    spacegroup_type = spglib.get_spacegroup_type_from_symmetry(
        [operation["real_rotation"] for operation in operations],
        [operation["translation"] for operation in operations],
        lattice=np.asarray(lattice, dtype=float),
        symprec=symprec,
    )
    if spacegroup_type is None:
        raise RuntimeError(
            "Could not identify the complete physical space-group operation "
            "set in the submitted basis."
        )
    if int(spacegroup_type.number) != int(expected_number):
        raise RuntimeError(
            "Physical operation-set identification changed the expected "
            f"space-group number {int(expected_number)} to "
            f"{int(spacegroup_type.number)}."
        )
    return {
        "number": int(spacegroup_type.number),
        "symbol": str(spacegroup_type.international_short),
        "point_group": str(spacegroup_type.pointgroup_international),
        "hall_number": int(spacegroup_type.hall_number),
    }


def _standard_physical_symmetry(spacegroup_number):
    """Return a standard G0 label when the submitted supercell setting cannot be identified."""
    fallback = None
    for hall_number in range(1, 531):
        spacegroup_type = spglib.get_spacegroup_type(hall_number)
        if (
            spacegroup_type is None
            or int(spacegroup_type.number) != int(spacegroup_number)
        ):
            continue
        candidate = {
            "number": int(spacegroup_type.number),
            "symbol": str(spacegroup_type.international_short),
            "point_group": str(spacegroup_type.pointgroup_international),
            "hall_number": int(spacegroup_type.hall_number),
        }
        if fallback is None:
            fallback = candidate
        if str(spacegroup_type.choice) == "":
            return candidate
    if fallback is None:
        raise RuntimeError(
            f"No spglib setting exists for space group {spacegroup_number}."
        )
    return fallback


def _marker_orbits_with_distinct_types(
    seeds,
    rotations,
    translations,
    reserved_type_numbers,
):
    """Generate each marker orbit with a distinct unused type number."""
    positions = []
    types = []
    marker_types = []
    used_types = {int(value) for value in reserved_type_numbers}
    next_marker_type = max(used_types, default=0) + 1
    for seed in seeds:
        orbit = _dedupe_frac_positions([
            seed @ rotation.T + translation
            for rotation, translation in zip(rotations, translations)
        ])
        while next_marker_type in used_types or next_marker_type <= 0:
            next_marker_type += 1
        orbit_type = next_marker_type
        used_types.add(orbit_type)
        next_marker_type += 1
        marker_types.append(orbit_type)
        positions.extend(orbit)
        types.extend([orbit_type] * len(orbit))
    return positions, types, marker_types


def _build_nonprimitive_bz_marker_cell(
    lattice,
    real_type_numbers,
    space_operations,
    *,
    symprec=1e-3,
):
    """Build a marker-only BZ helper for a nonprimitive submitted cell.

    Generate marker orbits with compatible G0 ``(R, 0)`` point operations so
    SeeK-path preserves the submitted conventional-cell or supercell translation
    lattice and its folded BZ. Real atoms are excluded because they obey the
    complete G0 ``(R, t)`` operations rather than this artificial ``(R, 0)`` set.
    """
    lattice = np.asarray(lattice, dtype=float)
    if lattice.shape != (3, 3) or not np.all(np.isfinite(lattice)):
        raise ValueError("Submitted lattice must be a finite 3x3 matrix.")
    if abs(np.linalg.det(lattice)) < 1e-12:
        raise ValueError("Submitted lattice is singular.")

    real_type_numbers = [int(value) for value in real_type_numbers]

    rotations, compatible_space_operations, source_space_operations = (
        _point_operations_preserving_submitted_lattice(
            lattice, space_operations
        )
    )
    translations = [np.zeros(3) for _rotation in rotations]
    operations = [
        {"real_rotation": rotation, "translation": np.zeros(3)}
        for rotation in rotations
    ]
    intended_keys = _point_operation_keys(rotations)
    intended_space_keys = _space_operation_keys(rotations, translations)
    failures = []
    for seeds in _MARKER_SEED_SETS:
        helper_positions, helper_types, marker_types = (
            _marker_orbits_with_distinct_types(
                seeds,
                rotations,
                translations,
                real_type_numbers,
            )
        )
        cell = (
            lattice.tolist(),
            [np.asarray(position, dtype=float).tolist()
             for position in helper_positions],
            helper_types,
        )
        dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
        seed_label = [seed.tolist() for seed in seeds]
        if dataset is None:
            failures.append(f"seeds {seed_label}: spglib found no symmetry")
            continue
        detected_keys = _point_operation_keys(dataset.rotations)
        detected_space_keys = _space_operation_keys(
            dataset.rotations, dataset.translations
        )
        missing_operations, unexpected_operations = _match_space_operations(
            lattice,
            list(zip(rotations, translations)),
            list(zip(dataset.rotations, dataset.translations)),
            symprec,
        )
        if missing_operations or unexpected_operations:
            failures.append(
                f"seeds {seed_label}: "
                + _space_operation_mismatch_report(
                    missing_operations,
                    unexpected_operations,
                    len(intended_space_keys),
                )
            )
            continue
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r".*dict interface is deprecated.*"
            )
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                module=r"seekpath\.hpkot(\..*)?",
            )
            sp_result = seekpath.get_path(
                cell,
                with_time_reversal=True,
                symprec=symprec,
            )
        volume_ratio = float(sp_result["volume_original_wrt_prim"])
        if not np.isfinite(volume_ratio) or volume_ratio <= 0.0:
            failures.append(
                f"seeds {seed_label}: invalid volume_original_wrt_prim="
                f"{volume_ratio!r}"
            )
            continue
        if not np.isclose(volume_ratio, 1.0, atol=1e-6, rtol=0.0):
            failures.append(
                f"seeds {seed_label}: helper reduced the submitted "
                f"translation lattice, volume_original_wrt_prim="
                f"{volume_ratio!r}"
            )
            continue
        reciprocal = 2 * np.pi * np.linalg.inv(lattice).T
        return {
            "cell": cell,
            "marker_types": marker_types,
            "marker_seeds": [seed.copy() for seed in seeds],
            "marker_count": len(helper_positions),
            "marker_min_distance": _min_periodic_cart_distance(
                helper_positions, lattice
            ),
            "point_operations": rotations,
            "space_operations": operations,
            "source_space_operations": source_space_operations,
            "source_space_operation_count": len(source_space_operations),
            "compatible_space_operation_count": len(
                compatible_space_operations
            ),
            "intended_point_operation_count": len(intended_keys),
            "detected_point_operation_count": len(detected_keys),
            "intended_space_operation_count": len(intended_space_keys),
            "detected_space_operation_count": len(detected_space_keys),
            "volume_original_wrt_prim": volume_ratio,
            "submitted_bz_volume": abs(float(np.linalg.det(reciprocal))),
            "seekpath_bravais": sp_result["bravais_lattice_extended"],
            "point_coords": dict(sp_result["point_coords"]),
            "analysis_spacegroup_number": int(dataset.number),
            "analysis_spacegroup_symbol": str(dataset.international),
            "analysis_point_group": str(dataset.pointgroup),
        }

    raise RuntimeError(
        "Could not validate the submitted-cell marker helper. "
        f"{'; '.join(failures)}"
    )


def _build_g0_marker_cell(
    lattice,
    real_positions,
    real_type_numbers,
    space_operations,
    *,
    symprec=1e-3,
    expected_spacegroup_number=None,
):
    """Add marker orbits that help spglib and SeeK-path recognize G0.

    Generate the marker orbits with the complete G0 ``(R, t)`` operations and
    add them to the submitted structure. The markers remove structural symmetries
    absent from G0 and verify its complete operation set. When the submitted
    translation lattice is primitive, the same augmented structure is passed to
    SeeK-path because its BZ already matches the requested one.
    """
    lattice = np.asarray(lattice, dtype=float)
    real_positions = np.mod(np.asarray(real_positions, dtype=float), 1.0)
    real_type_numbers = [int(value) for value in real_type_numbers]

    operations = _validated_space_operations(space_operations)
    rotations = [operation["real_rotation"] for operation in operations]
    translations = [operation["translation"] for operation in operations]
    intended_keys = _point_operation_keys(rotations)
    intended_space_keys = _space_operation_keys(rotations, translations)
    failures = []
    for seeds in _MARKER_SEED_SETS:
        markers, marker_type_numbers, marker_types = (
            _marker_orbits_with_distinct_types(
                seeds,
                rotations,
                translations,
                real_type_numbers,
            )
        )
        helper_positions = [*real_positions.tolist(), *markers]
        cell = (
            lattice.tolist(),
            [
                np.asarray(position, dtype=float).tolist()
                for position in helper_positions
            ],
            [*real_type_numbers, *marker_type_numbers],
        )
        dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
        seed_label = [seed.tolist() for seed in seeds]
        if dataset is None:
            failures.append(f"seeds {seed_label}: spglib found no symmetry")
            continue
        detected_keys = _point_operation_keys(dataset.rotations)
        detected_space_keys = _space_operation_keys(
            dataset.rotations, dataset.translations
        )
        missing_operations, unexpected_operations = _match_space_operations(
            lattice,
            list(zip(rotations, translations)),
            list(zip(dataset.rotations, dataset.translations)),
            symprec,
        )
        if missing_operations or unexpected_operations:
            failures.append(
                f"seeds {seed_label}: "
                + _space_operation_mismatch_report(
                    missing_operations,
                    unexpected_operations,
                    len(intended_space_keys),
                )
            )
            continue
        if (
            expected_spacegroup_number is not None
            and int(dataset.number) != int(expected_spacegroup_number)
        ):
            failures.append(
                f"seeds {seed_label}: expected space group "
                f"{int(expected_spacegroup_number)} but detected "
                f"{int(dataset.number)}"
            )
            continue
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r".*dict interface is deprecated.*"
            )
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                module=r"seekpath\.hpkot(\..*)?",
            )
            sp_result = seekpath.get_path(
                cell,
                with_time_reversal=True,
                symprec=symprec,
            )
        volume_ratio = float(sp_result["volume_original_wrt_prim"])
        if not np.isfinite(volume_ratio) or volume_ratio <= 0.0:
            failures.append(
                f"seeds {seed_label}: invalid volume_original_wrt_prim="
                f"{volume_ratio!r}"
            )
            continue
        reciprocal = 2 * np.pi * np.linalg.inv(lattice).T
        return {
            "cell": cell,
            "marker_types": marker_types,
            "marker_seeds": [seed.copy() for seed in seeds],
            "marker_count": len(markers),
            "marker_min_distance": _min_periodic_cart_distance(
                helper_positions, lattice
            ),
            "point_operations": rotations,
            "space_operations": operations,
            "source_space_operations": operations,
            "source_space_operation_count": len(operations),
            "compatible_space_operation_count": len(operations),
            "intended_point_operation_count": len(intended_keys),
            "detected_point_operation_count": len(detected_keys),
            "intended_space_operation_count": len(intended_space_keys),
            "detected_space_operation_count": len(detected_space_keys),
            "volume_original_wrt_prim": volume_ratio,
            "submitted_bz_volume": abs(float(np.linalg.det(reciprocal))),
            "seekpath_bravais": sp_result["bravais_lattice_extended"],
            "point_coords": dict(sp_result["point_coords"]),
            "analysis_spacegroup_number": int(dataset.number),
            "analysis_spacegroup_symbol": str(dataset.international),
            "analysis_point_group": str(dataset.pointgroup),
        }

    raise RuntimeError(
        "Could not validate the primitive-cell full-Seitz marker helper. "
        f"{'; '.join(failures)}"
    )


def _submitted_to_primitive_volume_index(
    submitted_lattice,
    primitive_lattice,
    *,
    tol=1e-5,
):
    """Return the integer index and volume ratio of the submitted to primitive cell."""
    submitted_volume = abs(float(np.linalg.det(submitted_lattice)))
    primitive_volume = abs(float(np.linalg.det(primitive_lattice)))
    if (
        not np.isfinite(submitted_volume)
        or not np.isfinite(primitive_volume)
        or submitted_volume <= 0.0
        or primitive_volume <= 0.0
    ):
        raise RuntimeError("Cannot compare invalid submitted/primitive cells.")
    ratio = submitted_volume / primitive_volume
    index = int(round(ratio))
    if index < 1 or not np.isclose(ratio, index, atol=tol, rtol=tol):
        raise RuntimeError(
            "The submitted-cell volume is not an integer multiple of the "
            f"physical primitive-cell volume (ratio={ratio:.12g})."
        )
    return index, ratio


def prepare_submitted_cell_analysis(
    structure_file,
    *,
    moments_str="",
    spin_axis_cart=None,
    output_dir=".",
    symprec=1e-3,
    write_magnetic_diagnostic=False,
    input_vacuum_axis=None,
):
    """Prepare the marker cell and symmetry data for submitted-cell BZ analysis.

    Use the G0 marker cell for a primitive input and the marker-only ``(R, 0)``
    helper for a conventional-cell or supercell input.
    """
    from ase.data import atomic_numbers

    lattice, positions, elements, moments, spin_setting = (
        _load_magnetic_input_data(structure_file, moments_str, spin_axis_cart)
    )
    real_types = [atomic_numbers[str(element)] for element in elements]
    fsg_result = None
    nonmagnetic_primitive_symmetry = None
    has_magnetic_moments = bool(np.any(
        np.linalg.norm(np.asarray(moments, dtype=float), axis=1) > 1e-10
    ))
    if has_magnetic_moments:
        fsg_result = find_spin_group_acc_primitive_from_data(
            structure_file,
            lattice,
            positions,
            elements,
            [1.0] * len(elements),
            moments,
            input_spin_setting=spin_setting,
        )
        expected_spacegroup_number = _g0_spacegroup_number(fsg_result)
        magnetic_primitive_lattice = np.asarray(
            fsg_result["acc_primitive_cell_detail"]["lattice"],
            dtype=float,
        )
        translation_index, translation_volume_ratio = (
            _submitted_to_primitive_volume_index(
                lattice, magnetic_primitive_lattice
            )
        )
        # FindSpinGroup's magnetic-primitive list contains one operation for each G0 rotation.
        # Express these operations in the submitted basis; the nonprimitive BZ helper later keeps
        # only rotations that preserve the submitted lattice.
        space_operations = _g0_representatives_in_submitted_basis(fsg_result)
        try:
            physical_operations = (
                _complete_g0_operations_in_submitted_basis(fsg_result)
            )
            physical_symmetry = _physical_symmetry_from_operations(
                lattice,
                physical_operations,
                expected_spacegroup_number,
                symprec,
            )
            physical_operation_set_verified = True
        except RuntimeError as exc:
            if translation_index == 1:
                if not str(exc).startswith(
                    "Multiple inequivalent standard Hall settings"
                ):
                    raise RuntimeError(
                        "Could not construct the complete physical G0 Seitz "
                        "set for a primitive submitted cell. The pure-rotation "
                        "conventional/supercell helper is not applicable."
                    ) from exc
                # Several conventional Hall settings may reduce to the same operations
                # in a primitive cell, so the setting cannot always be identified uniquely.
                # A primitive cell needs no additional centering operations, so the validated
                # magnetic-primitive G0 set can be used directly.
                physical_operations = space_operations
                physical_symmetry = _physical_symmetry_from_operations(
                    lattice,
                    physical_operations,
                    expected_spacegroup_number,
                    symprec,
                )
                physical_operation_set_verified = True
            else:
                # An anisotropic or non-diagonal supercell may not represent every G0
                # operation with an integer rotation matrix. The nonprimitive BZ helper
                # needs only rotations that preserve the submitted lattice, so use the
                # standard G0 label when the full operation set cannot be verified.
                physical_operations = space_operations
                physical_symmetry = _standard_physical_symmetry(
                    expected_spacegroup_number
                )
                physical_operation_set_verified = False
    else:
        dataset = spglib.get_symmetry_dataset(
            (lattice, positions, real_types),
            symprec=symprec,
        )
        if dataset is None:
            raise RuntimeError(
                "Could not determine submitted-cell structural operations."
            )
        expected_spacegroup_number = int(dataset.number)
        symmetry = spglib.get_symmetry(
            (lattice, positions, real_types),
            symprec=symprec,
        )
        if symmetry is None:
            raise RuntimeError(
                "Could not determine submitted-cell structural operations."
            )
        space_operations = _validated_space_operations([
            {
                "real_rotation": rotation,
                "translation": translation,
            }
            for rotation, translation in zip(
                symmetry["rotations"], symmetry["translations"]
            )
        ])
        physical_operations = space_operations
        physical_operation_set_verified = True
        physical_symmetry = {
            "number": int(dataset.number),
            "symbol": str(dataset.international),
            "point_group": str(dataset.pointgroup),
            "hall_number": int(dataset.hall_number),
        }

        primitive = spglib.find_primitive(
            (lattice, positions, real_types),
            symprec=symprec,
        )
        if primitive is None:
            raise RuntimeError(
                "Could not determine the physical primitive translation "
                "cell of the submitted structure."
            )
        primitive_dataset = spglib.get_symmetry_dataset(
            primitive,
            symprec=symprec,
        )
        if primitive_dataset is None:
            raise RuntimeError(
                "Could not determine nonmagnetic primitive-cell symmetry."
            )
        nonmagnetic_primitive_symmetry = {
            "number": int(primitive_dataset.number),
            "symbol": str(primitive_dataset.international),
            "point_group": str(primitive_dataset.pointgroup),
            "sites": len(primitive[1]),
        }
        translation_index, translation_volume_ratio = (
            _submitted_to_primitive_volume_index(lattice, primitive[0])
        )

    physical_helper = None
    if physical_operation_set_verified:
        try:
            physical_helper = _build_g0_marker_cell(
                lattice,
                positions,
                real_types,
                physical_operations,
                symprec=symprec,
                expected_spacegroup_number=expected_spacegroup_number,
            )
        except RuntimeError:
            if translation_index == 1:
                raise

    uses_conventional_supercell_bz = translation_index > 1
    if uses_conventional_supercell_bz:
        helper = _build_nonprimitive_bz_marker_cell(
            lattice,
            real_types,
            space_operations,
            symprec=symprec,
        )
    else:
        if physical_helper is None:
            raise RuntimeError(
                "Could not validate the physical input-cell analysis."
            )
        helper = physical_helper
    input_cell_symmetry = dict(physical_symmetry)
    input_cell_symmetry["seekpath_bravais"] = (
        physical_helper["seekpath_bravais"]
        if physical_helper is not None
        else None
    )
    if nonmagnetic_primitive_symmetry is not None:
        nonmagnetic_primitive_symmetry["seekpath_bravais"] = (
            input_cell_symmetry["seekpath_bravais"]
        )
    bz_helper_symmetry = {
        "number": helper["analysis_spacegroup_number"],
        "symbol": helper["analysis_spacegroup_symbol"],
        "point_group": helper["analysis_point_group"],
        "seekpath_bravais": helper["seekpath_bravais"],
    }
    basename = os.path.splitext(os.path.basename(structure_file))[0]
    result = {
        "analysis_cell": helper["cell"],
        "analysis_has_markers": bool(helper["marker_types"]),
        "submitted_lattice": np.array(lattice, dtype=float),
        "submitted_sites": len(elements),
        "operation_basis_label": (
            f"submitted structure '{os.path.basename(structure_file)}'"
        ),
        "physical_symmetry": physical_symmetry,
        "input_cell_symmetry": input_cell_symmetry,
        "nonmagnetic_primitive_symmetry": nonmagnetic_primitive_symmetry,
        "bz_helper_symmetry": bz_helper_symmetry,
        "uses_conventional_supercell_bz": uses_conventional_supercell_bz,
        # Internal consumers use this alias for the BZ helper symmetry.
        "analysis_symmetry": bz_helper_symmetry,
        "summary": {
            "marker_seeds": [
                seed.tolist() for seed in helper["marker_seeds"]
            ],
            "marker_count": helper["marker_count"],
            "marker_types": helper["marker_types"],
            "marker_min_distance": helper["marker_min_distance"],
            "intended_point_operations": helper[
                "intended_point_operation_count"
            ],
            "detected_point_operations": helper[
                "detected_point_operation_count"
            ],
            "intended_space_operations": helper[
                "intended_space_operation_count"
            ],
            "detected_space_operations": helper[
                "detected_space_operation_count"
            ],
            "source_space_operations": helper[
                "source_space_operation_count"
            ],
            "compatible_space_operations": helper[
                "compatible_space_operation_count"
            ],
            "volume_original_wrt_prim": helper[
                "volume_original_wrt_prim"
            ],
            "submitted_to_primitive_volume_index": translation_index,
            "submitted_to_primitive_volume_ratio": (
                translation_volume_ratio
            ),
            "uses_conventional_supercell_bz": (
                uses_conventional_supercell_bz
            ),
        },
    }
    result["summary"]["physical_space_operations"] = len(
        physical_operations
    )
    result["summary"]["physical_operation_set_verified"] = (
        physical_operation_set_verified
    )
    if input_vacuum_axis is not None:
        result["layer_cell_summary"] = _layer_cell_summary(
            lattice,
            positions,
            elements,
            moments,
            input_vacuum_axis,
            symprec,
        )

    if write_magnetic_diagnostic:
        if fsg_result is None:
            raise RuntimeError(
                "A magnetic primitive diagnostic was requested for a "
                "structure without nonzero magnetic moments."
            )
        magnetic_cell = fsg_result["acc_primitive_cell_detail"]
        magnetic_lattice = np.asarray(
            magnetic_cell["lattice"], dtype=float
        )
        magnetic_positions = [
            np.asarray(position, dtype=float)
            for position in magnetic_cell["positions"]
        ]
        magnetic_elements = [
            str(value) for value in magnetic_cell["elements"]
        ]
        # Equal lattice lengths can have higher metric symmetry than G0.
        # Classify the magnetic primitive structure with its spatial operations,
        # just as the physical input-cell helper does, rather than its metric alone.
        magnetic_helper = _build_g0_marker_cell(
            magnetic_lattice,
            magnetic_positions,
            [atomic_numbers[element] for element in magnetic_elements],
            _magnetic_primitive_nssg_operations(fsg_result),
            symprec=symprec,
            expected_spacegroup_number=expected_spacegroup_number,
        )
        magnetic_moments = np.asarray(
            magnetic_cell.get(
                "moments", np.zeros((len(magnetic_elements), 3))
            ),
            dtype=float,
        )
        _, _, grouped_positions, ordered_indices = _group_poscar_sites(
            magnetic_elements,
            magnetic_positions,
        )
        ordered_elements = [
            magnetic_elements[index] for index in ordered_indices
        ]
        ordered_moments = magnetic_moments[ordered_indices]
        os.makedirs(output_dir, exist_ok=True)
        mcif_path = os.path.join(
            output_dir, f"{basename}_magnetic_primitive.mcif"
        )
        _write_magnetic_mcif(
            mcif_path,
            f"{basename}_magnetic_primitive",
            magnetic_lattice,
            ordered_elements,
            grouped_positions,
            ordered_moments,
        )
        result.update({
            "mcif_path": mcif_path,
            "magnetic_primitive_lattice": magnetic_lattice,
            "magnetic_primitive_sites": len(magnetic_elements),
            "magnetic_primitive_lattice_tag": magnetic_helper["seekpath_bravais"],
            "magnetic_summary": {
                "index": fsg_result.get("index"),
                "acc_symbol": fsg_result.get("acc_symbol"),
                "setting": fsg_result.get("acc_primitive_cell_setting"),
            },
        })
    return result
