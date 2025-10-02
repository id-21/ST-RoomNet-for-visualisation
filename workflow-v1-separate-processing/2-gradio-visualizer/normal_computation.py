# File: normal_computation.py
"""
Phase 3B: Normal Vector Computation from Theta Parameters

Computes 3D normal vectors for room walls using ST-RoomNet's theta transformation
parameters. Normals transform by the inverse transpose of the homography matrix.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


# Reference cuboid normals (in reference coordinate system)
REFERENCE_NORMALS = {
    0: np.array([0.0, -1.0, 0.0]),   # ceiling (points down)
    1: np.array([-1.0, 0.0, 0.0]),   # left wall (points right)
    2: np.array([0.0, 0.0, -1.0]),   # front wall (points toward viewer)
    3: np.array([1.0, 0.0, 0.0]),    # right wall (points left)
    4: np.array([0.0, 1.0, 0.0]),    # floor (points up)
}


def build_homography_matrix(theta: np.ndarray) -> np.ndarray:
    """
    Build 3x3 homography matrix from 8 theta parameters.

    Args:
        theta: 8-element array of transformation parameters

    Returns:
        3x3 homography matrix H
    """
    if len(theta) != 8:
        raise ValueError(f"Expected 8 theta parameters, got {len(theta)}")

    # Append 1.0 as the 9th element and reshape to 3x3
    theta_with_one = np.append(theta, 1.0)
    H = theta_with_one.reshape(3, 3)

    return H


def compute_normal_vectors(theta: np.ndarray,
                          wall_labels: List[int]) -> Dict[int, np.ndarray]:
    """
    Compute 3D normal vectors for walls using theta transformation parameters.

    Mathematical approach:
    - Points transform: p' = H * p
    - Normals transform: n' = (H^-1)^T * n

    Args:
        theta: 8-element array of transformation parameters
        wall_labels: List of wall labels to compute normals for (e.g., [1, 2, 3])

    Returns:
        Dictionary mapping wall_label to normalized 3D normal vector

    Raises:
        ValueError: If theta is invalid or matrix is singular
    """
    try:
        # Build homography matrix
        H = build_homography_matrix(theta)

        # Compute inverse
        H_inverse = np.linalg.inv(H)

        # Compute inverse transpose for normal transformation
        H_inv_transpose = H_inverse.T

        computed_normals = {}

        for label in wall_labels:
            if label not in REFERENCE_NORMALS:
                print(f"Warning: Unknown label {label}, skipping normal computation")
                continue

            # Get reference normal
            ref_normal = REFERENCE_NORMALS[label]

            # Transform normal: n' = (H^-1)^T * n
            transformed_normal = H_inv_transpose @ ref_normal

            # Normalize to unit length
            magnitude = np.linalg.norm(transformed_normal)

            if magnitude > 1e-6:  # Avoid division by zero
                normalized_normal = transformed_normal / magnitude
            else:
                # Fallback to reference normal if transformation produces near-zero vector
                print(f"Warning: Near-zero magnitude for label {label}, using reference normal")
                normalized_normal = ref_normal

            computed_normals[label] = normalized_normal

        return computed_normals

    except np.linalg.LinAlgError as e:
        print(f"Error: Matrix inversion failed - {e}")
        print("Falling back to reference normals")
        # Return reference normals as fallback
        return {label: REFERENCE_NORMALS[label].copy() for label in wall_labels
                if label in REFERENCE_NORMALS}


def validate_normal_consistency(normals_dict: Dict[int, np.ndarray],
                               tolerance: float = 0.1) -> Dict[str, bool]:
    """
    Validate geometric consistency of computed normals.

    Checks:
    1. All normals are unit vectors (magnitude = 1.0)
    2. Adjacent walls have approximately perpendicular normals (dot product ≈ 0)
    3. Floor and ceiling normals are opposite (dot product ≈ -1)

    Args:
        normals_dict: Dictionary mapping label to normal vector
        tolerance: Tolerance for floating point comparisons

    Returns:
        Dictionary with validation results
    """
    results = {
        'all_unit_vectors': True,
        'walls_perpendicular': True,
        'floor_ceiling_opposite': True,
        'details': []
    }

    # Check 1: Unit vectors
    for label, normal in normals_dict.items():
        magnitude = np.linalg.norm(normal)
        is_unit = abs(magnitude - 1.0) < tolerance

        if not is_unit:
            results['all_unit_vectors'] = False
            results['details'].append(
                f"Label {label}: magnitude = {magnitude:.4f} (expected 1.0)"
            )

    # Check 2: Wall perpendicularity (left, front, right should be mutually perpendicular)
    wall_pairs = [(1, 2), (2, 3), (1, 3)]  # (left, front), (front, right), (left, right)

    for label1, label2 in wall_pairs:
        if label1 in normals_dict and label2 in normals_dict:
            dot_product = np.dot(normals_dict[label1], normals_dict[label2])
            is_perpendicular = abs(dot_product) < tolerance

            if not is_perpendicular:
                results['walls_perpendicular'] = False
                results['details'].append(
                    f"Walls {label1} and {label2}: dot product = {dot_product:.4f} (expected ≈0)"
                )

    # Check 3: Floor and ceiling opposition
    if 0 in normals_dict and 4 in normals_dict:
        dot_product = np.dot(normals_dict[0], normals_dict[4])
        is_opposite = abs(dot_product + 1.0) < tolerance

        if not is_opposite:
            results['floor_ceiling_opposite'] = False
            results['details'].append(
                f"Floor and ceiling: dot product = {dot_product:.4f} (expected ≈-1)"
            )

    return results


def get_reference_normal(label: int) -> Optional[np.ndarray]:
    """
    Get reference cuboid normal for a given label.

    Args:
        label: Wall label (0=ceiling, 1=left, 2=front, 3=right, 4=floor)

    Returns:
        3D normal vector, or None if label unknown
    """
    return REFERENCE_NORMALS.get(label, None)


def visualize_normals_info(normals_dict: Dict[int, np.ndarray]) -> str:
    """
    Generate human-readable summary of computed normals.

    Args:
        normals_dict: Dictionary mapping label to normal vector

    Returns:
        Formatted string with normal information
    """
    label_names = {
        0: "Ceiling",
        1: "Left Wall",
        2: "Front Wall",
        3: "Right Wall",
        4: "Floor"
    }

    lines = ["Computed Normal Vectors:", "=" * 40]

    for label in sorted(normals_dict.keys()):
        normal = normals_dict[label]
        name = label_names.get(label, f"Surface {label}")
        magnitude = np.linalg.norm(normal)

        lines.append(f"{name} (label={label}):")
        lines.append(f"  Vector: [{normal[0]:7.4f}, {normal[1]:7.4f}, {normal[2]:7.4f}]")
        lines.append(f"  Magnitude: {magnitude:.6f}")
        lines.append("")

    return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    # Test with identity-like theta (minimal transformation)
    theta_identity = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    print("Test 1: Identity transformation")
    print("-" * 40)
    normals = compute_normal_vectors(theta_identity, [1, 2, 3])
    print(visualize_normals_info(normals))

    validation = validate_normal_consistency(normals)
    print("Validation Results:")
    print(f"  All unit vectors: {validation['all_unit_vectors']}")
    print(f"  Walls perpendicular: {validation['walls_perpendicular']}")
    if validation['details']:
        print("  Details:")
        for detail in validation['details']:
            print(f"    - {detail}")

    print("\n" + "=" * 40)
    print("Test 2: Real theta from room1.jpeg")
    print("-" * 40)
    # Theta values extracted from room1.jpeg
    theta_real = np.array([
        0.304548, 0.006075, -0.017495,
        0.011261, 0.424205, 0.141033,
        0.145683, -0.013271
    ])

    normals_real = compute_normal_vectors(theta_real, [1, 2, 3, 4, 0])
    print(visualize_normals_info(normals_real))

    validation_real = validate_normal_consistency(normals_real)
    print("Validation Results:")
    print(f"  All unit vectors: {validation_real['all_unit_vectors']}")
    print(f"  Walls perpendicular: {validation_real['walls_perpendicular']}")
    print(f"  Floor/ceiling opposite: {validation_real['floor_ceiling_opposite']}")
    if validation_real['details']:
        print("  Details:")
        for detail in validation_real['details']:
            print(f"    - {detail}")
