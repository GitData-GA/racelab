import numpy as np
from shapely.geometry import Point, Polygon


def menger_curvature(pt1, pt2, pt3, atol):
    """Calculate Menger curvature for three points with optimizations."""
    x21, y21 = pt1[0] - pt2[0], pt1[1] - pt2[1]
    x23, y23 = pt3[0] - pt2[0], pt3[1] - pt2[1]
    
    norm21 = np.hypot(x21, y21)
    norm23 = np.hypot(x23, y23)
    
    if norm21 == 0 or norm23 == 0:
        return 0.0 
    
    dot_product = x21 * x23 + y21 * y23
    cos_theta = dot_product / (norm21 * norm23)
    
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    
    if np.isclose(theta, np.pi, atol=atol):
        theta = 0.0
    
    dist13 = np.hypot(x21 - x23, y21 - y23)
    
    if dist13 == 0:
        return 0.0  
    
    return 2 * np.sin(theta) / dist13


def refine_point(
    point_index,
    refined_line,
    inner_polygon,
    outer_polygon,
    target_curvature,
    xi_iterations,
    atol,
):
    """Refine a single point's position iteratively to match the target curvature."""
    total_points = len(refined_line)

    prev_idx = (point_index - 1 + total_points) % total_points
    next_idx = (point_index + 1 + total_points) % total_points

    current_point = refined_line[point_index]
    lower_bound = current_point
    upper_bound = (
        (refined_line[next_idx][0] + refined_line[prev_idx][0]) / 2.0,
        (refined_line[next_idx][1] + refined_line[prev_idx][1]) / 2.0,
    )
    prospective_point = current_point

    for _ in range(xi_iterations):
        prospective_curvature = menger_curvature(
            refined_line[prev_idx], prospective_point, refined_line[next_idx], atol
        )

        if np.isclose(prospective_curvature, target_curvature):
            break

        if prospective_curvature < target_curvature:
            # Too flat, adjust bounds
            upper_bound = prospective_point
            midpoint = (
                (lower_bound[0] + prospective_point[0]) / 2.0,
                (lower_bound[1] + prospective_point[1]) / 2.0,
            )
            if Point(midpoint).within(inner_polygon) or not Point(midpoint).within(
                outer_polygon
            ):
                lower_bound = midpoint
            else:
                prospective_point = midpoint
        else:
            # Too curved, adjust bounds
            lower_bound = prospective_point
            midpoint = (
                (upper_bound[0] + prospective_point[0]) / 2.0,
                (upper_bound[1] + prospective_point[1]) / 2.0,
            )
            if Point(midpoint).within(inner_polygon) or not Point(midpoint).within(
                outer_polygon
            ):
                upper_bound = midpoint
            else:
                prospective_point = midpoint

    return prospective_point


def refine_line(track, refined_line, inner_polygon, outer_polygon, xi_iterations, atol):
    """Refine the racing line for a single iteration."""
    num_points = len(refined_line)

    for point_index in range(num_points):
        # Surrounding points
        prev2_idx = (point_index - 2 + num_points) % num_points
        prev_idx = (point_index - 1 + num_points) % num_points
        next_idx = (point_index + 1 + num_points) % num_points
        next2_idx = (point_index + 2 + num_points) % num_points

        # Calculate curvatures
        curvature_before = menger_curvature(
            refined_line[prev2_idx],
            refined_line[prev_idx],
            refined_line[point_index],
            atol,
        )
        curvature_after = menger_curvature(
            refined_line[point_index],
            refined_line[next_idx],
            refined_line[next2_idx],
            atol,
        )
        target_curvature = (curvature_before + curvature_after) / 2

        # Refine the current point
        refined_line[point_index] = refine_point(
            point_index,
            refined_line,
            inner_polygon,
            outer_polygon,
            target_curvature,
            xi_iterations,
            atol,
        )

    return refined_line
