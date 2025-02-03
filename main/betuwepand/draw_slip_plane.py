import numpy as np
from geometry_equations import intersection


def draw_bishop_slip_plane(x_center, z_center, radius, surface_line):
    """
    Draw a Bishop slip plane given its center, radius, and a surface line.

    Args:
        x_center (float): X-coordinate of the center of the slip plane.
        z_center (float): Z-coordinate of the center of the slip plane.
        radius (float): Radius of the slip plane.
        surface_line (dictionary): Dictionary containing coordinates of a surface line ('s', 'z').

    Returns:
        tuple: A tuple containing X and Z coordinates of the portion of the slip plane within the surface line.
    """

    t = np.linspace(0, np.pi, 1000)
    x = x_center - radius * np.cos(t)
    z = z_center - radius * np.sin(t)

    intersections = intersection(np.asarray(surface_line['s']), np.asarray(surface_line['z']), x, z)

    try:
        mask = np.where((x > intersections[0][0]) & (x < intersections[0][1]))
    except IndexError:
        mask = np.where(x > intersections[0][0])

    return x[mask], z[mask]


def draw_lift_van_slip_plane(x_center_left, z_center_left, x_center_right, z_center_right, tangent_line, surface_line):
    """
    Draw a lift-van slip plane given its centers, tangent line, and a surface line.

    Args:
        x_center_left (float): X-coordinate of the left center of the slip plane.
        z_center_left (float): Z-coordinate of the left center of the slip plane.
        x_center_right (float): X-coordinate of the right center of the slip plane.
        z_center_right (float): Z-coordinate of the right center of the slip plane.
        tangent_line (float): Tangent line value.
        surface_line (dictionary): Dictionary containing coordinates of a surface line ('s', 'z').

    Returns:
        tuple: A tuple containing X and Z coordinates of the portion of the slip plane within the surface line.
    """

    temp1 = np.linspace(0, np.pi / 2, 1000)
    radius_left = z_center_left - tangent_line
    radius_right = z_center_right - tangent_line

    xr = radius_right * np.sin(temp1) + x_center_right
    zr = -radius_right * np.cos(temp1) + z_center_right
    xl = np.flip(-radius_left * np.sin(temp1) + x_center_left)
    zl = np.flip(-radius_left * np.cos(temp1) + z_center_left)

    x = np.concatenate((xl, xr), axis=0)
    z = np.concatenate((zl, zr), axis=0)

    intersections = intersection(np.asarray(surface_line['s']), np.asarray(surface_line['z']), x, z)

    try:
        mask = np.where((x > intersections[0][0]) & (x < intersections[0][1]))
    except IndexError:
        mask = np.where(x > intersections[0][0])

    return x[mask], z[mask]
