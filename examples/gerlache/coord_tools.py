import numpy as np

def sph2cart(r, lat, lon):
    """
    Transform spherical (meters, degrees) to cartesian (meters)
    Args:
        r: meters
        lat: degrees
        lon: degrees

    Returns:
    Cartesian xyz (meters)
    """
    x = r * cosd(lon) * cosd(lat)
    y = r * sind(lon) * cosd(lat)
    z = r * sind(lat)

    return x, y, z


def sind(x):
    return np.sin(np.deg2rad(x))


def cosd(x):
    return np.cos(np.deg2rad(x))


def unproject_stereographic(x, y, lon0, lat0, R):
    """
    Stereographic Coordinates unprojection
    Args:
        x: stereo coord
        y: stereo coord
        lon0: center of the projection (longitude, deg)
        lat0: center of the projection (latitude, deg)
        R: planet radius

    Returns:
    Longitude and latitude (deg) of points in cylindrical coordinates
    """
    rho = np.sqrt(np.power(x, 2) + np.power(y, 2))
    c = 2 * np.arctan2(rho, 2 * R)

    lat = np.rad2deg(np.arcsin(np.cos(c) * sind(lat0) + (cosd(lat0) * y * np.sin(c)) / rho))
    lon = np.mod(
        lon0 + np.rad2deg(np.arctan2(x * np.sin(c), cosd(lat0) * rho * np.cos(c) - sind(lat0) * y * np.sin(c))), 360)

    if (x == 0).any() and (y == 0).any():
        #    if x == 0 and y == 0:
        return lon0, lat0
    else:
        return lon, lat