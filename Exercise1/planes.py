import numpy as np


def point_to_plane_distance(points, normal, d):

    return np.abs(np.dot(points, normal) - d)


def fit_plane(p0, p1, p2):

    vec1 = p1 - p0
    vec2 = p2 - p0
    normal = np.cross(vec1, vec2)

    if np.linalg.norm(normal) == 0:
        return None, None

    normal = normal / np.linalg.norm(normal)
    d = np.dot(normal, p0)

    return normal, d
