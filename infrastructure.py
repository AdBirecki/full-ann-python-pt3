import numpy as np
from math import radians, cos, sin, asin, sqrt

# https://en.wikipedia.org/wiki/Haversine_formula


def haversine_distance(df, lat1, long1, lat2, long2):
    r = 6371

    phi1 = np.radians(df[lat1])
    phi2 = np.radians(df[lat2])

    delta_phi = np.radians(df[lat2]-df[lat1])
    delta_lambda = np.radians(df[long2]-df[long1])

    a = np.sin(delta_phi/2)**2 + np.cos(phi1) * \
        np.cos(phi2) * np.sin(delta_lambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = (r * c)

    return d
