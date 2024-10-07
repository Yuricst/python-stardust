"""Miscellaneous functions"""

import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt


def vbprint(msg: str, verbose: bool = True) -> None:
    """Print message if verbose is True"""
    if verbose:
        print(msg)


def plot_sphere_wireframe(ax, radius, center, color='grey'):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = center[0] + radius * np.cos(u)*np.sin(v)
    y = center[1] + radius * np.sin(u)*np.sin(v)
    z = center[2] + radius * np.cos(v)
    ax.plot_wireframe(x, y, z, color=color)
    return
