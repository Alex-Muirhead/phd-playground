import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import Delaunay
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.stats import qmc

# Set seed to fixed value for repeatability
# np.random.seed(0)

def get_edges(simplex):
    """Get edges from vertex index triplet"""
    vertices = np.tile(simplex, 2)
    face_dim = len(simplex)
    edge_dim = face_dim - 1
    return np.array([
        vertices[i:i+edge_dim]
        for i in range(1, face_dim+1)
    ])

rng = np.random.default_rng()
radius = 0.1
# Sample points somewhat evenly
engine = qmc.PoissonDisk(d=2, radius=radius, seed=rng)
points = engine.random(20)
rough_centre = np.median(points, axis=0)

tri = Delaunay(points)
vor = Voronoi(points)

edges = np.array([get_edges(simplex) for simplex in tri.simplices])

def normal_2D(vec):
    # Arbitary decision currently
    direction = np.array([vec[1], -vec[0]])
    return direction / np.linalg.norm(direction)

normals = np.array([
    [normal_2D(points[b] - points[a]) for (a, b) in simplex_edges]
    for simplex_edges in edges
])

source_index = tri.find_simplex(rough_centre)
source_simplex = tri.simplices[source_index]

centroids = np.asarray([np.mean(points[simplex], axis=0) for simplex in tri.simplices])
source_point = centroids[source_index, :]
source_x, source_y = source_point

fig, ax = plt.subplots()
ax.quiver(*np.tile(source_point, (3, 1)).T, *normals[source_index, :, :].T)

# Basic 4 cardinal directions
directions = np.array([
    [+1,  0],
    [ 0, +1],
    [-1,  0],
    [ 0, -1]
]) 

for i in range(tri.nsimplex):
    # Don't do any propagation on the source-point
    if i == source_index:
        continue
    neighbours = tri.neighbors[i, :]
    local_intensity = []
    for n, indx in enumerate(neighbours):
        # Oh boy, triple nested for-loop already...
        if indx == source_index:
            # Do something special here to get "source" intensities
            
            continue
        
        # for intensity in intensities[indx]:
            # pass            

x, y = np.meshgrid(np.linspace(0, 1, 101), np.linspace(0, 1, 101))
r = np.sqrt((x - source_x)**2 + (y - source_y)**2)
intensity = 1 / r**2
intensity = np.nan_to_num(intensity, posinf=np.nan)
intensity = np.clip(intensity, a_max=100, a_min=None)

ax.contourf(x, y, intensity, levels=50, alpha=0.2)
voronoi_plot_2d(vor, ax, line_colors='blue', show_vertices=False)
ax.plot(*centroids.T, color="black", ls="", marker="+")
ax.triplot(*points.T, tri.simplices, color="black")
ax.set(xlim=[None, 1], ylim=[None, 1])
plt.show()

