import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate


def within_cell(coord, vertices, start_index=0):
    # Use cross product to determine whether point is within cell
    n = len(vertices)
    i = start_index
    new = vertices[i]
    for _ in range(n):
        i = (i+1) % n
        old, new = new, vertices[i]
        if np.cross(new-old, coord-old) >= 0:
            return False
    return True


def cross_boundary(coord, vertices, start_index=0):
    # Use cross product to determine whether point is within cell
    n = len(vertices)
    i = start_index
    new = vertices[i]
    for _ in range(n):
        i = (i+1) % n
        old, new = new, vertices[i]
        if np.cross(new-old, coord-old) >= 0:
            return i
    return -1


if __name__ == "__main__":
    # Run some unit tests

    # Build up coordinate system (2D cartesian)
    Ni, Nj = (25, 25)
    x, y = np.mgrid[0:1:(Ni+1)*1j, 0:1:(Nj+1)*1j]  # From 0->1 for simplicity
    vertices = np.vstack((x.flatten(), y.flatten())).T

    cell_ids = np.arange(Ni*Nj)

    neighbour_ids = []
    for j in range(Nj):
        for i in range(Ni):
            s = i + j*Ni
            neighbours = []
            if 1 <= i:
                neighbours.append(s-1)
            else:
                neighbours.append(-1)
            if i < (Ni-1):
                neighbours.append(s+1)
            else:
                neighbours.append(-1)
            if 1 <= j:
                neighbours.append(s-Ni)
            else:
                neighbours.append(-1)
            if j < (Nj-1):
                neighbours.append(s+Ni)
            else:
                neighbours.append(-1)
            neighbour_ids.append(neighbours)

    vertex_ids = []
    for j in range(Nj):
        for i in range(Ni):
            s = i + j*(Ni+1)
            # Anti-clockwise listing. Width is (Ni+1) for vertices
            vertex_ids.append([s, s+1, s+(Ni+1)+1, s+(Ni+1)])

    cell_centers = np.zeros((Ni*Nj, 2))
    for j in range(Nj):
        for i in range(Ni):
            s = i + j*Ni
            cell_centers[s, :] = np.mean(vertices[vertex_ids[s], :], axis=0)

    start_cell = np.random.randint(0, Ni*Nj)
    start_i, start_j = start_cell % Ni, int(start_cell // Ni)
    # Initially just use the center of the cell
    start_coord = np.mean(vertices[vertex_ids[start_cell], :], axis=0)

    # Create constant step vector
    step_length = 1.0 / (Ni+Nj)
    direction = np.random.rand() * (2*np.pi)
    step = step_length * np.array([np.cos(direction), np.sin(direction)])

    n_steps   = 20
    ray_path  = np.zeros((n_steps+1, 2))
    ray_coord = start_coord.copy()
    ray_path[0, :] = ray_coord
    print(f"Starting at cell {start_cell}")

    for n in range(n_steps):
        ray_coord += step
        boundary = cross_boundary(ray_coord, vertices[vertex_ids[start_cell]])
        while boundary != -1:
            start_cell = neighbour_ids[start_cell][boundary]
            if start_cell == -1:
                print("Ray has exited domain")
                break
            boundary = cross_boundary(ray_coord, vertices[vertex_ids[start_cell]])
            print(f"Moved to cell {start_cell} over boundary {boundary}")
        ray_path[n+1, :] = ray_coord

    X, Y = cell_centers.T  # Unpack cell centeres

    # NOTE: reshaping, since pcolormesh requires a grid
    X, Y = X.reshape((Nj, Ni)), Y.reshape((Nj, Ni))
    p = np.exp(-(0.5 - X)**2 - (0.5 - Y)**2)

    fig, ax = plt.subplots()
    ax.pcolormesh(x, y, p, shading="flat")
    ax.invert_yaxis()
    ax.plot(*ray_path.T, 'k-', marker='*')
    ax.plot(*start_coord, 'r', marker='*')
    plt.show()
