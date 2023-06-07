import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate


if __name__ == "__main__":
	# Run some unit tests

	# Build up coordinate system (2D cartesian)
	Ni, Nj = (21, 21)
	x, y = np.mgrid[0:1:Ni*1j, 0:1:Nj*1j]  # From 0->1 for simplicity

	# Using single temperature model of gas
	T = np.full(x.shape, 20_000)    # [K]
	p = np.full(x.shape, 100_000)   # [Pa]
