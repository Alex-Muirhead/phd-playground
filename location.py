import heapq
import random

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from numba.typed import List

from util import Timer, endarange


@nb.njit
def reservoir_sample(weights, size, replacements=0):
    weight_stream = iter(weights)
    bins = replacements + 1
    k = size // bins

    # Add a single item to each heap to fix the type
    # Only required for Numba
    heap = List()
    w = next(weight_stream)
    for j in range(bins):
        u = random.random()
        item = (pow(u, 1/w), 0)
        heap.append(List([item]))

    # Add the next (k-1) numbers to each heap to fill them
    for i, w in zip(range(1, k), weight_stream):
        for j in range(bins):
            u = random.random()
            item = (pow(u, 1/w), i)
            heap[j].append(item)

    # Turn each "heap" into the proper structure
    for j in range(bins):
        heapq.heapify(heap[j])

    # Filter through the remaining weights using AE alg.
    for i, w in enumerate(weight_stream, k):
        for j in range(bins):
            u = random.random()
            item = (pow(u, 1/w), i)
            # If the item is bigger than any on the heap
            if item > heap[j][0]:
                heapq.heapreplace(heap[j], item)

    # Extract indices from the heap, discard random weights
    samples = List()
    for j in range(bins):
        for _, i in heap[j]:
            samples.append(i)

    return samples


if __name__ == "__main__":
    # Create some sample data
    from scipy import stats
    import pandas as pd

    dist = stats.norm
    domain = dist.interval(0.99)

    x = np.linspace(*domain, num=1000)
    y = dist.pdf(x)

    # The number of random numbers we want
    # size =   500
    # size = 1_000
    size = 2_000
    repeats = 100

    expected = stats.poisson.ppf(0.99, (size/1000)**2)
    print(f"Expected repeats: {int(expected)} or {expected/size:.2%}")

    fig, ax = plt.subplots()
    ax.plot(x, y)

    bin_edges = np.linspace(*domain, 11)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # Cold call to warm up the JIT
    _ = reservoir_sample(y, 50, replacements=0)

    data = pd.DataFrame({"x": bin_edges})

    for r_percent in endarange(0, 0.02, 0.002):
        r = int(size * r_percent)
        hist_avg = np.zeros(len(bin_centers))
        timer = Timer(f"Sample with {r:>2} repeats")

        for repeat_id in range(repeats):
            timer.start()
            sample_ids = reservoir_sample(y, size, replacements=r)
            timer.stop()

            # Use the value ids to collect the samples
            samples = [x[i] for i in sample_ids]
            hist, _ = np.histogram(samples, density=True, bins=bin_edges)
            hist_avg += hist

        hist_avg /= repeats
        ax.scatter(bin_centers, hist_avg, label=f"{r_percent:.1%} replacements")
        print(timer.report())

        data[f"{r}r"] = np.pad(hist_avg, (0, 1), mode='edge')

        err = np.mean((dist.pdf(bin_centers) - hist_avg)**2)
        print(f"L2 error: {err:.3g}")

    data.to_csv("data/oversample_loc.csv", index=False)

    ax.legend()
    plt.show(block=True)
