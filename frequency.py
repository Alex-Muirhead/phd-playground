import numpy as np
import matplotlib.pyplot as plt

from scipy.stats.sampling import NumericalInversePolynomial
from scipy.special import exprel


class Plank:

    @staticmethod
    def pdf(x: float) -> float:
        """Return a normalised value of the non-dimensional Plank's Law.
        Let x = hv/(kT), and use exprel = x/(exp(x)-1) for high precision.
        """
        return x**2 / exprel(x) * 15/(np.pi**4)


class Frequency:

    def __init__(self):
        # NOTE: Domain is [0, inf), however value is below machine
        # precision after x=45
        self._generator = NumericalInversePolynomial(Plank, domain=(0, 45))

    def sample(self, temperature: float, size: int) -> np.ndarray:
        from scipy.constants import h, c, k
        x = self._generator.rvs(size)
        wavelength = h*c/(k*x) * temperature
        return wavelength

    def cdf(self, x: float) -> float:
        return self._generator.cdf(x)


if __name__ == "__main__":

    from scipy.integrate import solve_ivp

    # NOTE: Using solve_ivp allows for dynamic capturing of curve data
    # Endpoint found empirically, more testing can be done to find the
    # better cut-off point
    cdf = solve_ivp(
        lambda t, _: Plank.pdf(t), (0, 25), [0],
        vectorized=True, rtol=1e-06, atol=1e-06
    )

    generator = Frequency()

    # Check visually
    x = np.linspace(0, 25, num=51)
    y = generator._generator.cdf(x)

    plt.plot(cdf.t, cdf.y[0, :])
    plt.scatter(x, y)
    plt.show(block=True)
