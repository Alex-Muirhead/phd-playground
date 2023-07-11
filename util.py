from statistics import mean, stdev
from time import perf_counter_ns


class Timer:

    def __init__(self, info=""):
        self.info = info
        self._timings = []
        self._laststart = None

    def start(self):
        self._laststart = perf_counter_ns()

    def stop(self):
        time = perf_counter_ns()
        if self._laststart is None:
            raise ValueError("Can't stop a timer that isn't started!")
        self._timings.append(time - self._laststart)
        self._laststart = None

    def report(self):
        loops = len(self._timings)
        avg = mean(self._timings)
        std = stdev(self._timings)
        message = f"{self.format_time(avg)} ± {self.format_time(std)} per loop ({loops:,} loops)"
        if self.info:
            message = self.info + ": " + message
        return message

    @staticmethod
    def format_time(time, precision=3):
        units  = ("ns", "µs", "ms", "s")
        scales = (1E3, 1E3, 1E3)
        for unit, scale in zip(units, scales):
            if time / scale < 1:
                break
            time /= scale
        return f"{time:.{precision}g} {unit}"


def endrange(start, stop=None, step=1):
    if stop is None:
        start, stop = 0, start
    return range(start, stop+step, step)


def endarange(start, stop=None, step=1, *args):
    from numpy import arange
    if stop is None:
        start, stop = 0, start
    return arange(start, stop+step, step, *args, dtype=type(step))
