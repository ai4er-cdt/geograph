"""General project util functions"""
import time
from functools import wraps


def timeit(method):
    """
    timeit is a wrapper for performance analysis which should
    return the time taken for a function to run,
    :param method: the function that it takes as an input
    :return: timed
    example usage:
    tmp_log_data={}
    part = spin_forward(400, co, particles=copy.deepcopy(particles),
                        log_time=tmp_log_d)
    # chuck it into part to stop interference.
    assert part != particles
    spin_round_time[key].append(tmp_log_data['SPIN_FORWARD'])
    @timeit
    """

    @wraps(method)
    def timed(*args, **kw):
        ts = time.perf_counter()
        result = method(*args, **kw)
        te = time.perf_counter()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = te - ts
        else:
            print("%r  %2.5f s\n" % (method.__name__, (te - ts)))
        return result

    return timed
