__all__ = ["run_profiler"]

from cProfile import Profile
from pstats import Stats, SortKey
from io import StringIO

# https://docs.python.org/3/library/profile.html#profile.Profile
def run_profiler(func, sort_key: SortKey = SortKey.CUMULATIVE, **kwargs):
    pr = Profile()
    pr.enable()
    func(**kwargs)
    pr.disable()
    s = StringIO()
    ps = Stats(pr, stream=s).sort_stats(sort_key)
    ps.print_stats()
    print(s.getvalue())

