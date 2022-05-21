import sys

sys.path.append("..")
import tightbinding.moire_tb as tbtb

from cProfile import Profile
from pstats import Stats

__author__ = "Wangqian Miao"


def test_profile(n_moire, n_g, n_k):
    tbtb.tightbinding_solver(n_moire, n_g, n_k, "atom", "+1", disp=False)


def main():
    test_profile(30, 5, 3)


if __name__ == "__main__":
    profiler = Profile()
    profiler.runcall(main)
    stats = Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats()
