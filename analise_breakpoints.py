# -*- coding: utf-8 -*-
import sys
import getopt
import numpy as np
from artificial_data import generate_artificial_data


def norm(xs):
    mmax = max(xs)
    mmin = min(xs)

    return (xs - mmin) / (mmax - mmin)


def get_args_terminal():
    argv = sys.argv[1:]

    source = None
    output = None
    N = 5
    try:
        opts, args = getopt.getopt(argv, "s:o:n:")
    except getopt.GetoptError:
        print('usage: example.py -s <source> -o <output> -N <n>')

    for opt, arg in opts:

        if opt == '-s':
            source = arg
        elif opt == '-o':
            output = arg
        elif opt == '-N':
            N = arg
    return source, output, N


if __name__ == "__main__":
    source, output, N = get_args_terminal()
    print(source, output, N)

    nx = 10
    ny = 10

    minx, maxx = 0, 90
    deltax = (maxx - minx) / nx
    intervalsx = np.arange(minx, maxx + deltax, deltax)

    miny, maxy = 0, 1
    deltay = (maxy - miny) / ny
    intervalsy = np.arange(miny, maxy + deltay, deltay)

    args = [intervalsx, intervalsy]

    # xs,ys = read_file_original(filename='data/plos_one_2019.txt')
    # xs = np.asarray([norm(x) for x in xs])
    # ys = np.asarray([norm(y) for y in ys])

    generate_artificial_data(N, intervalsx, intervalsy, maxx, source, output)
