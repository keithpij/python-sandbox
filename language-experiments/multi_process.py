from multiprocessing import Pool
import os


def f(x):
    print('process id:', os.getpid())
    return x*x


if __name__ == '__main__':
    with Pool(5) as p:
        print(p.map(f, [1, 2, 3, 4, 5, 6]))