'''
This module creates a race condition.
'''
from threading import Thread
import time

total_value = 0

def increment_total(loop):
    global total_value
    for _ in range(loop):
        new_value = total_value
        new_value = new_value + 1
        total_value = new_value


def fibonacci(sequence_size):
    sequence = []
    for i in range(0, sequence_size):
        if i < 2:
            sequence.append(i)
            continue
        sequence.append(sequence[i-1]+sequence[i-2])
    return sequence_size


def linear_equation(x_start, x_end):
    m = 5
    b = 7
    x_values = []
    y_values = []
    for x in range(x_start, x_end):
        x_values.append(x)
        y_values.append(m*x+b)
    return x_values, y_values


def create_threads():
    global total_value
    loop = 1000000
    t1 = Thread(target = increment_total, args = (loop, ))
    t2 = Thread(target = increment_total, args = (loop, ))
    t1.start()
    t2.start()
    t1.join()
    t2.join()


def main():
    create_threads()
    print(total_value)


if __name__ == '__main__':
    main()
