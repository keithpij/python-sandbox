'''
The built-in function zip.

zip(*iterables)
Make an iterator that aggregates elements from each of the iterables.

Returns an iterator of tuples, where the i-th tuple contains the i-th element from 
each of the argument sequences or iterables. The iterator stops when the shortest 
input iterable is exhausted. With a single iterable argument, it returns an iterator 
of 1-tuples. With no arguments, it returns an empty iterator. 

Another collection that can be iterated over is a zip. A zip is constructed from other collections
typically of the same length. Each element of the zip is a tuple consisting of one element from each
of the input collections. 
'''


# zip is equivalent to the following code.
def my_zip(*iterables):
    # zip('ABCD', 'xy') --> Ax By
    sentinel = object()
    iterators = [iter(it) for it in iterables]
    while iterators:
        result = []
        for it in iterators:
            elem = next(it, sentinel)
            if elem is sentinel:
                return
            result.append(elem)
        yield tuple(result)


# zip can be used to return matching tuples.
x = [1, 2, 3]
y = [4, 5, 6]
zipped = zip(x, y)
#zipped = my_zip(x, y) # my_zip will produce the same results.
print(list(zipped))


# zip can be used as to feed a loop.
for i, j in zip(x, y):
    print(str(i) + '  ' + str(j))


# This is from the online documentation. Not sure what purpose this serves but it can be done.
# This merely turns the inputs to tuples.
x2, y2 = zip(*zip(x, y))
print(x2)
print(y2)
same = (x == list(x2) and y == list(y2))
print(same)
