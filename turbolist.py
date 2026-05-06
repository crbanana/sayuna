class MyList:
    def __init__(self, iterable=None):
        self._data = list(iterable) if iterable is not None else []

    def __getitem__(self, index):
        return self._data[index]

    def __setitem__(self, index, value):
        self._data[index] = value

    def __delitem__(self, index):
        del self._data[index]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __contains__(self, item):
        return item in self._data

    def __repr__(self):
        return repr(self._data)

    def __str__(self):
        return str(self._data)

    def __eq__(self, other):
        if isinstance(other, MyList):
            return self._data == other._data
        return self._data == other

    def __add__(self, other):
        return self._data + list(other)

    def __radd__(self, other):
        return list(other) + self._data

    def __iadd__(self, other):
        self._data += list(other)
        return self

    def __mul__(self, n):
        return self._data * n

    def __rmul__(self, n):
        return n * self._data

    def __imul__(self, n):
        self._data *= n
        return self

    def append(self, item):
        self._data.append(item)

    def extend(self, iterable):
        self._data.extend(iterable)

    def insert(self, index, item):
        self._data.insert(index, item)

    def remove(self, item):
        self._data.remove(item)

    def pop(self, index=-1):
        return self._data.pop(index)

    def clear(self):
        self._data.clear()

    def index(self, item, *args):
        return self._data.index(item, *args)

    def count(self, item):
        return self._data.count(item)

    def sort(self, *args, **kwargs):
        self._data.sort(*args, **kwargs)

    def reverse(self):
        self._data.reverse()

    def copy(self):
        return MyList(self._data.copy())
