from collections import defaultdict
from typing import List, Iterable


class Memory:
    def __init__(self, columns):
        self.columns = columns
        self._storage = defaultdict(list, {col: [] for col in columns})

    def store(self, item: Iterable) -> None:
        for k, v in zip(self.columns, item):
            self._storage[k].append(v)

    def append_column(self, column, values):
        self._storage[column].append(values)

    def extend_column(self, column, values):
        self._storage[column].extend(values)

    def reset(self) -> None:
        for k in self._storage.keys():
            self._storage[k] = []

    def extend(self, mem: "Memory") -> None:
        assert mem.columns == self.columns, "Memories should share the same columns to be extended"
        for col in mem._storage.keys():
            self._storage[col].extend(mem.get_columns([col])[0])

    def get_rows(self) -> List:
        return list(zip(self._storage[col] for col in self.columns))

    def get_columns(self, columns):
        return [self._storage[column] for column in columns]
