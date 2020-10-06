from collections import defaultdict
from typing import List, Iterable

import numpy as np


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

    def get_columns(self, columns) -> List[List]:
        return [self._storage[column] for column in columns]

    def get_batch(self, columns, batch=None):
        column_batches = self.get_columns(columns)
        if not batch:
            yield [column_batches]
        else:
            n = len(column_batches[0])
            for i in range(n // batch):
                yield [col[i * batch: (i + 1) * batch] for col in column_batches]
            if n % batch:
                yield [col[(n // batch) * batch:] for col in column_batches]

    def normalize_columns(self, columns):
        for col in columns:
            mean = np.mean(self._storage[col])
            std = np.std(self._storage[col]) + np.finfo(float).eps
            self._storage[col] -= mean
            self._storage[col] /= std
