from typing import Tuple, List


class Memory:
    def __init__(self, reverse: bool = False):
        self.reverse = reverse
        self._storage = []

    def store(self, item: Tuple) -> None:
        self._storage.append(item)

    def reset(self) -> None:
        self._storage = []

    def get_values(self) -> List:
        if self.reverse:
            return list(zip(*self._storage))
        else:
            return self._storage
