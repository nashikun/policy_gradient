class Memory:
    def __init__(self, reverse=False):
        self.reverse = reverse
        self._storage = []

    def store(self, item):
        self._storage.append(item)

    def reset(self):
        self._storage = []

    def get_values(self):
        if self.reverse:
            return list(zip(*self._storage))
        else:
            return self._storage
