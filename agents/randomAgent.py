import numpy as np


class randomAgent:

    def __init__(self, cfg:dict):
        self._K = cfg["K"]
        self._U = cfg["U"]

    def predict(self):
        schedule = np.random.randint(2, size=(self._K, self._U))
        return schedule
