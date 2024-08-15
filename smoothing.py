import numpy as np


class TemporalSmoothing:
    def __init__(self, history_size=5):
        self.history_size = history_size
        self.depth_maps = []

    def add_depth_map(self, depth_map):
        if len(self.depth_maps) >= self.history_size:
            self.depth_maps.pop(0)
        self.depth_maps.append(depth_map)

    def get_smoothed_depth_map(self):
        if not self.depth_maps:
            return None
        return np.mean(np.stack(self.depth_maps), axis=0)
