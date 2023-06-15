import numpy as np


class Observe_State_Change:
    def state_update(self, winning, value, data_list):

         return (np.array([winning], dtype=np.float32),
                 np.array([value], dtype=np.float32),
                 np.array([data_list[0]], dtype=np.float32),
                 np.array([data_list[1]], dtype=np.float32),
                 np.array([data_list[2]], dtype=np.float32),
                 np.array([data_list[3]], dtype=np.float32),
                 np.array([data_list[4]], dtype=np.float32),
                 np.array([data_list[5]], dtype=np.float32),
                 np.array([data_list[6]], dtype=np.float32),
                 np.array([data_list[7]], dtype=np.float32),
                 np.array([data_list[8]], dtype=np.float32),
                 np.array([data_list[9]], dtype=np.float32))

