import numpy as np


class Observe_State_Change:
    def state_update(self, winning, value, data_list):

        state = np.array([winning, value], dtype= np.float32)
        for items in data_list:
                state = np.append(state, items)

        return state


