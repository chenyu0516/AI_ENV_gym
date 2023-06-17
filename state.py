import numpy as np


class Observe_State_Change:
    def state_update(self, winning, value, data_list):

        state = np.array([winning, value], dtype= np.float32)
        for items in data_list:
                state = np.append(state, items)

        return state

    def log(self, input):
        input = np.array(input)  # ensures the input is a numpy array
        output = np.empty_like(input)  # prepare an output array of the same size

        # Using numpy's array boolean indexing to apply the operation to elements that meet the condition
        mask = (-np.exp(1) <= input) & (input <= np.exp(1))
        output[mask] = input[mask]/np.exp(1)
    
        mask = input > np.exp(1)
        output[mask] = np.log(input[mask])+1

        mask = input < -np.exp(1)
        output[mask] = -np.log(-input[mask])-1
    
        return output



