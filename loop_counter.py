import numpy as np
import pickle

class LoopCounter:
    def __init__(self):
        self.loop_hashtable = {}
        self.loop_hashtable['overall'] = [[0, 0, [0,0]]]

    def add_loop_count(self, size, pos, steps):
        if str(size) not in self.loop_hashtable:
            self.loop_hashtable[str(size)] = [[1, steps, pos]]
        else:
            self.loop_hashtable[str(size)].append([self.loop_hashtable[str(size)][-1][0] + 1, steps, pos])

        self.inc_overall_loop_count(steps)

    def inc_overall_loop_count(self, steps):
        self.loop_hashtable['overall'].append([self.loop_hashtable['overall'][-1][0] + 1, steps])


    def save(self, filename):
        # open a file, where you ant to store the data
        file = open(filename, 'wb')

        # dump information to that file
        pickle.dump(self.loop_hashtable, file)

        # close the file
        file.close()
