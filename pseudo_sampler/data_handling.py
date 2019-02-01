import numpy as np


class BatchMaker(object):
    def __init__(self):
        self.batch_number = -1
    def load_data(self,size):
        self.size = size
        self.shuffle()
    
    def shuffle(self):
        self.current_q = np.random.permutation(self.size)
        self.head = 0
        self.batch_number += 1
    def get_batch(self,batch_size):
        if self.current_q.shape[0]-self.head >= batch_size:
            result = self.current_q[self.head:self.head+batch_size]
            self.head += batch_size
            return result
        else:
            if self.head == self.current_q.shape[0]:
                self.shuffle()
                self.head = batch_size
                return self.current_q[:batch_size]
            else:
                result = self.current_q[self.head:]
                self.shuffle()
                return result
        return None


def normalizer(data):
    min_con = np.min(data,axis=0)
    max_con = np.max(data,axis=0)
    data = data-min_con
    data = data/(max_con-min_con +(1e-10))
    return data