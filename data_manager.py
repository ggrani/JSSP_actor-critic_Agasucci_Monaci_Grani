import math
import random

class data_manager():
    def __init__(self, stacklength = 10000, seed = None):
        self._memoryx = []
        self._memoryy = []
        self._stacklength = stacklength
        self._length = 0
        self.seed = seed
        if seed != None:
            random.seed(a= seed)


    def memoryrecoverCSV(self, path):
        # read csv and add data to memory
        return True


    def add(self, instancex, instancey):
        self._length += 1
        if self._length <= self._stacklength:
            self._memoryx.append(instancex)
            self._memoryy.append(instancey)
        else:
            ind = self._length % self._stacklength
            self._memoryx[ind] = instancex
            self._memoryy[ind] = instancey


    def get_batch(self, batchsize, last = None):
        batchsize = min(self._length, batchsize, self._stacklength)
        if last is None:
            indices = random.sample(range(min(self._stacklength, self._length)), batchsize)
        else:
            bsize = round(batchsize*last)
            bbsize = batchsize-bsize
            indices = random.sample(range(min(self._stacklength, self._length)), bsize) + list(range(max(0,min(self._stacklength, self._length)-bbsize),min(self._stacklength, self._length)))
        xx = [self._memoryx[i] for i in indices]
        yy = [self._memoryy[i] for i in indices]
        return xx, yy

    def clear(self, newlength=None):
        self._memoryx = []
        self._memoryy = []
        self._length = 0

        if newlength is not None:
            self._stacklength = newlength
    def size(self):
        return self._length/self._stacklength









