import numpy as np


class Histo(object):
    def __init__(self, data=None):
        if data is not None:
            self.set_by_iterable(data)
        else:
            data = []
            self.set_by_iterable(data)

    def set_by_iterable(self, data):
        self.frequency = np.array(data)
        self.bucketSize = len(data)
        self.range = np.arange(self.bucketSize)

    def print(self):
        for i in range(self.bucketSize):
            print("{}: {}".format(self.range[i], self.frequency[i]))

    def publish(self):
        T = self.range[-1] + 1
        new_freq = []
        left = -1
        for freq, right in zip(self.frequency, self.range):
            bucket_len = right - left
            left = right
            if bucket_len <= 0:
                raise ValueError("bucketLen cannot be negative")

            new_freq.extend([freq / bucket_len] * bucket_len)
        self.frequency = np.array(new_freq, dtype=np.float64)
        self.range = np.arange(T, dtype=np.int32)
        self.bucketSize = T