import numpy as np

from alg.histo import Histo
from util import utils as ut


def calcSSE(cSum, csqrSum, i, j):
    if i == 0:
        out = csqrSum[j] - cSum[j] * cSum[j] / (j - i + 1)
    else:
        out = csqrSum[j] - csqrSum[i-1] - (cSum[j] - cSum[i-1]) * (cSum[j] - cSum[i-1]) / (j-i+1);
    if i == j:
        out = 0
    if out < 0:
        raise ValueError("SSE cannot be a negative value sse = {}, i = {}, j = {}".format(out, i, j))
    return out


def preSSE(cSum, csqrSum, row=None, col=None):
    bucketSize = len(cSum)
    if row is not None:
        precomp_SSE = np.zeros(bucketSize)
        if row == 0:
            precomp_SSE[:] = csqrSum[:] - cSum[:] * cSum[:] / np.arange(1, bucketSize+1)
        else:
            i = row
            precomp_SSE[i:] = csqrSum[i:] - csqrSum[i-1] - (cSum[i:] - cSum[i-1]) * (cSum[i:] - cSum[i-1]) / np.arange(1, bucketSize-i +1)
        return precomp_SSE

    if col is not None:
        precomp_SSE = np.zeros(bucketSize)
        if col == 0:
            raise NotImplementedError
        else:
            j = col
            precomp_SSE[0] = csqrSum[j] - cSum[j] * cSum[j] / (j+1)
            precomp_SSE[1:j+1] = csqrSum[j] - csqrSum[:j] - (cSum[j] - cSum[:j]) * (cSum[j] - cSum[:j]) \
                                / np.arange(j, 0, -1)
        return precomp_SSE


    precomp_SSE = np.zeros((bucketSize, bucketSize))
    precomp_SSE[0, :] = csqrSum[:] - cSum[:] * cSum[:] / np.arange(1, bucketSize+1)
    for i in range(1, bucketSize):
        precomp_SSE[i, i:] = csqrSum[i:] - csqrSum[i-1] - (cSum[i:] - cSum[i-1]) * (cSum[i:] - cSum[i-1]) / np.arange(1, bucketSize-i +1)
    return precomp_SSE


def minSSE(data, bucketSize):
    output = Histo([0.0] * bucketSize)
    cumulSum = data.frequency.cumsum()
    cumsqrSum = np.array([x**2 for x in data.frequency]).cumsum()
    # precomp_SSE = preSSE(cumulSum, cumsqrSum)

    SSE = np.zeros((data.bucketSize, bucketSize))
    DP_range = np.ones((data.bucketSize, bucketSize), dtype=np.int32) * -1

    SSE[:, 0] = preSSE(cumulSum, cumsqrSum, row=0)

    for j in range(1, bucketSize):
        for i in range(j, data.bucketSize):
            #tmp = SSE[:i, j-1] + precomp_SSE[1:i+1, i]
            tmp = SSE[j-1:i, j-1] + preSSE(cumulSum, cumsqrSum, col=i)[j:i+1]

            k = tmp.argmin()
            DP_range[i, j] = k + j - 1
            SSE[i, j] = tmp[k]


    k = data.bucketSize - 1
    for i in reversed(range(bucketSize)):
        if k == -1:
            raise ValueError("k error!")
        output.range[i] = data.range[k]
        k = DP_range[k][i]

    output.frequency[:] = 0

    j = 0
    for i in range(data.bucketSize):
        if output.range[j] >= data.range[i]:
            output.frequency[j] += data.frequency[i]
            if output.range[j] == data.range[i]:
                j += 1

    return output


def histoDP_shuffle_publish(eps, bucketSize, data, ratio=0.05, is_structure=False):
    return histoDP_plus_publish(eps, bucketSize, data, ratio=ratio, is_cluster=True, is_structure=is_structure)


def histoDP_plus_publish(eps, bucketSize, data, ratio=0.05, is_cluster=False, is_structure=False):
    eps1 = ratio * eps
    eps2 = (1-ratio) * eps

    copyHisto = Histo()
    copyHisto.set_by_iterable(data.frequency)
    if not is_structure:
        copyHisto.frequency = ut.applyLaplaceMechanism(copyHisto.frequency, eps1)
        copyHisto.frequency[copyHisto.frequency < 0.0] = 0.0
    if is_cluster:
        mapping = sorted(range(data.bucketSize), key=lambda x: data.frequency[x])
        mapping_r = sorted(range(data.bucketSize), key=lambda x: mapping[x])
        copyHisto.frequency.sort()

    out = minSSE(copyHisto, bucketSize)

    # recount the frequencies
    out.frequency[:] = 0

    if is_cluster:
        copyData = Histo()
        copyData.set_by_iterable(data.frequency)
        new_freq = np.zeros(data.bucketSize)
        for i, idx in enumerate(mapping):
            new_freq[i] = data.frequency[idx]
        data.frequency = new_freq


    for i in range(data.bucketSize):
        for t in range(out.bucketSize):
            if out.range[t] >= data.range[i]:
                out.frequency[t] += data.frequency[i]
                break
            if t == out.bucketSize-1:
                raise ValueError("histoDP is wrong")

    if not is_structure:
        out.frequency = ut.applyLaplaceMechanism(out.frequency, eps2)
        out.frequency[out.frequency < 0.0] = 0.0
    out.publish()

    if is_cluster:
        data.frequency = copyData.frequency
        new_freq = np.zeros(data.bucketSize)
        for i, idx in enumerate(mapping_r):
            new_freq[i] = out.frequency[idx]
        out.frequency = new_freq

    return out.frequency