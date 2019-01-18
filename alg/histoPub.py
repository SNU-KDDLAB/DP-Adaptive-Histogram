from alg.histo import Histo
import util.utils as ut
import numpy as np

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


def minSSE_eps(data, eps, ratio=None):
    bucketSize = data.bucketSize
    frequency_nz = data.frequency.copy()
    frequency_nz[frequency_nz < 0] = 0.0
    cumulSum = frequency_nz.cumsum()
    cumsqrSum = np.array([x**2 for x in frequency_nz]).cumsum()

    SSE = np.zeros((bucketSize, bucketSize))
    DP_range = np.zeros((bucketSize, bucketSize), dtype=np.int32)

    SSE[:, 0] = preSSE(cumulSum, cumsqrSum, row=0)

    for j in range(1, bucketSize):
        for i in range(j, bucketSize):
            tmp = SSE[j-1:i, j-1] + preSSE(cumulSum, cumsqrSum, col=i)[j:i+1]

            k = tmp.argmin()
            DP_range[i, j] = k + j - 1
            SSE[i, j] = tmp[k]

    if eps is not None:
        struct_err = SSE[bucketSize-1, :]
        noise_err = np.arange(1, bucketSize+1) * 2 / (eps * eps)
        optBucketSize = np.argmin(struct_err ** 2 + noise_err ** 2) + 1
        if optBucketSize < 10:
            optBucketSize = 10
    else:
        optBucketSize = np.argmin(SSE[bucketSize-1, :]) + 1
    output = HistoPub([0.0] * optBucketSize)

    k = bucketSize - 1
    for i in reversed(range(optBucketSize)):
        if k == -1:
            raise ValueError("k error!")

        output.range[i] = data.range[k]
        k = DP_range[k][i]

    j = 0
    for i in range(data.bucketSize):
        if output.range[j] >= data.range[i]:
            output.frequency[j] += data.frequency[i]
            if output.range[j] == data.range[i]:
                j += 1
    if j != output.bucketSize:
        raise ValueError("j error")

    return output


class HistoPub(Histo):
    def optHist(self, eps=None, ratio=0.05):
        if eps is not None:
            eps1 = ratio * eps
            eps2 = (1-ratio) * eps
        else:
            eps1 = None
            eps2 = None

        copyHisto = Histo()
        copyHisto.set_by_iterable(self.frequency)

        if eps1 is not None:
            copyHisto.frequency = ut.applyLaplaceMechanism(copyHisto.frequency, eps1)
            # copyHisto.frequency[copyHisto.frequency < 0.0] = 0.0

        # out = minSSE_eps(copyHisto, eps2/6)
        out = minSSE_eps(copyHisto, eps2 * float(np.sqrt(eps1)) / 2)

        frequency_origin = np.zeros(out.bucketSize)
        j = 0
        for i in range(self.bucketSize):
            if out.range[j] >= self.range[i]:
                frequency_origin[j] += self.frequency[i]
                if out.range[j] == self.range[i]:
                    j += 1

        frequency_first = out.frequency.copy()
        out.frequency[:] = 0
        j = 0
        for i in range(self.bucketSize):
            if out.range[j] >= self.range[i]:
                out.frequency[j] += self.frequency[i]
                if out.range[j] == self.range[i]:
                    j += 1

        if eps2 is not None:
            out.frequency = ut.applyLaplaceMechanism(out.frequency, eps2)

        frequency_second = out.frequency.copy()

        for i in range(out.bucketSize):
            if i == 0:
                tmp_n = out.range[i] + 1
            else:
                tmp_n = out.range[i] - out.range[i-1]
            weights = [eps1*eps1/tmp_n, eps2*eps2]
            # out.frequency[i] = ut.weighted_sum([frequency_first[i], frequency_second[i]], weights)
        # out.frequency[frequency_second < 0.0] = 0.0
        out.frequency[out.frequency < 0.0] = 0.0

        return out

