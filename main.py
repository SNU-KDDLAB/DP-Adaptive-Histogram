import numpy as np

from alg.histo import Histo
from alg.histoPlus import histoDP_shuffle_publish, histoDP_plus_publish
from alg.histoPub import HistoPub
import util.utils as ut
from alg.histoPub import minSSE_eps, preSSE


def exp2D():
    twitter_dataset = np.load("dataset/2D/twitter_256_256.npy")
    T = 256
    # twitter_dataset = twitter_dataset[:T, :T]
    # twitter_dataset = twitter_dataset[5:10, 155:160]
    # for i in range(256):
    #     for j in range(256):
    #         if twitter_dataset[i, j] > 30:
    #             print(i, j)
    #             exit(0)
    # print(twitter_dataset.shape)
    # nbin = 256
    nbin = T
    cmap = "Greens_r"
    # util.plots.plot_full_matrix(twitter_dataset, "pngs/twitter_{}.png".format(nbin), is_overwrite=True, nbins=nbin, cmap=cmap)
    twitter_histogram = twitter_dataset.reshape((-1, 1)).squeeze()
    # print(twitter_histogram.shape)
    # print(twitter_histogram.dtype)

    x = Histo()
    x.set_by_iterable(twitter_histogram)

    eps = 1.0
    bs = 10
    for eps in [0.01, 0.1, 1, 10]:
        for bs in [10, 20, 30, 40]:
            np.random.seed(0)
            output_histo = histoDP_plus_publish(eps, bs, x)
            output_histo = output_histo.reshape(T, T)

            np.save("result/twitter_histo_eps_{}_bs_{}.npy".format(eps, bs), output_histo)
            # util.plots.plot_full_matrix(output_histo, "pngs/twitter_histo_eps_{}_bs_{}.png".format(eps, bs), is_overwrite=True, nbins=nbin, cmap=cmap)

            np.random.seed(0)
            output_shuffle = histoDP_shuffle_publish(eps, bs, x)
            output_shuffle = output_shuffle.reshape(T, T)

            np.save("resuslt/twitter_shuffle_eps_{}_bs_{}.npy".format(eps, bs), output_shuffle)
            # util.plots.plot_full_matrix(output_shuffle, "pngs/twitter_shuffle_eps_{}_bs_{}png".format(eps, bs), is_overwrite=True, nbins=nbin, cmap=cmap)


def exp1D(dataset, dataname):
    print(dataname)
    T = len(dataset)

    x = Histo()
    x.set_by_iterable(dataset)

    eps = 1
    bs = 10

    print("eps", "bs", "mae_histo", "rmse_histo", "mae_cluster", "rmse_cluster", "mae_identity", "rmse_identity", "time_histo")
    for eps in [1]:
        np.random.seed(0)
        output_identity = ut.applyLaplaceMechanism(dataset, eps)
        l1_identity, l2_identity = ut.mean_err_histo(dataset, output_identity)
        for bs in [5, 10, 20, 40, 80, 160]:
            np.random.seed(0)
            start = ut.alg_begin()
            output_histo = histoDP_plus_publish(eps, bs, x, ratio=0.5, is_structure=True)
            time_histo = ut.alg_end(start)
            print(eps, bs, end=" ")

            np.save("result/{}_histo_eps_{}_bs_{}.npy".format(dataname, eps, bs), output_histo)
            # ut.plot_full_matrix(output_histo, "pngs/{}_histo_eps_{}_bs_{}.png".format(dataname, eps, bs), is_overwrite=True, nbins=nbin, cmap=cmap)
            # plot_1D(output_histo, "pngs/{}_histo_eps_{}_bs_{}.png".format(dataname, eps, bs), is_overwrite=True)
            # print(ut.mean_square_err_freq_vectors(dataset, output_histo), end=" ")
            l1, l2 = ut.mean_err_histo(dataset, output_histo)
            print("%.1f %.1f" % (l1, np.sqrt(l2)), end=" ")

            np.random.seed(0)
            output_shuffle = histoDP_shuffle_publish(eps, bs, x, ratio=0.5, is_structure=True)

            np.save("result/{}_shuffle_eps_{}_bs_{}.npy".format(dataname, eps, bs), output_shuffle)
            # ut.plot_full_matrix(output_shuffle, "pngs/{}_shuffle_eps_{}_bs_{}png".format(dataname, eps, bs), is_overwrite=True, nbins=nbin, cmap=cmap)
            # plot_1D(output_shuffle, "pngs/{}_shuffle_eps_{}_bs_{}.png".format(dataname, eps, bs), is_overwrite=True)
            # print(ut.mean_square_err_freq_vectors(dataset, output_shuffle), end=" ")
            l1, l2 = ut.mean_err_histo(dataset, output_shuffle)
            print("%.1f %.1f" % (l1, np.sqrt(l2)), end=" ")

            print("%.1f %.1f" % (l1_identity, np.sqrt(l2_identity)), end=" ")
            print(time_histo, end=" ")
            print()

def expB(dataset, dataname, seed=None):
    print(dataname)
    T = len(dataset)
    x = Histo()
    x.set_by_iterable(dataset)

    print("eps", "bs", "mae_histo", "rmse_histo", "mae_optB", "rmse_optB", "mae_identity", "rmse_identity")
    # for eps in [0.1, 1, 10, 100]:
    ratio = 0.2
    for eps in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    # for eps in [0.1, 0.2, 0.4, 0.8, 1.6]:
        np.random.seed(seed)
        output_identity = ut.applyLaplaceMechanism(dataset, eps)
        l1_identity, l2_identity = ut.mean_err_histo(dataset, output_identity)

        histo = HistoPub(dataset)
        np.random.seed(seed)
        output_optB = histo.optHist(eps, ratio=ratio)
        optB = output_optB.bucketSize
        output_optB.publish()
        l1_optB, l2_optB = ut.mean_err_histo(dataset, output_optB.frequency)

        # for bs in range(1, T+1):
        # for bs in [optB//3, optB//2, optB, optB*2, optB*4, optB*8]:
        for bs in [20]:
            if bs == 0:
                continue
            if bs > T:
                bs = T
        # for bs in [T]:
            np.random.seed(seed)
            output_histo = histoDP_plus_publish(eps, bs, x, ratio=ratio)
            print(eps, bs, end=" ")

            # np.save("result/{}_histo_eps_{}_bs_{}.npy".format(dataname, eps, bs), output_histo)
            l1, l2 = ut.mean_err_histo(dataset, output_histo)
            print("%.1f %.1f" % (l1, np.sqrt(l2)), end=" ")

            print("%.1f %.1f" % (l1_optB, np.sqrt(l2_optB)), end=" ")
            # if bs == optB:
            #     print("%.1f %.1f" % (l1_optB, np.sqrt(l2_optB)), end=" ")
            # else:
            #     print("%.1f %.1f" % (0, 0), end=" ")

            print("%.1f %.1f" % (l1_identity, np.sqrt(l2_identity)), end=" ")
            print()

def get_DPtable(data):
    data = np.array(data)
    bucketSize = len(data)
    cumulSum = data.cumsum()
    cumsqrSum = np.array([x**2 for x in data]).cumsum()

    SSE = np.zeros((bucketSize, bucketSize))
    DP_range = np.zeros((bucketSize, bucketSize), dtype=np.int32)

    SSE[:, 0] = preSSE(cumulSum, cumsqrSum, row=0)

    for j in range(1, bucketSize):
        for i in range(j, bucketSize):
            tmp = SSE[j-1:i, j-1] + preSSE(cumulSum, cumsqrSum, col=i)[j:i+1]

            k = tmp.argmin()
            DP_range[i, j] = k + j - 1
            SSE[i, j] = tmp[k]

    return SSE

if __name__ == '__main__':
    DPtable = get_DPtable([10, 30, 20, 40, 10])
    print(DPtable.T)
    exit()
    seed = 0
    dataset = np.load("dataset/1D/HEPTH.n4096.npy")
    T = 4
    dataset = dataset.reshape(-1,T).sum(axis=-1)
    N = 1000
    pval = dataset / dataset.sum()
    np.random.seed(seed)
    dataset = np.random.multinomial(N, pval)
    expB(dataset, "HEPTH", seed=seed)
    print()

    dataset = np.load("dataset/1D/PATENT.n4096.npy")
    dataset = dataset.reshape(-1,T).sum(axis=-1)
    pval = dataset / dataset.sum()
    np.random.seed(seed)
    dataset = np.random.multinomial(N, pval)
    expB(dataset, "PATENT", seed=seed)

