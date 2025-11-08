# import glob, pickle, gzip
#
# with open("data/icdar17_labels_train.txt") as f:
#     basename, writer = f.readline().split()
#
# # suppose basename = "1000-IMG_MAX_116390"
# path = f"data/local_features/train/{basename}_SIFT_patch_pr.pkl.gz"
# with gzip.open(path, "rb") as gz:
#     # tell pickle to map Python2 bytes → Python3 bytes
#     desc = pickle.load(gz, encoding="latin1")
#
# print(basename, "→ descriptor shape:", desc.shape)

import os
import shlex
import argparse

from sklearn.metrics import pairwise_distances_argmin, pairwise_distances
from tqdm import tqdm

# for python3: read in python2 pickled files
import _pickle as cPickle

import gzip
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize
import numpy as np
# import cv2
from parmap import parmap
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed


def parseArgs(parser):
    parser.add_argument('--labels_test',
                        help='contains test images/descriptors to load + labels')
    parser.add_argument('--labels_train',
                        help='contains training images/descriptors to load + labels')
    parser.add_argument('-s', '--suffix',
                        default='_SIFT_patch_pr.pkl.gz',
                        help='only chose those images with a specific suffix')
    parser.add_argument('--in_test',
                        help='the input folder of the test images / features')
    parser.add_argument('--in_train',
                        help='the input folder of the training images / features')
    parser.add_argument('--overwrite', action='store_true',
                        help='do not load pre-computed encodings')
    parser.add_argument('--powernorm', action='store_true',
                        help='use powernorm')
    parser.add_argument('--gmp', action='store_true',
                        help='use generalized max pooling')
    parser.add_argument('--gamma', default=1, type=float,
                        help='regularization parameter of GMP')
    parser.add_argument('--C', default=1000, type=float,
                        help='C parameter of the SVM')
    return parser


def getFiles(folder, pattern, labelfile):
    """
    returns files and associated labels by reading the labelfile
    parameters:
        folder: inputfolder
        pattern: new suffix
        labelfiles: contains a list of filename and labels
    return: absolute filenames + labels
    """
    # read labelfile
    with open(labelfile, 'r') as f:
        all_lines = f.readlines()

    # get filenames from labelfile
    all_files = []
    labels = []
    check = True
    for line in all_lines:
        # using shlex we also allow spaces in filenames when escaped w. ""
        splits = shlex.split(line)
        file_name = splits[0]
        class_id = splits[1]

        # strip all known endings, note: os.path.splitext() doesnt work for
        # '.' in the filenames, so let's do it this way...
        for p in ['.pkl.gz', '.txt', '.png', '.jpg', '.tif', '.ocvmb', '.csv']:
            if file_name.endswith(p):
                file_name = file_name.replace(p, '')

        # get now new file name
        true_file_name = os.path.join(folder, file_name + pattern)
        all_files.append(true_file_name)
        labels.append(class_id)

    return all_files, labels


def loadRandomDescriptors(files, max_descriptors):
    """
    load roughly `max_descriptors` random descriptors
    parameters:
        files: list of filenames containing local features of dimension D
        max_descriptors: maximum number of descriptors (Q)
    returns: QxD matrix of descriptors
    """
    # let's just take 100 files to speed up the process
    max_files = 100
    indices = np.random.permutation(max_files)
    files = np.array(files)[indices]

    # rough number of descriptors per file that we have to load
    max_descs_per_file = int(max_descriptors / len(files))

    descriptors = []
    for i in tqdm(range(len(files))):
        with gzip.open(files[i], 'rb') as ff:
            # for python2
            # desc = cPickle.load(ff)
            # for python3
            desc = cPickle.load(ff, encoding='latin1')

        # get some random ones
        indices = np.random.choice(len(desc),
                                   min(len(desc),
                                       int(max_descs_per_file)),
                                   replace=False)
        desc = desc[indices]
        descriptors.append(desc)

    descriptors = np.concatenate(descriptors, axis=0)
    return descriptors


def dictionary(descriptors, n_clusters):
    """
    return cluster centers for the descriptors
    parameters:
        descriptors: NxD matrix of local descriptors
        n_clusters: number of clusters = K
    returns: KxD matrix of K clusters
    """
    # TODO
    mbk = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=10_000,
        max_iter=200,
        verbose=1,
        random_state=0
    )
    mbk.fit(descriptors)
    return mbk.cluster_centers_


def assignments(descriptors, clusters):
    """
    compute assignment matrix
    parameters:
        descriptors: TxD descriptor matrix
        clusters: KxD cluster matrix
    returns: TxK assignment matrix
    """
    # compute nearest neighbors
    # TODO

    # create hard assignment
    # assignment = np.zeros((len(descriptors), len(clusters)))
    # TODO

    return pairwise_distances_argmin(descriptors, clusters, metric="euclidean")


def vlad(files, mus, powernorm, gmp=False, gamma=1000):
    """
    compute VLAD encoding for each files
    parameters:
        files: list of N files containing each T local descriptors of dimension
        D
        mus: KxD matrix of cluster centers
        gmp: if set to True use generalized max pooling instead of sum pooling
    returns: NxK*D matrix of encodings
    """
    K, D = mus.shape
    N = len(files)
    encodings = np.zeros((N, K * D), dtype=np.float32)

    for i, path in enumerate(tqdm(files, desc="VLAD encoding")):
        with gzip.open(path, 'rb') as ff:
            desc = cPickle.load(ff, encoding='latin1')

        idxs = assignments(desc, mus)

        f_enc = np.zeros((K, D), dtype=np.float32)
        for t, k in enumerate(idxs):
            # it's faster to select only those descriptors that have
            # this cluster as nearest neighbor and then compute the
            # difference to the cluster center than computing the differences
            # first and then select
            f_enc[k] += (desc[t] - mus[k])

        f_enc = f_enc.reshape(-1)

        # c) power normalization
        if powernorm:
            # TODO
            f_enc = np.sign(f_enc) * np.sqrt(np.abs(f_enc))

        # l2 normalization
        # TODO
        norm = np.linalg.norm(f_enc)
        if norm > 0:
            f_enc /= norm

        encodings[i] = f_enc
    return encodings


# def _esvm_one(i, encs_test, encs_train, C):
#     """
#     Train one exemplar SVM for test index i and return
#     its L2-normalized weight vector (shape (D,)).
#     """
#     # positive
#     X_pos = encs_test[i : i+1]
#     # negatives (you can subsample here if you like)
#     X_neg = encs_train
#     X = np.vstack([X_pos, X_neg])
#     y = np.empty((1 + len(X_neg),), dtype=int)
#     y[0] = 1; y[1:] = -1
#
#     clf = LinearSVC(C=C, class_weight='balanced',
#                     dual=False, max_iter=10000, tol=1e-4)
#     clf.fit(X, y)
#     w = clf.coef_.ravel()
#     w /= np.linalg.norm(w) + 1e-12
#     return w

# def _esvm_worker(i, encs_test, encs_train, C):
#     # build the tiny dataset
#     X_pos = encs_test[i : i+1]
#     X_neg = encs_train                # you can subsample here if you like
#     X = np.vstack([X_pos, X_neg])
#     y = np.empty((1 + len(X_neg),), dtype=int)
#     y[0] = 1; y[1:] = -1
#
#     clf = LinearSVC(C=C, class_weight='balanced',
#                     dual=False, max_iter=10000, tol=1e-4)
#     clf.fit(X, y)
#
#     w = clf.coef_.ravel()
#     w /= np.linalg.norm(w) + 1e-12
#     return i, w  # return the index so we can place it correctly

def esvm(encs_test, encs_train, C=1000):
    """
    compute a new embedding using Exemplar Classification
    compute for each encs_test encoding an E-SVM using the
    encs_train as negatives
    parameters:
        encs_test: NxD matrix
        encs_train: MxD matrix

    returns: new encs_test matrix (NxD)
    """
    # (1)
    # N_test, D = encs_test.shape
    #
    # # 1) build a delayed task for each test exemplar
    # tasks = [
    #     delayed(_esvm_one)(i, encs_test, encs_train, C)
    #     for i in range(N_test)
    # ]
    #
    # # 2) compute them in parallel using Dask’s default scheduler
    # #    you can pass scheduler="processes" or "threads"
    # with ProgressBar():
    #     results = compute(*tasks, scheduler="threads")
    # # results = compute(*tasks, scheduler="processes")
    #
    # # 3) stack into (N_test, D)
    # return np.vstack(results)

    # (2)
    # N_test, D = encs_test.shape
    # results = [None] * N_test
    #
    # # n_workers defaults to number of CPU cores
    # with ThreadPoolExecutor(max_workers=4) as exec:
    #     futures = {
    #         exec.submit(_esvm_worker, i, encs_test, encs_train, C): i
    #         for i in range(N_test)
    #     }
    #
    #     # as_completed yields futures as they finish; tqdm tracks them
    #     for future in tqdm(
    #             as_completed(futures),
    #             total=N_test,
    #             desc="ESVM",
    #             unit="it",
    #             leave=True
    #     ):
    #         i, w = future.result()
    #         results[i] = w
    #
    # # stack into (N_test, D)
    # return np.stack(results, axis=0)

    # (3) main
    # set up labels
    # TODO
    N_test, D = enc_test.shape
    N_train = encs_train.shape[0]

    def loop(i):
        # compute SVM
        # and make feature transformation
        # TODO
        # 1) Build the training data for this exemplar
        X_pos = encs_test[i:i + 1]  # shape (1, D)
        X_neg = encs_train  # shape (N_train, D)
        X = np.vstack([X_pos, X_neg])  # (1+N_train, D)

        # 2) Labels: +1 for exemplar, -1 for all negatives
        y = np.empty((1 + N_train,), dtype=np.int8)
        y[0] = 1
        y[1:] = -1

        # 3) Train a LinearSVC
        clf = LinearSVC(
            C=C,
            class_weight='balanced',
            dual=False,  # faster when D >> N
            max_iter=1000,
            tol=1e-2,
        )
        clf.fit(X, y)

        # 4) Extract & L2‐normalize the weight vector
        w = clf.coef_.ravel()  # shape (D,)
        norm = np.linalg.norm(w)
        if norm > 0:
            w /= norm

        # return as shape (1, D) so concatenation works
        return w[np.newaxis, :]

    #
    # let's do that in parallel:
    # if that doesn't work for you, just exchange 'parmap' with 'map'
    # Even better: use DASK arrays instead, then everything should be
    # parallelized
    # new_encs = list(map(loop, range(N_test)))
    new_encs = [loop(i) for i in tqdm(range(N_test), desc="ESVM")]
    new_encs = np.concatenate(new_encs, axis=0)
    # return new encodings
    return new_encs


def distances(encs):
    """
    compute pairwise distances

    parameters:
        encs:  TxK*D encoding matrix
    returns: TxT distance matrix
    """
    # compute cosine distance = 1 - dot product between l2-normalized
    # encodings
    # TODO
    # mask out distance with itself
    dists = pairwise_distances(encs, metric='cosine')
    np.fill_diagonal(dists, np.finfo(dists.dtype).max)
    return dists


def evaluate(encs, labels):
    """
    evaluate encodings assuming using associated labels
    parameters:
        encs: TxK*D encoding matrix
        labels: array/list of T labels
    """
    dist_matrix = distances(encs)
    # sort each row of the distance matrix
    indices = dist_matrix.argsort()

    n_encs = len(encs)

    mAP = []
    correct = 0
    for r in range(n_encs):
        precisions = []
        rel = 0
        for k in range(n_encs - 1):
            if labels[indices[r, k]] == labels[r]:
                rel += 1
                precisions.append(rel / float(k + 1))
                if k == 0:
                    correct += 1
        avg_precision = np.mean(precisions)
        mAP.append(avg_precision)
    mAP = np.mean(mAP)

    print('Top-1 accuracy: {} - mAP: {}'.format(float(correct) / n_encs, mAP))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('retrieval')
    parser = parseArgs(parser)
    args = parser.parse_args()
    np.random.seed(42)  # fix random seed

    # a) dictionary
    files_train, labels_train = getFiles(
        folder="data/local_features/train",
        pattern="_SIFT_patch_pr.pkl.gz",
        labelfile="data/icdar17_labels_train.txt"
    )
    print('#train: {}'.format(len(files_train)))

    files_test, labels_test = getFiles(
        folder="data/local_features/test",
        pattern="_SIFT_patch_pr.pkl.gz",
        labelfile="data/icdar17_labels_test.txt"
    )
    print('#test: {}'.format(len(files_test)))

    descriptors = loadRandomDescriptors(files_train, max_descriptors=500_000)
    print("Sampled descriptors:", descriptors.shape)

    if not os.path.exists('mus.pkl.gz'):
        # TODO
        print('> loaded {} descriptors:'.format(len(descriptors)))
        K = 100
        mus = dictionary(descriptors, n_clusters=K)
        print("Codebook centers shape:", mus.shape)
        # cluster centers
        print('> compute dictionary')
        # TODO
        with gzip.open('mus.pkl.gz', 'wb') as fOut:
            cPickle.dump(mus, fOut, -1)
    else:
        with gzip.open('mus.pkl.gz', 'rb') as f:
            mus = cPickle.load(f)

    # b) VLAD encoding
    fname = 'enc_train_gmp{}.pkl.gz'.format(
        args.gamma) if args.gmp else 'enc_train.pkl.gz'
    if not os.path.exists(fname) or args.overwrite:
        # TODO
        enc_train = vlad(
            files_train,
            mus,
            powernorm=args.powernorm,
            gmp=args.gmp,
            gamma=args.gamma
        )
        with gzip.open(fname, 'wb') as fOut:
            cPickle.dump(enc_train, fOut, -1)
    else:
        with gzip.open(fname, 'rb') as f:
            enc_train = cPickle.load(f)

    fname = 'enc_test_gmp{}.pkl.gz'.format(
        args.gamma) if args.gmp else 'enc_test.pkl.gz'
    if not os.path.exists(fname) or args.overwrite:
        # TODO
        enc_test = vlad(
            files_test,
            mus,
            powernorm=args.powernorm,
            gmp=args.gmp,
            gamma=args.gamma
        )
        with gzip.open(fname, 'wb') as fOut:
            cPickle.dump(enc_test, fOut, -1)
    else:
        with gzip.open(fname, 'rb') as f:
            enc_test = cPickle.load(f)

    # cross-evaluate test encodings
    print('> evaluate')
    evaluate(enc_test, labels_test)

    # d) compute exemplar svms

    print('> esvm computation')
    # TODO
    # eval
    print("> running Exemplar‐SVM refinement")
    enc_test = esvm(enc_test, enc_train, C=args.C)
    print(f"> refined TEST encodings via ESVM → {enc_test.shape}")
    print('> evaluate')
    evaluate(enc_test, labels_test)
