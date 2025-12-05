import os
import shlex
import gzip
import numpy as np
import _pickle as cPickle
from tqdm import tqdm
from sklearn.metrics import pairwise_distances_argmin, pairwise_distances
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from parmap import parmap


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
    max_files = min(100, len(files))
    indices = np.random.permutation(len(files))[:max_files]
    files = np.array(files)[indices]

    # rough number of descriptors per file that we have to load
    max_descs_per_file = int(max_descriptors / max(1, len(files)))

    descriptors = []
    for i in tqdm(range(len(files)), desc="Loading descriptors"):
        with gzip.open(files[i], 'rb') as ff:
            desc = cPickle.load(ff, encoding='latin1')

        # handle empty descriptors
        if desc is None or len(desc) == 0:
            continue

        # get some random ones
        indices = np.random.choice(len(desc),
                                   min(len(desc),
                                       int(max_descs_per_file)),
                                   replace=False)
        desc = desc[indices]
        descriptors.append(desc)

    if len(descriptors) == 0:
        return np.zeros((0, 128), dtype=np.float32)

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
    idx = pairwise_distances_argmin(descriptors, clusters)

    # create hard assignment
    assignment = np.zeros((len(descriptors), len(clusters)), dtype=np.uint8)
    assignment[np.arange(len(descriptors)), idx] = 1

    return assignment


def sum_pooling(desc, mus, assignment_matrix):
    """
    Standard VLAD sum pooling per cluster

    parameters:
        desc: TxD descriptor matrix
        mus: KxD cluster centers
        assignment_matrix: TxK assignment matrix
    returns: KxD pooled residuals
    """
    K, D = mus.shape
    pooled_residuals = np.zeros((K, D), dtype=np.float32)

    for k in range(K):
        mask = assignment_matrix[:, k] > 0
        if not np.any(mask):
            continue

        cluster_descriptors = desc[mask]  # (#assigned, D)
        residuals = cluster_descriptors - mus[k]  # (#assigned, D)
        pooled_residuals[k] = residuals.sum(axis=0)

    return pooled_residuals


def gmp_pooling(desc, mus, assignment_matrix, gamma):
    """
    Generalized Max Pooling with Ridge regression (Part f - Bonus)

    parameters:
        desc: TxD descriptor matrix
        mus: KxD cluster centers
        assignment_matrix: TxK assignment matrix
        gamma: regularization parameter
    returns: KxD pooled residuals
    """
    K, D = mus.shape
    pooled_residuals = np.zeros((K, D), dtype=np.float32)

    for k in range(K):
        mask = assignment_matrix[:, k] > 0
        descriptors_in_cluster = int(np.sum(mask))
        if descriptors_in_cluster == 0:
            continue

        residuals = desc[mask] - mus[k]  # (#assigned, D)

        if descriptors_in_cluster == 1:
            pooled_residuals[k] = residuals[0].astype(np.float32)
            continue

        try:
            ridge = Ridge(
                alpha=float(gamma),
                solver='sparse_cg',
                fit_intercept=False,
                max_iter=500
            )
            y = np.ones(descriptors_in_cluster, dtype=np.float32)
            ridge.fit(residuals, y)
            pooled_residuals[k] = ridge.coef_.astype(np.float32)
        except (np.linalg.LinAlgError, ValueError):
            # fallback to sum pooling
            pooled_residuals[k] = np.sum(residuals, axis=0).astype(np.float32)

    return pooled_residuals


def vlad(files, mus, powernorm, gmp=False, gamma=1000):
    """
    Compute VLAD encoding for each file

    This ONE function handles BOTH sum pooling (for group) and GMP (for individual)
    based on the gmp flag.

    parameters:
        files: list of N files containing each T local descriptors of dimension D
        mus:   K x D matrix of cluster centers
        powernorm: if True, apply signed sqrt (power normalization)
        gmp:   if True, use Generalized Max Pooling (GMP) - Ridge regression (bonus)
               if False, use standard sum pooling (default for group solution)
        gamma: regularization parameter for GMP (only used if gmp=True)

    returns:
        encodings: N x (K*D) matrix of encodings
    """
    K, D = mus.shape
    N = len(files)
    encodings = np.zeros((N, K * D), dtype=np.float32)

    for i, path in enumerate(tqdm(files, desc="VLAD encoding")):
        with gzip.open(path, 'rb') as ff:
            desc = cPickle.load(ff, encoding='latin1')   # T x D

        # guard against empty descriptors
        if desc is None or len(desc) == 0:
            continue

        # handle dimension mismatch
        if desc.shape[1] != D:
            desc = desc[:, :D]

        # hard assignments: T x K one-hot
        assignment_matrix = assignments(desc, mus)      # T x K

        # Choose pooling method based on gmp flag
        if gmp:
            # Use GMP (Generalized Max Pooling) - Ridge regression
            pooled_residuals = gmp_pooling(desc, mus, assignment_matrix, gamma)
        else:
            # Use standard sum pooling (default)
            pooled_residuals = sum_pooling(desc, mus, assignment_matrix)

        # flatten to 1D: K*D
        f_enc = pooled_residuals.reshape(-1)

        # power normalization (signed sqrt)
        if powernorm:
            f_enc = np.sign(f_enc) * np.sqrt(np.abs(f_enc))

        # L2 normalization
        norm = np.linalg.norm(f_enc)
        if norm > 0:
            f_enc /= norm

        encodings[i] = f_enc

    return encodings


def esvm(encs_test, encs_train, C=1000):
    """ 
    compute a new embedding using Exemplar Classification
    compute for each encs_test encoding an E-SVM using the
    encs_train as negatives   

    encs_test: N x D
    encs_train: M x D
    returns: N x D matrix (new encs_test)
    """
    N_test, D = encs_test.shape
    N_train = encs_train.shape[0]

    y = np.empty((1 + N_train,), dtype=np.int8)
    y[0] = 1
    y[1:] = -1

    def loop(i):
        # build tiny training set: 1 positive, many negatives
        X_pos = encs_test[i:i+1]        # shape (1, D)
        X_neg = encs_train              # shape (M, D)
        X = np.vstack([X_pos, X_neg])   # (1+M, D)

        y = np.empty((1 + N_train,), dtype=np.int8)
        y[0] = 1
        y[1:] = -1

        # train Linear SVM for this exemplar
        clf = LinearSVC(
            C=C,
            class_weight='balanced',
            dual=False,
            max_iter=1000,
            tol=1e-2,
        )
        clf.fit(X, y)

        # use normalized weight vector as new embedding
        w = clf.coef_.ravel()           # (D,)
        norm = np.linalg.norm(w)
        if norm > 0:
            w /= norm

        # we return shape (1, D) so concatenation works
        return w[np.newaxis, :]

    # parallel over all test exemplars
    indices = range(N_test)
    new_encs_list = list(parmap(loop, tqdm(indices)))
    new_encs = np.concatenate(new_encs_list, axis=0)   # N x D
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
    dists = pairwise_distances(encs, metric='cosine')
    # mask out distance with itself
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


def fetch_encodings(files, mus, fname, powernorm, gmp, gamma, overwrite=False):
    """
    Load encodings from disk if present; otherwise compute and save them.

    parameters:
        files: list of descriptor files
        mus: codebook (K x D)
        fname: filename to save/load encodings
        powernorm: use power normalization
        gmp: use generalized max pooling
        gamma: GMP regularization parameter
        overwrite: if True, recompute even if file exists

    returns: N x (K*D) encoding matrix
    """
    if not os.path.exists(fname) or overwrite:
        encodings = vlad(
            files,
            mus,
            powernorm=powernorm,
            gmp=gmp,
            gamma=gamma
        )
        with gzip.open(fname, 'wb') as fOut:
            cPickle.dump(encodings, fOut, -1)
    else:
        with gzip.open(fname, 'rb') as f:
            encodings = cPickle.load(f)
    return encodings
