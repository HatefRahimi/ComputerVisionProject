import os
import shlex
import argparse

from sklearn.metrics import pairwise_distances_argmin, pairwise_distances
from tqdm import tqdm

import _pickle as cPickle

import gzip
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize
import numpy as np
from parmap import parmap
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed

from custom_sift_extractor import CustomSIFTExtractor


# Use the BINARY folders
TRAIN_IMAGE_DIRS = ["data/icdar2017-training-binary"]
TEST_IMAGE_DIRS = ["data/icdar2017-testing-binary"]

OUT_TRAIN = "data/local_features/train1"
OUT_TEST = "data/local_features/test1"


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

    parser.add_argument('--multi_vlad', action='store_true',
                        help='use multi-VLAD with multiple codebooks (part g)')
    parser.add_argument('--n_codebooks', default=5, type=int,
                        help='number of codebooks for multi-VLAD (default: 5)')
    parser.add_argument('--n_clusters', default=100, type=int,
                        help='clusters per codebook (default: 100)')
    parser.add_argument('--pca_components', default=1000, type=int,
                        help='PCA output dimensionality (with whitening) for part g')
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
        # using shlex also allows spaces in filenames when escaped w. ""
        splits = shlex.split(line)
        file_name = splits[0]
        class_id = splits[1]

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
    # Take 100 files to speed up the process
    max_files = min(100, len(files))
    indices = np.random.permutation(len(files))[:max_files]
    files = np.array(files)[indices]

    # rough number of descriptors per file that we have to load
    max_descs_per_file = int(max_descriptors / max(1, len(files)))

    descriptors = []
    for i in tqdm(range(len(files))):
        with gzip.open(files[i], 'rb') as ff:
            desc = cPickle.load(ff, encoding='latin1')

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
        files: list of N files containing each T local descriptors of dimension D
        mus:   KxD matrix of cluster centers
        gmp:   if set to True use generalized max pooling instead of sum pooling
    returns: NxK*D matrix of encodings
    """
    K, D = mus.shape
    N = len(files)
    encodings = np.zeros((N, K * D), dtype=np.float32)

    for i, path in enumerate(tqdm(files, desc="VLAD encoding")):
        with gzip.open(path, 'rb') as ff:
            desc = cPickle.load(ff, encoding='latin1')

        # guard against empty descriptors
        if desc is None or len(desc) == 0:
            # leave zero row to keep alignment
            continue

        # ensure descriptor dims match codebook
        if desc.shape[1] != D:
            desc = desc[:, :D]

        idxs = assignments(desc, mus)

        # pooled residuals per cluster
        f_enc = np.zeros((K, D), dtype=np.float32)

        if gmp:
            # Generalized Max Pooling via Ridge
            for k in range(K):
                mask = (idxs == k)
                n_k = int(np.sum(mask))
                if n_k == 0:
                    continue

                R = desc[mask] - mus[k]  # (n_k, D)

                if n_k == 1:
                    f_enc[k] = R[0].astype(np.float32)
                    continue

                try:
                    ridge = Ridge(
                        alpha=float(gamma),
                        solver='sparse_cg',
                        fit_intercept=False,
                        max_iter=500
                    )
                    ridge.fit(R, np.ones((n_k,), dtype=np.float32))
                    coef = ridge.coef_                # (D,)
                    f_enc[k] = coef.astype(np.float32)
                except Exception:
                    # fallback: mean residual
                    f_enc[k] = np.mean(R, axis=0).astype(np.float32)
        else:
            # Standard VLAD (sum pooling)
            for t, k in enumerate(idxs):
                f_enc[k] += (desc[t] - mus[k])

        # c) power normalization
        if powernorm:
            # TODO
            f_enc = np.sign(f_enc) * np.sqrt(np.abs(f_enc))

        # l2 normalization of final vector
        f_enc = f_enc.reshape(-1).astype(np.float32)
        norm = np.linalg.norm(f_enc)
        if norm > 0:
            f_enc /= norm

        encodings[i] = f_enc

    return encodings


def esvm(encs_test, encs_train, C=1000):
    """
    compute a new embedding using Exemplar Classification
    parameters:
        encs_test: NxD matrix   (each row is a test exemplar)
        encs_train: MxD matrix  (negatives)
    returns:
        new encs_test matrix (NxD) with per-exemplar SVM weights
    """
    # shapes
    N_test, D = encs_test.shape
    N_train = encs_train.shape[0]

    y = np.empty((1 + N_train,), dtype=np.int8)
    y[0] = 1
    y[1:] = -1

    def fit_one(i):
        # compute SVM
        # and make feature transformation
        # TODO
        # 1) Build the training data for this exemplar
        X_pos = encs_test[i:i+1]
        X = np.vstack([X_pos, encs_train])

        # Train linear SVM
        clf = LinearSVC(
            C=C,
            class_weight='balanced',
            dual=False,            # faster when D >> N
            max_iter=1000,
            tol=1e-2,
        )
        clf.fit(X, y)

        # L2-normalize the weight vector
        w = clf.coef_.ravel().astype(np.float32)
        n = np.linalg.norm(w)
        if n > 0:
            w /= n
        return w

    Ws = [fit_one(i) for i in tqdm(range(N_test), desc="ESVM")]
    return np.vstack(Ws)


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


# multi-VLAD + PCA-whitening

def create_multiple_codebooks(files, n_codebooks=5, n_clusters=100,
                              max_descriptors=1_000_000, seed=42):
    """
    Create multiple codebooks with different seeds and subsets.
    Returns list of KxD arrays.
    """
    codebooks = []
    for i in range(n_codebooks):
        print(f"> Building codebook {i+1}/{n_codebooks}")
        # different subset per codebook
        desc_subset = loadRandomDescriptors(
            files, max_descriptors=max_descriptors // max(1, n_codebooks))
        if desc_subset is None or len(desc_subset) == 0:
            raise RuntimeError("No descriptors for codebook creation.")
        mus_i = dictionary(desc_subset, n_clusters=n_clusters)
        codebooks.append(mus_i.astype(np.float32))
    return codebooks


def multi_vlad_encode(files, codebooks, powernorm, gmp=False, gamma=1000):
    """
    Compute VLAD for each codebook, then concatenate features horizontally.
    Returns N x (sum_i K_i*D) matrix (usually N x (n_codebooks*K*D)).
    """
    enc_list = []
    for ci, mus in enumerate(codebooks):
        print(
            f"> Encoding with codebook {ci+1}/{len(codebooks)}  (K={mus.shape[0]}, D={mus.shape[1]})")
        enc = vlad(files, mus, powernorm=powernorm, gmp=gmp, gamma=gamma)
        enc_list.append(enc.astype(np.float32))
    # horizontal concat → feature dimension grows
    return np.concatenate(enc_list, axis=1)


def apply_pca_whitening(enc_train, enc_test, n_components=1000, seed=42):
    """
    PCA with whitening, fit on train, transform test.
    Caps components to valid range: min(n_components, train_dim, train_size-1).
    """
    from sklearn.decomposition import PCA
    if enc_train.size == 0 or enc_test.size == 0:
        print("PCA skipped: empty encodings.")
        return enc_train, enc_test, None

    max_comp = min(n_components, enc_train.shape[1], max(
        1, enc_train.shape[0]-1))
    if max_comp <= 0:
        print("PCA skipped: invalid target dimensionality.")
        return enc_train, enc_test, None

    print(f"> PCA whitening: {enc_train.shape[1]} → {max_comp}")
    pca = PCA(n_components=max_comp, whiten=True, random_state=seed)
    enc_train_p = pca.fit_transform(enc_train)
    enc_test_p = pca.transform(enc_test)
    print(
        f"  Explained variance (sum): {np.sum(pca.explained_variance_ratio_):.4f}")
    return enc_train_p.astype(np.float32), enc_test_p.astype(np.float32), pca


if __name__ == '__main__':
    parser = argparse.ArgumentParser('retrieval')
    parser = parseArgs(parser)
    args = parser.parse_args()
    np.random.seed(42)

    # build custom descriptors into data/local_features/{train1,test1} ===
    os.makedirs(OUT_TRAIN, exist_ok=True)
    os.makedirs(OUT_TEST, exist_ok=True)

    extractor = CustomSIFTExtractor()  # central place to tweak SIFT params if needed

    extractor.build_for_split(
        labels_file="data/icdar17_labels_train.txt",
        out_folder=OUT_TRAIN,
        search_dirs=TRAIN_IMAGE_DIRS
    )
    extractor.build_for_split(
        labels_file="data/icdar17_labels_test.txt",
        out_folder=OUT_TEST,
        search_dirs=TEST_IMAGE_DIRS
    )

    # a) dictionary
    files_train, labels_train = getFiles(
        folder=OUT_TRAIN,  # ← use train1
        pattern="_SIFT_patch_pr.pkl.gz",
        labelfile="data/icdar17_labels_train.txt"
    )
    print('#train: {}'.format(len(files_train)))

    files_test, labels_test = getFiles(
        folder=OUT_TEST,  # ← use test1
        pattern="_SIFT_patch_pr.pkl.gz",
        labelfile="data/icdar17_labels_test.txt"
    )
    print('#test: {}'.format(len(files_test)))

    # Report on how many descriptor files exist and are not empty
    def _count_desc_stats(file_list):
        exist = nonempty = rows = 0
        for p in file_list:
            if not os.path.exists(p):
                continue
            exist += 1
            try:
                with gzip.open(p, 'rb') as f:
                    arr = cPickle.load(f, encoding='latin1')
                if arr is not None and getattr(arr, "shape", (0, 0))[0] > 0:
                    nonempty += 1
                    rows += arr.shape[0]
            except Exception:
                pass
        return exist, nonempty, rows

    tr_exist, tr_nonempty, tr_rows = _count_desc_stats(files_train)
    te_exist, te_nonempty, te_rows = _count_desc_stats(files_test)
    print(f"[Part (e)] Train1 descriptors: {tr_exist} files present, "
          f"{tr_nonempty} non-empty, {tr_rows} total SIFT vectors.")
    print(f"[Part (e)] Test1  descriptors: {te_exist} files present, "
          f"{te_nonempty} non-empty, {te_rows} total SIFT vectors.")

    # multi-VLAD + PCA (or single VLAD baseline)
    descriptors = loadRandomDescriptors(files_train, max_descriptors=500_000)
    print("Sampled descriptors:", descriptors.shape)

    if args.multi_vlad:
        print("> Part (g): multi-VLAD enabled")
        # build multiple codebooks (different subsets)
        codebooks = create_multiple_codebooks(
            files_train,
            n_codebooks=args.n_codebooks,
            n_clusters=args.n_clusters,
            max_descriptors=1_000_000,  # total budget across codebooks
            seed=42
        )

        # multi-VLAD encoding
        enc_train = multi_vlad_encode(
            files_train, codebooks,
            powernorm=args.powernorm,
            gmp=args.gmp, gamma=args.gamma
        )
        enc_test = multi_vlad_encode(
            files_test, codebooks,
            powernorm=args.powernorm,
            gmp=args.gmp, gamma=args.gamma
        )
        print(
            f"> Multi-VLAD shapes: train {enc_train.shape}, test {enc_test.shape}")

        # PCA whitening
        enc_train_p, enc_test_p, pca = apply_pca_whitening(
            enc_train, enc_test, n_components=args.pca_components, seed=42
        )

        # Evaluate VLAD(+PCA)
        print('> evaluate (multi-VLAD + PCA)')
        evaluate(enc_test_p, labels_test)

        # E-SVM on PCA-reduced features (optional comparison)
        print('> esvm computation (multi-VLAD + PCA)')
        enc_test_esvm = esvm(enc_test_p, enc_train_p, C=args.C)
        print('> evaluate (multi-VLAD + PCA + E-SVM)')
        evaluate(enc_test_esvm, labels_test)

    else:
        # single-codebook path (as before)
        if not os.path.exists('mus.pkl.gz'):
            print('> loaded {} descriptors:'.format(len(descriptors)))
            K = args.n_clusters  # configurable
            mus = dictionary(descriptors, n_clusters=K)
            print("Codebook centers shape:", mus.shape)
            print('> compute dictionary')
            with gzip.open('mus.pkl.gz', 'wb') as fOut:
                cPickle.dump(mus, fOut, -1)
        else:
            with gzip.open('mus.pkl.gz', 'rb') as f:
                mus = cPickle.load(f)

        # b) VLAD encoding
        fname = 'enc_train_gmp{}.pkl.gz'.format(
            args.gamma) if args.gmp else 'enc_train.pkl.gz'
        if not os.path.exists(fname) or args.overwrite:
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

        print('> esvm computation')
        print("> running Exemplar‐SVM refinement")
        enc_test = esvm(enc_test, enc_train, C=args.C)
        print(f"> refined TEST encodings via ESVM → {enc_test.shape}")
        print('> evaluate')
        evaluate(enc_test, labels_test)
