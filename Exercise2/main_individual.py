import os
import argparse
import gzip
import numpy as np
import _pickle as cPickle
from tqdm import tqdm

# Import shared functions - vlad() handles both sum and GMP!
from skeleton import (
    getFiles,
    loadRandomDescriptors,
    dictionary,
    vlad,          # ONE function, use gmp=True for GMP, gmp=False for sum
    esvm,
    evaluate,
    fetch_encodings
)

# Custom SIFT extractor for part (e)
from custom_sift_extractor import CustomSIFTExtractor


# Use the BINARY folders
TRAIN_IMAGE_DIRS = ["data/icdar2017-training-color"]
TEST_IMAGE_DIRS = ["data/icdar2017-testing-color"]

OUT_TRAIN = "data/local_features/trainColor1"
OUT_TEST = "data/local_features/testColor1"


def parseArgs(parser):
    """Extended argument parser with bonus feature flags"""
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
                        help='use generalized max pooling (Part f bonus)')
    parser.add_argument('--gamma', default=1, type=float,
                        help='regularization parameter of GMP')
    parser.add_argument('--C', default=1000, type=float,
                        help='C parameter of the SVM')

    # Bonus part (g)
    parser.add_argument('--multi_vlad', action='store_true',
                        help='use multi-VLAD with multiple codebooks (part g)')
    parser.add_argument('--n_codebooks', default=5, type=int,
                        help='number of codebooks for multi-VLAD (default: 5)')
    parser.add_argument('--n_clusters', default=100, type=int,
                        help='clusters per codebook (default: 100)')
    parser.add_argument('--pca_components', default=1000, type=int,
                        help='PCA output dimensionality (with whitening) for part g')
    return parser


# ========== BONUS PART (g): MULTI-VLAD + PCA WHITENING ==========

def create_multiple_codebooks(files, n_codebooks=5, n_clusters=100,
                              max_descriptors=1_000_000, seed=42):
    """
    Create multiple codebooks with different seeds and subsets (Part g)

    parameters:
        files: descriptor files
        n_codebooks: number of codebooks to create
        n_clusters: K clusters per codebook
        max_descriptors: total descriptor budget
        seed: random seed
    returns: list of KxD codebook arrays
    """
    codebooks = []
    for i in range(n_codebooks):
        print(f"> Building codebook {i+1}/{n_codebooks}")
        np.random.seed(seed + i)  # different seed per codebook

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
    Compute VLAD for each codebook, then concatenate features (Part g)
    Uses the shared vlad() function with gmp parameter!

    parameters:
        files: descriptor files
        codebooks: list of KxD codebooks
        powernorm: use power normalization
        gmp: use GMP (passed to vlad function)
        gamma: GMP regularization (passed to vlad function)
    returns: N x (n_codebooks*K*D) matrix
    """
    enc_list = []
    for ci, mus in enumerate(codebooks):
        print(
            f"> Encoding with codebook {ci+1}/{len(codebooks)} (K={mus.shape[0]}, D={mus.shape[1]})")
        # Just call the shared vlad() with gmp parameter!
        enc = vlad(files, mus, powernorm=powernorm, gmp=gmp, gamma=gamma)
        enc_list.append(enc.astype(np.float32))
    # horizontal concat → feature dimension grows
    return np.concatenate(enc_list, axis=1)


def pca_whitening(enc_train, enc_test, n_components=1000, seed=42):
    """
    PCA with whitening (Part g)
    Fit on train, transform test

    parameters:
        enc_train: NxD training encodings
        enc_test: MxD test encodings
        n_components: target dimensionality
        seed: random seed
    returns: (enc_train_pca, enc_test_pca, pca_model)
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


# ========== UTILITY FUNCTIONS ==========

def descriptor_stats(file_list):
    """Get statistics about descriptor files"""
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


# ========== MAIN EXECUTION ==========

if __name__ == '__main__':
    parser = argparse.ArgumentParser('retrieval')
    parser = parseArgs(parser)
    args = parser.parse_args()
    np.random.seed(42)

    # ===== PART (e): Build custom descriptors =====
    os.makedirs(OUT_TRAIN, exist_ok=True)
    os.makedirs(OUT_TEST, exist_ok=True)

    extractor = CustomSIFTExtractor()

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

    # Load files
    files_train, labels_train = getFiles(
        folder=OUT_TRAIN,
        pattern="_SIFT_patch_pr.pkl.gz",
        labelfile="data/icdar17_labels_train.txt"
    )
    print('#train: {}'.format(len(files_train)))

    files_test, labels_test = getFiles(
        folder=OUT_TEST,
        pattern="_SIFT_patch_pr.pkl.gz",
        labelfile="data/icdar17_labels_test.txt"
    )
    print('#test: {}'.format(len(files_test)))

    # Descriptor statistics
    tr_exist, tr_nonempty, tr_rows = descriptor_stats(files_train)
    te_exist, te_nonempty, te_rows = descriptor_stats(files_test)
    print(f"[Part (e)] Train descriptors: {tr_exist} files present, "
          f"{tr_nonempty} non-empty, {tr_rows} total SIFT vectors.")
    print(f"[Part (e)] Test descriptors: {te_exist} files present, "
          f"{te_nonempty} non-empty, {te_rows} total SIFT vectors.")

    # Load descriptors for codebook
    descriptors = loadRandomDescriptors(files_train, max_descriptors=500_000)
    print("Sampled descriptors:", descriptors.shape)

    # ===== PART (g): MULTI-VLAD PATH =====
    if args.multi_vlad:
        print("> Part (g): multi-VLAD enabled")
        print(
            f"> Using {'GMP' if args.gmp else 'SUM'} pooling with gamma={args.gamma}")

        # Build multiple codebooks
        codebooks = create_multiple_codebooks(
            files_train,
            n_codebooks=args.n_codebooks,
            n_clusters=args.n_clusters,
            max_descriptors=1_000_000,
            seed=42
        )

        # Multi-VLAD encoding - vlad() handles GMP automatically!
        enc_train = multi_vlad_encode(
            files_train, codebooks,
            powernorm=args.powernorm,
            gmp=args.gmp,      # ← Pass GMP flag to shared vlad()
            gamma=args.gamma
        )
        enc_test = multi_vlad_encode(
            files_test, codebooks,
            powernorm=args.powernorm,
            gmp=args.gmp,      # ← Pass GMP flag to shared vlad()
            gamma=args.gamma
        )
        print(
            f"> Multi-VLAD shapes: train {enc_train.shape}, test {enc_test.shape}")

        # PCA whitening
        enc_train_p, enc_test_p, pca = pca_whitening(
            enc_train, enc_test, n_components=args.pca_components, seed=42
        )

        # Evaluate VLAD+PCA
        print('> evaluate (multi-VLAD + PCA)')
        evaluate(enc_test_p, labels_test)

        # E-SVM on PCA-reduced features
        print('> esvm computation (multi-VLAD + PCA)')
        enc_test_esvm = esvm(enc_test_p, enc_train_p, C=args.C)
        print('> evaluate (multi-VLAD + PCA + E-SVM)')
        evaluate(enc_test_esvm, labels_test)

    # ===== SINGLE-CODEBOOK PATH (with optional GMP) =====
    else:
        print(
            f"> Using {'GMP' if args.gmp else 'SUM'} pooling with gamma={args.gamma}")

        if not os.path.exists('mus.pkl.gz'):
            print('> loaded {} descriptors:'.format(len(descriptors)))
            K = args.n_clusters
            mus = dictionary(descriptors, n_clusters=K)
            print("Codebook centers shape:", mus.shape)
            print('> compute dictionary')
            with gzip.open('mus.pkl.gz', 'wb') as fOut:
                cPickle.dump(mus, fOut, -1)
        else:
            with gzip.open('mus.pkl.gz', 'rb') as f:
                mus = cPickle.load(f)

        # VLAD encoding - just pass gmp flag! vlad() handles the rest
        fname = 'enc_train_gmp{}.pkl.gz'.format(
            args.gamma) if args.gmp else 'enc_train.pkl.gz'
        enc_train = fetch_encodings(
            files_train,
            mus,
            fname,
            powernorm=args.powernorm,
            gmp=args.gmp,      # ← Just pass the flag!
            gamma=args.gamma,
            overwrite=args.overwrite
        )

        fname = 'enc_test_gmp{}.pkl.gz'.format(
            args.gamma) if args.gmp else 'enc_test.pkl.gz'
        enc_test = fetch_encodings(
            files_test,
            mus,
            fname,
            powernorm=args.powernorm,
            gmp=args.gmp,      # ← Just pass the flag!
            gamma=args.gamma,
            overwrite=args.overwrite
        )

        # Evaluate
        print('> evaluate')
        evaluate(enc_test, labels_test)

        # E-SVM refinement
        print('> esvm computation')
        print("> running Exemplar‐SVM refinement")
        enc_test = esvm(enc_test, enc_train, C=args.C)
        print(f"> refined TEST encodings via ESVM → {enc_test.shape}")
        print('> evaluate')
        evaluate(enc_test, labels_test)
