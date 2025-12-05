"""
Group Solution: VLAD-based Writer Identification (Parts a-d)
Uses shared vlad_core module - calls vlad() with gmp=False for sum pooling
"""
import os
import argparse
import gzip
import numpy as np
import _pickle as cPickle

# Import all shared functions
from skeleton import (
    getFiles,
    loadRandomDescriptors,
    dictionary,
    vlad,          # ONE vlad function that handles both sum and GMP
    esvm,
    evaluate,
    fetch_encodings
)


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
    parser.add_argument('--C', default=1000, type=float,
                        help='C parameter of the SVM')
    return parser


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

    if not os.path.exists('mus_binary.pkl.gz'):
        print('> loaded {} descriptors:'.format(len(descriptors)))
        K = 100
        mus = dictionary(descriptors, n_clusters=K)
        print("Codebook centers shape:", mus.shape)
        print('> compute dictionary')
        with gzip.open('mus_binary.pkl.gz', 'wb') as fOut:
            cPickle.dump(mus, fOut, -1)
    else:
        with gzip.open('mus_binary.pkl.gz', 'rb') as f:
            mus = cPickle.load(f)

    # b) VLAD encoding - use separate filenames for binary images
    fname = 'enc_train_binary.pkl.gz'
    enc_train = fetch_encodings(
        files_train,
        mus,
        fname,
        powernorm=args.powernorm,
        gmp=False,      # ← Sum pooling for group solution
        gamma=1,        # Not used when gmp=False
        overwrite=args.overwrite
    )

    fname = 'enc_test_binary.pkl.gz'
    enc_test = fetch_encodings(
        files_test,
        mus,
        fname,
        powernorm=args.powernorm,
        gmp=False,      # ← Sum pooling for group solution
        gamma=1,        # Not used when gmp=False
        overwrite=args.overwrite
    )

    # cross-evaluate test encodings
    print('> evaluate')
    evaluate(enc_test, labels_test)

    # d) compute exemplar svms
    print('> esvm computation')
    print("> running Exemplar‐SVM refinement")
    enc_test = esvm(enc_test, enc_train, C=args.C)
    print(f"> refined TEST encodings via ESVM → {enc_test.shape}")
    print('> evaluate')
    evaluate(enc_test, labels_test)
