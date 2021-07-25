""" Create reference list of audio files and transcripts, 
      then create Pytorch-NLP tokenizer for TIMIT.
"""
import torch
import os
import glob
import argparse
import pandas as pd
from featurizers import build_featurizer

dataset_dirnames = {
    "timit": "timit",
    "arcticl2": "arctic_l2",
    "arabicsc": "arabic_speech_corpus",
    "buckeye": "buckeye"
}


def main():
    parser = argparse.ArgumentParser(
        description="Make lists of audio files and transcripts, and create tokenizer.")
    parser.add_argument('--root', default="/mount/studenten/arbeitsdaten-studenten1/rileyic/timit/data",
                        type=str, help="base data directory")
    parser.add_argument("--datasets", default="timit", type=str,
                        help="comma-separated dataset names (timit / arcticl2 / arabicsc / buckeye)")
    parser.add_argument("--features", default="phones", type=str,
                        help="comma-separated feature types (phones / cont / bin)")
    args = parser.parse_args()

    print(args.root)
    assert os.path.exists(args.root)
    
    for dataset in args.datasets.split(","):
        if dataset == "timit":
            from timit import process_dataset
            # TODO
        elif dataset == "arcticl2":
            from arcticl2 import process_dataset
            # TODO
        elif dataset == "arabicsc":
            from arabicsc import process_dataset
            # TODO
        elif dataset == "buckeye":
            from buckeye import process_dataset
            # TODO
        else:
            print(
                "Dataset {args.dataset} not available. Check spelling and try again.")
            continue

        data_dir = os.path.join(args.root, dataset_dirnames[dataset])
        process_dataset(data_dir)

        if dataset == "arcticl2":
            dataset = "arcticl2_all"

        for feat_type in args.features.split(","):
            build_featurizer(dataset, feat_type)
    
    print("Data preparation is complete !")


if __name__ == '__main__':
    main()
