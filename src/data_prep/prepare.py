""" Create reference list of audio files and transcripts, 
      then create Pytorch-NLP tokenizer for TIMIT.
"""
import torch
import os
import glob
import argparse
import pandas as pd
#import data_utils

from featurizers import build_featurizer


# def create_tokenizer():
#    """
#    Create and save Pytorch-NLP tokenizer.
#    Args:
#        root (string): Directory of TIMIT.
#    """
#    transcripts = pd.read_csv('TRAIN.csv')['transcript']
#    tokenizer = StaticTokenizerEncoder(transcripts,
#                                       append_sos=True,
#                                       append_eos=True,
#                                       tokenize=data_utils.encode_fn,
#                                       detokenize=data_utils.decode_fn)
#    torch.save(tokenizer, 'tokenizer.pth')

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
    # TODO: fix this section here
    #process_dataset(args.root, 'train')
    #process_dataset(args.root, 'dev')
    #process_dataset(args.root, 'test')

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

        for feat_type in args.features.split(","):
            build_featurizer(dataset, feat_type)
    
    print("Data preparation is complete !")


if __name__ == '__main__':
    main()
