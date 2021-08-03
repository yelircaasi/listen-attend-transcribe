""" Load and preprocess data.
"""
import sys
import torch
import torchaudio
import os
import argparse
import pandas as pd
import numpy as np
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader
sys.path.append("resources")
sys.path.append("src/data_prep")
import featurizers

dataset_dict = { # map dataset codes todirectory names
    "timit": "timit",
    "arcticl2": "arctic_l2",
    "arabicsc": "arabic_speech_corpus",
    "buckeye": "buckeye",
    "arcticl2_all": "arctic_l2"
}

class ASR(Dataset):
    """
    Stores a Pandas DataFrame in __init__, and reads and preprocesses examples in __getitem__.
    """
    def __init__(self, split, data_dir, dataset_key, output_type, stack_frames):
        split = split.upper()
        self.df = pd.read_csv(f"resources/datalists/{dataset_key}_{split}.csv")
        self.data_dir = data_dir
        self.dataset_key = dataset_key
        self.dataset_folder = dataset_dict[dataset_key]
        self.output_type = output_type
        self.dataset_name = dataset_key.replace("_all", "")
        self.tokenizer = torch.load(f"resources/featurizers/featurizer_{self.dataset_name}_{output_type}.pth")
        self.stack_frames = stack_frames
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
            x (torch.FloatTensor, [seq_length, dim_features]): The FBANK features.
            y (torch.LongTensor, [n_tokens]): The label sequence.
        """
        x, _, y = self.df.iloc[idx]
        x, _ = torchaudio.load(os.path.join(self.data_dir, x))
        # Compute filter bank features
        x = torchaudio.compliance.kaldi.fbank(x, num_mel_bins=80)   # [n_windows, 80]
        # Stack every 3 frames and down-sample frame rate by 3, following https://arxiv.org/pdf/1712.01769.pdf.
        n = self.stack_frames
        x = x[:(x.shape[0]//n)*n].view(-1,n*80)   # [n_windows, 80] --> [n_windows//3, 240]
        # Tokenization
        y = self.tokenizer.encode(y)
        return x, y

    def generateBatch(self, batch):
        """
        Generate a mini-batch of data. For DataLoader's 'collate_fn'.
        Args:
            batch (list(tuple)): A mini-batch of (FBANK features, label sequences) pairs.
        Returns:
            xs (torch.FloatTensor, [batch_size, (padded) seq_length, dim_features]): A mini-batch of FBANK features.
            xlens (torch.LongTensor, [batch_size]): Sequence lengths before padding.
            ys (torch.LongTensor, [batch_size, (padded) n_tokens]): A mini-batch of label sequences.
        """
        xs, ys = zip(*batch)
        xlens = torch.tensor([x.shape[0] for x in xs])
        xs = rnn_utils.pad_sequence(xs, batch_first=True)   # [batch_size, (padded) seq_length, dim_features]
        ys = rnn_utils.pad_sequence(ys, batch_first=True)   # [batch_size, (padded) n_tokens]
        return xs, xlens, ys


def load(split, batch_size, data_dir, dataset_key, output, nstack, workers=0):
    """
    Args:
        split (string): Which of the subset of data to take. One of 'train', 'dev' or 'test'.
        batch_size (integer): Batch size.
        workers (integer): How many subprocesses to use for data loading.
    Returns:
        loader (DataLoader): A DataLoader can generate batches of (FBANK features, FBANK lengths, label sequence).
    """
    split = split.upper()
    assert split in ['TRAIN', 'DEV', 'TEST']
    #data_dir = dataset_dict[dataset_key]

    dataset = ASR(split, data_dir, dataset_key, output, nstack)
    n = dataset.__len__()
    print (f"{split} set size: {n}")
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        collate_fn=dataset.generateBatch,
                        shuffle=True,
                        num_workers=workers,
                        pin_memory=True)
    return loader


def inspect_data(data_prefix, dataset, split, batch_size, output, nstack):
    """
    Test the functionality of input pipeline and visualize a few samples.
    """
    import matplotlib.pyplot as plt
    print("\n********************************")
    print(f"Dataset: {dataset}\nSplit: {split}")
    print(f"Batch size: {batch_size}")
    print(f"Output type: {output}")
    print(f"Frames stacked: {nstack}")
    print("********************************")

    loader = load(split, batch_size, data_prefix, dataset, output, nstack)
    dataset = dataset.replace("_all", "")
    tokenizer = torch.load(f"resources/featurizers/featurizer_{dataset}_{output}.pth")
    print ("Vocabulary size:", len(tokenizer.vocab))
    print ("Vocabulary: ", tokenizer.vocab)

    xs, xlens, ys = next(iter(loader))
    print ("Dimensions of X: ", xs.shape) 
    print("Dimensions of y: ", ys.shape)
    print()
    for i in range(batch_size):
        print ("y indices:\t", ys[i])
        print ("y labels:\t", tokenizer.decode(ys[i]))
        plt.figure()
        plt.imshow(xs[i].T)
        plt.show()
        print()
    print("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inspect data.")
    parser.add_argument('--dataset', type=str, help="Which data to inspect.")
    parser.add_argument('--split', default='dev', type=str,
                        help="Specify which split of data to evaluate (train / test / dev).")
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument('--output', default="phones", type=str,
                        help="Output type: phones / bin feats / cont feats.")
    parser.add_argument("--nstack", default=3, type=int, 
                        help="number of frames to stack")
    args = parser.parse_args()
    inspect_data(args.dataset, args.split, args.batch_size, 
                 args.output, args.nstack)
