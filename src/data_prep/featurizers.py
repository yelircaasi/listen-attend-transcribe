""" Define useful functions for data I/O.
"""
import os
import sys
import torch
from pathlib import Path
import numpy as np
import pandas as pd

#from pandas.io.parsers import ParserBase
from torchnlp.encoders.text import StaticTokenizerEncoder
# https://pytorchnlp.readthedocs.io/en/latest/source/torchnlp.encoders.html

sys.path.append(os.path.join(Path(__file__).parent.parent.parent, "resources"))
import ipa


def tokenize_fn(s_in):
    """
    A function for Pytorch-NLP tokenizer to encode sequences.
    Args:
        s_in (string): Sentence.
    Returns:
        s_out (list(string)): Words.
    """
    s_out = s_in.split()
    return s_out


def detokenize_fn(s_in):
    """
    A function for the Pytorch-NLP tokenizer to decode sequences.
    Args:
        s_in (list of strings): Words.
    Returns:
        s_out (string): Sentence.
    """
    s_out = []
    for w in s_in:
        if w == '<s>':
            continue
        elif w == '</s>':
            break
        s_out.append(w)
    s_out = ' '.join(s_out)
    return s_out


#def get_feats(filepath, continuous=False, as_dict=True):
#    """
#    A function to 
#    """
#    if continuous:
#        featdf = pd.read_csv("../ipa/features_cont.tsv",
#                             sep="\t", header=None, index_col=0)
#    else:
#        featdf = pd.read_csv("../ipa/features_bin.tsv",
#                             sep="\t", header=None, index_col=0)
#    if as_dict:
#        feats = {}
#        for i in featdf.index:
#            feats[i] = np.array(featdf.loc[i])
#            return feats
#    return featdf


class FeaturizerCont(StaticTokenizerEncoder):
    """Override 'encode' method to allow for dense continuous feature representations.
            """
    def encode(self, ipa_sequence):
        """Encodes a sequence.
                Args:
                    sequence (string): String sequence to encode.
                Returns:
                    torch.Tensor: Encoding of the sequence.
                """
        orig_seq = ipa_sequence.split()
        n = len(orig_seq)
        matrix = np.zeros((n, 33))
        for i, seg in enumerate(orig_seq):
            matrix[i] = ipa.contfeats[seg]
        return torch.tensor(matrix)
    
    def decode(self, data):
        return None

    def decode_beam_search(self, preds, beam_size):
        """
        Implements beam search to find best phonetic sequence
            given feature predictions.
        """
        return None


class FeaturizerBin(StaticTokenizerEncoder):
    """Override 'encode' method to allow for dense binary feature representations.
    """
    def encode(self, ipa_sequence):
        orig_seq = ipa_sequence.split()
        n = len(orig_seq)
        matrix = np.zeros((n, 43))
        for i, seg in enumerate(orig_seq):
            matrix[i] = ipa.binfeats[seg]
        return torch.tensor(matrix)

    def decode(self, data):
        return None

    def decode_beam_search(self, preds, beam_size):
        """
        Implements beam search to find best phonetic sequence
            given feature predictions.
        """
        return None




def build_featurizer(dataset, feature_type):
    """
    Builds a featurizer class for a specified dataset and feature type.
    Args:
        dataset (string): One of ["timit", "arcticl2_all", "arabicsc", "buckeye"]
        feature_type (string): One of ["phones", "cont", "bin"]
    Returns:
        Instance of class 'torchnlp.encoders.text.static_tokenizer_encoder'
    """
    transcripts = pd.read_csv(f"resources/datalists/{dataset}_TRAIN.csv").append(
                  pd.read_csv(f"resources/datalists/{dataset}_TEST.csv")).append(
                  pd.read_csv(f"resources/datalists/{dataset}_DEV.csv"))["ipa"]
    if feature_type == "phones":
        featurizer = StaticTokenizerEncoder(transcripts,
                                            append_sos=True,
                                            append_eos=True,
                                            tokenize=tokenize_fn,
                                            detokenize=detokenize_fn)
    elif feature_type == "cont":
        featurizer = FeaturizerCont(transcripts,
                                    append_sos=True,
                                    append_eos=True,
                                    tokenize=tokenize_fn,
                                    detokenize=detokenize_fn)
    elif feature_type == "bin":
        featurizer = FeaturizerBin(transcripts,
                                   append_sos=True,
                                   append_eos=True,
                                   tokenize=tokenize_fn,
                                   detokenize=detokenize_fn)
    
    dataset = dataset.split("_")[0]
    save_path = f"resources/featurizers/featurizer_{dataset}_{feature_type}.pth"
    torch.save(featurizer, save_path)
    print(f"{save_path} saved.")
