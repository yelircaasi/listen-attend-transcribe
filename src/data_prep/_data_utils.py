""" Define useful functions for data I/O.
"""
import os
import glob
import numpy as np
import pandas as pd

from pandas.io.parsers import ParserBase


def encode_fn(s_in):
    """
    A function for Pytorch-NLP tokenizer to encode sequences.
    Args:
        s_in (string): Sentence.
    Returns:
        s_out (list(string)): Words.
    """
    s_out = s_in.split()
    return s_out


def decode_fn(s_in):
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
        elif w=='</s>':
            break
        s_out.append(w)
    s_out = ' '.join(s_out)
    return s_out


def get_feats(filepath, continuous=False, as_dict=True):
    """
    A function to 
    """
    if continuous:
        featdf = pd.read_csv("../ipa/features_cont.tsv", sep="\t", header=None, index_col=0)
    else: 
        featdf = pd.read_csv("../ipa/features_bin.tsv", sep="\t", header=None, index_col=0)
    if as_dict:
        feats = {}
        for i in featdf.index:
            feats[i] = np.array(featdf.loc[i])
            return feats
    return featdf
