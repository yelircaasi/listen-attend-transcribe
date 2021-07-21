""" Define useful functions for data I/O.
"""
import os
import numpy as np
import pandas as pd

#from pandas.io.parsers import ParserBase
from torchnlp.encoders.text import StaticTokenizerEncoder


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
        elif w == '</s>':
            break
        s_out.append(w)
    s_out = ' '.join(s_out)
    return s_out


def get_feats(filepath, continuous=False, as_dict=True):
    """
    A function to 
    """
    if continuous:
        featdf = pd.read_csv("../ipa/features_cont.tsv",
                             sep="\t", header=None, index_col=0)
    else:
        featdf = pd.read_csv("../ipa/features_bin.tsv",
                             sep="\t", header=None, index_col=0)
    if as_dict:
        feats = {}
        for i in featdf.index:
            feats[i] = np.array(featdf.loc[i])
            return feats
    return featdf


encoders = {
    "": #TODO
}

decoders = {
    "": #TODO
}


class FeaturizerCont(StaticTokenizerEncoder):
    """Override 'encode' method to allow for dense continuous feature representations.
            """
    def encode(self, sequence):
        """Encodes a sequence.
                Args:
                    sequence (string): String sequence to encode.
                Returns:
                    torch.Tensor: Encoding of the sequence.
                """
        #TODO
        return torch.tensor


class FeaturizerBin(StaticTokenizerEncoder):
    """Override 'encode' method to allow for dense binary feature representations.
    """
    





def build_featurizer(dataset, feature_type):
    """
    Builds a featurizer class for a specified dataset and feature type.
    Args:
        dataset (string): One of ["timit", "arcticl2", "arabic_speech_corpus", "buckeye"]
        feature_type (string): One of ["phones", "cont", "bin"]
    Returns:
        Instance of class 'torchnlp.encoders.text.static_tokenizer_encoder'
    """
    if feature_type == "phones":
        featurizer = StaticTokenizerEncoder(transcripts,
                                            append_sos=True,
                                            append_eos=True,
                                            tokenize=encoders[dataset][feature_type],
                                            detokenize=decoders[dataset][feature_type])
    
    elif feature_type == "cont":
        featurizer = FeaturizerCont(transcripts,
                                    append_sos=True,
                                    append_eos=True,
                                    tokenize=encoders[dataset][feature_type],
                                    detokenize=decoders[dataset][feature_type])

    elif feature_type == "bin":
        featurizer = FeaturizerBin(transcripts,
                                    append_sos=True,
                                    append_eos=True,
                                    tokenize=encoders[dataset][feature_type],
                                    detokenize=decoders[dataset][feature_type])

            
    save_path = f"resources/featurizer_{dataset}_{feature_type}.pth"
    torch.save(featurizer, save_path)
    print(f"{save_path} saved.")



#def create_tokenizer():
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
