""" Create reference list of audio files and transcripts, 
      then create Pytorch-NLP tokenizer for Arctic L2.
"""
import torch
import os
import glob
import argparse
import pandas as pd
import data_utils
from torchnlp.encoders.text import StaticTokenizerEncoder


def read_phonemes():
    """
    Given an audio file path, return its labels.
    Args:
        audio_file (string): Audio file path
        label_type (string): Default (phones)
    Returns:
        phonemes (string): A sequence of phonemes for this audio file
    """


"""
"""


def process_dataset(dataset_directory, feature_type):

    return None

