"""Script to process TIMIT dataset and save train/test/dev CSV files containing lists of
     audio files and their corresponding labels
"""

import ipa
import sys
import os
#import glob
import re
from pathlib import Path
import numpy as np
sys.path.append(os.path.join(Path(__file__).parent.parent.parent, "resources"))
#print(sys.path)
assert os.path.exists(sys.path[-1])


np.random.seed(7)


def read_phonemes(textgrid_path):
    """
    Given an audio file path in the 'textgrid' folder (automatically generated, ARPA phonemes),
      return its labels.
    Args:
        audio_file (string): Audio file path
        label_type (string): Default (phones), continuous features, or binary features
    Returns:
        phonemes (string): A sequence of ARPA phonemes for this audio file,
                             as contained in the original file.
        ipa_seq (string): The sequence of IPA characters, converted from
    """
    with open(textgrid_path) as f:
        text = f.read().split("=name = \"phones\"")[-1].split("name = \"words\"")[0]
    phon_seq = re.findall("(?<=text = \").+?(?=\")", text)
    #print(phon_seq)
    phonemes = " ".join(phon_seq)
    ipa_seq = " ".join([ipa.ascdict.get(p, "?") for p in phon_seq])
    return phonemes, ipa_seq


#def read_annotation(textgrid_path):
#    """
#    Given an audio file path in the 'textgrid' folder (automatically generated, ARPA phonemes),
#      return its labels.
#    Args:
#        audio_file (string): Audio file path
#        label_type (string): Default (phones), continuous features, or binary features
#    Returns:
#        phonemes (string): A sequence of ARPA phonemes for this audio file,
#                             as contained in the original file.
#        ipa_seq (string): The sequence of IPA characters, converted from
#    """
#    return ipa_seq


def process_dataset(root):
    """
    List audio files and transcripts for a certain partition of TIMIT dataset.
    Args:
        root (string): Directory of TIMIT.
        split (string): Which of the subset of data to take. One of 'train', 'dev' or 'test'.
    """
    print(root)
    root_test = os.path.join(root, "test set")
    print(root_test)
    #audio_files = []
    #phones_files = []
    
    def get_lines(directory):
        lines_ = []
        wavpath = os.path.join(directory, "wav")
        audio = [os.path.join(wavpath, file) for file in os.listdir(wavpath)]
        for file in audio:
            tg = file.replace("/wav/", "/textgrid/").replace(".wav", ".TextGrid")
            if os.path.exists(tg):
                phonseq, ipaseq = read_phonemes(tg)
                lines_.append(f"{file},{phonseq},{ipaseq}")
        return lines_
    
    trainlines = get_lines(root)
    testlines = get_lines(root_test)
    #print(trainlines)
    #print(testlines)
    np.random.shuffle(trainlines)
    np.random.shuffle(testlines)

    nobs = len(trainlines)
    split = round(0.9 * nobs)
    fname = f"resources/datalists/arabicsc_TRAIN.csv"
    header = "audio,phonemes,ipa\n"
    with open(fname, "w") as f:
        f.write(header + "\n".join(trainlines[:split]))
    print(f"{fname} is created.")
    fname = f"resources/datalists/arabicsc_DEV.csv"
    with open(fname, "w") as f:
        f.write(header + "\n".join(trainlines[split:]))
    print(f"{fname} is created.")
    fname = f"resources/datalists/arabicsc_TEST.csv"
    with open(fname, "w") as f:
        f.write(header + "\n".join(testlines))
    print(f"{fname} is created.")
    
