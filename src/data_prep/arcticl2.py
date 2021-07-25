"""Script to process TIMIT dataset and save train/test/dev CSV files containing lists of 
     audio files and their corresponding labels
"""

import sys 
import os
import re
from pathlib import Path
import numpy as np
sys.path.append(os.path.join(Path(__file__).parent.parent.parent, "resources"))
#print(sys.path)
assert os.path.exists(sys.path[-1])

import ipa


np.random.seed(7)

# Core test set 24 speakers
SPEAKERS = [
    "ABA", "ASI", "BWC", "EBVS", "ERMS", "HJK", "HKK", "HQTV", "LXC", "MBMPS", 
    "NCC", "NJS", "PNV", "RRBI", "SKA", "SVBI", "THV", "TLV", "TNI", "TXHC", 
    "YBAA", "YDCK", "YKWK", "ZHAA", "suitcase_corpus"
]    


def read_textgrid(textgrid_path):
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
        text = f.read().split("name = \"phones\"")[-1].split("name = \"")[0]
    phon_seq = re.findall("(?<=text = \").+?(?=\")", text)
    print(phon_seq)
    phonemes = " ".join(phon_seq)
    ipa_seq = " ".join([ipa.arpadict.get(p, "%") for p in phon_seq])
    return phonemes, ipa_seq


def read_annotation(textgrid_path):
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
        text = f.read().split("name = \"phones\"")[-1].split("name = \"")
    phon_seq = re.findall("text = \".+?\"", text)
    phon_seq = [x.split(",")[1] if "," in x else x for x in phon_seq]
    #phonemes = " ".join(phon_seq)
    ipa_seq = " ".join([ipa.arpadict.get(p, "%") for p in phon_seq])
    phon_seq = " ".join(phon_seq)
    return ipa_seq
    

def process_dataset(root):
    """
    List audio files and transcripts for a certain partition of TIMIT dataset.
    Args:
        root (string): Directory of TIMIT.
        split (string): Which of the subset of data to take. One of 'train', 'dev' or 'test'.
    """
    print(root)
    audio_files = []
    for speaker in SPEAKERS:
        folder = os.path.join(root, speaker, "wav")
        audio_files.extend([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".wav")])

    seq_manual = []
    seq_auto = []
    seq_all = []
    for file in audio_files[:10]:
        manual = file[:-4].replace("/wav/", "/annotations/") + ".TextGrid" 
        auto = file[:-4].replace("/wav/", "/textgrid/") + ".TextGrid" 
        phonseq, ipaseq = read_textgrid(auto)
        seq_all.append(f"{file},{phonseq},{ipaseq}")
        if os.path.exists(manual):
            phonseq_, ipaseq_ = read_annotation(manual)
            seq_manual.append(f"{file},{phonseq_},{ipaseq_}")
        else:
            seq_auto.append(f"{file},{phonseq},{ipaseq}")

    for name, list_ in zip(["manual", "auto", "all"], [seq_manual, seq_auto, seq_all]):
        nobs = len(list_)
        split1 = int(0.8 * nobs)
        split2 = int(0.9 * nobs)
        np.random.shuffle(list_)
        fname = f"resources/datalists/arcticl2_{name}_TRAIN.csv"
        header = "audio,phonemes,ipa\n"
        with open(fname, "w") as f:
            f.write(header + "\n".join(list_[:split1]))
        print(f"{fname} is created.")
        fname = f"resources/datalists/arcticl2_{name}_TEST.csv"
        with open(fname, "w") as f:
            f.write(header + "\n".join(list_[split1:split2]))
        print(f"{fname} is created.")
        fname = f"resources/datalists/arcticl2_{name}_DEV.csv"
        with open(fname, "w") as f:
            f.write(header + "\n".join(list_[split2:]))
        print(f"{fname} is created.")
