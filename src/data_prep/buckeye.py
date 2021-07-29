"""Script to process TIMIT dataset and save train/test/dev CSV files containing lists of
     audio files and their corresponding labels
"""

import ipa
import sys
import os
import glob
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
        lines = [x for x in f.read().split("\n#\n")[-1].split("\n") if x]
    phon_seq = [l.split()[-1] for l in lines]
    ipa_seq = " ".join([ipa.arpadict.get(p, "?") for p in phon_seq])
    phonemes = " ".join(phon_seq)
    return phonemes, ipa_seq


def process_dataset(root, buckeye_dir):
    """
    List audio files and transcripts for a certain partition of TIMIT dataset.
    Args:
        root (string): Directory of TIMIT.
        split (string): Which of the subset of data to take. One of 'train', 'dev' or 'test'.
    """
    print(root)
    print(buckeye_dir)
    #audio_files = []
    #phones_files = []

    relativize = lambda file_path: file_path.replace(root, "").strip("/")

    lines = []
    for sXX in [os.path.join(root, buckeye_dir, f) for f in os.listdir(os.path.join(root, buckeye_dir)) if len(f) == 3 and f.startswith("s")]:
        #print("sXX", sXX)
        for sXXXXX in os.listdir(sXX):
            folder = os.path.join(root, sXX, sXXXXX)
            #print(sXXXXX)
            line = []
            for file in os.listdir(folder):
                file = os.path.join(folder, file)
                if file.endswith(".wav"):
                    line.append(os.path.join(root, sXX, sXXXXX, file))
                elif file.endswith(".phones"):
                    line.extend(read_phonemes(file))
            line.sort(key=lambda x: x.startswith("/"), reverse=True)
            line[0] = relativize(line[0])
            lines.append(",".join(line))
    nobs = len(lines)
    np.random.shuffle(lines)
    split1 = round(0.8 * nobs)
    split2 = round(0.9 * nobs)
    fname = f"resources/datalists/buckeye_TRAIN.csv"
    header = "audio,phonemes,ipa\n"
    with open(fname, "w") as f:
        f.write(header + "\n".join(lines[:split1]))
    print(f"{fname} is created.")
    fname = f"resources/datalists/buckeye_TEST.csv"
    with open(fname, "w") as f:
        f.write(header + "\n".join(lines[split1:split2]))
    print(f"{fname} is created.")
    fname = f"resources/datalists/buckeye_DEV.csv"
    with open(fname, "w") as f:
        f.write(header + "\n".join(lines[split2:]))
    print(f"{fname} is created.")
    