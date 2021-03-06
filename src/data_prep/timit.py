"""Script to process TIMIT dataset and save train/test/dev CSV files containing lists of 
     audio files and their corresponding labels
"""

import sys 
import os
import glob
from pathlib import Path
sys.path.append(os.path.join(Path(__file__).parent.parent.parent, "resources"))
#print(sys.path)
assert os.path.exists(sys.path[-1])

import ipa


# Core test set 24 speakers
SPEAKERS_TEST = [
    'MDAB0', 'MWBT0', 'FELC0', 'MTAS1', 'MWEW0', 'FPAS0', 'MJMP0', 'MLNT0', 'FPKT0', 
    'MLLL0', 'MTLS0', 'FJLM0', 'MBPM0', 'MKLT0', 'FNLP0', 'MCMJ0', 'MJDH0', 'FMGD0', 
    'MGRT0', 'MNJM0', 'FDHC0', 'MJLN0', 'MPAM0', 'FMLD0']


def read_phonemes(audio_file):
    """
    Given an audio file path, return its labels.
    Args:
        audio_file (string): Audio file path
        label_type (string): Default (phones), continuous features, or binary features
    Returns:
        phonemes (string): A sequence of phonemes for this audio file
    """
    phn_file = audio_file[:-8] + '.PHN'
    with open(phn_file) as f:
        phonemes = f.readlines()
    phon_seq = [p.strip().split()[-1] for p in phonemes]
    phonemes = " ".join(phon_seq)
    ipa_seq = " ".join([ipa.arpadict.get(p) for p in phon_seq])
    return phonemes, ipa_seq


def process_dataset(root, timit_dir):
    """
    List audio files and transcripts for a certain partition of TIMIT dataset.
    Args:
        root (string): Directory of TIMIT.
        split (string): Which of the subset of data to take. One of 'train', 'dev' or 'test'.
    """
    print(root)
    print(timit_dir)

    relativize = lambda file_path: file_path.replace(root, "").strip("/")

    for split in ["train", "test", "dev"]:
        if split == "train":
            audio_files = glob.glob(os.path.join(
                root, timit_dir, "data/TRAIN/**/*.WAV.wav"), recursive=True)
        else:
            audio_files = glob.glob(os.path.join(
                root, timit_dir, "data/TEST/**/*.WAV.wav"), recursive=True)
            if split == 'dev':
                audio_files = [p for p in audio_files if p.split(
                    '/')[-2] not in SPEAKERS_TEST]
            else:
                audio_files = [p for p in audio_files if p.split(
                    '/')[-2] in SPEAKERS_TEST]
        # Remove all 'SA' records.
        audio_files = [p for p in audio_files if 'SA' not in os.path.basename(p)]
        transcripts = [read_phonemes(p) for p in audio_files]
        #print(audio_files[:10])
        #print(transcripts[:10])

        fname = f"resources/datalists/timit_{split.upper()}.csv"
        with open(fname, 'w') as f:
            f.write(f"audio,phonemes,ipa\n")
            for (x, (y, z)) in zip(audio_files, transcripts):
                relpath = relativize(x)
                f.write(f"{relpath},{y},{z}\n")
        print(f"{fname} is created.")
