"""
"""
import os
import sys
import yaml
import argparse
import time
import torch
sys.path.append(os.path.join(sys.path[0].rsplit("/", 1)[0], "src/models"))
sys.path.append(os.path.join(sys.path[0].rsplit("/", 1)[0], "resources"))
#sys.path.append(os.path.join(sys.path[0].rsplit("/", 1)[0], "resources/featurizers"))
sys.path.append(os.path.join(sys.path[0].rsplit("/", 1)[0], "src/data_prep"))
import featurizers
import seq2seq

def main():
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument('cfg', type=str, help="Specify which experiment config file to use.")
    parser.add_argument('--gpu_id', default=0, type=int, help="CUDA visible GPU ID. Currently only support single GPU.")
    args = parser.parse_args()
    
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    featurizer = torch.load(f"resources/featurizers/featurizer_{cfg['dataset']}_{cfg['output_type']}.pth")

    model = seq2seq.Seq2Seq(len(featurizer.vocab),
                                hidden_size=cfg['model']['hidden_size'],
                                encoder_layers=cfg['model']['encoder_layers'],
                                decoder_layers=cfg['model']['decoder_layers'],
                                output_type=cfg["output_type"],
                                drop_p=cfg['model']['drop_p'])
    if torch.cuda.is_available():
        model = model.cuda()
    print(model)


if __name__ == '__main__':
    main()