#!/bin/bash

# make sure we are in the right directory
CURRENT=$(pwd)
if [[ $CURRENT == *scripts ]]
then
    cd ..;
    CURRENT=$(pwd)
fi
echo $CURRENT

# script to train the model on on binary features
python train.py configs/default.yaml