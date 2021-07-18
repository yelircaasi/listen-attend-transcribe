#!/bin/bash

# script to train the model on on binary features

# make sure we are in the right directory
CURRENT=$(pwd)
if [[ $CURRENT == *scripts ]]
then
    cd ..;
    CURRENT=$(pwd)
fi
echo $CURRENT

python train.py configs/default.yaml