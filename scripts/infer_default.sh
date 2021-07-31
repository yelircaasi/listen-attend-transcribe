#!/usr/bin/zsh

# make sure we are in the right directory
CURRENT=$(pwd)
if [[ $CURRENT == *scripts ]]
then
    cd ..;
    CURRENT=$(pwd)
fi
echo $CURRENT

# script to perform inference on the default model (timit, standard phone set)

