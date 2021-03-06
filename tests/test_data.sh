#!/usr/bin/zsh

# make sure we are in the right directory
CURRENT=$(pwd)
if [[ $CURRENT == *scripts ]]
then
    cd ..;
    CURRENT=$(pwd)
fi
echo $CURRENT

if [[ $CURRENT == /home/isaac/* ]]
then
    echo "Running on local machine."
    python tests/test_data.py /home/isaac/Projects/Thesis/data
elif [[ $CURRENT == /mount/studenten/* ]]
then
    echo "Running on IMS server."
    python test/test_data.py
fi