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
elif [[ $CURRENT == /mount/studenten/* ]]
then
    echo "Running on IMS server."
fi

python tests/test_models.py configs/binfeat_arcticl2.yaml --gpu_id 0
