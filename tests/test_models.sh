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

echo "\n*** Timit, phone features, stack 3 ***\n"
python tests/test_models.py configs/default.yaml --gpu_id 0
echo "\n*** Arctic L2, binary features, no stacking ***\n"
python tests/test_models.py configs/binfeat_arcticl2_stack1.yaml --gpu_id 0
echo "\n*** Arctic L2, continuous features, stack 2 ***\n"
python tests/test_models.py configs/contfeat_arcticl2_stack2.yaml --gpu_id 0
echo "\n*** Arctic L2, binary features, stack 3 ***\n"
python tests/test_models.py configs/binfeat_arcticl2_stack3.yaml --gpu_id 0
echo "\n*** Arctic L2, continuous features, stack 4 ***\n"
python tests/test_models.py configs/contfeat_arcticl2_stack4.yaml --gpu_id 0
