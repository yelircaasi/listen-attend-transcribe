#!/usr/bin/zsh

# make sure we are in the right directory
CURRENT=$(pwd)
if [[ $CURRENT == *scripts ]]
then
    cd ..;
    CURRENT=$(pwd)
fi
echo "Current directory: $CURRENT"

# find whether we are on IMS server or personal machine
if [[ $CURRENT == /home/isaac/* ]]
then
    DATA_ROOT="/home/isaac/Projects/Thesis/data"
elif [[ $CURRENT == /mount/studenten/* ]]
then 
    DATA_ROOT="/mount/studenten/arbeitsdaten-studenten1/rileyic/timit/data"
else
    echo "Device not recognized; modify this script with the correct data directory."
fi

echo "Data directory: $DATA_ROOT"

# script to prepare all data sources
python ./src/data_prep/prepare.py --root $DATA_ROOT --datasets timit,arcticl2,arabicsc,buckeye --features phones,cont,bin
#python ./src/data_prep/prepare.py --root $DATA_ROOT --datasets arabicsc --features phones,cont,bin
#python ./src/data_prep/prepare.py --root $DATA_ROOT --datasets timit --features phones,cont,bin
#python ./src/data_prep/prepare.py --root $DATA_ROOT --datasets arcticl2 --features phones,cont,bin
#python ./src/data_prep/prepare.py --root $DATA_ROOT --datasets buckeye --features phones,cont,bin
