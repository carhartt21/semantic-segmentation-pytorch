#!/bin/sh

for i in $(seq 100 200)
do
    echo "i: $i"
    python3 eval.py --cfg config/outside30k-hrnetv2-pyramid.yaml VAL.checkpoint "epoch_$i.pth" 
done
