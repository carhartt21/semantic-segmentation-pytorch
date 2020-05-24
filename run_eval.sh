#!/bin/sh

for i in $(seq 10 50)
do
    echo "i: $i"
    python3 eval.py --cfg config/outside30k-hrnetv2-ocr.yaml VAL.checkpoint "epoch_$i.pth" 
done
