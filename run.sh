#!/bin/sh
python train.py --mnist ./mnist.pkl.gz --opt adam --batch_size 20 --num_hidden 1 --sizes 300 --lr 0.03 --loss ce --activation sigmoid --anneal true

