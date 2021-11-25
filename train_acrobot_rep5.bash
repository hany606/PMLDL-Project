#!/bin/bash

for i in {1..5}
do
  seed=$((1 + $RANDOM % 100))
  python3 train.py --algo $1 --env acrobot --seed $seed
done