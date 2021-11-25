#!/bin/bash

for i in {1..5}
do
  seed=$((1 + $RANDOM % 100))
  python3 train.py --algo $1 --env $2 --seed $seed
done