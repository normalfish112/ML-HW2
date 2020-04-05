#!/bin/bash

preprocessed training feature=$3
preprocessed testing feature (X_test)=$5
output path (prediction.csv)=$6

python hw2-logistic.py $3 $5 $6