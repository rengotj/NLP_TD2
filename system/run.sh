#!/bin/sh

echo "Use prepare_data : $1"
echo "Use extract pcfg : $2"
echo "Use extract OOV : $3"
echo "Use extract OOV / reversal : $4"
echo "Use extract OOV / proba : $5"
echo "Use extract OOV / proba / context : $6"
echo "Use extract eval : $7"
echo "Current file path : $8"
echo "Python path : $9"

export PATH="$PATH:$9"
python evaluate.py
#python main.py $1 $2 $3 $4 $5 $6 $7 $8

echo "done"