#!/bin/bash --login
#$ -cwd
#$ -l nvidia_v100=1

python Layer/rank_score.py \
  --taxo_name WDC
