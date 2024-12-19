#!/bin/bash --login
#$ -cwd
#$ -l nvidia_a100=1

python Layer/Infer.py \
--llm gpt-4-1106-preview \
--taxo_name GDS \
--numofExamples 10 \
--run True \
--save_path_model_response ./results/taxo_ChainofLayers_filter_zero/ \
--demo_path ./Layer/demos/demo_gen/ \
--ChainofLayers True \
--iteratively False \
--filter_mode lm_score_ensemble \
--filter_model scibert_scivocab_uncased \
--filter_topk 15