#!/bin/bash

python infer.py \
  --openai_key your_openai_key \
  --taxo_name WDC \
  --numofExamples 5 \
  --run True \
  --save_path_model_response ./results/taxo_ChainofLayers_filter_zero/ \
  --demo_path ./demos/demo_gen/ \
  --ChainofLayers True \
  --iteratively True \
  --filter_mode lm_score_ensemble \
  --filter_model scibert_scivocab_uncased \
  --filter_topk 10