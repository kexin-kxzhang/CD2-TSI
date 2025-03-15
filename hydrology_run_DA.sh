#!/bin/bash

dataset="hydrology"
miu_align="45e-1"
taul="2e-1"
tauh="7e-1"
freq_interpolation="1"
mixup_lambda="2e-1"
nsample="100"
device="cuda:0"
train_missing_pattern="block"
eval_missing_pattern="block"

nohup python main_domain_adaptation.py \
  --dataset ${dataset} \
  --miu_align ${miu_align} \
  --taul ${taul} \
  --tauh ${tauh} \
  --freq_interpolation ${freq_interpolation} \
  --mixup_lambda ${mixup_lambda} \
  --nsample ${nsample} \
  --device ${device} \
  --train_missing_pattern ${train_missing_pattern} \
  --eval_missing_pattern ${eval_missing_pattern} > ./logs/${dataset}/CD2-TSI-train_missing_pattern_${train_missing_pattern}-eval_missing_pattern_${eval_missing_pattern}-miu_align_${miu_align}-taul_${taul}-tauh_${tauh}-freq_interpolation_${freq_interpolation}-mixup_lambda_${mixup_lambda}.log 2>&1 &
wait