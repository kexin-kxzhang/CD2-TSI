# Cross-Domain Conditional Diffusion Models for Time Series Imputation (ECML-PKDD 2025)

This repository contains the official implementation of our ECML-PKDD 2025 paper CD2-TSI:
📘: Cross-Domain Conditional Diffusion Models for Time Series Imputation.

Here, take Hydrology dataset as an example:
## File Structure

- Data:
  - Hydrology dataset: Discharge (source domain) and Pooled (target domain)
- Code: Implementation code for our approach

## Experiments

### training and imputation for the Air Quality dataset
Run <pre style="background: #f0f0f0; display: inline-block;">bash hydrology_run_DA.sh</pre>
<pre style="background: #f0f0f0; padding: 10px;">
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
</pre>
