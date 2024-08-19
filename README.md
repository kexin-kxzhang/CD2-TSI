# CD2-TSI

## File Structure
- Data:
  - Air Quality dataset: Beijing as source domain, Tianjin as target domain
  - Electricity dataset: Etth1 as source domain, Etth2 as target domain
- Code: Code of our implementation

## Experiments

### training and imputation for the Air Quality dataset
Run <pre style="background: #f0f0f0; display: inline-block;">bash air_run_da.sh</pre>
<pre style="background: #f0f0f0; padding: 10px;">
nohup python main_domain_adaptation.py \
  --dataset air_quality \
  --lambda_T 1e-1 \
  --lambda_C 45e-1 \
  --freq_interpolation 1 --ratio 2e-1 \
  --nsample 100 --testmissingratio 1e-1 \
  --device cuda:0 \
  --targetstrategy random > ./logs/air_quality/CD2_TSI/targetstrategy_random-mr_1e-1-transfer_1e-1-consistency_45e-1-freq_interpolation-3e-3_2e-1.log 2>&1 &
wait
</pre>

### training and imputation for the Electricity dataset
Run <pre style="background: #f0f0f0; display: inline-block;">bash etth_run_da.sh</pre>
<pre style="background: #f0f0f0; padding: 10px;">
nohup python main_domain_adaptation.py \
  --dataset electricity \
  --miu_using_mse 5 \
  --miu_using_multi_domain 1e-1 \
  --freq_interpolation 1 --ratio 4e-1 \
  --nsample 100 --seed 1 \
  --targetstrategy block \
  --missing_pattern block > ./logs/electricity/CD2_TSI/targetstrategy_block-missing_pattern_block-transfer_1e-1-consistency_5-freq_interpolation-3e-3-4e-1.log 2>&1 &
wait
</pre>
