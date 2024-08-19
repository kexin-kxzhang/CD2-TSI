nohup python main_domain_adaptation.py \
  --dataset air_quality \
  --lambda_T 1e-1 \
  --lambda_C 45e-1 \
  --freq_interpolation 1 --ratio 2e-1 \
  --nsample 100 --testmissingratio 1e-1 \
  --device cuda:0 \
  --targetstrategy random > ./logs/air_quality/CD2_TSI/targetstrategy_random-mr_1e-1-transfer_1e-1-consistency_45e-1-freq_interpolation-3e-3_2e-1.log 2>&1 &
wait