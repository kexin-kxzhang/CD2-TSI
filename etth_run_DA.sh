nohup python main_domain_adaptation.py \
  --dataset electricity \
  --miu_using_mse 5 \
  --miu_using_multi_domain 1e-1 \
  --freq_interpolation 1 --ratio 4e-1 \
  --nsample 100 --seed 1 \
  --targetstrategy block \
  --missing_pattern block > ./logs/electricity/CD2_TSI/targetstrategy_block-missing_pattern_block-transfer_1e-1-consistency_5-freq_interpolation-3e-3-4e-1.log 2>&1 &
wait