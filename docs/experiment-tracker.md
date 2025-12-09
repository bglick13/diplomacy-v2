Competitive self play with ELO pressue
- `python scripts/benchmark_training.py \
  --league-training \
  --rollout-horizon-years 5 \
  --winner-threshold-sc 7 \
  --buffer-depth 3 \
  --num-groups-per-step 16 \
  --total-steps 250`