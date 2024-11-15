#!/bin/sh

env="SingleCombat"
scenario="1v1/ShootMissile/HierarchicalSelfplay"
algo="ppo"
exp="v1"
seed=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=1 python train/train_jsbsim.py \
    --env-name ${env} \
    --algorithm-name ${algo} \
    --experiment-name ${exp} \
    --seed ${seed} \
    --cuda \
    --n-training-threads 1 \
    --n-rollout-threads 32 \
    --num-env-steps 1e8 \
    --use-wandb \
    --user-name "zzp" \
    --wandb-name "zzp" \
    --scenario-name ${scenario} \
    --gamma 0.99 \
    --buffer-size 3000 \
    --hidden-size "128 128" \
    --act-hidden-size "128 128" \
    --use-feature-normalization \
    --use-prior \
    --recurrent-hidden-size 128 \
    --recurrent-hidden-layers 1 \
    --data-chunk-length 8 \
    --lr 3e-4 \
    --ppo-epoch 4 \
    --clip-params 0.2 \
    --use-clip-value-loss \
    --num-mini-batch 5 \
    --entropy-coef 1e-3 \
    --max-grad-norm 2 \
    --use-selfplay \
    --selfplay-algorithm "fsp" \
    --n-choose-opponents 1 \
    --save-interval 1 \
    --log-interval 1 \
    --use-eval \
    --n-eval-rollout-threads 1 \
    --eval-interval 1 \
    --eval-episodes 1 \
    