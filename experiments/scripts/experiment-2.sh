#!/bin/sh
save=$1
model=$2
m=$3
batch=20
seed=1
gamma_tr=0.4
gamma_gu=0.0

export PYTHONPATH="."

run=0
for n in 64 128 256 512 1024 2048 4096
do
    run=$((run+1)); echo run $run
    python experiments/c4-experiment.py --save $save/m$m-run-$run-trst.p --n $n --batch_size $batch --m $m --model $model --seed $seed
    python experiments/c4-experiment.py --save $save/m$m-run-$run-gust.p --n $n --method gumbel --batch_size $batch  --m $m --model $model --seed $seed
    python experiments/c4-experiment.py --save $save/m$m-run-$run-tred.p --n $n --batch_size $batch --m $m --model $model --seed $seed --edit --gamma $gamma_tr
    python experiments/c4-experiment.py --save $save/m$m-run-$run-gued.p --n $n --method gumbel --batch_size $batch  --m $m --model $model --seed $seed --edit --gamma $gamma_gu
done