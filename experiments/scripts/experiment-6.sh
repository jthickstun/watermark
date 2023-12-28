#!/bin/sh
save=$1
model=$2
lang=$3
batch=20
gamma_tr=0.4
gamma_gu=0.0
n=256
buffer=100
seed=1

export PYTHONPATH="."

run=0
for m in 10 15 20 25 30 40 50
do
    run=$((run+1)); echo run $run
    python experiments/c4-experiment.py --save $save/$lang-run-$run-trst.p --n $n --batch_size $batch --m $m --rt_translate --model $model --language $lang --buffer_tokens $buffer --seed $seed
    python experiments/c4-experiment.py --save $save/$lang-run-$run-gust.p --n $n --method gumbel --batch_size $batch --m $m --rt_translate --model $model --language $lang --buffer_tokens $buffer --seed $seed
    python experiments/c4-experiment.py --save $save/$lang-run-$run-tred.p --n $n --gamma $gamma_tr --edit --batch_size $batch --m $m --rt_translate --model $model --language $lang --buffer_tokens $buffer --seed $seed
    python experiments/c4-experiment.py --save $save/$lang-run-$run-gued.p --n $n --gamma $gamma_gu --edit --method gumbel --batch_size $batch --m $m --rt_translate --model $model --language $lang --buffer_tokens $buffer --seed $seed
    python experiments/c4-experiment.py --save $save/$lang-run-$run-ki10.p --method kirchenbauer --batch_size $batch --m $m --kirch_delta 1.0 --rt_translate --model $model --language $lang --buffer_tokens $buffer --seed $seed
    python experiments/c4-experiment.py --save $save/$lang-run-$run-ki15.p --method kirchenbauer --batch_size $batch --m $m --kirch_delta 1.5 --rt_translate --model $model --language $lang --buffer_tokens $buffer --seed $seed
    python experiments/c4-experiment.py --save $save/$lang-run-$run-ki20.p --method kirchenbauer --batch_size $batch --m $m --kirch_delta 2.0 --rt_translate --model $model --language $lang --buffer_tokens $buffer --seed $seed
done