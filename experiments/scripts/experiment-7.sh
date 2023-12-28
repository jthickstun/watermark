#!/bin/sh
save=$1
gamma_tr=0.4
gamma_gu=0.0
n=256
T=200
seed=1

export PYTHONPATH="."

run=0

run=$((run+1)); echo run $run
python experiments/instruct-experiment.py --save $save/run-$run-gued.p --n $n --gamma $gamma_gu --edit --method gumbel --T $T --seed $seed
python experiments/instruct-experiment.py --save $save/run-$run-ki10.p --method kirchenbauer --kirch_delta 1.0 --T $T --seed $seed
python experiments/instruct-experiment.py --save $save/run-$run-ki15.p --method kirchenbauer --kirch_delta 1.5 --T $T --seed $seed
python experiments/instruct-experiment.py --save $save/run-$run-ki20.p --method kirchenbauer --kirch_delta 2.0 --T $T --seed $seed

run=$((run+1)); echo run $run
python experiments/instruct-experiment.py --save $save/run-$run-gued.p --n $n --gamma $gamma_gu --edit --method gumbel --T $T --seed $seed --rt_translate --language french
python experiments/instruct-experiment.py --save $save/run-$run-ki10.p --method kirchenbauer --kirch_delta 1.0 --T $T --seed $seed --rt_translate --language french
python experiments/instruct-experiment.py --save $save/run-$run-ki15.p --method kirchenbauer --kirch_delta 1.5 --T $T --seed $seed --rt_translate --language french
python experiments/instruct-experiment.py --save $save/run-$run-ki20.p --method kirchenbauer --kirch_delta 2.0 --T $T --seed $seed --rt_translate --language french

run=$((run+1)); echo run $run
python experiments/instruct-experiment.py --save $save/run-$run-gued.p --n $n --gamma $gamma_gu --edit --method gumbel --T $T --seed $seed --rt_translate --language russian
python experiments/instruct-experiment.py --save $save/run-$run-ki10.p --method kirchenbauer --kirch_delta 1.0 --T $T --seed $seed --rt_translate --language russian
python experiments/instruct-experiment.py --save $save/run-$run-ki15.p --method kirchenbauer --kirch_delta 1.5 --T $T --seed $seed --rt_translate --language russian
python experiments/instruct-experiment.py --save $save/run-$run-ki20.p --method kirchenbauer --kirch_delta 2.0 --T $T --seed $seed --rt_translate --language russian
