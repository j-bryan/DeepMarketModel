NTRIALS=128
# NPROC=$(nproc --all)
# If the nproc command is not available (e.g. on macOS), you can set NPROC manually
NPROC=8
TRIALS_PER_PROC=$((NTRIALS / NPROC))

for model_type in LR KNN RF GBT;
do
    echo "Tuning $model_type"

    for proc in $(seq 1 $NPROC);
    do
        python baseline_models.py --model-type=$model_type --n-trials=$TRIALS_PER_PROC &
    done

    wait
    echo "Finished tuning $model_type"
    echo ""
done
