while true; do

MINIGO_MODELS=outputs/models

PREV_MODEL=$(ls -d $MINIGO_MODELS/* | tail -1 | cut -f 1 -d '.')

echo "Previous model" $PREV_MODEL

LAST_MODEL=$(ls -d $MINIGO_MODELS/* | tail -1 | cut -f 3 -d '/' | cut -f 1 -d '-')

NEXT_MODEL=$MINIGO_MODELS/$(printf "%06d\n" $((10#$LAST_MODEL + 1)))-hunt

echo "The next model" $NEXT_MODEL

echo "Selfplay................................."
python3 selfplay.py \
  --load_file=$PREV_MODEL \
  --num_readouts 10 \
  --verbose 3 \
  --selfplay_dir=outputs/data/selfplay \
  --holdout_dir=outputs/data/holdout \
  --sgf_dir=outputs/sgf #> /dev/null 2>&1

echo "Training................................."
python3 train.py \
  outputs/data/selfplay/* \
  --work_dir=estimator_working_dir \
  --export_path=$NEXT_MODEL #> /dev/null 2>&1

echo "Sleeping................................."
sleep 10

done
