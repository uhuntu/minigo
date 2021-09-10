rm estimator_working_dir/ -rfv

rm outputs/ -rfv

python3 bootstrap.py \
  --work_dir=estimator_working_dir \
  --export_path=outputs/models/000000-hunt

