# interactive. claudiaa.

srun --gres=gpu:t4:1 --time=12:00:00 --pty singularity shell --nv ~/pytorch_23.10-py3.sif 

1. generate info from log files.
```
python evaluations/01_read_eval_logs.py evaluations/eval_logs.csv
```
output in `eval_logs/`

2. check if any results are missing for correctors

```
python evaluations/02_check_logs.py

 python evaluations/02_check_logs.py eval_logs/monolingual/
```

3. collect results
```
# for multilingual
python evaluations/03_collect_eval_results.py

python evaluations/03_collect_eval_results.py monolingual results/mt5_mono/
```


4. json2df

```
python -m evaluations.04_json2df results/mt5_mono/
```
output `results/mt5_me5/`