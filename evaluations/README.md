
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
python evaluations/04_json2df.py

```