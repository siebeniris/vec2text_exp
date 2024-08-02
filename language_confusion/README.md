# Post evaluation and Language Confusion


## Tokenization and evaluate sequence token metrics and language detection


```
python language_confusion/post_eval.py multilingual inverter

python language_confusion/post_eval.py multilingual corrector
```


## Collect token set evaluations

```

python -m language_confusion.collect_token_set_eval multilingual token_set_f1

python -m language_confusion.collect_token_set_eval multilingual num_true_words

python -m language_confusion.collect_token_set_eval multilingual num_pred_words

```

## Get Dataset2language dictionary

```
python language_confusion/get_dataset2langdist.py
```

