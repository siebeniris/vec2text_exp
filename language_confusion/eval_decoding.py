import os.path

import pandas as pd
import plac
import scipy.stats
import collections
import evaluate
import numpy as np
from typing import List, Union

from lingua import Language, LanguageDetectorBuilder
import jieba
import nltk
import hebrew_tokenizer
from fugashi import Tagger

japaneseTagger = Tagger("-Owakati")
tk = nltk.WordPunctTokenizer()

from kiwipiepy import Kiwi

kiwi = Kiwi()

# NO YIDDISH, AMHARIC, SIN
# NO MALTESE (TRAIN)
languages = [Language.ENGLISH, Language.GERMAN, Language.HEBREW, Language.ARABIC,
             Language.HINDI, Language.GUJARATI, Language.PUNJABI, Language.URDU,
             Language.TURKISH, Language.KAZAKH,
             Language.CHINESE, Language.MONGOLIAN, Language.KOREAN, Language.JAPANESE,
             Language.HUNGARIAN, Language.FINNISH]
detector = LanguageDetectorBuilder.from_languages(*languages).build()

print(f"{len(languages)} languages")


def tokenize_one_text_by_language(language, text):
    if language == Language.CHINESE:
        return list(jieba.cut(text, cut_all=True))
    elif language == Language.ENGLISH:
        return nltk.word_tokenize(text)
    elif language == Language.GERMAN:
        return nltk.word_tokenize(text, language="german")
    elif language == Language.FINNISH:
        return nltk.word_tokenize(text, language="finnish")
    elif language == Language.TURKISH:
        return nltk.word_tokenize(text, language="turkish")
    elif language == Language.HEBREW:
        # https://github.com/YontiLevin/Hebrew-Tokenizer?tab=readme-ov-file
        return [x for _, x, _, _ in hebrew_tokenizer.tokenize(text)]
    elif language == Language.ARABIC:
        return tk.tokenize(text)
    elif language == Language.HINDI:
        return tk.tokenize(text)
    elif language == Language.KAZAKH:
        return tk.tokenize(text)
    elif language == Language.MONGOLIAN:
        return tk.tokenize(text)
    elif language == Language.KOREAN:
        # https://github.com/bab2min/kiwipiepy
        return [x.form for x in kiwi.tokenize(text)]
    elif language == Language.JAPANESE:
        # https://github.com/polm/fugashi?tab=readme-ov-file
        # pip install 'fugashi[unidic]'
        # python -m unidic download
        return [x for x in japaneseTagger.parse(text)]
    else:
        # maltese is latin.
        return nltk.word_tokenize(text)


def sem(L: List[float]) -> float:
    result = scipy.stats.sem(np.array(L))
    if isinstance(result, np.ndarray):
        return result.mean().item()
    return result


def mean(L: Union[List[int], List[float]]) -> float:
    return sum(L) / len(L)


metric_bleu = evaluate.load("sacrebleu")
metric_rouge = evaluate.load("rouge")
metric_accuracy = evaluate.load("accuracy")


def count_overlapping_ngrams(s1: str, s2: str, n: int) -> int:
    ngrams_1 = nltk.ngrams(s1, n)
    ngrams_2 = nltk.ngrams(s2, n)
    ngram_counts_1 = collections.Counter(ngrams_1)
    ngram_counts_2 = collections.Counter(ngrams_2)
    total = 0
    for ngram, count in ngram_counts_1.items():
        total += min(count, ngram_counts_2[ngram])
    return total


def text_comparison_metics(predictions, references, ):
    num_preds = len(predictions)

    precision_sum = 0.0
    recall_sum = 0.0
    num_overlapping_words = []
    num_overlapping_bigrams = []
    num_overlapping_trigrams = []
    num_true_words = []
    num_pred_words = []
    f1s = []
    for i in range(num_preds):
        lang = detector.compute_language_confidence_values(references[i])[0].language
        true_words = tokenize_one_text_by_language(lang, references[i])
        pred_words = tokenize_one_text_by_language(lang, predictions[i])
        # true_words = nltk.tokenize.word_tokenize(references[i])
        # print(true_words)
        # pred_words = nltk.tokenize.word_tokenize(predictions[i])
        # print(pred_words)
        num_true_words.append(len(true_words))
        num_pred_words.append(len(pred_words))

        true_words_set = set(true_words)
        pred_words_set = set(pred_words)
        TP = len(true_words_set & pred_words_set)
        FP = len(true_words_set) - len(true_words_set & pred_words_set)
        FN = len(pred_words_set) - len(true_words_set & pred_words_set)
        precision = (TP) / (TP + FP + 1e-20)
        recall = (TP) / (TP + FN + 1e-20)

        try:
            f1 = (2 * precision * recall) / (precision + recall + 1e-20)
        except ZeroDivisionError:
            f1 = 0.0
        f1s.append(f1)

        precision_sum += precision
        recall_sum += recall

        ############################################################
        num_overlapping_words.append(
            count_overlapping_ngrams(true_words, pred_words, 1)
        )
        num_overlapping_bigrams.append(
            count_overlapping_ngrams(true_words, pred_words, 2)
        )
        num_overlapping_trigrams.append(
            count_overlapping_ngrams(true_words, pred_words, 3)
        )

    set_token_metrics = {
        "token_set_precision": (precision_sum / num_preds),
        "token_set_recall": (recall_sum / num_preds),
        "token_set_f1": mean(f1s),
        "token_set_f1_sem": sem(f1s),
        "n_ngrams_match_1": mean(num_overlapping_words),
        "n_ngrams_match_2": mean(num_overlapping_bigrams),
        "n_ngrams_match_3": mean(num_overlapping_trigrams),
        "num_true_words": mean(num_true_words),
        "num_pred_words": mean(num_pred_words),
    }

    bleu_results = np.array(
        [
            metric_bleu.compute(predictions=[p], references=[r])["score"]
            for p, r in zip(predictions, references)
        ]
    )

    rouge_result = metric_rouge.compute(
        predictions=predictions, references=references
    )

    bleu_results = (bleu_results.tolist())

    exact_matches = np.array(predictions) == np.array(references)

    gen_metrics = {
        "bleu_score": np.mean(bleu_results),
        "bleu_score_sem": sem(bleu_results),
        "rouge_score": rouge_result[
            "rouge1"
        ],  # ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        # "bert_score": statistics.fmean(bertscore_result["f1"]),
        "exact_match": mean(exact_matches),
        "exact_match_sem": sem(exact_matches),
    }

    all_metrics = {**set_token_metrics, **gen_metrics}
    # return bleu_results_list, exact_matches, f1s
    return bleu_results, exact_matches, f1s


def generating_for_sequences(filepath):
    df = pd.read_csv(filepath)
    print(df.head())

    filedir = os.path.dirname(filepath)

    preds = df["pred"].tolist()
    # translations = df["pred_translated"].str.lower().tolist()
    references = df["labels"].tolist()

    preds_bleu_results, preds_exact_matches, preds_f1s = text_comparison_metics(predictions=preds,
                                                                                references=references)
    df["pred_bleu"] = preds_bleu_results
    df["pred_exact_match"] = preds_exact_matches
    df["pred_tokens_f1"] = preds_f1s

    df.to_csv(os.path.join(filedir, "eval_sequences.csv"))


if __name__ == '__main__':
    filepath = "saves/yiyic__mt5_me5_deu_Latn_32_2layers_corrector/decoded_eval_1720486468/decoded_sequences.csv"
    generating_for_sequences(filepath)
