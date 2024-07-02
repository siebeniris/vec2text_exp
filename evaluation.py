import functools
import json
import os
import plac

from vec2text import analyze_utils
from vec2text.utils import dataset_map_multi_worker





def eval_and_save_results(trainer, dataset, output_dir, corrector=False ):
    if corrector:
        for sbeam in [4,8]:
            for correction_step in [0, 20, 50]:
                pass


def eval_one_model(model_name, batch_size=8, lang=None):
    print(f"loading experiment and trainer from {model_name}")
    experiment, trainer = analyze_utils.load_experiment_and_trainer_from_pretrained(model_name, use_less_data=3000)
    print(experiment.data_args)
    # change the max_eval_samples
    # experiment.data_args["max_eval_samples"] = 1000
    if "corrector" in model_name:
        trainer.model.call_embedding_model = trainer.inversion_trainer.model.call_embedding_model
        trainer.model.tokenizer = trainer.inversion_trainer.model.tokenizer
        trainer.model.embedder_tokenizer = trainer.inversion_trainer.model.embedder_tokenizer
        trainer.model.embedder = trainer.inversion_trainer.embedder

    # get models parameters number
    n_params = sum({p.data_ptr(): p.numel() for p in trainer.model.parameters()}.values())
    print(f"model parameters {n_params}")

    # directly load the val_datasets for all languages
    val_datasets = experiment._load_val_datasets_uncached(
        model=trainer.model,
        tokenizer=trainer.tokenizer,
        embedder_tokenizer=trainer.embedder_tokenizer
    )





