###########################################################
################## EVALUATION SCRIPT ######################
###########################################################


import functools
import json
import os
import plac

from vec2text import analyze_utils


def eval_function(trainer, dataset, filepath):
    if not os.path.exists(filepath):
        try:
            metrics_results = trainer.evaluate(eval_dataset=dataset)
            print(f"saving results to {filepath}")
            with open(filepath, 'w') as f:
                json.dump(metrics_results, f)
        except Exception as msg:
            print(f"{msg}, eval did not finish")

    else:
        print(f"{filepath} already exists")


def eval_and_save_results(trainer, dataset, dataset_name, output_dir, corrector=False):
    if corrector:
        for correction_step in [1, 20, 50]:
            if correction_step == 1:
                trainer.args.per_device_eval_batch_size = 4
            else:
                trainer.args.per_device_eval_batch_size = 2
            trainer.num_gen_recursive_steps = correction_step
            print(f"evaluating corrector with steps {correction_step}")
            filepath = os.path.join(output_dir, f"{dataset_name}_steps-{correction_step}.json")
            eval_function(trainer, dataset, filepath)

        for sbeam in [4, 8]:
            trainer.args.per_device_eval_batch_size = 2
            trainer.num_gen_recursive_steps = 50
            trainer.sequence_beam_width = sbeam
            # beam_width with only 50 steps.
            print(f"evaluating corrector with beam width {sbeam}")
            filepath = os.path.join(output_dir, f"{dataset_name}_steps-50_sbeam-{sbeam}.json")
            eval_function(trainer, dataset, filepath)

    else:
        trainer.args.per_device_eval_batch_size = 8
        filepath = os.path.join(output_dir, f"{dataset_name}_inversion_base.json")
        eval_function(trainer, dataset, filepath)


def eval_one_model(model_name):
    print(f"loading experiment and trainer from {model_name}")
    experiment, trainer = analyze_utils.load_experiment_and_trainer_from_pretrained(model_name, use_less_data=3000)

    trainer.args.max_eval_samples = 500
    print(f"data arguments for experiment: {experiment.data_args}")

    # change the max_eval_samples
    # experiment.data_args["max_eval_samples"] = 1000
    if "corrector" in model_name:
        trainer.model.call_embedding_model = trainer.inversion_trainer.model.call_embedding_model
        trainer.model.tokenizer = trainer.inversion_trainer.model.tokenizer
        trainer.model.embedder_tokenizer = trainer.inversion_trainer.model.embedder_tokenizer
        trainer.model.embedder = trainer.inversion_trainer.embedder

    # get models parameters number
    n_params = sum({p.data_ptr(): p.numel() for p in trainer.model.parameters()}.values())
    print(f"model {model_name} parameters {n_params}")

    # directly load the val_datasets for all languages
    val_datasets = experiment._load_val_datasets_uncached(
        model=trainer.model,
        tokenizer=trainer.tokenizer,
        embedder_tokenizer=trainer.embedder_tokenizer
    )

    output_dir = trainer.model.config.output_dir
    output_dir = os.path.join(output_dir, "evaluations")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"output dir {output_dir}")

    # 20 languages.
    for name, val_dataset in val_datasets.items():
        print(f"evaluating {name} val_dataset")
        if "corrector" in model_name:
            print("evaluating corrector ")
            eval_and_save_results(trainer, dataset=val_dataset, dataset_name=name, output_dir=output_dir,
                corrector=True)
        else:
            print("evaluating inversion base model")
            eval_and_save_results(trainer, dataset=val_dataset, dataset_name=name, output_dir=output_dir,
                corrector=False)


if __name__ == '__main__':
    import plac
    plac.call(eval_one_model)

