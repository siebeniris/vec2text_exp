import os
import json
from collections import defaultdict


def read_one_file(filepath, metric="eval_bleu_score"):
    """
    Read the result of the metric
    """
    with open(filepath, 'r') as f:
        eval_results = json.load(f)
    if metric in eval_results:
        return eval_results[metric]
    else:
        return None


def read_results_files(lingual="multilingual", outputfolder="results/mt5_me5"):
    # eval_bleu_score
    # eval_exact_match
    # eval_emb_cos_sim
    # eval_rouge_score
    metrics = ["eval_bleu_score", "eval_exact_match", "eval_emb_cos_sim", "eval_rouge_score"]

    for metric in metrics:
        print(f"extracting metric results {metric}")
        results_model_dict = defaultdict(dict)

        if not os.path.exists(outputfolder):
            os.makedirs(outputfolder)
        # {model:{deu_latn:{"base":x1, "step1":x2, "50+sbeam8":x3}...}...}
        for file in os.listdir(f"eval_logs/{lingual}/"):
            if file.endswith(".json"):
                filepath = os.path.join(f"eval_logs/{lingual}/", file)
                # read the collection of log files.
                with open(filepath, 'r') as f:
                    eval_files = json.load(f)
                model = eval_files["model"]
                if model not in results_model_dict:
                    results_model_dict[model] = defaultdict(dict)

                # reading inverter
                if "inverter" in file:
                    for eval in eval_files["evaluations"]:
                        # eval_dataset
                        eval_dataset = eval["dataset"]
                        if eval_dataset not in results_model_dict[model]:
                            results_model_dict[model][eval_dataset] = dict()

                        try:

                            result_file = eval["results_file"]
                            result_file = result_file.replace("./", "")
                            with open(result_file) as f:
                                results_dataset = json.load(f)
                            metric_result = results_dataset[metric]
                            results_model_dict[model][eval_dataset]["Base"] = metric_result
                        except Exception as msg:
                            print(f'{filepath} ==> {msg}')

                else:
                    for eval_dataset, eval_steps in eval_files["evaluations"].items():
                        # eval_dataset: {eval_step: results}
                        # eval_dataset = eval_dataset["dataset"]
                        if eval_dataset not in results_model_dict[model]:
                            results_model_dict[model][eval_dataset] = dict()

                        for step, eval_files in eval_steps.items():
                            if "steps 1" in step:

                                result_filepath = eval_files.get("results_files", None)
                                if result_filepath and "Step1" not in results_model_dict[model][eval_dataset]:
                                    result_filepath = result_filepath.replace("./", "")
                                    with open(result_filepath) as f:
                                        results = json.load(f)
                                    metric_result = results[metric]
                                    results_model_dict[model][eval_dataset]["Step1"] = metric_result

                            if "beam width 8" in step:
                                result_filepath = eval_files.get("results_files", None)
                                if result_filepath and "Step50_sbeam8" not in results_model_dict[model][eval_dataset]:
                                    result_filepath = result_filepath.replace("./", "")
                                    with open(result_filepath) as f:
                                        results = json.load(f)
                                    metric_result = results[metric]
                                    results_model_dict[model][eval_dataset]["Step50_sbeam8"] = metric_result

        outputfile = os.path.join(outputfolder, f"{lingual}_{metric}.json")
        print(f"saving to {outputfile}")
        with open(outputfile, "w") as f:
            json.dump(results_model_dict, f)


if __name__ == '__main__':
    import plac

    plac.call(read_results_files)
