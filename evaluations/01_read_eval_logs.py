import re
import json
import os
from collections import defaultdict

import pandas as pd
import yaml


def get_log_info_eval_inverter(filepath, outputfile):
    print(f"extracting information from file {filepath}...")

    with open(filepath) as f:
        log_data = f.read()

    # Define a pattern to match each type of information
    patterns = {
        'model_params': re.compile(r'model (.+) parameters (\d+)'),
        'output_dir': re.compile(r'output dir (.+)'),
        'evaluating': re.compile(r'evaluating (\w+_\w+) val_dataset'),
        'evaluating_model': re.compile(r'evaluating inversion base model'),
        'output_file': re.compile(r'outptufile for decoded sequences: (.+)'),
        'saved_embeddings': re.compile(r'saving embeddings for preds and labels to (.+)'),
        'results': re.compile(r'saving results to (.+)')
    }

    # Container to hold the parsed data
    parsed_data = {
        'output_dir': None,
        'model': None,
        'parameters': None,
        'evaluations': []
    }

    # Split the log into lines
    log_lines = log_data.strip().split('\n')

    # Temporary variables to hold intermediate data
    current_eval = {}

    # Process each line
    for line in log_lines:
        # Match patterns
        for key, pattern in patterns.items():
            match = pattern.match(line)
            if match:
                if key == 'output_dir':
                    parsed_data['output_dir'] = match.group(1)
                elif key == 'model_params':
                    parsed_data['model'] = match.group(1)
                    parsed_data['parameters'] = match.group(2)
                elif key == 'evaluating':
                    # If there's already an evaluation in progress, save it
                    if current_eval:
                        parsed_data['evaluations'].append(current_eval)
                        current_eval = {}
                    current_eval['dataset'] = match.group(1)
                elif key == 'output_file':
                    current_eval['output_file'] = match.group(1)
                elif key == 'saved_embeddings':
                    current_eval['embeddings_file'] = match.group(1)
                elif key == 'results':
                    current_eval['results_file'] = match.group(1)
                break

    # Save the last evaluation in progress
    if current_eval:
        parsed_data['evaluations'].append(current_eval)

    # Save the parsed data to a JSON file for further use
    model_name = parsed_data["model"].replace("yiyic/", "")

    # parsed_json_path = f'eval_logs/eval_{model_name}.json'
    with open(outputfile, 'w') as json_file:
        json.dump(parsed_data, json_file, indent=4)

    print(f"Parsed data saved to {outputfile}")


def get_log_info_eval_corrector(log_file_path, outputfile):
    # Define a pattern to match each type of information
    patterns = {
        'output_dir': re.compile(r'output dir (.+)'),
        'model_params': re.compile(r'model (.+) parameters (\d+)'),
        'evaluating_corrector': re.compile(r'evaluating (\w+_\w+) val_dataset'),
        "correction_steps": re.compile(r'evaluating corrector with (.+)'),
        'output_file': re.compile(r'outptufile for decoded sequences: (.+)'),
        'saved_embeddings': re.compile(r'saving embeddings for preds and labels to (.+)'),
        'results': re.compile(r'saving results to (.+)'),
        'cuda_error': re.compile(r'CUDA out of memory'),
        'results_exist': re.compile(r'(.+) already exists')
    }

    # Container to hold the parsed data
    parsed_data = {
        'output_dir': None,
        'model': None,
        'parameters': None,
        'evaluations': defaultdict(dict)
    }

    def parse_one_file(log_file_path):
        try:
            # Read the log data from the file
            with open(log_file_path, 'r') as file:
                log_data = file.read()

            # Split the log into lines
            log_lines = log_data.strip().split('\n')

            # Temporary variables to hold intermediate data
            # {"1": {"output_files":xx, "embeddings_files":xxx, "results_files":xxx}}
            dataset = None
            correction_step = None

            for line in log_lines:
                # Match patterns
                for key, pattern in patterns.items():
                    match = pattern.match(line)
                    if match:
                        if key == 'output_dir':
                            # Experiment output_dir = saves/yiyic__mt5_me5_cmn_Hani_32_2layers_inverter
                            parsed_data['output_dir'] = match.group(1)
                        if key == 'model_params':
                            # model yiyic/mt5_me5_cmn_Hani_32_2layers_inverter parameters 870484992e
                            parsed_data['model'] = match.group(1)
                            parsed_data['parameters'] = match.group(2)

                        if key == "evaluating_corrector":
                            # evaluating deu_Latn val_dataset
                            # current dataset.
                            dataset = match.group(1)
                        elif key == 'correction_steps':
                            correction_step = match.group(1)
                            if correction_step not in parsed_data['evaluations'][dataset]:
                                parsed_data['evaluations'][dataset][correction_step] = dict()

                        elif key == 'output_file':
                            parsed_data['evaluations'][dataset][correction_step]["output_files"] = match.group(1)
                        elif key == 'saved_embeddings':
                            parsed_data['evaluations'][dataset][correction_step]["embeddings_files"] = match.group(1)
                        elif key == 'results':
                            parsed_data['evaluations'][dataset][correction_step]["results_files"] = match.group(1)
                        break

        except Exception as e:
            print(f"Exception {e}")

    if len(log_file_path) > 1:
        # a list of log file ids.
        # the later files come first
        logfile_names = sorted([int(x) for x in log_file_path], reverse=True)
        # file_paths = [f"eval_logs/eval_{x}.out" for x in logfile_names]
        file_paths = [f"eval_{x}.out" for x in logfile_names]
        for filepath in file_paths:
            print(filepath)
            if os.path.exists(filepath):
                parse_one_file(filepath)
    if len(log_file_path) == 1:
        # parse_one_file(f"eval_logs/eval_{log_file_path[0]}.out")
        parse_one_file(f"eval_{log_file_path[0]}.out")

        # Save the parsed data to a JSON file for further use
    if parsed_data["model"]:
        with open(outputfile, 'w') as json_file:
            json.dump(parsed_data, json_file, indent=4)


def read_yamlfile(filepath):
    with open(filepath, 'r') as yamlfile:
        try:
            eval_logs = yaml.safe_load(yamlfile)
            return eval_logs
        except yaml.YAMLError as exc:
            print(exc)
            return False


def main(eval_logfile="evaluations/eval_logs.yaml"):
    # file path can be multiples.
    output_folder = "eval_logs"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if eval_logfile.endswith(".yaml"):
        eval_logs = read_yamlfile(eval_logfile)
        if eval_logs:
            for l_, models_fileid in eval_logs.items():
                outputfolder = os.path.join(output_folder, l_)
                if not os.path.exists(outputfolder):
                    os.makedirs(outputfolder)
                for model_name, eval_file_id in models_fileid.items():
                    if "inverter" in model_name:
                        print(f"processing inverter {model_name} from {eval_file_id}")
                        # the list for inverter is only one.
                        assert len(eval_file_id) == 1
                        # logfilepath = f"eval_logs/eval_{eval_file_id[0]}.out"
                        logfilepath = f"eval_{eval_file_id[0]}.out"
                        outputfile = f"{outputfolder}/eval_{model_name}.json"
                        get_log_info_eval_inverter(logfilepath, outputfile)
                    if "corrector" in model_name:
                        assert len(eval_file_id) >= 1
                        print(f"processing corrector {model_name} from {eval_file_id}")
                        outputfile = f"{outputfolder}/eval_{model_name}.json"
                        get_log_info_eval_corrector(eval_file_id, outputfile)

    elif eval_logfile.endswith(".csv"):
        eval_logs = pd.read_csv("evaluations/eval_logs.csv")

        model2logs = dict(zip(eval_logs["Model"], eval_logs["JobID"]))
        for model_name, eval_file_id in model2logs.items():
            if "_me5_" in model_name:
                outputfolder = os.path.join(output_folder, "multilingual")
                if not os.path.exists(outputfolder):
                    os.makedirs(outputfolder)

                if "inverter" in model_name:
                    print(f"processing inverter {model_name} from {eval_file_id}")
                    # the list for inverter is only one.
                    assert len(eval_file_id) == 1
                    # logfilepath = f"eval_logs/eval_{eval_file_id[0]}.out"
                    logfilepath = f"eval_{eval_file_id[0]}.out"
                    outputfile = f"{outputfolder}/eval_{model_name}.json"
                    get_log_info_eval_inverter(logfilepath, outputfile)
                if "corrector" in model_name:
                    assert len(eval_file_id) >= 1
                    print(f"processing corrector {model_name} from {eval_file_id}")
                    outputfile = f"{outputfolder}/eval_{model_name}.json"
                    get_log_info_eval_corrector(eval_file_id, outputfile)



    else:
        print("choose eval log file between csv and yaml.")


if __name__ == '__main__':
    import plac

    plac.call(main)
