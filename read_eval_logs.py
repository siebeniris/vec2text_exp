import re
import json
import os


# Log data as a string (assuming you have it stored in a variable or a file)


def get_log_info_eval_inverter(filepath):
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
                        current_preds = []
                        current_trues = []
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
    parsed_json_path = 'parsed_log_data.json'
    with open(parsed_json_path, 'w') as json_file:
        json.dump(parsed_data, json_file, indent=4)

    print(f"Parsed data saved to {parsed_json_path}")


def get_log_info_eval_corrector(log_file_path):
    # Define a pattern to match each type of information
    patterns={
    'output_dir': re.compile(r'output dir (.+)'),
    'model_params': re.compile(r'model (.+) parameters (\d+)'),
    'evaluating_corrector': re.compile(r'evaluating (\w+_\w+) val_dataset with steps (\d+)'),
    'beam_width': re.compile(r'beam width (\d+)'),
    'output_file': re.compile(r'outptufile for decoded sequences: (.+)'),
    'saved_embeddings': re.compile(r'saving embeddings for preds and labels to (.+)'),
    'results': re.compile(r'saving results to (.+)'),
    'cuda_error': re.compile(r'CUDA out of memory')}

    # Container to hold the parsed data
    parsed_data = {
    'output_dir': None,
    'model': None,
    'parameters': None,
    'evaluations': []
    }

    try:
        # Read the log data from the file
        with open(log_file_path, 'r') as file:
            log_data = file.read()

        # Split the log into lines
        log_lines = log_data.strip().split('\n')

        # Temporary variables to hold intermediate data
        current_eval = {}

        cuda_error_detected = False

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
                    elif key == 'evaluating_corrector':
                        # If there's already an evaluation in progress, save it
                        if current_eval:

                            parsed_data['evaluations'].append(current_eval)
                            current_eval = {}
                        current_eval['dataset'] = match.group(1)
                        current_eval['steps'] = match.group(2)
                        cuda_error_detected = False
                    elif key == 'beam_width':
                        current_eval['beam_width'] = match.group(1)
                    elif key == 'output_file' and not cuda_error_detected:
                        current_eval['output_file'] = match.group(1)
                    elif key == 'saved_embeddings':
                        current_eval['embeddings_file'] = match.group(1)
                    elif key == 'results':
                        current_eval['results_file'] = match.group(1)
                    elif key == 'cuda_error':
                        cuda_error_detected = True
                    break

        # Save the last evaluation in progress
        if current_eval:

            parsed_data['evaluations'].append(current_eval)

        # Save the parsed data to a JSON file for further use
        parsed_json_path = 'parsed_log_data_corrector.json'
        with open(parsed_json_path, 'w') as json_file:
            json.dump(parsed_data, json_file, indent=4)


    except Exception as e:
        # Handle exceptions silently
        pass


def main(filepath, model_type):
    if model_type == "inverter":
        get_log_info_eval_inverter(filepath)
    elif model_type == "corrector":
        get_log_info_eval_corrector(filepath)
    else:
        print("choose model type between inverter and corrector")


if __name__ == '__main__':
    import plac

    plac.call(main)
