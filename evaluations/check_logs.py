import json
import os


def check_logs(folderpath="eval_logs/multilingual"):
    for file in os.listdir(folderpath):
        if file.endswith(".json"):
            if "corrector" in file:
                filepath = os.path.join(folderpath, file)
                # print(f"reading {filepath}")
                with open(filepath) as f:
                    evals = json.load(f)
                model = evals["model"]

                for eval_lang, eval in evals["evaluations"].items():

                    for step, eval_step in eval.items():
                        if step == "steps 1":
                            if eval_step["output_files"] is None:
                                print(f"model {model}")
                                print(f"eval {eval_lang} step 1 missing")
                        if step in ["beam width 8", "steps 50 and beam width 8"]:
                            if eval_step["output_files"] is None:
                                print(f"model {model}")
                                print(f"eval {eval_lang} step 50 and beam width 8 missing")
                print("*" * 10)


if __name__ == '__main__':
    import plac

    plac.call(check_logs)
