import json
import os


### check the extracted information from logs. whether results are missing.

def check_logs(folderpath="eval_logs/multilingual"):
    for file in os.listdir(folderpath):

        if file.endswith(".json"):
            print(file)
            if "corrector" in file:
                filepath = os.path.join(folderpath, file)
                # print(f"reading {filepath}")
                with open(filepath) as f:
                    evals = json.load(f)
                model = evals["model"]
                print(f"model {model}")
                if "semitic-fami" not in model:

                    for eval_lang, eval in evals["evaluations"].items():
                        # deu_latn, {"steps 1":.....}
                        sbeam8 = False
                        for step, eval_step in eval.items():

                            if step == "steps 1":
                                if "output_files" not in eval_step:
                                    print(f"eval {eval_lang} step 1 missing")
                            if step in ["beam width 8", "steps 50 and beam width 8"]:
                                if "output_files" in eval_step:
                                    sbeam8 = True
                        if not sbeam8:
                            print(f"eval {eval_lang} step 50 and beam width 8 missing")


            elif "inverter" in file:
                filepath = os.path.join(folderpath, file)
                # print(f"reading {filepath}")
                with open(filepath) as f:
                    evals = json.load(f)
                model = evals["model"]
                print(f"model {model}")
                for eval in evals["evaluations"]:
                    dataset = eval["dataset"]
                    if "output_file" not in eval:
                        print(f"eval {dataset} inverter eval missing")

            print("*" * 10)



if __name__ == '__main__':
    import plac

    plac.call(check_logs)
