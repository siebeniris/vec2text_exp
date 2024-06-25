import hashlib
import os
import json

cwd = os.getcwd()
DATASET_CACHE_PATH = os.environ.get(
    "VEC2TEXT_CACHE", os.path.expanduser(f"{cwd}/.cache/inversion")
)

def md5_hash_kwargs(**kwargs) -> str:
    # We ignore special hf args that start with _ like '__cached__setup_devices'.
    safe_kwargs = {k: str(v) for k, v in kwargs.items() if not k.startswith("_")}
    s = json.dumps(safe_kwargs, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()


dataset_kwargs: Dict[str, str] = self.dataset_kwargs
dataset_kwargs["use_frozen_embeddings_as_input"] = "True"
dataset_kwargs["suffix_conditioning"] = "False"

train_dataset_kwargs = {
            "dataset_name": self.data_args.dataset_name,
            **dataset_kwargs,
        }

train_dataset_path = os.path.join(
            DATASET_CACHE_PATH, (md5_hash_kwargs(**train_dataset_kwargs) + ".arrow")
        )