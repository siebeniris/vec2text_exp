import logging
import os
import random
from typing import Dict, List

import yaml
import datasets
import json
import numpy as np
import torch

from vec2text.run_args import DataArguments
from vec2text.utils import dataset_map_multi_worker, get_num_proc


def retain_dataset_columns(
        d: datasets.Dataset, allowed_columns: List[str]
) -> datasets.Dataset:
    column_names_to_remove = [c for c in d.features if c not in allowed_columns]
    return d.remove_columns(column_names_to_remove)


def load_nq_dpr_corpus() -> datasets.Dataset:
    return datasets.load_dataset("jxm/nq_corpus_dpr")


def load_msmarco_corpus() -> datasets.Dataset:
    # has columns ["title", "text"]. only one split ("train")
    dataset_dict = datasets.load_dataset("Tevatron/msmarco-passage-corpus")
    return dataset_dict["train"]


def create_omi_ex(ex: Dict[str, str]) -> Dict[str, str]:
    ex["text"] = ex["user"]
    return ex


def create_ompi_ex(ex: Dict[str, str]) -> Dict[str, str]:
    ex["user"] = ex["user"].strip()
    ex["system"] = ex["system"].strip()
    ex["text"] = ex["system"] + "\n\n" + ex["user"]
    ex["prefix"] = ex["system"] + "\n\n"
    ex["suffix"] = ex["user"]
    return ex


def get_world_size() -> int:
    try:
        return torch.distributed.get_world_size()
    except (RuntimeError, ValueError):
        return 1


def load_one_million_paired_instructions() -> datasets.Dataset:
    # has only "train" split, and "system" (system prompt)
    # and "user" (user input) columns
    dataset_dict = datasets.load_dataset("wentingzhao/one-million-paired-instructions")
    dataset_dict = dataset_map_multi_worker(
        dataset_dict,
        map_fn=create_ompi_ex,
        num_proc=get_num_proc(),
    )

    return dataset_dict["train"]


def load_one_million_instructions() -> datasets.Dataset:
    # has only "train" split, and "system" (system prompt)
    # and "user" (user input) columns
    dataset_dict = datasets.load_dataset("wentingzhao/one-million-instructions")
    dataset_dict = dataset_map_multi_worker(dataset_dict, create_ompi_ex)

    return dataset_dict["train"]


def load_anthropic_toxic_prompts() -> datasets.Dataset:
    d = datasets.load_dataset("wentingzhao/anthropic-hh-first-prompt")["train"]
    d = d.rename_column("user", "text")
    return d


def load_luar_reddit() -> datasets.Dataset:
    d = datasets.load_dataset("friendshipkim/reddit_eval_embeddings_luar")
    d = d.rename_column("full_text", "text")
    d = d.rename_column("embedding", "frozen_embeddings")
    return d


def _load_json(path: str):
    with open(path) as f:
        return json.load(f)


def _build_coco_nomic_pair_dataset(
        embeddings_path: str,
        image_ids_path: str,
        captions_path: str,
) -> datasets.Dataset:
    image_ids = _load_json(image_ids_path)
    image_to_caption_data = _load_json(captions_path)
    embeddings = np.load(embeddings_path)

    if len(image_ids) != len(embeddings):
        raise ValueError(
            f"mismatched counts for {embeddings_path}: "
            f"{len(image_ids)} ids vs {len(embeddings)} embeddings"
        )

    missing_ids = [image_id for image_id in image_ids if image_id not in image_to_caption_data]
    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} image ids from {image_ids_path} were missing in {captions_path}. "
            f"First few missing ids: {missing_ids[:5]}"
        )

    rows = {
        "image_id": [],
        "text": [],
        "frozen_embeddings": [],
        "captions": [],
    }
    for image_id, embedding in zip(image_ids, embeddings):
        caption_data = image_to_caption_data[image_id]
        if isinstance(caption_data, dict):
            captions = caption_data["caption"]
        elif isinstance(caption_data, list):
            captions = caption_data
        elif isinstance(caption_data, str):
            captions = [caption_data]
        else:
            raise ValueError(
                f"unsupported caption format for image {image_id} in {captions_path}: "
                f"{type(caption_data).__name__}"
            )
        if not captions:
            raise ValueError(f"image {image_id} in {captions_path} has no captions")

        rows["image_id"].append(image_id)
        rows["text"].append(captions[0])
        rows["frozen_embeddings"].append(embedding)
        rows["captions"].append(captions)

    dataset = datasets.Dataset.from_dict(rows)
    dataset = dataset.with_format("torch")
    return dataset


def _resolve_victim_embedding_files(victim_embedding_name: str):
    repo_root = os.getcwd()
    victim_root = os.path.join(repo_root, "data", "embeds", "victim_embeddings")

    embed_dir = os.path.join(victim_root, victim_embedding_name)
    if not os.path.isdir(embed_dir):
        raise ValueError(
            f"unsupported victim embedding source '{victim_embedding_name}'. "
            f"Directory not found: {embed_dir}"
        )

    train_npy = os.path.join(embed_dir, "train", "train.npy")
    train_ids = os.path.join(embed_dir, "train", "train_image_ids.json")
    test_npy = os.path.join(embed_dir, "test", "test.npy")
    test_ids = os.path.join(embed_dir, "test", "test_image_ids.json")
    if not all(os.path.exists(p) for p in [train_npy, train_ids, test_npy, test_ids]):
        raise ValueError(
            f"victim embedding source '{victim_embedding_name}' is missing train/test files under {embed_dir}"
        )

    return {
        "train_npy": train_npy,
        "train_ids": train_ids,
        "test_npy": test_npy,
        "test_ids": test_ids,
    }


def _build_coco_victim_pair_dataset(
        embeddings_path: str,
        image_ids_path: str,
        captions_path: str,
        randomize_embeddings: bool = False,
        random_seed: int = 42,
) -> datasets.Dataset:
    image_ids = _load_json(image_ids_path)
    image_to_caption_data = _load_json(captions_path)
    embeddings = np.load(embeddings_path)

    if randomize_embeddings:
        rng = np.random.default_rng(random_seed)
        embeddings = rng.standard_normal(size=embeddings.shape, dtype=np.float32)

    if len(image_ids) != len(embeddings):
        raise ValueError(
            f"mismatched counts for {embeddings_path}: "
            f"{len(image_ids)} ids vs {len(embeddings)} embeddings"
        )

    missing_ids = [image_id for image_id in image_ids if image_id not in image_to_caption_data]
    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} image ids from {image_ids_path} were missing in {captions_path}. "
            f"First few missing ids: {missing_ids[:5]}"
        )

    rows = {
        "image_id": [],
        "text": [],
        "frozen_embeddings": [],
        "captions": [],
    }
    for image_id, embedding in zip(image_ids, embeddings):
        caption_data = image_to_caption_data[image_id]
        if isinstance(caption_data, dict):
            captions = caption_data["caption"]
        elif isinstance(caption_data, list):
            captions = caption_data
        elif isinstance(caption_data, str):
            captions = [caption_data]
        else:
            raise ValueError(
                f"unsupported caption format for image {image_id} in {captions_path}: "
                f"{type(caption_data).__name__}"
            )
        if not captions:
            raise ValueError(f"image {image_id} in {captions_path} has no captions")

        rows["image_id"].append(image_id)
        rows["text"].append(captions[0])
        rows["frozen_embeddings"].append(embedding)
        rows["captions"].append(captions)

    dataset = datasets.Dataset.from_dict(rows)
    dataset = dataset.with_format("torch")
    return dataset


def load_coco_victim_first_caption(
        victim_embedding_name: str = "nomic",
        use_random_embeddings: bool = False,
        random_embedding_seed: int = 42,
) -> datasets.DatasetDict:
    repo_root = os.getcwd()
    caption_dir = os.path.join(repo_root, "data", "victim_embeds_data")
    embedding_files = _resolve_victim_embedding_files(victim_embedding_name)

    train_dataset = _build_coco_victim_pair_dataset(
        embeddings_path=embedding_files["train_npy"],
        image_ids_path=embedding_files["train_ids"],
        captions_path=os.path.join(caption_dir, "train_dict.json"),
        randomize_embeddings=use_random_embeddings,
        random_seed=random_embedding_seed,
    )
    validation_dataset = _build_coco_victim_pair_dataset(
        embeddings_path=embedding_files["test_npy"],
        image_ids_path=embedding_files["test_ids"],
        captions_path=os.path.join(caption_dir, "test_dict.json"),
        randomize_embeddings=use_random_embeddings,
        random_seed=random_embedding_seed + 1,
    )
    return datasets.DatasetDict(
        {
            "train": train_dataset,
            "validation": validation_dataset,
        }
    )


def load_coco_nomic_first_caption() -> datasets.DatasetDict:
    return load_coco_victim_first_caption(victim_embedding_name="nomic")


def load_xnli(lang) -> datasets.Dataset:
    def concat_pre_hyp(sample):
        sample["text"] = sample["premise"] + " " + sample["hypothesis"]
        return sample

    dataset = datasets.load_dataset("xnli", lang)
    dataset = dataset.map(concat_pre_hyp, remove_columns=["premise", "hypothesis"])
    return dataset


def load_xnli_test(lang) -> datasets.Dataset:
    def concat_pre_hyp(sample):
        sample["text"] = sample["premise"] + " " + sample["hypothesis"]
        return sample

    dataset = datasets.load_dataset("xnli", lang)
    dataset = dataset.map(concat_pre_hyp, remove_columns=["premise", "hypothesis"])["test"]
    return dataset


def dataset_from_args(data_args: DataArguments) -> datasets.DatasetDict:
    """Loads a dataset from data_args create in `run_args`."""
    if data_args.dataset_name == "nq":
        raw_datasets = load_nq_dpr_corpus()
        raw_datasets["validation"] = raw_datasets["dev"]
    elif data_args.dataset_name.startswith("mt-ms"):
        # 12.06.2024 mt-ms dataset.
        assert "_" in data_args.dataset_name  # mt-ms_lat_scrp
        lang = data_args.dataset_name.replace("mt-ms_", "")
        # assert len(lang) == 8
        raw_datasets = load_mt_ms(lang)
    elif data_args.dataset_name.startswith("yiyic/multiHPLT_"):
        raw_datasets = datasets.load_dataset(data_args.dataset_name)
        raw_datasets["validation"] = raw_datasets["dev"]
    elif data_args.dataset_name.startswith("yiyic/mmarco_"):
        raw_datasets = datasets.load_dataset(data_args.dataset_name)
        raw_datasets["validation"] = raw_datasets["dev"]
    elif data_args.dataset_name == "msmarco":
        raw_datasets = load_msmarco_corpus()
        raw_datasets = raw_datasets.train_test_split(test_size=0.01)
        raw_datasets["validation"] = raw_datasets["test"]
    elif data_args.dataset_name == "one_million_instructions":
        raw_datasets = load_one_million_instructions()
        raw_datasets = raw_datasets.train_test_split(test_size=0.01)
        raw_datasets["validation"] = raw_datasets["test"]
    elif data_args.dataset_name == "one_million_paired_instructions":
        raw_datasets = load_one_million_paired_instructions()
        raw_datasets = raw_datasets.train_test_split(test_size=0.01)
        raw_datasets["validation"] = raw_datasets["test"]
    elif data_args.dataset_name == "one_million_instructions":
        raw_datasets = load_one_million_instructions()
        raw_datasets = raw_datasets.train_test_split(test_size=0.01)
        raw_datasets["validation"] = raw_datasets["test"]
    elif data_args.dataset_name == "luar_reddit":
        all_luar_datasets = load_luar_reddit()
        raw_datasets = datasets.DatasetDict(
            {
                "train": all_luar_datasets["candidates"],
                "validation": all_luar_datasets["queries"],
            }
        )
    elif data_args.dataset_name in {"coco_nomic_first_caption", "coco_victim_first_caption"}:
        raw_datasets = load_coco_victim_first_caption(
            victim_embedding_name=data_args.victim_embedding_name,
            use_random_embeddings=data_args.use_random_embeddings,
            random_embedding_seed=data_args.random_embedding_seed,
        )
    else:
        raise ValueError(f"unsupported dataset {data_args.dataset_name}")
    return raw_datasets


def load_ag_news_test() -> datasets.Dataset:
    return datasets.load_dataset("ag_news")["test"]


def load_xsum_val(col: str) -> datasets.Dataset:
    d = datasets.load_dataset("xsum")["validation"]
    d = d.rename_column(col, "text")
    return d


def load_wikibio_val() -> datasets.Dataset:
    d = datasets.load_dataset("wiki_bio", trust_remote_code=True)["val"]
    d = d.rename_column("target_text", "text")
    return d


def load_arxiv_val() -> datasets.Dataset:
    d = datasets.load_dataset("ccdv/arxiv-summarization")["validation"]
    d = d.rename_column("abstract", "text")
    return d


def load_python_code_instructions_18k_alpaca() -> datasets.Dataset:
    d = datasets.load_dataset("iamtarun/python_code_instructions_18k_alpaca")["train"]
    d = d.rename_column("instruction", "text")
    return d


def load_beir_corpus(name: str) -> List[str]:
    from beir import util as beir_util
    from beir.datasets.data_loader import GenericDataLoader

    #### Download scifact.zip dataset and unzip the dataset
    beir_datasets_cache_dir = "/home/jxm3/research/retrieval/distractor_exp"

    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
        name
    )
    out_dir = os.path.join(beir_datasets_cache_dir, "datasets")
    data_path = beir_util.download_and_unzip(url, out_dir)

    # Limit each corpus to first 100k documents.
    MAX_N = 100_000

    if name == "cqadupstack":
        full_corpus = []
        for folder in [
            "android",
            "english",
            "gaming",
            "gis",
            "mathematica",
            "physics",
            "programmers",
            "stats",
            "tex",
            "unix",
            "webmasters",
            "wordpress",
        ]:
            corpus, _queries, _qrels = GenericDataLoader(
                data_folder=os.path.join(data_path, folder)
            ).load(split="test")
            full_corpus.extend([k["text"] for k in corpus.values()])
        random.shuffle(full_corpus)
        return full_corpus[:MAX_N]
    else:
        corpus, _queries, _qrels = GenericDataLoader(data_folder=data_path).load(
            split="test"
        )
        corpus = [k["text"] for k in corpus.values()]
        return corpus[:MAX_N]


def load_beir_dataset(name: str) -> datasets.Dataset:
    cache_path = (
        datasets.config.HF_DATASETS_CACHE
    )  # something like /home/jxm3/.cache/huggingface/datasets
    dataset_path = os.path.join(cache_path, "emb_inv_beir", name)
    # print(f"loading BEIR dataset: {name}")
    if os.path.exists(dataset_path):
        logging.info("Loading BEIR dataset %s path %s", dataset_path)
        dataset = datasets.load_from_disk(dataset_path)
    else:
        logging.info(
            "Loading BEIR dataset %s from JSON (slow) at path %s", dataset_path
        )
        corpus = load_beir_corpus(name=name)
        dataset = datasets.Dataset.from_list([{"text": t} for t in corpus])
        os.makedirs(os.path.join(cache_path, "emb_inv_beir"), exist_ok=True)
        dataset.save_to_disk(dataset_path)
        logging.info("Saved BEIR dataset as HF path %s", dataset_path)
    return dataset


def load_beir_datasets() -> datasets.DatasetDict:
    all_beir_datasets = [
        ####### public datasets #######
        "arguana",
        "climate-fever",
        "cqadupstack",
        "dbpedia-entity",
        "fever",
        "fiqa",
        "hotpotqa",
        "msmarco",
        "nfcorpus",
        "nq",
        "quora",
        "scidocs",
        "scifact",
        "trec-covid",
        "webis-touche2020",
        ####### private datasets #######
        "signal1m",
        "trec-news",
        "robust04",
        "bioasq",
    ]
    return datasets.DatasetDict({k: load_beir_dataset(k) for k in all_beir_datasets})


def load_mt_ms(lang) -> datasets.DatasetDict:
    # load multilingual multi-script dataset
    with open("vec2text/lang2file.yaml") as f:
        lang2file = yaml.safe_load(f)

    file = lang2file[lang]
    print(f"loading data from {file} for {lang}")
    train_dataset = datasets.load_dataset(file)["train"].select_columns(["text"])
    # loading the validation dataset.
    validation_file = file.replace("_train", "_dev")
    print(f"loading data from {validation_file} for {lang}")
    validation_dataset = datasets.load_dataset(validation_file)["train"].select_columns(["text"])
    raw_datasets = datasets.DatasetDict(
        {
            "train": train_dataset,
            "validation": validation_dataset,
        }
    )
    return raw_datasets


def load_mt_ms_test() -> datasets.DatasetDict:
    """
    Multilingual multi-script test dataset.
    """
    test_dataset = datasets.load_dataset("yiyic/mt_ms_test")
    return test_dataset


def load_standard_val_datasets(data_args: DataArguments) -> datasets.DatasetDict:
    """Loads a pre-defined set of standard val datasets."""
    # d = {
    #     "ag_news": load_ag_news_test(),
    #     "anthropic_toxic_prompts": load_anthropic_toxic_prompts(),
    #     "arxiv": load_arxiv_val(),
    #     "python_code_alpaca": load_python_code_instructions_18k_alpaca(),
    #     # "xsum_doc": load_xsum_val("document"),
    #     # "xsum_summ": load_xsum_val("summary"),
    #     "wikibio": load_wikibio_val(),
    # }
    # langs = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]
    # d = {f"nxli_{lang}": load_xnli_test(lang) for lang in langs}
    # d = {k: retain_dataset_columns(v, ["text"]) for k, v in d.items()}

    # d = load_mt_ms_test()

    d = datasets.load_dataset(data_args.dataset_name)
    d_test = datasets.DatasetDict({
        "test": d["test"]
    })
    return d_test
