from typing import Callable, Dict

import torch
import transformers

from vec2text.models import InversionModel


def tokenize_function(
        tokenizer: transformers.PreTrainedTokenizer,
        embedder_tokenizer: transformers.PreTrainedTokenizer,
        text_column_name: str,
        max_seq_length: int,
        padding: bool = False,
        prefix: str = None,
        lang_id: bool = False,
        script_id: bool = False,
) -> Callable[[Dict], Dict]:
    def tokenize_function_inner(examples) -> Dict[str, torch.Tensor]:
        if prefix:
            if lang_id and not script_id:
                examples[text_column_name] = [f"{prefix}: [{lang.split('_')[0]}] {text}" for text, lang in
                                              zip(examples[text_column_name],
                                                  examples["lang"])]
            elif lang_id and script_id:
                examples[text_column_name] = [f"{prefix}: [{lang.split('_')[0]}] [{lang.split('_')[1]}] {text}" for
                                              text, lang in
                                              zip(examples[text_column_name],
                                                  examples["lang"])]
            else:
                examples[text_column_name] = [f"{prefix}: {text}" for text in examples[text_column_name]]

        output = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
        )

        # copy to 'labels' for language modeling loss
        # but set padding to -100
        # github.com/huggingface/transformers/blob/cbe63949d76efd153a1f389f38fe9ce1287e06b0/src/transformers/models/t5/modeling_t5.py#L1504-L1507
        output["labels"] = [
            [
                (-100 if token_id == tokenizer.pad_token_id else token_id)
                for token_id in ids
            ]
            for ids in output["input_ids"]
        ]
        embedder_output = embedder_tokenizer(
            examples[text_column_name],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        embedder_output = {f"embedder_{k}": v for k, v in embedder_output.items()}

        output["length"] = [
            (torch.tensor(input_ids) != tokenizer.pad_token_id).sum().item()
            for input_ids in output["input_ids"]
        ]

        return {**output, **embedder_output}

    return tokenize_function_inner


def tokenize_function_llama_chat(
        tokenizer,
        embedder_tokenizer,
        text_column_name,
        max_seq_length,
        padding: bool = False,
        # no-op for compatibility with other tokenization functions
        prefix: str = None,
) -> Callable[[Dict], Dict]:
    """Use special tokenization for LLAMA chat models."""

    def tokenize_function_inner(examples) -> Dict[str, torch.Tensor]:
        if "prefix" not in examples:
            # hacky way to turn datasets into the right format for LLAMA chat.
            # "real" prompt datasets like one_million_paired_instructions
            # have "prefix" and "suffix" already.
            #
            # so this is only for evaluation datasets that may not have
            # actual prefix-suffix pairing.
            #
            examples["prefix"] = [""] * len(examples[text_column_name])
            examples["suffix"] = examples[text_column_name]

        output = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
        )

        # copy to 'labels' for language modeling loss
        # but set padding to -100
        # github.com/huggingface/transformers/blob/cbe63949d76efd153a1f389f38fe9ce1287e06b0/src/transformers/models/t5/modeling_t5.py#L1504-L1507
        output["labels"] = [
            [
                (-100 if token_id == tokenizer.pad_token_id else token_id)
                for token_id in ids
            ]
            for ids in output["input_ids"]
        ]
        embedder_output = embedder_tokenizer(
            text=[
                f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n {instruction} [/INST]"
                for (system_message, instruction) in zip(
                    examples["prefix"], examples["suffix"]
                )
            ],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        embedder_output = {f"embedder_{k}": v for k, v in embedder_output.items()}

        output["length"] = [
            (torch.tensor(input_ids) != tokenizer.pad_token_id).sum().item()
            for input_ids in output["input_ids"]
        ]

        return {**output, **embedder_output}

    return tokenize_function_inner


def embed_dataset_batch(model: InversionModel, batch: Dict) -> Dict:
    assert "input_ids" in batch.keys(), f"invalid keys {batch.keys()}"
    assert hasattr(model, "call_embedding_model")

    input_ids = batch["input_ids"]
    inputs_str = model.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    emb_input_ids = model.embedder_tokenizer(
        inputs_str,
        max_length=model.config.max_seq_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    ).to(next(model.parameters()).device)

    with torch.no_grad():
        batch["frozen_embeddings"] = model.call_embedding_model(**emb_input_ids)
    return batch


def get_tokenizer_mapping(
        lm: str, inverter: str, inverter_vocab_size: int
) -> torch.Tensor:
    """Computes the mapping from token outputs in `lm`'s vocabulary to those in `inverter's
    vocabulary. Makes some assumptions about spacing.
    """
    lm_tokenizer = transformers.AutoTokenizer.from_pretrained(lm)
    inverter_tokenizer = transformers.AutoTokenizer.from_pretrained(inverter)

    lm_vocab = lm_tokenizer.vocab
    mapping = torch.zeros(len(lm_vocab), dtype=torch.long)
    for k, idx in lm_tokenizer.vocab.items():
        # We replace space tokens with nothing and allow the call to
        # inverter_tokenizer.decode to determine this. We also
        # filter out 2 and 3 as first tokens which are extremely common
        # when the T5 tokenizer processes unicode. (These are hacks
        # specific to the LLAMA-T5 lm-inverter pairing, and it would
        # be better to find an automated wa to do this later.)
        mapping[idx] = inverter_tokenizer.encode(k.replace("▁", " "))[0]
        if mapping[idx] in [2, 3]:
            mapping[idx] = inverter_tokenizer.encode(k.replace("▁", " "))[1]

    preservation = len(set(mapping.tolist())) / len(lm_vocab)
    print(
        f"Mapped tokenizer {lm} to {inverter}. Preserved {preservation * 100:.1f}% of unique tokens."
    )
    return mapping


def whiten_embeddings(X: torch.Tensor):
    """
    Whitening the embeddings with SVD.
    """
    # Step 1: Compute the mean
    mu = torch.mean(X, dim=0, keepdim=True)

    # Step 2: Center the data
    X_centered = X - mu

    # Step 3: Compute the covariance matrix
    covariance_matrix = torch.cov(X_centered.T)

    # Step 4: Singular Value Decomposition (SVD)
    U, S, V = torch.linalg.svd(covariance_matrix)

    # Step 5: Whiten the data
    whitening_matrix = U @ torch.diag(1.0 / torch.sqrt(S)) @ U.T
    X_whitened = torch.mm(X_centered, whitening_matrix)

    return X_whitened, mu, S, U


def update_whitening_batch(X_whitened, mu, S, U, X_new):
    """
    Whitening new embeddings.
    """
    n = X_whitened.size(0)
    m = X_new.size(0)

    # Update mean
    mu_new = (n * mu + torch.sum(X_new, dim=0)) / (n + m)

    # Center the new embeddings
    X_new_centered = X_new - mu_new

    # Reconstruct the original centered data
    X_centered = X_whitened @ torch.inverse(U @ torch.diag(1.0 / torch.sqrt(S)) @ U.T)

    # Combine the original and new centered data
    X_combined = torch.cat((X_centered, X_new_centered), dim=0)

    # Compute the new covariance matrix
    covariance_matrix_new = torch.cov(X_combined.T)

    # Singular Value Decomposition (SVD) of the updated covariance matrix
    U_new, S_new, V_new = torch.linalg.svd(covariance_matrix_new)

    # Whiten the new embeddings
    whitening_matrix_new = U_new @ torch.diag(1.0 / torch.sqrt(S_new)) @ U_new.T
    X_new_whitened = torch.mm(X_new_centered, whitening_matrix_new)

    # Whiten the existing embeddings with the updated whitening transformation
    # X_whitened_new = torch.mm(X_centered, whitening_matrix_new)

    return X_new_whitened, mu_new, S_new, U_new
