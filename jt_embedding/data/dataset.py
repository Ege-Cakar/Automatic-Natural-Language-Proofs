from typing import Optional

import datasets
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from abc import ABC, abstractmethod



class MathTheoremDataset(Dataset):
    """Dataset for mathematical theorem pairs (formal + informal)"""

    def __init__(
        self,
        formal_theorems: list[str],
        informal_theorems: list[str],
        fl_tokenizer: AutoTokenizer,
        nl_tokenizer: AutoTokenizer,
        max_length: int = 512,
        labels: Optional[list[int]] = None,
    ):
        assert len(formal_theorems) == len(informal_theorems)
        self.formal_theorems = formal_theorems
        self.informal_theorems = informal_theorems
        self.fl_tokenizer = fl_tokenizer
        self.nl_tokenizer = nl_tokenizer
        self.max_length = max_length
        self.labels = labels

    def __len__(self):
        return len(self.formal_theorems)

    def __getitem__(self, idx):
        formal = self.formal_theorems[idx]
        informal = self.informal_theorems[idx]

        # Tokenize formal theorem (LEAN code)
        fl_tokens = self.fl_tokenizer(
            formal,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        # Tokenize informal theorem (natural language)
        nl_tokens = self.nl_tokenizer(
            informal,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        batch = {
            "formal_input_ids": fl_tokens["input_ids"].squeeze(),
            "formal_attention_mask": fl_tokens["attention_mask"].squeeze(),
            "informal_input_ids": nl_tokens["input_ids"].squeeze(),
            "informal_attention_mask": nl_tokens["attention_mask"].squeeze(),
            "index": idx,
        }

        if self.labels is not None:
            batch["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return batch
    
    @abstractmethod
    def get_lean_code(self, idx: int) -> str:
        """Return a compilable Lean code string for the given index."""
        pass


class FrenzyMathDataset(MathTheoremDataset):
    """Dataset loader for FrenzyMath/Herald proofs"""

    def __init__(
        self,
        headers: list[str],
        formal_proofs: list[str],
        informal_theorems: list[str],
        fl_tokenizer: AutoTokenizer,
        nl_tokenizer: AutoTokenizer,
        **kwargs,
    ):
        super().__init__(
            formal_theorems=formal_proofs,
            informal_theorems=informal_theorems,
            fl_tokenizer=fl_tokenizer,
            nl_tokenizer=nl_tokenizer,
            **kwargs,
        )
        self.headers = headers
        self.formal_proofs = formal_proofs

    @classmethod
    def from_datasets(
        cls,
        fl_tokenizer: AutoTokenizer,
        nl_tokenizer: AutoTokenizer,
        **kwargs,
    ):
        """Load dataset from Hugging Face datasets library"""
        ds = datasets.load_dataset("FrenzyMath/Herald_proofs", split="train")

        headers = []
        formal_proofs = []
        informal_theorems = []

        for item in ds:
            # Extract formal theorem from LEAN proof
            formal_theorem = item.get("formal_theorem", "")
            if "formal_proof" in item:
                formal_theorem += "\n" + item["formal_proof"]


            informal = item.get("informal_theorem", "")
            if "informal_proof" in item:
                informal += "\n" + item["informal_proof"]
            informal_theorems.append(informal)

        return cls(
            headers, formal_proofs, informal_theorems, fl_tokenizer, nl_tokenizer, **kwargs
        )

    def get_lean_code(self, idx: int, namespace: str = "Scratch") -> str:
        header = self.headers[idx]
        proof = self.formal_proofs[idx]

        import_lines = []
        other_header_lines = []

        for line in header.strip().splitlines():
            if line.strip().startswith("import"):
                import_lines.append(line.strip())
            else:
                other_header_lines.append(line.strip())

        return "\n".join([
            *import_lines,
            "",
            f"namespace {namespace}",
            "",
            *other_header_lines,
            "",
            proof,
            "",
            f"end {namespace}",
        ])

class GoedelDataset(MathTheoremDataset):
    """Dataset loader for Goedel-LM/Lean-workbook-proofs"""

    @classmethod
    def from_datasets(
        cls,
        fl_tokenizer: AutoTokenizer,
        nl_tokenizer: AutoTokenizer,
        **kwargs,
    ):
        """Load dataset from Hugging Face datasets library"""
        ds = datasets.load_dataset("Goedel-LM/Lean-workbook-proofs", split="train")

        formal_theorems = []
        informal_theorems = []

        for item in ds:
            # Extract formal theorem
            formal_theorem = item.get("formal_theorem", "")
            if "formal_proof" in item:
                formal_theorem += "\n" + item["formal_proof"]

            # Extract informal theorem
            informal_theorem = item.get("informal_theorem", "")
            if "informal_proof" in item:
                informal_theorem += "\n" + item["informal_proof"]

            formal_theorems.append(formal_theorem)
            informal_theorems.append(informal_theorem)

        return cls(
            formal_theorems, informal_theorems, fl_tokenizer, nl_tokenizer, **kwargs
        )


class FormL4Dataset(MathTheoremDataset):
    """Dataset loader for FormL4 dataset"""

    @classmethod
    def from_datasets(
        cls,
        fl_tokenizer: AutoTokenizer,
        nl_tokenizer: AutoTokenizer,
        **kwargs,
    ):
        """Load dataset from Hugging Face datasets library"""
        ds = datasets.load_dataset("FormL4", split="train")

        formal_theorems = []
        informal_theorems = []

        for item in ds:
            # Extract formal theorem
            formal_theorem = item.get("formal", "")
            if "formal_proof" in item:
                formal_theorem += "\n" + item["formal_proof"]

            # Extract informal theorem
            informal_theorem = item.get("informal", "")
            if "informal_proof" in item:
                informal_theorem += "\n" + item["informal_proof"]

            formal_theorems.append(formal_theorem)
            informal_theorems.append(informal_theorem)

        return cls(
            formal_theorems, informal_theorems, fl_tokenizer, nl_tokenizer, **kwargs
        )


def load_dataset(
    dataset_name: str,
    fl_tokenizer: AutoTokenizer,
    nl_tokenizer: AutoTokenizer,
    **kwargs,
) -> MathTheoremDataset:
    """Factory function to load datasets"""

    dataset_loaders = {
        "frenzy": FrenzyMathDataset.from_datasets,
        "goedel": GoedelDataset.from_datasets,
        "forml4": FormL4Dataset.from_datasets,
    }

    if dataset_name not in dataset_loaders:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available: {list(dataset_loaders.keys())}"
        )


    return dataset_loaders[dataset_name](fl_tokenizer, nl_tokenizer, **kwargs)
