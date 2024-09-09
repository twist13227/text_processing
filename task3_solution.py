import os
import lzma
import pickle
from pathlib import Path
from typing import Iterable, Set

from transformers import AutoTokenizer, BertForTokenClassification


class Solution:
    def __init__(self):
        LABELS_RAW = [
            "AGE",
            "FAMILY",
            "PENALTY",
            "AWARD",
            "IDEOLOGY",
            "PERCENT",
            "CITY",
            "LANGUAGE",
            "PERSON",
            "COUNTRY",
            "LAW",
            "PRODUCT",
            "CRIME",
            "LOCATION",
            "PROFESSION",
            "DATE",
            "MONEY",
            "RELIGION",
            "DISEASE",
            "NATIONALITY",
            "STATE_OR_PROVINCE",
            "DISTRICT",
            "NUMBER",
            "TIME",
            "EVENT",
            "ORDINAL",
            "WORK_OF_ART",
            "FACILITY",
            "ORGANIZATION",
        ]
        self.LABELS = (
            ["O"]
            + ["B_" + label for label in LABELS_RAW]
            + ["I_" + label for label in LABELS_RAW]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(Path(os.getcwd()) / "tokenizer")
        self.model = self._load_model()

    def predict(self, texts: list[str]) -> Iterable[Set[tuple[int, int, str]]]:
        for text in texts:
            tokens = self.tokenizer.encode_plus(
                text, return_offsets_mapping=True, return_tensors="pt"
            )
            model_output = self.model(tokens.input_ids)
            labels = model_output.logits.argmax(dim=2).tolist()[0]
            entities = set()
            cur_type = ""
            cur = [0, 0]
            for (a, b), label_int in zip(tokens["offset_mapping"][0], labels):
                label = self.LABELS[label_int]
                if label == "O" and cur_type:
                    entities.add((cur[0], cur[1], cur_type))
                    cur_type = ""
                    continue
                if label.startswith("B_"):
                    if cur_type:
                        entities.add((cur[0], cur[1], cur_type))
                    cur_type = label[2:]
                    cur = [int(a), int(b)]
                    continue
                if label.startswith("I_"):
                    if label[2:] != cur_type:
                        entities.add((cur[0], cur[1], cur_type))
                        cur_type = label[2:]
                        cur = [int(a), int(b)]
                    else:
                        cur[1] = int(b)
                    continue
            yield entities

    def _load_model(self):
        with open(Path(os.getcwd()) / "model_config", "rb") as f:
            model_config_bin = f.read()
        with open(Path(os.getcwd()) / "model_state_dict", "rb") as f:
            model_state_dict_bin = f.read()
        model_config = pickle.loads(
            lzma.decompress(
                model_config_bin,
                format=lzma.FORMAT_RAW,
                filters=[
                    {
                        "id": lzma.FILTER_LZMA2,
                        "dict_size": 268435456,
                        "preset": 9,
                        "mf": lzma.MF_HC3,
                        "depth": 0,
                        "lc": 3,
                    }
                ],
            )
        )
        model_state_dict = pickle.loads(
            lzma.decompress(
                model_state_dict_bin,
                format=lzma.FORMAT_RAW,
                filters=[
                    {
                        "id": lzma.FILTER_LZMA2,
                        "dict_size": 268435456,
                        "preset": 9,
                        "mf": lzma.MF_HC3,
                        "depth": 0,
                        "lc": 3,
                    }
                ],
            )
        )
        return BertForTokenClassification.from_pretrained(
            config=model_config,
            state_dict=model_state_dict,
            pretrained_model_name_or_path=None,
        )


Solution()
