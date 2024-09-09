import lzma
import os
import pickle
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
)


IGNORED_FILES = [
    "165459_text.txt",
    "176167_text.txt",
    "178485_text.txt",
    "192238_text.txt",
    "193267_text.txt",
    "193946_text.txt",
    "194112_text.txt",
    "2021.txt",
    "202294_text.txt",
    "2031.txt",
    "209438_text.txt",
    "209731_text.txt",
    "546860_text.txt",
]

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

LABELS = (
    ["O"]
    + ["B_" + label for label in LABELS_RAW]
    + ["I_" + label for label in LABELS_RAW]
)


def line_to_tuple(s):
    _, t, a, b, text = s.rstrip().split(maxsplit=4)
    return int(a), int(b), t, text


def load_seqs_from_dir(path):
    token_seqs, label_seqs = [], []
    for file_path in Path(path).rglob("*.txt"):
        if file_path.name in IGNORED_FILES:
            continue
        with open(file_path) as f:
            text = f.read()
        with open(file_path.with_name(file_path.name.replace("txt", "ann"))) as f:
            spans_raw = sorted(
                [
                    line_to_tuple(line)
                    for line in f
                    if ";" not in line and line[0] == "T"
                ]
            )
            spans = []
            for x in spans_raw:
                if not any(x[0] <= y[0] and y[1] <= x[1] and x != y for y in spans_raw):
                    spans.append(x)
        tokens = tokenizer.encode_plus(text, return_offsets_mapping=True)
        tokens_map = tokens["offset_mapping"]
        if len(tokens_map) > 2048:
            continue
        labels_seq = []
        idx = 0
        for span in spans:
            if span[0] == span[1]:
                continue
            while tokens_map[idx][0] < span[0] and tokens_map[idx + 1][0] <= span[1]:
                labels_seq.append("O")
                idx += 1
            labels_seq.append("B_" + span[2])
            idx += 1
            while idx < len(tokens_map) and tokens_map[idx][1] <= span[1]:
                labels_seq.append("I_" + span[2])
                idx += 1
        while len(labels_seq) != len(tokens_map):
            labels_seq.append("O")
        token_seqs.append(tokens)
        label_seqs.append(labels_seq)
    return token_seqs, label_seqs


def get_label2idx(label_set: list[str]) -> dict[str, int]:
    label2idx: dict[str, int] = {}
    i = 0
    for label in label_set:
        label2idx[label] = i
        i += 1
    return label2idx


def save_model(model):
    with open(Path(os.getcwd()) / "model_state_dict", "wb") as f:
        compressed = lzma.compress(
            pickle.dumps(model.state_dict()),
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
        f.write(compressed)
    with open(Path(os.getcwd()) / "model_config", "wb") as f:
        compressed = lzma.compress(
            pickle.dumps(model.config),
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
        f.write(compressed)


class TransformersDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        token_seqs,
        label_seqs,
    ):
        self.token_seqs = token_seqs
        for t in self.token_seqs:
            if "offset_mapping" in t:
                t.pop("offset_mapping")
        self.label_seqs = [
            self.process_labels(labels, get_label2idx(LABELS)) for labels in label_seqs
        ]

    def __len__(self):
        return len(self.token_seqs)

    def __getitem__(
        self,
        idx: int,
    ):
        return {**self.token_seqs[idx], "labels": self.label_seqs[idx]}

    @staticmethod
    def process_labels(
        labels: list[str],
        label2idx: dict[str, int],
    ) -> list[int]:
        return torch.tensor([label2idx[label] for label in labels])


tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
model = AutoModelForTokenClassification.from_pretrained(
    "cointegrated/rubert-tiny2", num_labels=len(LABELS)
).cuda()
train_tokens, train_labels = load_seqs_from_dir(Path(os.getcwd()) / "train")
dev_tokens, dev_labels = load_seqs_from_dir(Path(os.getcwd()) / "dev")
test_tokens, test_labels = load_seqs_from_dir(Path(os.getcwd()) / "test")
train_dataset = TransformersDataset(
    train_tokens,
    train_labels,
)
dev_dataset = TransformersDataset(
    dev_tokens,
    dev_labels,
)
os.environ["WANDB_DISABLED"] = "true"
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./res",
        logging_dir="./logs",
        num_train_epochs=25,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        weight_decay=0.01,
        logging_steps=10,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=250,
        evaluation_strategy="epoch",
        save_strategy="no",
    ),
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
)
trainer.train()
model.cpu()
save_model(model)
tokenizer.save_pretrained(Path(os.getcwd()) / "tokenizer")
