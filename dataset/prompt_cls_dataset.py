from torch.utils.data import Dataset
import torch
import random
import numpy as np
import scipy.io as scio


class PromptClsDataset(Dataset):
    def __init__(self, captions, labels, bert_tokenizer, max_len=32):
        self.captions = captions
        self.labels = labels
        self.bert_tokenizer = bert_tokenizer
        self.max_len = max_len
        self.length = len(labels)

    def __len__(self):
        return self.length

    def _sample_raw_text(self, index):
        captions = self.captions[index]
        use_cap = captions[random.randint(0, len(captions) - 1)]

        if isinstance(use_cap, bytes):
            use_cap = use_cap.decode("utf-8")
        else:
            use_cap = str(use_cap)

        return use_cap.strip()

    def __getitem__(self, index):
        raw_text = self._sample_raw_text(index)

        encoding = self.bert_tokenizer(
            raw_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)          # [L]
        attention_mask = encoding["attention_mask"].squeeze(0) # [L]
        label = torch.from_numpy(self.labels[index]).float()   # [C]

        return input_ids, attention_mask, label


def split_data(captions, labels, query_num=5000, train_num=10000, seed=None):
    np.random.seed(seed=seed)
    random_index = np.random.permutation(range(len(labels)))

    query_index = random_index[: query_num]
    train_index = random_index[query_num: query_num + train_num]
    retrieval_index = random_index[query_num:]

    query_captions = captions[query_index]
    query_labels = labels[query_index]

    train_captions = captions[train_index]
    train_labels = labels[train_index]

    retrieval_captions = captions[retrieval_index]
    retrieval_labels = labels[retrieval_index]

    split_captions = (query_captions, train_captions, retrieval_captions)
    split_labels = (query_labels, train_labels, retrieval_labels)
    return split_captions, split_labels


def build_prompt_cls_datasets(
    captionFile: str,
    labelFile: str,
    bert_tokenizer,
    max_len=32,
    query_num=5000,
    train_num=10000,
    seed=None
):
    if captionFile.endswith("mat"):
        captions = scio.loadmat(captionFile)["caption"]
        captions = captions[0] if captions.shape[0] == 1 else captions
    elif captionFile.endswith("txt"):
        with open(captionFile, "r") as f:
            captions = f.readlines()
        captions = np.asarray([[item.strip()] for item in captions])
    else:
        raise ValueError("captionFile only supports [txt, mat].")

    labels = scio.loadmat(labelFile)["category"]

    split_captions, split_labels = split_data(
        captions, labels,
        query_num=query_num,
        train_num=train_num,
        seed=seed
    )

    train_data = PromptClsDataset(
        captions=split_captions[1],
        labels=split_labels[1],
        bert_tokenizer=bert_tokenizer,
        max_len=max_len
    )

    query_data = PromptClsDataset(
        captions=split_captions[0],
        labels=split_labels[0],
        bert_tokenizer=bert_tokenizer,
        max_len=max_len
    )

    retrieval_data = PromptClsDataset(
        captions=split_captions[2],
        labels=split_labels[2],
        bert_tokenizer=bert_tokenizer,
        max_len=max_len
    )

    return train_data, query_data, retrieval_data