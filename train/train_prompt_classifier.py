# import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from transformers import BertTokenizer

# from dataset.prompt_cls_dataset import build_prompt_cls_datasets
# from model.bert_prompt_classifier import BertPromptClassifier


# def compute_micro_f1(logits, labels, threshold=0.5):
#     probs = torch.sigmoid(logits)
#     preds = (probs > threshold).float()
#     labels = labels.float()

#     tp = (preds * labels).sum()
#     fp = (preds * (1 - labels)).sum()
#     fn = ((1 - preds) * labels).sum()

#     precision = tp / (tp + fp + 1e-8)
#     recall = tp / (tp + fn + 1e-8)
#     f1 = 2 * precision * recall / (precision + recall + 1e-8)

#     return precision.item(), recall.item(), f1.item()


# def compute_macro_f1(logits, labels, threshold=0.5):
#     probs = torch.sigmoid(logits)
#     preds = (probs > threshold).float()
#     labels = labels.float()

#     num_classes = labels.size(1)
#     f1_list = []

#     for c in range(num_classes):
#         pred_c = preds[:, c]
#         label_c = labels[:, c]

#         tp = (pred_c * label_c).sum()
#         fp = (pred_c * (1 - label_c)).sum()
#         fn = ((1 - pred_c) * label_c).sum()

#         precision = tp / (tp + fp + 1e-8)
#         recall = tp / (tp + fn + 1e-8)
#         f1 = 2 * precision * recall / (precision + recall + 1e-8)
#         f1_list.append(f1)

#     macro_f1 = torch.stack(f1_list).mean()
#     return macro_f1.item()


# def evaluate(model, loader, device):
#     model.eval()

#     all_logits = []
#     all_labels = []
#     total_loss = 0.0
#     total_num = 0

#     with torch.no_grad():
#         for input_ids, attention_mask, label in loader:
#             input_ids = input_ids.to(device)
#             attention_mask = attention_mask.to(device)
#             label = label.to(device).float()

#             logits = model(input_ids, attention_mask)
#             loss = F.binary_cross_entropy_with_logits(logits, label)

#             bs = input_ids.size(0)
#             total_loss += loss.item() * bs
#             total_num += bs

#             all_logits.append(logits.cpu())
#             all_labels.append(label.cpu())

#     all_logits = torch.cat(all_logits, dim=0)
#     all_labels = torch.cat(all_labels, dim=0)

#     precision, recall, micro_f1 = compute_micro_f1(all_logits, all_labels, threshold=0.5)
#     macro_f1 = compute_macro_f1(all_logits, all_labels, threshold=0.5)

#     return {
#         "loss": total_loss / total_num,
#         "precision": precision,
#         "recall": recall,
#         "micro_f1": micro_f1,
#         "macro_f1": macro_f1,
#     }


# def train_one_epoch(model, loader, optimizer, device):
#     model.train()

#     total_loss = 0.0
#     total_num = 0

#     for input_ids, attention_mask, label in loader:
#         input_ids = input_ids.to(device)
#         attention_mask = attention_mask.to(device)
#         label = label.to(device).float()

#         logits = model(input_ids, attention_mask)
#         loss = F.binary_cross_entropy_with_logits(logits, label)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         bs = input_ids.size(0)
#         total_loss += loss.item() * bs
#         total_num += bs

#     return total_loss / total_num


# def main():
#     # =========================
#     # 你先手动写死 Flickr25K
#     # =========================
#     caption_file = "dataset/flickr25k/caption.mat"
#     label_file = "dataset/flickr25k/label.mat"

#     bert_path = "/home/yuck/bert-base-uncased"
#     save_path = "./checkpoints/prompt_cls_flickr25k.pt"

#     num_classes = 24
#     max_len = 32
#     batch_size = 32
#     epochs = 10
#     lr = 2e-5
#     num_workers = 4
#     seed = 1814

#     query_num = 5000
#     train_num = 10000

#     os.makedirs("./checkpoints", exist_ok=True)

#     torch.manual_seed(seed)
#     device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

#     bert_tokenizer = BertTokenizer.from_pretrained(
#         bert_path,
#         local_files_only=True
#     )

#     train_data, val_data, retrieval_data = build_prompt_cls_datasets(
#         captionFile=caption_file,
#         labelFile=label_file,
#         bert_tokenizer=bert_tokenizer,
#         max_len=max_len,
#         query_num=query_num,
#         train_num=train_num,
#         seed=seed
#     )

#     train_loader = DataLoader(
#         train_data,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=True
#     )

#     val_loader = DataLoader(
#         val_data,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True
#     )

#     model = BertPromptClassifier(
#         num_classes=num_classes,
#         bert_path=bert_path,
#         dropout=0.1
#     ).to(device)

#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

#     best_macro_f1 = -1.0

#     for epoch in range(epochs):
#         train_loss = train_one_epoch(model, train_loader, optimizer, device)
#         val_metrics = evaluate(model, val_loader, device)

#         print(
#             f"Epoch [{epoch+1}/{epochs}] | "
#             f"train_loss={train_loss:.4f} | "
#             f"val_loss={val_metrics['loss']:.4f} | "
#             f"micro_f1={val_metrics['micro_f1']:.4f} | "
#             f"macro_f1={val_metrics['macro_f1']:.4f}"
#         )

#         if val_metrics["macro_f1"] > best_macro_f1:
#             best_macro_f1 = val_metrics["macro_f1"]
#             torch.save(
#                 {
#                     "model_state_dict": model.state_dict(),
#                     "best_macro_f1": best_macro_f1,
#                     "num_classes": num_classes,
#                     "bert_path": bert_path,
#                     "max_len": max_len,
#                 },
#                 save_path
#             )
#             print(f"Saved best model to: {save_path}")

#     print("Training finished.")
#     print(f"Best macro_f1: {best_macro_f1:.4f}")


# if __name__ == "__main__":
#     main()


# import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from transformers import BertTokenizer

# from dataset.prompt_cls_dataset import build_prompt_cls_datasets
# from model.bert_prompt_classifier import BertPromptClassifier


# def compute_micro_f1(logits, labels, threshold=0.5):
#     probs = torch.sigmoid(logits)
#     preds = (probs > threshold).float()
#     labels = labels.float()

#     tp = (preds * labels).sum()
#     fp = (preds * (1 - labels)).sum()
#     fn = ((1 - preds) * labels).sum()

#     precision = tp / (tp + fp + 1e-8)
#     recall = tp / (tp + fn + 1e-8)
#     f1 = 2 * precision * recall / (precision + recall + 1e-8)

#     return precision.item(), recall.item(), f1.item()


# def compute_macro_f1(logits, labels, threshold=0.5):
#     probs = torch.sigmoid(logits)
#     preds = (probs > threshold).float()
#     labels = labels.float()

#     num_classes = labels.size(1)
#     f1_list = []

#     for c in range(num_classes):
#         pred_c = preds[:, c]
#         label_c = labels[:, c]

#         tp = (pred_c * label_c).sum()
#         fp = (pred_c * (1 - label_c)).sum()
#         fn = ((1 - pred_c) * label_c).sum()

#         precision = tp / (tp + fp + 1e-8)
#         recall = tp / (tp + fn + 1e-8)
#         f1 = 2 * precision * recall / (precision + recall + 1e-8)
#         f1_list.append(f1)

#     macro_f1 = torch.stack(f1_list).mean()
#     return macro_f1.item()


# def evaluate(model, loader, device):
#     model.eval()

#     all_logits = []
#     all_labels = []
#     total_loss = 0.0
#     total_num = 0

#     with torch.no_grad():
#         for input_ids, attention_mask, label in loader:
#             input_ids = input_ids.to(device)
#             attention_mask = attention_mask.to(device)
#             label = label.to(device).float()

#             logits = model(input_ids, attention_mask)
#             loss = F.binary_cross_entropy_with_logits(logits, label)

#             bs = input_ids.size(0)
#             total_loss += loss.item() * bs
#             total_num += bs

#             all_logits.append(logits.cpu())
#             all_labels.append(label.cpu())

#     all_logits = torch.cat(all_logits, dim=0)
#     all_labels = torch.cat(all_labels, dim=0)

#     precision, recall, micro_f1 = compute_micro_f1(all_logits, all_labels, threshold=0.5)
#     macro_f1 = compute_macro_f1(all_logits, all_labels, threshold=0.5)

#     return {
#         "loss": total_loss / total_num,
#         "precision": precision,
#         "recall": recall,
#         "micro_f1": micro_f1,
#         "macro_f1": macro_f1,
#     }


# def train_one_epoch(model, loader, optimizer, device):
#     model.train()

#     total_loss = 0.0
#     total_num = 0

#     for input_ids, attention_mask, label in loader:
#         input_ids = input_ids.to(device)
#         attention_mask = attention_mask.to(device)
#         label = label.to(device).float()

#         logits = model(input_ids, attention_mask)
#         loss = F.binary_cross_entropy_with_logits(logits, label)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         bs = input_ids.size(0)
#         total_loss += loss.item() * bs
#         total_num += bs

#     return total_loss / total_num


# def main():
#     # =========================
#     # MSCOCO
#     # =========================
#     caption_file = "dataset/coco/caption.mat"
#     label_file = "dataset/coco/label.mat"

#     bert_path = "/home/yuck/bert-base-uncased"
#     save_path = "./checkpoints/prompt_cls_coco.pt"

#     num_classes = 80
#     max_len = 32
#     batch_size = 128
#     epochs = 20
#     lr = 2e-5
#     num_workers = 4
#     seed = 1814

#     query_num = 5000
#     train_num = 10000

#     os.makedirs("./checkpoints", exist_ok=True)

#     torch.manual_seed(seed)
#     device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

#     bert_tokenizer = BertTokenizer.from_pretrained(
#         bert_path,
#         local_files_only=True
#     )

#     train_data, val_data, retrieval_data = build_prompt_cls_datasets(
#         captionFile=caption_file,
#         labelFile=label_file,
#         bert_tokenizer=bert_tokenizer,
#         max_len=max_len,
#         query_num=query_num,
#         train_num=train_num,
#         seed=seed
#     )

#     train_loader = DataLoader(
#         train_data,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=True
#     )

#     val_loader = DataLoader(
#         val_data,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True
#     )

#     model = BertPromptClassifier(
#         num_classes=num_classes,
#         bert_path=bert_path,
#         dropout=0.1
#     ).to(device)

#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

#     best_macro_f1 = -1.0

#     for epoch in range(epochs):
#         train_loss = train_one_epoch(model, train_loader, optimizer, device)
#         val_metrics = evaluate(model, val_loader, device)

#         print(
#             f"Epoch [{epoch+1}/{epochs}] | "
#             f"train_loss={train_loss:.4f} | "
#             f"val_loss={val_metrics['loss']:.4f} | "
#             f"micro_f1={val_metrics['micro_f1']:.4f} | "
#             f"macro_f1={val_metrics['macro_f1']:.4f}"
#         )

#         if val_metrics["macro_f1"] > best_macro_f1:
#             best_macro_f1 = val_metrics["macro_f1"]
#             torch.save(
#                 {
#                     "model_state_dict": model.state_dict(),
#                     "best_macro_f1": best_macro_f1,
#                     "num_classes": num_classes,
#                     "bert_path": bert_path,
#                     "max_len": max_len,
#                 },
#                 save_path
#             )
#             print(f"Saved best model to: {save_path}")

#     print("Training finished.")
#     print(f"Best macro_f1: {best_macro_f1:.4f}")


# if __name__ == "__main__":
#     main()



import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from dataset.prompt_cls_dataset import build_prompt_cls_datasets
from model.bert_prompt_classifier import BertPromptClassifier


def compute_micro_f1(logits, labels, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    labels = labels.float()

    tp = (preds * labels).sum()
    fp = (preds * (1 - labels)).sum()
    fn = ((1 - preds) * labels).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return precision.item(), recall.item(), f1.item()


def compute_macro_f1(logits, labels, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    labels = labels.float()

    num_classes = labels.size(1)
    f1_list = []

    for c in range(num_classes):
        pred_c = preds[:, c]
        label_c = labels[:, c]

        tp = (pred_c * label_c).sum()
        fp = (pred_c * (1 - label_c)).sum()
        fn = ((1 - pred_c) * label_c).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_list.append(f1)

    macro_f1 = torch.stack(f1_list).mean()
    return macro_f1.item()


def evaluate(model, loader, device):
    model.eval()

    all_logits = []
    all_labels = []
    total_loss = 0.0
    total_num = 0

    with torch.no_grad():
        for input_ids, attention_mask, label in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label = label.to(device).float()

            logits = model(input_ids, attention_mask)
            loss = F.binary_cross_entropy_with_logits(logits, label)

            bs = input_ids.size(0)
            total_loss += loss.item() * bs
            total_num += bs

            all_logits.append(logits.cpu())
            all_labels.append(label.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    precision, recall, micro_f1 = compute_micro_f1(all_logits, all_labels, threshold=0.5)
    macro_f1 = compute_macro_f1(all_logits, all_labels, threshold=0.5)

    return {
        "loss": total_loss / total_num,
        "precision": precision,
        "recall": recall,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
    }


def train_one_epoch(model, loader, optimizer, device):
    model.train()

    total_loss = 0.0
    total_num = 0

    for input_ids, attention_mask, label in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label = label.to(device).float()

        logits = model(input_ids, attention_mask)
        loss = F.binary_cross_entropy_with_logits(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = input_ids.size(0)
        total_loss += loss.item() * bs
        total_num += bs

    return total_loss / total_num


def main():
    # =========================
    # NUS-WIDE
    # =========================
    caption_file = "dataset/nuswide/caption.txt"
    label_file = "dataset/nuswide/label.mat"

    bert_path = "/home/yuck/bert-base-uncased"
    save_path = "./checkpoints/prompt_cls_nuswide.pt"

    num_classes = 21
    max_len = 32
    batch_size = 128
    epochs = 20
    lr = 2e-5
    num_workers = 4
    seed = 1814

    # 这里按你的主实验划分来
    query_num = 5000
    train_num = 10000

    os.makedirs("./checkpoints", exist_ok=True)

    torch.manual_seed(seed)
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

    bert_tokenizer = BertTokenizer.from_pretrained(
        bert_path,
        local_files_only=True
    )

    train_data, val_data, retrieval_data = build_prompt_cls_datasets(
        captionFile=caption_file,
        labelFile=label_file,
        bert_tokenizer=bert_tokenizer,
        max_len=max_len,
        query_num=query_num,
        train_num=train_num,
        seed=seed
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    model = BertPromptClassifier(
        num_classes=num_classes,
        bert_path=bert_path,
        dropout=0.1
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_macro_f1 = -1.0

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"micro_f1={val_metrics['micro_f1']:.4f} | "
            f"macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_macro_f1": best_macro_f1,
                    "num_classes": num_classes,
                    "bert_path": bert_path,
                    "max_len": max_len,
                },
                save_path
            )
            print(f"Saved best model to: {save_path}")

    print("Training finished.")
    print(f"Best macro_f1: {best_macro_f1:.4f}")


if __name__ == "__main__":
    main()