import torch
import torch.nn as nn
from transformers import BertTokenizer

from model.simple_tokenizer import SimpleTokenizer as ClipTokenizer
from model.bert_prompt_classifier import BertPromptClassifier


# =========================
# 数据集类别名
# =========================
MIR_LABELS = [
    'river', 'dog', 'food', 'transport', 'plant_life', 'bird', 'people', 'sunset',
    'male', 'baby', 'flower', 'lake', 'animals', 'night', 'clouds', 'sky', 'car',
    'tree', 'water', 'indoor', 'structures', 'sea', 'female', 'portrait'
]

NUS_LABELS = [
    'animal', 'beach', 'buildings', 'clouds', 'flowers', 'grass', 'lake', 'mountain',
    'ocean', 'person', 'plants', 'reflection', 'road', 'rocks', 'sky', 'snow',
    'sunset', 'tree', 'vehicle', 'water', 'window'
]

COCO_LABELS = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush"
]


# =========================
# 类名规范化
# =========================
MIR_LABEL_MAP = {
    'river': 'river',
    'dog': 'dog',
    'food': 'food',
    'transport': 'transportation',
    'plant_life': 'plant life',
    'bird': 'bird',
    'people': 'people',
    'sunset': 'sunset',
    'male': 'man',
    'baby': 'baby',
    'flower': 'flower',
    'lake': 'lake',
    'animals': 'animals',
    'night': 'night scene',
    'clouds': 'clouds',
    'sky': 'sky',
    'car': 'car',
    'tree': 'tree',
    'water': 'water',
    'indoor': 'indoor scene',
    'structures': 'buildings and structures',
    'sea': 'sea',
    'female': 'woman',
    'portrait': 'portrait'
}

NUS_LABEL_MAP = {
    'animal': 'animal',
    'beach': 'beach',
    'buildings': 'buildings',
    'clouds': 'clouds',
    'flowers': 'flowers',
    'grass': 'grass',
    'lake': 'lake',
    'mountain': 'mountain',
    'ocean': 'ocean',
    'person': 'person',
    'plants': 'plants',
    'reflection': 'reflection',
    'road': 'road',
    'rocks': 'rocks',
    'sky': 'sky',
    'snow': 'snow',
    'sunset': 'sunset',
    'tree': 'tree',
    'vehicle': 'vehicle',
    'water': 'water',
    'window': 'window'
}

COCO_LABEL_MAP = {
    "person": "person",
    "bicycle": "bicycle",
    "car": "car",
    "motorcycle": "motorcycle",
    "airplane": "airplane",
    "bus": "bus",
    "train": "train",
    "truck": "truck",
    "boat": "boat",
    "traffic light": "traffic light",
    "fire hydrant": "fire hydrant",
    "stop sign": "stop sign",
    "parking meter": "parking meter",
    "bench": "bench",
    "bird": "bird",
    "cat": "cat",
    "dog": "dog",
    "horse": "horse",
    "sheep": "sheep",
    "cow": "cow",
    "elephant": "elephant",
    "bear": "bear",
    "zebra": "zebra",
    "giraffe": "giraffe",
    "backpack": "backpack",
    "umbrella": "umbrella",
    "handbag": "handbag",
    "tie": "tie",
    "suitcase": "suitcase",
    "frisbee": "frisbee",
    "skis": "skis",
    "snowboard": "snowboard",
    "sports ball": "sports ball",
    "kite": "kite",
    "baseball bat": "baseball bat",
    "baseball glove": "baseball glove",
    "skateboard": "skateboard",
    "surfboard": "surfboard",
    "tennis racket": "tennis racket",
    "bottle": "bottle",
    "wine glass": "wine glass",
    "cup": "cup",
    "fork": "fork",
    "knife": "knife",
    "spoon": "spoon",
    "bowl": "bowl",
    "banana": "banana",
    "apple": "apple",
    "sandwich": "sandwich",
    "orange": "orange",
    "broccoli": "broccoli",
    "carrot": "carrot",
    "hot dog": "hot dog",
    "pizza": "pizza",
    "donut": "donut",
    "cake": "cake",
    "chair": "chair",
    "couch": "couch",
    "potted plant": "potted plant",
    "bed": "bed",
    "dining table": "dining table",
    "toilet": "toilet",
    "tv": "television",
    "laptop": "laptop",
    "mouse": "computer mouse",
    "remote": "remote control",
    "keyboard": "keyboard",
    "cell phone": "mobile phone",
    "microwave": "microwave oven",
    "oven": "oven",
    "toaster": "toaster",
    "sink": "sink",
    "refrigerator": "refrigerator",
    "book": "book",
    "clock": "clock",
    "vase": "vase",
    "scissors": "scissors",
    "teddy bear": "teddy bear",
    "hair drier": "hair dryer",
    "toothbrush": "toothbrush"
}


SPECIAL_TOKEN = {
    "CLS_TOKEN": "<|startoftext|>",
    "SEP_TOKEN": "<|endoftext|>",
    "MASK_TOKEN": "[MASK]",
    "UNK_TOKEN": "[UNK]",
    "PAD_TOKEN": "[PAD]",
}


def get_class_info(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name in ["mir", "mirflickr", "mirflickr25k", "flickr25k"]:
        return MIR_LABELS, MIR_LABEL_MAP
    elif dataset_name in ["nus", "nuswide"]:
        return NUS_LABELS, NUS_LABEL_MAP
    elif dataset_name in ["coco", "mscoco", "ms-coco"]:
        return COCO_LABELS, COCO_LABEL_MAP
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def normalize_label_text(name, label_map=None):
    if label_map is not None and name in label_map:
        return label_map[name]
    return name.replace("_", " ")


def tokenize_clip_texts(texts, clip_tokenizer, maxWords=32):
    """
    texts: list[str]
    return: LongTensor [B, maxWords]
    """
    all_tokens = []

    for text in texts:
        words = clip_tokenizer.tokenize(text)
        words = [SPECIAL_TOKEN["CLS_TOKEN"]] + words

        total_length_with_cls = maxWords - 1
        if len(words) > total_length_with_cls:
            words = words[:total_length_with_cls]

        words = words + [SPECIAL_TOKEN["SEP_TOKEN"]]
        token_ids = clip_tokenizer.convert_tokens_to_ids(words)

        while len(token_ids) < maxWords:
            token_ids.append(0)

        all_tokens.append(token_ids)

    return torch.tensor(all_tokens, dtype=torch.long)


def build_oracle_prompt_texts(label, class_names, label_map=None, topk=None):
    """
    label: [B, C] 的0/1多标签
    return: list[str]
    """
    label = label.detach().cpu()
    prompt_texts = []

    for i in range(label.size(0)):
        true_idx = torch.nonzero(label[i] > 0, as_tuple=False).squeeze(-1).tolist()

        if topk is not None:
            true_idx = true_idx[:topk]

        class_texts = []
        for idx in true_idx:
            name = class_names[idx]
            if label_map is not None and name in label_map:
                name = label_map[name]
            else:
                name = name.replace("_", " ")
            class_texts.append(name)

        prompt_text = "" if len(class_texts) == 0 else " ".join(class_texts)
        prompt_texts.append(prompt_text)

    return prompt_texts


class PromptGenerator(nn.Module):
    """
    输入:
        raw_texts: list[str]

    输出:
        prompt_ids:       [B, maxWords]
        topk_indices:     [B, num_classes]，实际存“选中的类别索引”，不足补 -1
        selected_counts:  [B]
        probs:            [B, C]
        prompt_texts:     list[str]
    """
    def __init__(
        self,
        dataset_name,
        classifier_ckpt,
        bert_path="/home/yuck/bert-base-uncased",
        device="cuda:5",
        maxWords=32,
        prob_threshold=0.7
    ):
        super().__init__()

        self.dataset_name = dataset_name
        self.device = torch.device(device) if isinstance(device, str) else device
        self.maxWords = maxWords

        self.class_names, self.label_map = get_class_info(dataset_name)
        self.num_classes = len(self.class_names)

        self.bert_tokenizer = BertTokenizer.from_pretrained(
            bert_path,
            local_files_only=True
        )

        self.clip_tokenizer = ClipTokenizer()

        self.classifier = BertPromptClassifier(
            num_classes=self.num_classes,
            bert_path=bert_path,
            dropout=0.1
        )

        ckpt = torch.load(classifier_ckpt, map_location="cpu")
        self.classifier.load_state_dict(ckpt["model_state_dict"], strict=True)

        # 优先读取训练时保存的 best_threshold
        if prob_threshold is None:
            self.prob_threshold = ckpt.get("best_threshold", 0.5)
        else:
            self.prob_threshold = prob_threshold

        self.classifier = self.classifier.to(self.device)
        self.classifier.eval()

        # 冻结分类器参数
        for p in self.classifier.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, raw_texts):
        """
        raw_texts: list[str]

        return:
            prompt_ids: [B, maxWords]
            topk_indices: [B, num_classes]，不足补 -1
            selected_counts: [B]
            probs: [B, C]
            prompt_texts: list[str]
        """
        # ===== 1. 清洗文本 =====
        cleaned_texts = []
        empty_mask = []

        for t in raw_texts:
            if t is None:
                t = ""
            elif not isinstance(t, str):
                t = str(t)

            t = t.strip()
            cleaned_texts.append(t)
            empty_mask.append(t == "")

        B = len(cleaned_texts)

        # ===== 2. 初始化输出 =====
        probs = torch.zeros(B, self.num_classes, device=self.device)

        # 虽然名字还叫 topk_indices，但现在不再受 topk 限制
        # 这里存的是所有选中的类别索引，不足补 -1
        topk_indices = torch.full(
            (B, self.num_classes),
            -1,
            dtype=torch.long,
            device=self.device
        )

        selected_counts = torch.zeros(
            B,
            dtype=torch.long,
            device=self.device
        )

        prompt_texts = [""] * B

        # ===== 3. 只对非空文本跑 BERT =====
        non_empty_indices = [i for i, is_empty in enumerate(empty_mask) if not is_empty]

        if len(non_empty_indices) > 0:
            non_empty_texts = [cleaned_texts[i] for i in non_empty_indices]

            enc = self.bert_tokenizer(
                non_empty_texts,
                padding="max_length",
                truncation=True,
                max_length=self.maxWords,
                return_tensors="pt"
            )

            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            logits = self.classifier(input_ids, attention_mask)   # [N, C]
            non_empty_probs = torch.sigmoid(logits)               # [N, C]

            for j, ori_idx in enumerate(non_empty_indices):
                sample_probs = non_empty_probs[j]   # [C]
                probs[ori_idx] = sample_probs

                # ===== 按概率从高到低排序 =====
                sorted_scores, sorted_indices = torch.sort(sample_probs, descending=True)

                # ===== top2保底 =====
                min_keep = min(2, self.num_classes)
                selected = sorted_indices[:min_keep].tolist()

                # ===== threshold补充：剩余类别里，超过阈值的都加进来 =====
                for idx in sorted_indices[min_keep:]:
                    idx_item = idx.item()
                    if sample_probs[idx_item] >= self.prob_threshold:
                        selected.append(idx_item)

                # 去重，防御性处理
                selected = list(dict.fromkeys(selected))

                selected_indices = torch.tensor(
                    selected,
                    dtype=torch.long,
                    device=self.device
                )

                num_selected = selected_indices.numel()
                selected_counts[ori_idx] = num_selected

                if num_selected > 0:
                    topk_indices[ori_idx, :num_selected] = selected_indices

                    idx_list = selected_indices.tolist()
                    class_texts = [
                        normalize_label_text(self.class_names[idx], self.label_map)
                        for idx in idx_list
                    ]
                    prompt_texts[ori_idx] = " ".join(class_texts)
                else:
                    prompt_texts[ori_idx] = ""

        # ===== 4. tokenize 成 CLIP 输入 =====
        prompt_ids = tokenize_clip_texts(
            texts=prompt_texts,
            clip_tokenizer=self.clip_tokenizer,
            maxWords=self.maxWords
        ).to(self.device)

        return prompt_ids, topk_indices, selected_counts, probs, prompt_texts