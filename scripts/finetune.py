import pickle
import torch
import torch.nn.functional as F
from torch.optim import Adam
from src.model.text_augmentation import augment_text_labels, average_augmented_embeddings

import autorootcwd
from src.model.model import FrozenInTime, compute_similarity
from src.data.data_loader import TextVideoDataLoader
import torch.optim as optim
import torch
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_loader = TextVideoDataLoader(
    dataset_name= "NTU",
    text_params= {"input": "text",},
    video_params= {"input_res": 224, "num_frames": 4,},
    data_dir= "data/nturgbd",
    metadata_dir= "data/nturgbd",
    split= 'train',
    tsfm_params= None,
    tsfm_split= None,
    cut= None,
    subsample= 1,
    sliding_window_stride= -1,
    reader= 'decord',
    batch_size=32,
    num_workers=1,
    shuffle=True,
)

model = FrozenInTime(
    video_params={"model": "SpaceTimeTransformer", "num_frames": 4, "arch_config":"base_patch16_224", "vit_init": "imagenet-21k", "attention_style":"frozen-in-time", "pretrained":True},
    text_params={"model": "distilbert-base-uncased", "pretrained": True},
    projection_dim=256,
).to(device)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

batch = next(iter(data_loader))
video_data = batch['video'].to(device)
text_data = tokenizer(
    batch['text'],
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=tokenizer.model_max_length
).to(device)

# 1) backbone freeze
for p in model.video_model.parameters(): p.requires_grad = False
for p in model.text_model.parameters():  p.requires_grad = False

# 2) 사전 계산: 모든 라벨에 대해 증강 텍스트 → 임베딩 → 평균
# dataset.metadata['raw_captions'] 에 모든 캡션(라벨) 저장돼 있다고 가정
unique_labels = sorted(set(data_loader.dataset.metadata['raw_captions']))
aug_data       = augment_text_labels(unique_labels)
augmented_texts, label_groups = aug_data['augmented_texts'], aug_data['label_groups']

all_aug_embeds = []
model.eval()
with torch.no_grad():
    # 한번에 너무 많이 넣으면 OOM 날 수 있으니 쪼개서
    chunk = 32
    for i in range(0, len(augmented_texts), chunk):
        batch_txts = augmented_texts[i:i+chunk]
        toks = tokenizer(batch_txts, return_tensors="pt",
                         padding=True, truncation=True,
                         max_length=tokenizer.model_max_length).to(device)
        emb = model.compute_text(toks)           # [chunk, dim]
        all_aug_embeds.append(emb.cpu())
all_aug_embeds = torch.cat(all_aug_embeds, dim=0).to(device)  # [num_templates, dim]

averaged_embeds = average_augmented_embeddings(all_aug_embeds, label_groups)
# averaged_embeds: [num_labels, dim]

# 라벨→인덱스 매핑
label_to_idx = {lbl:i for i,lbl in enumerate(unique_labels)}

# 3) optimizer: head만
optimizer = Adam(
    list(model.vid_proj.parameters()) +
    list(model.txt_proj.parameters()),
    lr=1e-5
)

# 4) fine-tuning loop
num_epochs   = 10
log_interval = 100
temperature  = 0.07

model.train()
for epoch in range(1, num_epochs+1):
    running_loss = 0.0
    epoch_loss   = 0.0
    n_batches    = 0

    for batch_idx, batch in enumerate(data_loader, start=1):
        video_data = batch["video"].to(device)
        # 원래 batch['text'] 가 raw caption 문자열
        batch_labels = batch['text']
        idxs = [label_to_idx[lbl] for lbl in batch_labels]
        t_emb = averaged_embeds[idxs]           # [B, dim]

        v_emb = model.compute_video(video_data) # [B, dim]

        # contrastive InfoNCE
        logits, _ = compute_similarity(v_emb, t_emb)
        logits = logits / temperature
        labels = torch.arange(len(v_emb), device=device)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epoch_loss   += loss.item()
        n_batches    += 1

        if batch_idx % log_interval == 0:
            print(f"[Epoch {epoch}/{num_epochs}  Batch {batch_idx}]"
                  f"  batch-loss = {running_loss/log_interval:.4f}")
            running_loss = 0.0

    avg_epoch_loss = epoch_loss / n_batches
    print(f"--- Epoch {epoch}/{num_epochs} avg loss = {avg_epoch_loss:.4f} ---")

    # 5) 체크포인트 저장
    ckpt = {"state_dict": model.state_dict()}
    with open(f"finetuned_epoch{epoch}.pkl", "wb") as f:
        pickle.dump(ckpt, f)
    print(f"Saved checkpoint to finetuned_epoch{epoch}.pkl\n")
