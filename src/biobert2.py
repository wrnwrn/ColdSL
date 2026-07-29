


# -*- coding: utf-8 -*-
"""
Generate BioBERT entity embeddings aligned to kg.pt's entity_to_index (0..N-1).

Inputs:
  ../data_raw/SLKG2/node_info.tsv   columns: _id, name, _labels, description
  ../data/pan/kg.pt                dict containing "entity_to_index"

Outputs:
  ../data/pan/E_biobert.npy        shape [num_entities, 768]  (row index == entity idx)
  ../data/pan/entity_ids.json      list of entity_id ordered by idx (sanity check)

Notes:
  - Uses mean pooling with attention_mask (PAD excluded).
  - entity_to_index keys and node_info _id are coerced to str for robust matching.
"""

import os
import json
import math
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# ---------------------------
# Config
# ---------------------------
NODE_INFO_TSV = "../data_raw/SLKG2/node_info.tsv"
KG_PT = "../data/pan/kg.pt"

OUT_DIR = "../data/pan"
OUT_NPY = os.path.join(OUT_DIR, "E_biobert.npy")
OUT_ENTITY_IDS = os.path.join(OUT_DIR, "entity_ids.json")

MODEL_PATH = "../data_raw/biobert-v1.1/" # TODO
BATCH_SIZE = 64
MAX_LENGTH = 256 # TODO
NORMALIZE_L2 = True
SAVE_DTYPE = np.float16  # set np.float32 if you prefer


# ---------------------------
# Helpers
# ---------------------------
def _clean_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x).strip()


def build_entity_text(name: str, labels: str, desc: str, entity_id: str) -> str:
    """
    Minimal but robust text template for BioBERT.
    You can refine it later (synonyms, ids, etc.).
    """
    name = _clean_str(name)
    labels = _clean_str(labels)
    desc = _clean_str(desc)
    entity_id = _clean_str(entity_id)

    labels_clean = labels.replace(":", " ").replace("|", " ").replace(";", " ").strip()
    labels_clean = " ".join([w for w in labels_clean.split() if w])

    if labels_clean and name:
        head = f"{labels_clean}: {name}."
    elif name:
        head = f"Entity: {name}."
    else:
        head = f"EntityID: {entity_id}."

    if desc:
        return f"{head} Description: {desc}" # TODO 
    return head


@torch.no_grad()
def biobert_encode_texts(
    texts,
    tokenizer,
    model,
    device,
    batch_size=64,
    max_length=256,
    normalize=True,
):
    """
    Mean pooling excluding PAD using attention_mask.
    Returns np.ndarray [N, hidden_size] float16/float32 depending on SAVE_DTYPE.
    """
    hidden = model.config.hidden_size
    N = len(texts)

    # Use memmap to avoid huge RAM usage
    tmp_path = OUT_NPY + ".mmap"
    emb_mmap = np.memmap(tmp_path, dtype=np.float32, mode="w+", shape=(N, hidden))

    for start in tqdm(range(0, N, batch_size), desc="BioBERT encoding"):
        batch = texts[start : start + batch_size]

        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        out = model(**enc)  # last_hidden_state: [B, L, H]
        token_emb = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).type_as(token_emb)  # [B, L, 1]

        summed = (token_emb * mask).sum(dim=1)  # [B, H]
        counts = mask.sum(dim=1).clamp(min=1e-9)  # [B, 1]
        sent_emb = summed / counts  # [B, H]

        if normalize:
            sent_emb = torch.nn.functional.normalize(sent_emb, p=2, dim=1)

        sent_emb = sent_emb.cpu().numpy().astype(np.float32)
        emb_mmap[start : start + sent_emb.shape[0], :] = sent_emb

    emb_mmap.flush()
    E = np.array(emb_mmap, dtype=SAVE_DTYPE, copy=True)
    del emb_mmap
    try:
        os.remove(tmp_path)
    except OSError:
        pass
    return E


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) load kg.pt
    kg = torch.load(KG_PT, map_location="cpu")
    if "entity_to_index" not in kg:
        raise KeyError(f'kg.pt missing key "entity_to_index". Keys: {list(kg.keys())}')

    # Ensure keys are str and idx are int
    entity_to_index = {str(k): int(v) for k, v in kg["entity_to_index"].items()}
    num_entities = len(entity_to_index)
    print(f"[INFO] num_entities in KG: {num_entities}")

    # Build index -> entity_id (ordered)
    index_to_entity = [None] * num_entities
    for eid, idx in entity_to_index.items():
        if idx < 0 or idx >= num_entities:
            raise ValueError(f"Invalid idx {idx} for entity {eid}")
        index_to_entity[idx] = eid

    # 2) load node_info.tsv
    df = pd.read_csv(NODE_INFO_TSV, sep="\t", dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]  # defensive

    for c in ["_id", "name", "_labels", "description"]:
        if c not in df.columns:
            raise KeyError(f"node_info.tsv missing column {c}. Got columns: {list(df.columns)}")

    df["_id"] = df["_id"].astype(str)
    df = df.drop_duplicates(subset=["_id"], keep="first").set_index("_id")
    print(f"[INFO] node_info rows (after dedup): {len(df)}")

    # 3) build texts aligned to idx
    texts = [None] * num_entities
    hit = 0
    fallback = 0

    for idx, eid in enumerate(index_to_entity):
        if eid is None:
            # rare but handle
            texts[idx] = "Entity: UNKNOWN."
            fallback += 1
            continue

        if eid in df.index:
            r = df.loc[eid]
            texts[idx] = build_entity_text(
                name=r["name"],
                labels=r["_labels"],
                desc=r["description"],
                entity_id=eid,
            )
            hit += 1
        else:
            # If node_info is missing this entity, keep minimal text
            texts[idx] = f"EntityID: {eid}."
            fallback += 1

    print(f"[INFO] matched entities (KG ∩ node_info): {hit}")
    print(f"[INFO] fallback used (missing in node_info): {fallback}")

    # 4) load BioBERT
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True,trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_PATH).to(device).eval()

    # 5) encode
    E = biobert_encode_texts(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        normalize=NORMALIZE_L2,
    )
    print(f"[INFO] E shape: {E.shape}, dtype={E.dtype}")

    # 6) save
    np.save(OUT_NPY, E)
    with open(OUT_ENTITY_IDS, "w", encoding="utf-8") as f:
        json.dump(index_to_entity, f, ensure_ascii=False)

    # quick sanity: first few
    print("[INFO] first 3 entity texts:")
    for i in range(min(3, num_entities)):
        print(f"  idx={i} id={index_to_entity[i]} text={texts[i][:120]}...")

    print(f"[OK] saved embeddings to: {OUT_NPY}")
    print(f"[OK] saved entity id order to: {OUT_ENTITY_IDS}")


if __name__ == "__main__":
    main()
