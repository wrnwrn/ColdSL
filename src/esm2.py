import os
import csv
import json
from typing import Dict, Tuple, Union, List, Optional

import numpy as np
import torch
from tqdm import tqdm

# ---------------------------
# Config (default)
# ---------------------------
SEQUENCE_CSV = "../data/pan/sequence.csv"
GENE_PT = "../data/pan/gene.pt"

OUT_DIR = "../data/pan"
OUT_NPY = os.path.join(OUT_DIR, "E_esm2_600.npy")
OUT_GENE_IDS = os.path.join(OUT_DIR, "gene_ids.json")

ESM2_MODEL_DIR = "../data_raw/esm2_t33_650M_UR50D"  # Hugging Face local dir or model id
POOLING = "mean"
BATCH_SIZE = 2048
USE_FP16 = True
MAX_RESIDUES_PER_BATCH = 20000
MAX_SEQ_LEN = 1000
LONG_SEQ_OVERLAP = 200
FILL_MISSING = "zero"
SAVE_DTYPE = np.float16


def _load_gene_idx_to_name(gene_idx_to_name: Union[str, Dict[int, str], List[str], Tuple[str, ...]]):
    """
    Load the gene-index-to-name mapping.

    Supported inputs:
        - A dict, list, or tuple supplied directly.
        - A string path to an object loaded with torch.load.

    Returns:
        names: A list of gene names whose indices correspond to gene_idx.
    """
    if isinstance(gene_idx_to_name, str):
        obj = torch.load(gene_idx_to_name, weights_only=False)
    else:
        obj = gene_idx_to_name

    if isinstance(obj, dict):
        
        max_idx = max(obj.keys()) if len(obj) > 0 else -1
        names = [None] * (max_idx + 1)
        for k, v in obj.items():
            names[int(k)] = str(v)
        
        for i in range(len(names)):
            if names[i] is None:
                names[i] = f"<UNK_{i}>"
        return names

    if isinstance(obj, (list, tuple)):
        return [str(x) for x in obj]

    raise TypeError(f"Unsupported type for gene_idx_to_name: {type(obj)}")


def _read_sequence_csv(sequence_csv_path: str) -> Dict[str, str]:
    """
    Read a CSV file containing the columns Hugo_Symbol and Sequence.

    Returns:
        A mapping from gene names to protein sequences.
    """
    seq_map: Dict[str, str] = {}
    with open(sequence_csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert "Hugo_Symbol" in reader.fieldnames and "Sequence" in reader.fieldnames, \
            f"CSV header must contain Hugo_Symbol and Sequence, got {reader.fieldnames}"
        for row in reader:
            name = (row["Hugo_Symbol"] or "").strip()
            seq = (row["Sequence"] or "").strip()
            if not name or not seq:
                continue
            seq_map[name] = seq
    return seq_map


def _clean_protein_sequence(seq: str) -> str:
    """
    Normalize a protein sequence for ESM-2.

    Lowercase letters are converted to uppercase, and unsupported characters are replaced with "X".
    """
    seq = seq.strip().upper()
    
    allowed = set(list("ACDEFGHIKLMNPQRSTVWYBXZJUO"))  
    cleaned = []
    for ch in seq:
        if "A" <= ch <= "Z" and ch in allowed:
            cleaned.append(ch)
        elif "A" <= ch <= "Z":
            cleaned.append("X")
        else:
            
            continue
    return "".join(cleaned)


def _chunk_sequence(seq: str, max_len: int, overlap: int) -> List[str]:
    """
    Split a long sequence into overlapping windows for subsequent embedding aggregation.

    Args:
        max_len: Length of each window, excluding special tokens.
        overlap: Number of residues shared by adjacent windows.
    """
    if len(seq) <= max_len:
        return [seq]
    assert max_len > overlap >= 0
    chunks = []
    step = max_len - overlap
    for start in range(0, len(seq), step):
        chunk = seq[start:start + max_len]
        if len(chunk) < 1:
            break
        chunks.append(chunk)
        if start + max_len >= len(seq):
            break
    return chunks


@torch.no_grad()
def build_esm2_gene_features(
    sequence_csv_path: str,
    gene_idx_to_name: Union[str, Dict[int, str], List[str], Tuple[str, ...]],
    out_path: Optional[str] = None,
    model_dir_or_id: str = "facebook/esm2_t12_35M_UR50D",
    repr_layer: Optional[int] = None,
    pooling: str = "mean",
    batch_size: int = 128,
    device: Optional[str] = None,
    use_fp16: bool = True,
    max_residues_per_batch: int = 20000,
    max_seq_len: int = 1000,
    long_seq_overlap: int = 200,
    fill_missing: str = "zero",
    local_files_only: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    
    
    idx2name = _load_gene_idx_to_name(gene_idx_to_name)
    seq_map_raw = _read_sequence_csv(sequence_csv_path)
    seq_map = {k: _clean_protein_sequence(v) for k, v in seq_map_raw.items()}

    
    from transformers import AutoModel, AutoTokenizer
    print(f"[INFO] Loading model from {model_dir_or_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir_or_id, local_files_only=local_files_only, do_lower_case=False)
    model = AutoModel.from_pretrained(model_dir_or_id, local_files_only=local_files_only)
    model.eval()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    if repr_layer is None:
        repr_layer = getattr(model.config, "num_hidden_layers", 33)

    embed_dim = model.config.hidden_size
    num_genes = len(idx2name)

    
    gene_feat = torch.zeros((num_genes, embed_dim), dtype=torch.float32)
    agg_cnt = torch.zeros((num_genes,), dtype=torch.float32) 
    has_seq_mask = torch.zeros((num_genes,), dtype=torch.bool)

    
    
    all_tasks = []
    print("[INFO] Pre-processing and chunking sequences...")
    for idx, name in enumerate(idx2name):
        if name in seq_map and len(seq_map[name]) > 0:
            full_seq = seq_map[name]
            
            chunks = _chunk_sequence(full_seq, max_len=max_seq_len, overlap=long_seq_overlap)
            for chunk in chunks:
                all_tasks.append((idx, chunk))
            has_seq_mask[idx] = True 
        else:
            
            pass 

    if fill_missing == "random":
        gene_feat.normal_(mean=0.0, std=0.02)
    # create default zero is implied by torch.zeros

    print(f"[INFO] Total chunks to process: {len(all_tasks)}")

    
    def process_batch_tensors(batch_tasks: List[Tuple[int, str]]):
        """
        Encode a batch of sequence windows.

        Args:
            batch_items: List of (gidx, sequence) tuples.

        Returns:
            A list of (gidx, embedding_vector) tuples.
        """
        seqs = [seq for (_, seq) in batch_tasks]
        gidxs = [gidx for (gidx, _) in batch_tasks]

        inputs = tokenizer(seqs, return_tensors="pt", padding=True, truncation=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # AutoCast
        ctx = torch.cuda.amp.autocast() if (device.startswith("cuda") and use_fp16) else torch.no_grad()
        
        with ctx:
            out = model(**inputs, output_hidden_states=True)
            
            if repr_layer == model.config.num_hidden_layers:
                reps = out.last_hidden_state
            else:
                
                
                reps = out.hidden_states[repr_layer]

        # Pooling
        reps = reps.float() # [B, L, D]
        attn_mask = inputs["attention_mask"] # [B, L] 1 for token, 0 for pad
        
        
        # ESM tokenizer: [CLS] seq [EOS] [PAD]
        
        
        
        results = []
        for b in range(len(batch_tasks)):
            
            valid_len = attn_mask[b].sum().item()
            
            
            if valid_len <= 2:
                
                vec = torch.zeros(embed_dim)
            else:
                
                residue_reps = reps[b, 1 : valid_len - 1] 
                
                if pooling == "mean":
                    vec = residue_reps.mean(dim=0)
                elif pooling == "cls":
                    vec = reps[b, 0] # CLS token
                else:
                    raise ValueError(f"Unknown pooling: {pooling}")
            
            results.append((gidxs[b], vec.detach().cpu()))
            
        return results

    
    
    cursor = 0
    pbar = tqdm(total=len(all_tasks), desc="Embedding")
    
    while cursor < len(all_tasks):
        batch_tasks = []
        current_residues = 0
        
        
        while cursor < len(all_tasks) and len(batch_tasks) < batch_size:
            idx, seq = all_tasks[cursor]
            seq_len = len(seq)
            
            
            if len(batch_tasks) > 0 and (current_residues + seq_len) > max_residues_per_batch:
                break
            
            batch_tasks.append((idx, seq))
            current_residues += seq_len
            cursor += 1
        
        
        batch_results = process_batch_tensors(batch_tasks)
        
        
        for gidx, vec in batch_results:
            gene_feat[gidx] += vec
            agg_cnt[gidx] += 1.0
            
        pbar.update(len(batch_tasks))
    
    pbar.close()

    
    
    
    mask_indices = torch.where(agg_cnt > 0)[0]
    # gene_feat[mask_indices] /= agg_cnt[mask_indices].unsqueeze(1) # shape broadcasting
    
    for idx in mask_indices:
        gene_feat[idx] = gene_feat[idx] / agg_cnt[idx]

    meta = {"embed_dim": embed_dim, "repr_layer": repr_layer, "variant_id": model_dir_or_id}
    
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(
            {"gene_feat": gene_feat, "has_seq_mask": has_seq_mask, "meta": meta},
            out_path
        )

    return gene_feat, has_seq_mask, meta


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    gene_ids = _load_gene_idx_to_name(GENE_PT)
    num_genes = len(gene_ids)
    print(f"[INFO] num_genes: {num_genes}")

    gene_feat, mask, meta = build_esm2_gene_features(
        sequence_csv_path=SEQUENCE_CSV,
        gene_idx_to_name=gene_ids,
        out_path=None,
        model_dir_or_id=ESM2_MODEL_DIR,
        pooling=POOLING,
        batch_size=BATCH_SIZE,
        use_fp16=USE_FP16,
        max_residues_per_batch=MAX_RESIDUES_PER_BATCH,
        max_seq_len=MAX_SEQ_LEN,
        long_seq_overlap=LONG_SEQ_OVERLAP,
        fill_missing=FILL_MISSING,
        local_files_only=True,
    )

    hit = int(mask.sum().item())
    print(f"[INFO] sequences found: {hit} / {num_genes}")
    print(f"[INFO] embed_dim: {meta.get('embed_dim')}, repr_layer: {meta.get('repr_layer')}")

    E = gene_feat.cpu().numpy().astype(SAVE_DTYPE, copy=True)
    np.save(OUT_NPY, E)
    with open(OUT_GENE_IDS, "w", encoding="utf-8") as f:
        json.dump(gene_ids, f, ensure_ascii=False)

    print(f"[OK] saved embeddings to: {OUT_NPY}")
    print(f"[OK] saved gene id order to: {OUT_GENE_IDS}")