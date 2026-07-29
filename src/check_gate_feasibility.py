#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import MultiModalDataset, SLDataset
from model import UMTModel
from util.my import load_args_from_json, rank_main_print, set_seed


def _load_config(path: str) -> Dict[str, Any]:
    if path and os.path.isfile(path):
        return vars(load_args_from_json(path))
    return {}


def _build_parser(defaults: Dict[str, Any]) -> argparse.ArgumentParser:
    """
    python check_gate_feasibility.py   --seq_ckpt_path ../result/mlp_ln_pan_cv_3_fold_2_only_seq/checkpoint.pth   --kg_ckpt_path ../result/4_omics_into_onlyKG_with_BioBERT/kg_biobert_pan_cv_3_fold_2_only_kg_kg_experiment_C/checkpoint.pth   --output_path gate_features.npz
    """
    d = defaults or {}
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--seq_ckpt_path", type=str, required=True)
    parser.add_argument("--kg_ckpt_path", type=str, required=True)
    parser.add_argument("--seq_config_path", type=str, default="")
    parser.add_argument("--kg_config_path", type=str, default="")

    parser.add_argument("--output_path", type=str, default="gate_features.npz")
    parser.add_argument("--feature_mode", type=str, default="logits_plus_emb", choices=["logits_only", "logits_plus_emb"])
    parser.add_argument("--batch_size", type=int, default=int(d.get("batch_size", 2048)))
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # data
    parser.add_argument("--cancer_type", type=str, default=d.get("cancer_type", "pan"))
    parser.add_argument("--metric", type=int, default=int(d.get("metric", 3)))
    parser.add_argument("--train_fold", type=int, default=int(d.get("train_fold", 2)))
    parser.add_argument("--omics_types", nargs="+", type=str, default=d.get("omics_types", ["cna", "exp", "mut"]))

    # model params (must align with checkpoints)
    parser.add_argument("--hid_dim", type=int, default=int(d.get("hid_dim", 128)))
    parser.add_argument("--vae_hidden_dims", nargs="+", type=int, default=d.get("vae_hidden_dims", [2048, 1024, 512, 256]))
    parser.add_argument("--vae_dropout", type=float, default=float(d.get("vae_dropout", 0.2)))

    parser.add_argument("--in_channels", type=int, default=int(d.get("in_channels", 128)))
    parser.add_argument("--hidden_channels", type=int, default=int(d.get("hidden_channels", 256)))
    parser.add_argument("--gcn_layers", type=int, default=int(d.get("gcn_layers", 2)))
    parser.add_argument("--graph_dropout", type=float, default=float(d.get("graph_dropout", 0.5)))
    parser.add_argument("--kg_experiment", type=str, default=d.get("kg_experiment", "C"))
    parser.add_argument("--biobert_embedding_path", type=str, default=d.get("biobert_embedding_path", "../data/pan/E_biobert.npy"))
    parser.add_argument("--seq_embedding_path", type=str, default=d.get("seq_embedding_path", "../data/pan/E_esm2_600.npy"))
    parser.add_argument("--p_rel", type=float, default=0.3, help="relation dropout prob for KG subgraph")
    parser.add_argument("--kg_biobert_fusion_type", type=int, default=3, choices=[0, 1, 2, 3], help="0: only use RGCN output, 1: concat fusion, 2: residual sum, 3: residual gate")
    parser.add_argument("--kg_node_type_emb_switch", type=bool, default=True, help="only useful when kg_biobert_fusion_type choose 3")
    parser.add_argument("--kg_node_type_emb_dim", type=int, default=32)

    parser.add_argument("--gene_final_dim", type=int, default=int(d.get("gene_final_dim", 128)))
    parser.add_argument("--final_mlp_dropout", type=float, default=float(d.get("final_mlp_dropout", 0.5)))
    parser.add_argument("--seq_encoder_mlp_dropout", type=float, default=0.3)

    return parser


def _auto_config_path(ckpt_path: str) -> str:
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    candidate = os.path.join(ckpt_dir, "hyper_parameters.json")
    return candidate if os.path.isfile(candidate) else ""


def _warn_if_mismatch(seq_cfg: Dict[str, Any], kg_cfg: Dict[str, Any], keys) -> None:
    for key in keys:
        if key in seq_cfg and key in kg_cfg and seq_cfg[key] != kg_cfg[key]:
            print(f"[warn] config mismatch for {key}: seq={seq_cfg[key]} vs kg={kg_cfg[key]}")


def _load_ckpt_model(model: UMTModel, ckpt_path: str, device: torch.device) -> None:
    print(f"loading ckpt from {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=True, map_location=device)
    model_dicts = ckpt["model"]
    UMTModel.load_state_dicts(model, model_dicts)


def _collect_features(
    dataloader: DataLoader,
    model: UMTModel,
    kg_graph: Any,
    loss_fn: nn.Module,
    device: torch.device,
    feature_mode: str,
) -> Dict[str, np.ndarray]:
    all_features = []
    all_gate_labels = []
    all_labels = []
    all_meta = []
    all_logit_seq = []
    all_logit_kg = []
    all_loss_seq = []
    all_loss_kg = []

    with torch.no_grad():
        for batch in dataloader:
            gene_1_gene_entity = batch["gene_1_gene_entity"].to(device)
            gene_2_gene_entity = batch["gene_2_gene_entity"].to(device)
            gene_1_gene_idx = batch["gene_1_gene_idx"].to(device)
            gene_2_gene_idx = batch["gene_2_gene_idx"].to(device)
            label = batch["label"].to(device).view(-1)

            kg1 = model.pretrain_kg_encoder({"gene_entity": gene_1_gene_entity, "kg_graph": kg_graph})
            kg2 = model.pretrain_kg_encoder({"gene_entity": gene_2_gene_entity, "kg_graph": kg_graph})
            logit_kg = model.kg_classifier(kg1, kg2).view(-1)

            s1 = model.seq_encoder({"gene_idx": gene_1_gene_idx})
            s2 = model.seq_encoder({"gene_idx": gene_2_gene_idx})
            logit_seq = model.seq_classifier(s1, s2).view(-1)

            loss_kg = loss_fn(logit_kg, label)
            loss_seq = loss_fn(logit_seq, label)
            gate_label = (loss_seq < loss_kg).float()

            if feature_mode == "logits_only":
                features = torch.stack([logit_seq, logit_kg], dim=1)
            else:
                emb_seq_pair = torch.cat([s1, s2], dim=1)
                emb_kg_pair = torch.cat([kg1, kg2], dim=1)
                features = torch.cat(
                    [logit_seq.unsqueeze(1), logit_kg.unsqueeze(1), emb_seq_pair, emb_kg_pair],
                    dim=1,
                )

            all_features.append(features.cpu())
            all_gate_labels.append(gate_label.cpu())
            all_labels.append(label.cpu())
            all_logit_seq.append(logit_seq.cpu())
            all_logit_kg.append(logit_kg.cpu())
            all_loss_seq.append(loss_seq.cpu())
            all_loss_kg.append(loss_kg.cpu())
            all_meta.append(
                torch.stack([gene_1_gene_idx.view(-1), gene_2_gene_idx.view(-1)], dim=1).cpu()
            )

    return {
        "X": torch.cat(all_features, dim=0).numpy(),
        "y": torch.cat(all_gate_labels, dim=0).numpy(),
        "labels": torch.cat(all_labels, dim=0).numpy(),
        "gene_pair_idx": torch.cat(all_meta, dim=0).numpy(),
        "logit_seq": torch.cat(all_logit_seq, dim=0).numpy(),
        "logit_kg": torch.cat(all_logit_kg, dim=0).numpy(),
        "loss_seq": torch.cat(all_loss_seq, dim=0).numpy(),
        "loss_kg": torch.cat(all_loss_kg, dim=0).numpy(),
    }


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--seq_ckpt_path", type=str, required=True)
    pre_parser.add_argument("--kg_ckpt_path", type=str, required=True)
    pre_parser.add_argument("--seq_config_path", type=str, default="")
    pre_parser.add_argument("--kg_config_path", type=str, default="")
    pre_args, _ = pre_parser.parse_known_args()

    if not pre_args.seq_config_path:
        pre_args.seq_config_path = _auto_config_path(pre_args.seq_ckpt_path)
    if not pre_args.kg_config_path:
        pre_args.kg_config_path = _auto_config_path(pre_args.kg_ckpt_path)

    seq_cfg = _load_config(pre_args.seq_config_path)
    kg_cfg = _load_config(pre_args.kg_config_path)
    _warn_if_mismatch(
        seq_cfg,
        kg_cfg,
        keys=["hid_dim", "gene_final_dim", "in_channels", "hidden_channels", "gcn_layers", "graph_dropout"],
    )

    defaults = dict(seq_cfg)
    for key in ["kg_experiment", "biobert_embedding_path", "in_channels", "hidden_channels", "gcn_layers", "graph_dropout",
                "p_rel","kg_biobert_fusion_type","kg_node_type_emb_switch","kg_node_type_emb_dim"
                ]:

        if key in kg_cfg:
            defaults[key] = kg_cfg[key]

    parser = _build_parser(defaults)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    class DummyAccelerator:
        def __init__(self):
            self.local_process_index = 0
            self.is_local_main_process = True

    dummy_acc = DummyAccelerator()

    rank_main_print(dummy_acc, f"seq_ckpt: {args.seq_ckpt_path}")
    rank_main_print(dummy_acc, f"kg_ckpt: {args.kg_ckpt_path}")

    data_base_path = f"../data/{args.cancer_type}"
    mm_dataset = MultiModalDataset(
        accelerator=dummy_acc,
        gene_path=os.path.join(data_base_path, "gene.pt"),
        omics_path_dict={omics_type: f"{data_base_path}/{omics_type}.npy" for omics_type in args.omics_types},
        kg_graph_path=os.path.join(data_base_path, "kg.pt"),
    )
    sl_train_dataset = SLDataset(
        accelerator=dummy_acc,
        sl_path=f"../data/{args.cancer_type}/cv_{args.metric}_fold_{args.train_fold}/train_sl.npy",
        mm_dataset=mm_dataset,
    )
    sl_val_dataset = SLDataset(
        accelerator=dummy_acc,
        sl_path=f"../data/{args.cancer_type}/cv_{args.metric}_fold_{args.train_fold}/val_sl.npy",
        mm_dataset=mm_dataset,
    )
    sl_test_dataset = SLDataset(
        accelerator=dummy_acc,
        sl_path=f"../data/{args.cancer_type}/cv_{args.metric}_fold_{args.train_fold}/test_sl.npy",
        mm_dataset=mm_dataset,
    )

    kg_experiment = args.kg_experiment.upper()
    use_graph = kg_experiment in {"A", "C"}
    entity_init = "random" if kg_experiment in {"A", "B"} else "biobert"
    biobert_embeddings = None
    if entity_init == "biobert":
        biobert_embeddings = torch.from_numpy(np.load(args.biobert_embedding_path)).float()
        if biobert_embeddings.shape[0] != mm_dataset.get_entity_vocab_size():
            raise ValueError(
                f"BioBERT embeddings row count {biobert_embeddings.shape[0]} "
                f"does not match entity_vocab_size {mm_dataset.get_entity_vocab_size()}"
            )

    seq_embeddings = torch.from_numpy(np.load(args.seq_embedding_path)).float()
    if seq_embeddings.shape[0] != len(mm_dataset):
        raise ValueError(
            f"ESM2 embeddings row count {seq_embeddings.shape[0]} "
            f"does not match gene count {len(mm_dataset)}"
        )

    omics_encoder_params = {
        "omics_count": len(args.omics_types),
        "omics_input_dim": mm_dataset.get_omics_input_dim(),
        "vae_hidden_dims": args.vae_hidden_dims,
        "vae_dropout": args.vae_dropout,
        "hid_dim": args.hid_dim,
    }
    kg_encoder_params = {
        "entity_vocab_size": mm_dataset.get_entity_vocab_size(),
        "hid_dim": args.hid_dim,
        "in_channels": args.in_channels,
        "hidden_channels": args.hidden_channels,
        "gcn_layers": args.gcn_layers,
        "graph_dropout": args.graph_dropout,
        "num_relations": mm_dataset.get_kg_relations_count(),
        "entity_init": entity_init,
        "use_graph": use_graph,
        "biobert_embeddings": biobert_embeddings,
        "num_node_types": int(mm_dataset.kg_graph.node_type.max().item() + 1),
        "p_rel": args.p_rel,
        "kg_biobert_fusion_type": args.kg_biobert_fusion_type,
        "kg_node_type_emb_switch":args.kg_node_type_emb_switch,
        "kg_node_type_emb_dim":args.kg_node_type_emb_dim,
    }
    seq_encoder_params = {"seq_embeddings": seq_embeddings, "gene_final_dim": args.gene_final_dim,
                                      "seq_encoder_mlp_dropout":args.seq_encoder_mlp_dropout,
                          }

    model = UMTModel(
        omics_encoder_params=omics_encoder_params,
        kg_encoder_params=kg_encoder_params,
        gene_final_dim=args.gene_final_dim,
        final_mlp_dropout=args.final_mlp_dropout,
        task_type = 'default', 
        seq_encoder_params=seq_encoder_params,
    ).to(device)

    _load_ckpt_model(model, args.seq_ckpt_path, device)
    _load_ckpt_model(model, args.kg_ckpt_path, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    train_loader = DataLoader(sl_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(sl_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(sl_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    kg_graph = mm_dataset.get_kg_graph()

    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    train_data = _collect_features(
        dataloader=train_loader,
        model=model,
        kg_graph=kg_graph,
        loss_fn=loss_fn,
        device=device,
        feature_mode=args.feature_mode,
    )
    val_data = _collect_features(
        dataloader=val_loader,
        model=model,
        kg_graph=kg_graph,
        loss_fn=loss_fn,
        device=device,
        feature_mode=args.feature_mode,
    )
    test_data = _collect_features(
        dataloader=test_loader,
        model=model,
        kg_graph=kg_graph,
        loss_fn=loss_fn,
        device=device,
        feature_mode=args.feature_mode,
    )

    np.savez(
        args.output_path,
        X=train_data["X"],
        y=train_data["y"],
        labels=train_data["labels"],
        gene_pair_idx=train_data["gene_pair_idx"],
        logit_seq=train_data["logit_seq"],
        logit_kg=train_data["logit_kg"],
        loss_seq=train_data["loss_seq"],
        loss_kg=train_data["loss_kg"],
        val_X=val_data["X"],
        val_y=val_data["y"],
        val_labels=val_data["labels"],
        val_gene_pair_idx=val_data["gene_pair_idx"],
        val_logit_seq=val_data["logit_seq"],
        val_logit_kg=val_data["logit_kg"],
        val_loss_seq=val_data["loss_seq"],
        val_loss_kg=val_data["loss_kg"],
        test_X=test_data["X"],
        test_y=test_data["y"],
        test_labels=test_data["labels"],
        test_gene_pair_idx=test_data["gene_pair_idx"],
        test_logit_seq=test_data["logit_seq"],
        test_logit_kg=test_data["logit_kg"],
        test_loss_seq=test_data["loss_seq"],
        test_loss_kg=test_data["loss_kg"],
        feature_mode=args.feature_mode,
    )

    pos_rate = float(train_data["y"].mean()) if train_data["y"].size else 0.0
    print(f"saved features to {args.output_path}, N={train_data['y'].size}, gate_pos_rate={pos_rate:.4f}")

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score, accuracy_score

        def _sigmoid(x: np.ndarray) -> np.ndarray:
            return 1.0 / (1.0 + np.exp(-x))

        def _eval_gate_selection(split_name: str, data: Dict[str, np.ndarray], gate_prob: np.ndarray) -> None:
            
            from executor import evaluate_performance
            
            
            gate_true = data["y"]
            gate_pred = (gate_prob >= 0.5).astype(np.float32)
            gate_auc = roc_auc_score(gate_true, gate_prob)
            gate_acc = accuracy_score(gate_true, gate_pred)

            
            logit_sel = np.where(gate_pred > 0.5, data["logit_seq"], data["logit_kg"])
            logit_oracle = np.where(gate_true > 0.5, data["logit_seq"], data["logit_kg"])
            
            prob_seq = _sigmoid(data["logit_seq"])
            prob_kg = _sigmoid(data["logit_kg"])
            prob_sel = _sigmoid(logit_sel)
            prob_oracle = _sigmoid(logit_oracle)

            # --- Soft Selection (MoE) ---
            prob_soft = gate_prob * prob_seq + (1.0 - gate_prob) * prob_kg

            
            y_true = data["labels"]
            
            perf_seq = evaluate_performance(y_true, prob_seq)
            perf_kg = evaluate_performance(y_true, prob_kg)
            perf_sel = evaluate_performance(y_true, prob_sel)
            perf_soft = evaluate_performance(y_true, prob_soft)
            perf_oracle = evaluate_performance(y_true, prob_oracle)

            
            acc_sel = accuracy_score(y_true, (prob_sel >= 0.5).astype(np.float32))
            acc_seq = accuracy_score(y_true, (prob_seq >= 0.5).astype(np.float32))
            acc_kg = accuracy_score(y_true, (prob_kg >= 0.5).astype(np.float32))
            acc_soft = accuracy_score(y_true, (prob_soft >= 0.5).astype(np.float32))
            acc_oracle = accuracy_score(y_true, (prob_oracle >= 0.5).astype(np.float32))

            
            print(
                f"[{split_name}] gate AUC: {gate_auc:.4f}, acc: {gate_acc:.4f} | \n"
                f"  Seq    -> AUC: {perf_seq['AUC']:.4f}, AUPR: {perf_seq['AUPR']:.4f}, F1: {perf_seq['F1']:.4f}, acc: {acc_seq:.4f} | \n"
                f"  KG     -> AUC: {perf_kg['AUC']:.4f}, AUPR: {perf_kg['AUPR']:.4f}, F1: {perf_kg['F1']:.4f}, acc: {acc_kg:.4f} | \n"
                f"  Select -> AUC: {perf_sel['AUC']:.4f}, AUPR: {perf_sel['AUPR']:.4f}, F1: {perf_sel['F1']:.4f}, acc: {acc_sel:.4f} | \n"
                f"  Soft   -> AUC: {perf_soft['AUC']:.4f}, AUPR: {perf_soft['AUPR']:.4f}, F1: {perf_soft['F1']:.4f}, acc: {acc_soft:.4f} (Real MoE) | \n"
                f"  Oracle -> AUC: {perf_oracle['AUC']:.4f}, AUPR: {perf_oracle['AUPR']:.4f}, F1: {perf_oracle['F1']:.4f}, acc: {acc_oracle:.4f}"
            )

        clf = LogisticRegression(max_iter=1000)
        clf.fit(train_data["X"], train_data["y"])

        val_gate_prob = clf.predict_proba(val_data["X"])[:, 1]
        test_gate_prob = clf.predict_proba(test_data["X"])[:, 1]

        _eval_gate_selection("val", val_data, val_gate_prob)
        _eval_gate_selection("test", test_data, test_gate_prob)
    except Exception as exc:
        print(f"[warn] sklearn baseline skipped: {exc}")


if __name__ == "__main__":
    main()
