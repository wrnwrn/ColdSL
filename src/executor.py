
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

# Binary Cross-Entropy (Task Loss)
def task_loss_function(predictions, labels):
    return nn.BCEWithLogitsLoss()(predictions, labels)


# Mean Squared Error (Distillation Loss)
def distill_loss_function(student_features, teacher_features, temperature=1.0):
    return nn.MSELoss()(student_features / temperature, teacher_features / temperature)


def evaluate_performance(label, pred):
    """
    Evaluate performance on the synthetic lethality prediction task.

    Args:
        label: Ground-truth labels as a NumPy array.
        pred: Predicted scores as a NumPy array.
    """

    auc = roc_auc_score(label, pred)
    aupr = average_precision_score(label, pred)

    precision, recall, _ = precision_recall_curve(label, pred)
    denominator = precision + recall  
    
    
    f1_scores = np.zeros_like(denominator)  
    valid_mask = denominator > 0  
    f1_scores[valid_mask] = 2 * (precision[valid_mask] * recall[valid_mask]) / denominator[valid_mask]
    f1 = np.max(f1_scores)

    performance_dict = {"AUC": auc, "AUPR": aupr, "F1": f1}
    return performance_dict


def safe_auc_aupr(label, pred):
    label = np.asarray(label)
    pred = np.asarray(pred)
    if label.size == 0 or np.unique(label).size < 2:
        return np.nan, np.nan
    return roc_auc_score(label, pred), average_precision_score(label, pred)


def _format_bucket_edge(value):
    if value is None:
        return "inf"
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    if text == "":
        text = "0"
    return text.replace(".", "p")


DEFAULT_BUCKETS = [(0.0, 0.05), (0.05, 0.15), (0.15, None)]


def compute_bucket_metrics(labels, p_kg, p_seq, alpha=None, buckets=None):
    if buckets is None:
        buckets = DEFAULT_BUCKETS
    labels = np.asarray(labels)
    p_kg = np.asarray(p_kg)
    p_seq = np.asarray(p_seq)
    alpha_arr = None if alpha is None else np.asarray(alpha)

    d = np.abs(p_kg - p_seq)
    avg_pred = 0.5 * (p_kg + p_seq)
    select_pred = None
    if alpha_arr is not None and alpha_arr.size == p_kg.size:
        select_pred = alpha_arr * p_seq + (1.0 - alpha_arr) * p_kg

    results = {}
    for low, high in buckets:
        low_str = _format_bucket_edge(low)
        high_str = _format_bucket_edge(high)
        bucket_name = f"d_{low_str}_{high_str}"
        if high is None:
            mask = d > low
        else:
            if low == 0:
                mask = (d >= low) & (d <= high)
            else:
                mask = (d > low) & (d <= high)
        count = int(mask.sum())
        results[f"bucket_{bucket_name}_count"] = count

        if count == 0:
            for pred_name in ("avg", "select", "kg", "seq"):
                results[f"bucket_{bucket_name}_{pred_name}_auc"] = np.nan
                results[f"bucket_{bucket_name}_{pred_name}_aupr"] = np.nan
            continue

        pred_map = {
            "avg": avg_pred[mask],
            "kg": p_kg[mask],
            "seq": p_seq[mask],
        }
        if select_pred is not None:
            pred_map["select"] = select_pred[mask]
        else:
            pred_map["select"] = None

        label_slice = labels[mask]
        for pred_name, pred_values in pred_map.items():
            if pred_values is None:
                results[f"bucket_{bucket_name}_{pred_name}_auc"] = np.nan
                results[f"bucket_{bucket_name}_{pred_name}_aupr"] = np.nan
                continue
            auc, aupr = safe_auc_aupr(label_slice, pred_values)
            results[f"bucket_{bucket_name}_{pred_name}_auc"] = auc
            results[f"bucket_{bucket_name}_{pred_name}_aupr"] = aupr

    return results


class MoEStatsTracker:
    def __init__(self, accelerator, feature_dim=11, logit_feature_dim=5, buckets=None):
        self.accelerator = accelerator
        self.feature_dim = feature_dim
        self.logit_feature_dim = logit_feature_dim
        self.buckets = buckets if buckets is not None else DEFAULT_BUCKETS
        self.prev_logit_kg = None
        self.prev_logit_seq = None
        self.prev_gate_feat_mean = None
        self.prev_gate_feat_var = None
        self.prev_gate_logit_feat_mean = None
        self.prev_gate_logit_feat_var = None
        self.reset()

    def reset(self):
        device = self.accelerator.device
        self.pos_count = torch.tensor(0.0, device=device)
        self.pos_total = torch.tensor(0.0, device=device)
        self.gate_feat_sum = torch.zeros(self.feature_dim, device=device)
        self.gate_feat_sum_sq = torch.zeros(self.feature_dim, device=device)
        self.gate_feat_count = torch.tensor(0.0, device=device)
        self.gate_logit_feat_sum = torch.zeros(self.logit_feature_dim, device=device)
        self.gate_logit_feat_sum_sq = torch.zeros(self.logit_feature_dim, device=device)
        self.gate_logit_feat_count = torch.tensor(0.0, device=device)
        self.all_p_kg = torch.tensor([], device=device)
        self.all_p_seq = torch.tensor([], device=device)
        self.all_alpha = torch.tensor([], device=device)
        self.all_labels = torch.tensor([], device=device)
        self.all_logit_kg = torch.tensor([], device=device)
        self.all_logit_seq = torch.tensor([], device=device)
        self.all_sample_idx = torch.tensor([], device=device, dtype=torch.long)

    def _ensure_gate_feature_dims(self, gate_features):
        feat_dim = gate_features.shape[1]
        if feat_dim != self.feature_dim:
            self.feature_dim = feat_dim
            device = gate_features.device
            self.gate_feat_sum = torch.zeros(self.feature_dim, device=device)
            self.gate_feat_sum_sq = torch.zeros(self.feature_dim, device=device)
            self.gate_feat_count = torch.tensor(0.0, device=device)
        if self.logit_feature_dim > self.feature_dim:
            self.logit_feature_dim = self.feature_dim
            device = gate_features.device
            self.gate_logit_feat_sum = torch.zeros(self.logit_feature_dim, device=device)
            self.gate_logit_feat_sum_sq = torch.zeros(self.logit_feature_dim, device=device)
            self.gate_logit_feat_count = torch.tensor(0.0, device=device)

    def update(self, logit_kg, logit_seq, labels, sample_idx, alpha=None, loss_seq=None, loss_kg=None, gate_features=None):
        if logit_kg.dim() > 1:
            logit_kg = logit_kg.view(-1)
        if logit_seq.dim() > 1:
            logit_seq = logit_seq.view(-1)
        labels = labels.view(-1)

        if loss_seq is not None and loss_kg is not None:
            self.pos_count += (loss_seq < loss_kg).float().sum()
            self.pos_total += torch.tensor(labels.numel(), device=self.accelerator.device)

        p_kg = torch.sigmoid(logit_kg.detach())
        p_seq = torch.sigmoid(logit_seq.detach())
        self.all_p_kg = torch.cat((self.all_p_kg, p_kg), dim=0)
        self.all_p_seq = torch.cat((self.all_p_seq, p_seq), dim=0)
        self.all_labels = torch.cat((self.all_labels, labels.detach()), dim=0)
        self.all_logit_kg = torch.cat((self.all_logit_kg, logit_kg.detach()), dim=0)
        self.all_logit_seq = torch.cat((self.all_logit_seq, logit_seq.detach()), dim=0)

        if alpha is not None:
            alpha = alpha.view(-1).detach()
            self.all_alpha = torch.cat((self.all_alpha, alpha), dim=0)

        if sample_idx is not None:
            sample_idx = sample_idx.view(-1).to(self.accelerator.device)
            self.all_sample_idx = torch.cat((self.all_sample_idx, sample_idx), dim=0)

        if gate_features is not None:
            gate_features = gate_features.detach()
            self._ensure_gate_feature_dims(gate_features)
            self.gate_feat_sum += gate_features.sum(dim=0)
            self.gate_feat_sum_sq += (gate_features * gate_features).sum(dim=0)
            self.gate_feat_count += torch.tensor(gate_features.size(0), device=self.accelerator.device)
            if self.logit_feature_dim > 0:
                logit_feats = gate_features[:, : self.logit_feature_dim]
                self.gate_logit_feat_sum += logit_feats.sum(dim=0)
                self.gate_logit_feat_sum_sq += (logit_feats * logit_feats).sum(dim=0)
                self.gate_logit_feat_count += torch.tensor(logit_feats.size(0), device=self.accelerator.device)

    def finalize(self):
        results = {}
        pos_total = self.accelerator.reduce(self.pos_total)
        pos_count = self.accelerator.reduce(self.pos_count)
        if pos_total.item() > 0:
            results["pos_rate"] = (pos_count / pos_total).item()

        gate_feat_count = self.accelerator.reduce(self.gate_feat_count)
        if gate_feat_count.item() > 0:
            gate_feat_sum = self.accelerator.reduce(self.gate_feat_sum)
            gate_feat_sum_sq = self.accelerator.reduce(self.gate_feat_sum_sq)
            gate_feat_mean_t = gate_feat_sum / gate_feat_count
            gate_feat_var_t = gate_feat_sum_sq / gate_feat_count - gate_feat_mean_t ** 2
            gate_feat_mean = gate_feat_mean_t.float().cpu().numpy()
            gate_feat_var = gate_feat_var_t.float().cpu().numpy()
            if self.accelerator.is_main_process:
                if self.prev_gate_feat_mean is None:
                    results["gate_feat_mean_drift"] = np.nan
                    results["gate_feat_var_drift"] = np.nan
                else:
                    results["gate_feat_mean_drift"] = float(np.mean(np.abs(gate_feat_mean - self.prev_gate_feat_mean)))
                    results["gate_feat_var_drift"] = float(np.mean(np.abs(gate_feat_var - self.prev_gate_feat_var)))
                self.prev_gate_feat_mean = gate_feat_mean
                self.prev_gate_feat_var = gate_feat_var

        gate_logit_feat_count = self.accelerator.reduce(self.gate_logit_feat_count)
        if gate_logit_feat_count.item() > 0:
            gate_logit_feat_sum = self.accelerator.reduce(self.gate_logit_feat_sum)
            gate_logit_feat_sum_sq = self.accelerator.reduce(self.gate_logit_feat_sum_sq)
            gate_logit_feat_mean_t = gate_logit_feat_sum / gate_logit_feat_count
            gate_logit_feat_var_t = gate_logit_feat_sum_sq / gate_logit_feat_count - gate_logit_feat_mean_t ** 2
            gate_logit_feat_mean = gate_logit_feat_mean_t.float().cpu().numpy()
            gate_logit_feat_var = gate_logit_feat_var_t.float().cpu().numpy()
            if self.accelerator.is_main_process:
                if self.prev_gate_logit_feat_mean is None:
                    results["gate_logit_feat_mean_drift"] = np.nan
                    results["gate_logit_feat_var_drift"] = np.nan
                else:
                    results["gate_logit_feat_mean_drift"] = float(np.mean(np.abs(gate_logit_feat_mean - self.prev_gate_logit_feat_mean)))
                    results["gate_logit_feat_var_drift"] = float(np.mean(np.abs(gate_logit_feat_var - self.prev_gate_logit_feat_var)))
                self.prev_gate_logit_feat_mean = gate_logit_feat_mean
                self.prev_gate_logit_feat_var = gate_logit_feat_var

        if self.all_labels.numel() == 0:
            return results

        labels = self.accelerator.gather(self.all_labels).detach().cpu().numpy()
        p_kg = self.accelerator.gather(self.all_p_kg).detach().cpu().numpy()
        p_seq = self.accelerator.gather(self.all_p_seq).detach().cpu().numpy()
        alpha = None
        if self.all_alpha.numel() > 0:
            alpha = self.accelerator.gather(self.all_alpha).detach().cpu().numpy()

        if self.accelerator.is_main_process:
            results.update(compute_bucket_metrics(labels, p_kg, p_seq, alpha, buckets=self.buckets))

        if self.all_logit_kg.numel() > 0 and self.all_logit_seq.numel() > 0 and self.all_sample_idx.numel() > 0:
            logit_kg = self.accelerator.gather(self.all_logit_kg).detach().cpu().numpy()
            logit_seq = self.accelerator.gather(self.all_logit_seq).detach().cpu().numpy()
            sample_idx = self.accelerator.gather(self.all_sample_idx).detach().cpu().numpy().astype(int)
            if self.accelerator.is_main_process:
                max_idx = int(sample_idx.max()) if sample_idx.size > 0 else -1
                if max_idx >= 0:
                    curr_logit_kg = np.full(max_idx + 1, np.nan, dtype=np.float32)
                    curr_logit_seq = np.full(max_idx + 1, np.nan, dtype=np.float32)
                    curr_logit_kg[sample_idx] = logit_kg
                    curr_logit_seq[sample_idx] = logit_seq

                    if self.prev_logit_kg is None or self.prev_logit_seq is None:
                        results["logit_drift_kg"] = np.nan
                        results["logit_drift_seq"] = np.nan
                    else:
                        common_mask = ~np.isnan(curr_logit_kg) & ~np.isnan(self.prev_logit_kg)
                        results["logit_drift_kg"] = float(np.mean(np.abs(curr_logit_kg[common_mask] - self.prev_logit_kg[common_mask]))) if np.any(common_mask) else np.nan
                        common_mask = ~np.isnan(curr_logit_seq) & ~np.isnan(self.prev_logit_seq)
                        results["logit_drift_seq"] = float(np.mean(np.abs(curr_logit_seq[common_mask] - self.prev_logit_seq[common_mask]))) if np.any(common_mask) else np.nan

                    self.prev_logit_kg = curr_logit_kg
                    self.prev_logit_seq = curr_logit_seq

        return results

class Losser(object):
    """
    Accumulate and summarize loss values.
    """

    def __init__(self, accelerator, all_batch_count):
        self.accelerator = accelerator
        self.all_batch_count = torch.tensor(all_batch_count).to(accelerator.device)  
        self.all_loss_dict = {}

    def multi_incr(self, value_dict):
        for key, value in value_dict.items():
            if key not in self.all_loss_dict:
                self.all_loss_dict[key] = torch.tensor(0.0).to(self.accelerator.device)
            self.all_loss_dict[key] += value

    def get_results(self):
        
        all_batch_count_reduce = self.accelerator.reduce(self.all_batch_count)

        results = {}
        for key, value in self.all_loss_dict.items():
            value_reduce = self.accelerator.reduce(value)
            value_per_batch = value_reduce / all_batch_count_reduce
            results[key] = value_per_batch.item()  

        return results


class Trainer(object):
    """
    Run model training.
    """

    def __init__(self, model, batch_size, multi_optimizer, lambda_distill, self_kl_loss_weight, cross_kl_loss_weight, gate_aux_weight=0.0, gate_main_weight=1.0):
        """
        Initialize the trainer.

        Args:
            model: Complete model.
            batch_size: Batch size.
            multi_optimizer: Custom multi-optimizer.
            lambda_distill: Distillation-loss weight.
            self_kl_loss_weight: Self-KL loss weight used by the omics branch.
            cross_kl_loss_weight: Cross-KL loss weight used by the omics branch.
        """

        self.model = model
        self.batch_size = batch_size
        self.multi_optimizer = multi_optimizer  
        self.lambda_distill = lambda_distill
        self.self_kl_loss_weight = self_kl_loss_weight
        self.cross_kl_loss_weight = cross_kl_loss_weight
        self.gate_aux_weight = gate_aux_weight
        self.gate_main_weight = gate_main_weight
        self.moe_stats = None

    def train(self, accelerator, sl_dataloader, kg_graph, task_type):
        
        self.model.train()

        dataloader = sl_dataloader

        
        losser = Losser(accelerator=accelerator, all_batch_count=len(dataloader))
        all_predicts = torch.tensor([], device=accelerator.device)  
        all_labels = torch.tensor([], device=accelerator.device)  
        moe_stats = None
        if task_type in ("kg_seq_moe", "kg_seq_moe_ft"):
            if self.moe_stats is None:
                self.moe_stats = MoEStatsTracker(accelerator)
            self.moe_stats.reset()
            moe_stats = self.moe_stats

        self.multi_optimizer.zero_grad(task_type)
        
        with tqdm(dataloader, unit="batch", disable=not accelerator.is_local_main_process) as tepoch:
            for step, data in enumerate(tepoch):
                with accelerator.accumulate(self.model):  
                    
                    gene_1_omics_data_list = data["gene_1_omics_data_list"]
                    gene_1_gene_entity = data["gene_1_gene_entity"]
                    gene_1_gene_idx = data["gene_1_gene_idx"]

                    gene_2_omics_data_list = data["gene_2_omics_data_list"]
                    gene_2_gene_entity = data["gene_2_gene_entity"]
                    gene_2_gene_idx = data["gene_2_gene_idx"]

                    label = data["label"]

                    
                    gene_1_omics_params = {"omics_data_list": gene_1_omics_data_list, "is_training": True}
                    gene_2_omics_params = {"omics_data_list": gene_2_omics_data_list, "is_training": True}
                    gene_1_kg_params = {"gene_entity": gene_1_gene_entity, "kg_graph": kg_graph}
                    gene_2_kg_params = {"gene_entity": gene_2_gene_entity, "kg_graph": kg_graph}
                    gene_1_seq_params = {"gene_idx": gene_1_gene_idx}
                    gene_2_seq_params = {"gene_idx": gene_2_gene_idx}

                    
                    
                    # target_model = self.model.module if hasattr(self.model, "module") else self.model
                    
                    # if kg_enc is not None:
                    
                    #     gene_1_kg_params["reset_aux"] = True

                    
                    logit = None
                    loss = None
                    if task_type == "only_omics":
                        logit, self_kl_loss, cross_kl_loss = self.model(task_type, gene_1_omics_params, gene_2_omics_params, gene_1_kg_params, gene_2_kg_params)
                        logit = logit.view(-1)  
                        label_loss = task_loss_function(logit, label)

                        loss = label_loss + self.self_kl_loss_weight * self_kl_loss + self.cross_kl_loss_weight * cross_kl_loss  
                        losser.multi_incr({"loss-all": loss, "loss-label": label_loss, "loss-self-kl": self_kl_loss, "loss-cross-kl": cross_kl_loss})
                    elif task_type == "only_kg":
                        logit = self.model(task_type, gene_1_omics_params, gene_2_omics_params, gene_1_kg_params, gene_2_kg_params)
                        logit = logit.view(-1)
                        label_loss = task_loss_function(logit, label)

                        
                        # aux_cons = torch.tensor(0.0, device=accelerator.device)
                        # if kg_enc is not None and getattr(kg_enc, "aux_sem_cons_loss", None) is not None:
                        #     aux_cons = kg_enc.aux_sem_cons_loss

                        # loss = label_loss + aux_cons
                        # losser.multi_incr({"loss-label": label_loss, "loss-sem-cons": aux_cons})

                        loss = label_loss
                        losser.multi_incr({"loss-label": label_loss})
                    elif task_type == "only_seq":
                        logit = self.model(task_type, gene_1_omics_params, gene_2_omics_params, gene_1_kg_params, gene_2_kg_params, gene_1_seq_params, gene_2_seq_params)
                        logit = logit.view(-1)
                        label_loss = task_loss_function(logit, label)

                        loss = label_loss
                        losser.multi_incr({"loss-label": label_loss})
                    elif task_type == "kg_seq":
                        logit = self.model(task_type, gene_1_omics_params, gene_2_omics_params, gene_1_kg_params, gene_2_kg_params, gene_1_seq_params, gene_2_seq_params)
                        logit = logit.view(-1)
                        label_loss = task_loss_function(logit, label)

                        loss = label_loss
                        losser.multi_incr({"loss-label": label_loss})
                    elif task_type in ("kg_seq_moe", "kg_seq_moe_ft"):
                        logit, logit_kg, logit_seq, w, gate_logit = self.model(
                            task_type, gene_1_omics_params, gene_2_omics_params,
                            gene_1_kg_params, gene_2_kg_params,
                            gene_1_seq_params, gene_2_seq_params
                        )
                        logit = logit.view(-1)
                        label_loss = task_loss_function(logit, label)

                        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
                        loss_seq = loss_fn(logit_seq, label)
                        loss_kg = loss_fn(logit_kg, label)

                        gate_target = (loss_seq < loss_kg).float()

                        
                        
                        gate_loss_cal_type  = 'soft' # TODO  hard # soft
                        if gate_loss_cal_type =='origin':
                            gate_loss = nn.BCEWithLogitsLoss()(gate_logit, gate_target)
                        elif gate_loss_cal_type =='hard':
                            # loss_seq, loss_kg: [Batch]
                            gap = torch.abs(loss_seq - loss_kg)
                            
                            
                            mask = (gap > 0.05).float()
                            gate_loss = F.binary_cross_entropy_with_logits(gate_logit, gate_target, weight=mask)
                        elif gate_loss_cal_type =='soft':
                            # loss_seq, loss_kg: [Batch]
                            gap = torch.abs(loss_seq - loss_kg)
                            
                            
                            gate_loss = F.binary_cross_entropy_with_logits(gate_logit, gate_target, reduction='none')
                            gate_loss = (gate_loss * gap).mean() 
                        else:
                            gate_loss = 0
                        # ---------------------------------------------------------------------
                        
                        
                        # logit = gate_logit
                        # label = gate_target 

                        loss = self.gate_main_weight * label_loss + self.gate_aux_weight * gate_loss
                        losser.multi_incr({"loss-all": loss, "loss-label": label_loss, "loss-gate-aux": gate_loss})

                        if moe_stats is not None:
                            target_model = self.model.module if hasattr(self.model, "module") else self.model
                            gate_module = getattr(target_model, "kg_seq_moe_gate", None)
                            gate_features = None if gate_module is None else getattr(gate_module, "aux_features", None)
                            sample_idx = data.get("sample_idx", None)
                            moe_stats.update(
                                logit_kg=logit_kg,
                                logit_seq=logit_seq,
                                labels=label,
                                sample_idx=sample_idx,
                                alpha=w,
                                loss_seq=loss_seq,
                                loss_kg=loss_kg,
                                gate_features=gate_features,
                            )
                    elif task_type == "umt":
                        logit, combined_pretrain_seq_emb, combined_distill_seq_emb, combined_pretrain_kg_emb, combined_distill_kg_emb, self_kl_loss, cross_kl_loss = self.model(
                            task_type, gene_1_omics_params, gene_2_omics_params,
                            gene_1_kg_params, gene_2_kg_params,
                            gene_1_seq_params, gene_2_seq_params
                        )
                        logit = logit.view(-1)
                        # Compute task loss (binary cross-entropy)
                        label_loss = task_loss_function(logit, label)
                        # Compute distillation loss (MSE)
                        distill_loss = distill_loss_function(combined_distill_seq_emb, combined_pretrain_seq_emb) + distill_loss_function(combined_distill_kg_emb, combined_pretrain_kg_emb)

                        # Combine losses
                        loss = label_loss + self.lambda_distill * distill_loss + self.self_kl_loss_weight * self_kl_loss + self.cross_kl_loss_weight * cross_kl_loss
                        losser.multi_incr({"loss-all": loss, "loss-label": label_loss, "loss-distill": distill_loss, "loss-self-kl": self_kl_loss, "loss-cross-kl": cross_kl_loss})
                    elif task_type == "umt_naive":
                        logit = self.model(
                            task_type, gene_1_omics_params, gene_2_omics_params,
                            gene_1_kg_params, gene_2_kg_params,
                            gene_1_seq_params, gene_2_seq_params
                        )
                        logit = logit.view(-1)
                        label_loss = task_loss_function(logit, label)
                        loss = label_loss
                        losser.multi_incr({"loss-all": loss, "loss-label": label_loss})

                    
                    accelerator.backward(loss)
                    self.multi_optimizer.step(task_type)
                    self.multi_optimizer.zero_grad(task_type)

                    
                    logit = logit.sigmoid()  
                    all_predicts = torch.cat((all_predicts, logit), dim=0)
                    all_labels = torch.cat((all_labels, label), dim=0)

        results = {}  

        
        all_predicts_gather = accelerator.gather(all_predicts)  
        all_labels_gather = accelerator.gather(all_labels)  
        if accelerator.is_main_process:  
            # TODO
            
            # target_model = self.model.module if hasattr(self.model, 'module') else self.model
            # print("gnn_gate(sigmoid)=", float(torch.sigmoid(target_model.pretrain_kg_encoder.gnn_gate).detach().cpu()))
            results = evaluate_performance(all_labels_gather.detach().cpu().numpy(), all_predicts_gather.detach().cpu().numpy())

        if task_type == "only_kg":
            target_model = self.model.module if hasattr(self.model, "module") else self.model
            kg_enc = getattr(target_model, "pretrain_kg_encoder", None)  
            if accelerator.is_main_process :
                if kg_enc.aux_gate_mean is not None:
                    print("gate mean/std/q10/q50/q90:",
                        float(kg_enc.aux_gate_mean.cpu()),
                        float(kg_enc.aux_gate_std.cpu()),
                        [float(x) for x in kg_enc.aux_gate_q.cpu()])
        
        if task_type == 'kg_seq':
            target_model = self.model.module if hasattr(self.model, "module") else self.model
            kg_enc = getattr(target_model, "pretrain_kg_encoder", None)  
            kg_seq_fuser = getattr(target_model, "kg_seq_fuser", None)
            if accelerator.is_main_process and kg_enc is not None and kg_enc.aux_gate_mean is not None:
                print(f"kg_seq: gate mean{kg_seq_fuser.aux_gate_mean.cpu()}, Gate quantiles: {kg_seq_fuser.aux_gate_q.cpu()}. bioBERT+KG: gate mean/std/q10/q50/q90:",
                    float(kg_enc.aux_gate_mean.cpu()),
                    float(kg_enc.aux_gate_std.cpu()),
                    [float(x) for x in kg_enc.aux_gate_q.cpu()])
                
        if task_type in ("kg_seq_moe", "kg_seq_moe_ft"):
            target_model = self.model.module if hasattr(self.model, "module") else self.model
            kg_enc = getattr(target_model, "pretrain_kg_encoder", None)  
            kg_seq_moe_gate = getattr(target_model, "kg_seq_moe_gate", None)
            if accelerator.is_main_process and kg_enc is not None and kg_enc.aux_gate_mean is not None:
                print(f"kg_seq: gate mean{kg_seq_moe_gate.aux_gate_mean.cpu()}, Gate quantiles: {kg_seq_moe_gate.aux_gate_q.cpu()}. bioBERT+KG: gate mean/std/q10/q50/q90:",
                    float(kg_enc.aux_gate_mean.cpu()),
                    float(kg_enc.aux_gate_std.cpu()),
                    [float(x) for x in kg_enc.aux_gate_q.cpu()])

        
        results_loss = losser.get_results()
        results = results | results_loss
        if moe_stats is not None:
            results = results | moe_stats.finalize()

        return results


class Tester(object):
    """
    Run evaluation on the validation or test set.
    """

    def __init__(self, model, batch_size, lambda_distill, self_kl_loss_weight, cross_kl_loss_weight, epochs, gate_aux_weight=0.0):
        """
        Initialize the evaluator.

        Args:
            model: Complete model.
            batch_size: Batch size.
            lambda_distill: Distillation-loss weight.
            self_kl_loss_weight: Self-KL loss weight used by the omics branch.
            cross_kl_loss_weight: Cross-KL loss weight used by the omics branch.
            epochs: Number of training epochs, retained for UME hyperparameter tuning.
        """

        self.model = model
        self.batch_size = batch_size
        self.lambda_distill = lambda_distill
        self.self_kl_loss_weight = self_kl_loss_weight
        self.cross_kl_loss_weight = cross_kl_loss_weight
        self.gate_aux_weight = gate_aux_weight
        self.moe_stats = None

    def test(self, accelerator, sl_dataloader, kg_graph, task_type):
        
        self.model.eval()

        dataloader = sl_dataloader

        
        losser = Losser(accelerator=accelerator, all_batch_count=len(dataloader))
        all_predicts = torch.tensor([], device=accelerator.device)  
        all_labels = torch.tensor([], device=accelerator.device)  
        moe_stats = None
        if task_type in ("kg_seq_moe", "kg_seq_moe_ft", "ume"):
            if self.moe_stats is None:
                self.moe_stats = MoEStatsTracker(accelerator)
            self.moe_stats.reset()
            moe_stats = self.moe_stats

        
        with tqdm(dataloader, unit="batch", disable=not accelerator.is_local_main_process) as tepoch:
            for step, data in enumerate(tepoch):
                
                gene_1_omics_data_list = data["gene_1_omics_data_list"]
                gene_1_gene_entity = data["gene_1_gene_entity"]
                gene_1_gene_idx = data["gene_1_gene_idx"]

                gene_2_omics_data_list = data["gene_2_omics_data_list"]
                gene_2_gene_entity = data["gene_2_gene_entity"]
                gene_2_gene_idx = data["gene_2_gene_idx"]

                label = data["label"]

                
                gene_1_omics_params = {"omics_data_list": gene_1_omics_data_list, "is_training": False}
                gene_2_omics_params = {"omics_data_list": gene_2_omics_data_list, "is_training": False}
                gene_1_kg_params = {"gene_entity": gene_1_gene_entity, "kg_graph": kg_graph}
                gene_2_kg_params = {"gene_entity": gene_2_gene_entity, "kg_graph": kg_graph}
                gene_1_seq_params = {"gene_idx": gene_1_gene_idx}
                gene_2_seq_params = {"gene_idx": gene_2_gene_idx}

                
                with torch.no_grad():  
                    logit = None
                    loss = None
                    if task_type == "only_omics":
                        logit, self_kl_loss, cross_kl_loss = self.model(task_type, gene_1_omics_params, gene_2_omics_params, gene_1_kg_params, gene_2_kg_params)
                        logit = logit.view(-1)  
                        label_loss = task_loss_function(logit, label)  

                        loss = label_loss + self.self_kl_loss_weight * self_kl_loss + self.cross_kl_loss_weight * cross_kl_loss  
                        losser.multi_incr({"loss-all": loss, "loss-label": label_loss, "loss-self-kl": self_kl_loss, "loss-cross-kl": cross_kl_loss})
                    elif task_type == "only_kg":
                        logit = self.model(task_type, gene_1_omics_params, gene_2_omics_params, gene_1_kg_params, gene_2_kg_params)
                        logit = logit.view(-1)
                        label_loss = task_loss_function(logit, label)

                        loss = label_loss
                        losser.multi_incr({"loss-label": label_loss})
                    elif task_type == "only_seq":
                        logit = self.model(task_type, gene_1_omics_params, gene_2_omics_params, gene_1_kg_params, gene_2_kg_params, gene_1_seq_params, gene_2_seq_params)
                        logit = logit.view(-1)
                        label_loss = task_loss_function(logit, label)

                        loss = label_loss
                        losser.multi_incr({"loss-label": label_loss})
                    elif task_type == "kg_seq":
                        logit = self.model(task_type, gene_1_omics_params, gene_2_omics_params, gene_1_kg_params, gene_2_kg_params, gene_1_seq_params, gene_2_seq_params)
                        logit = logit.view(-1)
                        label_loss = task_loss_function(logit, label)

                        loss = label_loss
                        losser.multi_incr({"loss-label": label_loss})
                    elif task_type in ("kg_seq_moe", "kg_seq_moe_ft"):
                        logit, logit_kg, logit_seq, w, gate_logit = self.model(
                            task_type, gene_1_omics_params, gene_2_omics_params,
                            gene_1_kg_params, gene_2_kg_params,
                            gene_1_seq_params, gene_2_seq_params
                        )
                        logit = logit.view(-1)
                        label_loss = task_loss_function(logit, label)
                        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
                        loss_seq = loss_fn(logit_seq, label)
                        loss_kg = loss_fn(logit_kg, label)
                        gate_target = (loss_seq < loss_kg).float()

                        
                        
                        gate_loss_cal_type  = 'soft' # TODO  hard # soft
                        if gate_loss_cal_type =='origin':
                            gate_loss = nn.BCEWithLogitsLoss()(gate_logit, gate_target)
                        elif gate_loss_cal_type =='hard':
                            # loss_seq, loss_kg: [Batch]
                            gap = torch.abs(loss_seq - loss_kg)
                            
                            
                            mask = (gap > 0.05).float()
                            gate_loss = F.binary_cross_entropy_with_logits(gate_logit, gate_target, weight=mask)
                        elif gate_loss_cal_type =='soft':
                            # loss_seq, loss_kg: [Batch]
                            gap = torch.abs(loss_seq - loss_kg)
                            
                            
                            gate_loss = F.binary_cross_entropy_with_logits(gate_logit, gate_target, reduction='none')
                            gate_loss = (gate_loss * gap).mean() 
                        else:
                            gate_loss = 0
                        # ---------------------------------------------------------------------

                        
                        # logit = gate_logit
                        # label = gate_target 

                        loss = label_loss + self.gate_aux_weight * gate_loss 
                        losser.multi_incr({"loss-all": loss, "loss-label": label_loss, "loss-gate-aux": gate_loss})

                        if moe_stats is not None:
                            target_model = self.model.module if hasattr(self.model, "module") else self.model
                            gate_module = getattr(target_model, "kg_seq_moe_gate", None)
                            gate_features = None if gate_module is None else getattr(gate_module, "aux_features", None)
                            sample_idx = data.get("sample_idx", None)
                            moe_stats.update(
                                logit_kg=logit_kg,
                                logit_seq=logit_seq,
                                labels=label,
                                sample_idx=sample_idx,
                                alpha=w,
                                loss_seq=loss_seq,
                                loss_kg=loss_kg,
                                gate_features=gate_features,
                            )
                    elif task_type == "umt":
                        logit, combined_pretrain_seq_emb, combined_distill_seq_emb, combined_pretrain_kg_emb, combined_distill_kg_emb, self_kl_loss, cross_kl_loss = self.model(
                            task_type, gene_1_omics_params, gene_2_omics_params,
                            gene_1_kg_params, gene_2_kg_params,
                            gene_1_seq_params, gene_2_seq_params
                        )
                        logit = logit.view(-1)
                        # Compute task loss (binary cross-entropy)
                        label_loss = task_loss_function(logit, label)
                        # Compute distillation loss (MSE)
                        distill_loss = distill_loss_function(combined_distill_seq_emb, combined_pretrain_seq_emb) + distill_loss_function(combined_distill_kg_emb, combined_pretrain_kg_emb)

                        # Combine losses
                        loss = label_loss + self.lambda_distill * distill_loss + self.self_kl_loss_weight * self_kl_loss + self.cross_kl_loss_weight * cross_kl_loss
                        losser.multi_incr({"loss-all": loss, "loss-label": label_loss, "loss-distill": distill_loss, "loss-self-kl": self_kl_loss, "loss-cross-kl": cross_kl_loss})

                    elif task_type == "umt_naive":
                        logit = self.model(
                            task_type, gene_1_omics_params, gene_2_omics_params,
                            gene_1_kg_params, gene_2_kg_params,
                            gene_1_seq_params, gene_2_seq_params
                        )
                        logit = logit.view(-1)
                        label_loss = task_loss_function(logit, label)
                        loss = label_loss
                        losser.multi_incr({"loss-all": loss, "loss-label": label_loss})

                    elif task_type == "ume":
                        logit_omics, logit_kg = self.model(task_type, gene_1_omics_params, gene_2_omics_params, gene_1_kg_params, gene_2_kg_params, gene_1_seq_params, gene_2_seq_params)
                        if moe_stats is not None:
                            loss_fn = nn.BCEWithLogitsLoss(reduction="none")
                            loss_seq = loss_fn(logit_omics.view(-1), label)
                            loss_kg = loss_fn(logit_kg.view(-1), label)
                            sample_idx = data.get("sample_idx", None)
                            moe_stats.update(
                                logit_kg=logit_kg,
                                logit_seq=logit_omics,
                                labels=label,
                                sample_idx=sample_idx,
                                alpha=None,
                                loss_seq=loss_seq,
                                loss_kg=loss_kg,
                                gate_features=None,
                            )

                
                if task_type == "ume":
                    
                    prob_omics  = logit_omics.sigmoid()
                    prob_kg  = logit_kg.sigmoid()
                    logit = (prob_omics + prob_kg) * 0.5

                    
                    # logit = 0.5*(logit_omics+logit_kg)
                    # logit = logit.sigmoid()
                    
                    
                    
                    # target = label.view_as(prob_omics) 
                    
                    
                    # diff_omics = torch.abs(prob_omics - target)
                    # diff_kg = torch.abs(prob_kg - target)
                    
                    
                    # logit = torch.where(diff_omics < diff_kg, prob_omics, prob_kg)
                    

                else:
                    logit = logit.sigmoid()  

                all_predicts = torch.cat((all_predicts, logit), dim=0)
                all_labels = torch.cat((all_labels, label), dim=0)

        
        all_predicts_gather = accelerator.gather(all_predicts)  
        all_labels_gather = accelerator.gather(all_labels)  
        
        results = evaluate_performance(all_labels_gather.detach().cpu().numpy(), all_predicts_gather.detach().cpu().numpy())

        
        results_loss = losser.get_results()
        results = results | results_loss
        if moe_stats is not None:
            results = results | moe_stats.finalize()

        return results
