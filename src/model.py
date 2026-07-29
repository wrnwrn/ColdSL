
import numpy as np
import math
import torch
from torch_geometric.nn import RGCNConv
from torch_geometric.utils import k_hop_subgraph
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RAdam,Adam
from util.lookahead import Lookahead
from collections import defaultdict
from typing import Dict, List, Tuple



from util.my import rank_main_print


class VAE(nn.Module):
    """
    Variational autoencoder for a single modality.
    """

    def __init__(self, input_dim, hidden_dims, latent_dim, dropout):
        """
        Initialize the variational autoencoder.

        Args:
            input_dim: Input dimension.
            hidden_dims: List of hidden-layer dimensions.
            latent_dim: Latent output dimension.
            dropout: Dropout rate applied at each layer.
        """
        super().__init__()

        
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Dropout(dropout))
            
            layers.append(nn.SyncBatchNorm(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.feature_encoder = nn.Sequential(*layers)

        # μ
        self.mu_predictor = nn.Sequential(nn.Linear(in_dim, latent_dim), nn.ReLU())

        """
        The VAE predicts both mu and log_var. Predicting log_var instead of variance provides two benefits.

        Numerical stability: variance must be positive, whereas directly predicted variance can be negative. Predicting log_var avoids this issue.

        Computational convenience: operations involving inverse variance or standard deviation can be obtained efficiently from log_var through exponentiation.
        """
        self.log_var_predictor = nn.Sequential(nn.Linear(in_dim, latent_dim), nn.ReLU())

        
        self.LOG_VAR_MIN = -10.0
        self.LOG_VAR_MAX = 10.0

    def forward(self, x):
        """
        Encode an input batch into the parameters of its latent distribution.

        Args:
            x: Input features with shape [batch_size, input_dim].

        Returns:
            mu: Mean of the latent distribution.
            log_var: Log variance of the latent distribution.
        """

        for layer in self.feature_encoder:
            x = layer(x)
        mu = self.mu_predictor(x)
        log_var = self.log_var_predictor(x)

        log_var = torch.clamp(log_var, self.LOG_VAR_MIN, self.LOG_VAR_MAX)

        return mu, log_var


class RGCN(nn.Module):
    """
    Relational Graph Convolutional Networks
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, n_layers, graph_dropout):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        if n_layers == 1:  
            self.convs.append(RGCNConv(in_channels, out_channels, num_relations))
        else:  
            self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations))
            for i in range(n_layers - 2):
                self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations))
            self.convs.append(RGCNConv(hidden_channels, out_channels, num_relations))

        self.graph_dropout = graph_dropout

    def forward(self, x, edge_index, edge_type):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_type)
            x = F.dropout(x, p=self.graph_dropout, training=self.training)
            x = F.relu(x)

        x = self.convs[-1](x, edge_index, edge_type)
        return x


class OmicsEncoder(nn.Module):
    """
    Encoder for tissue-omics data.
    """

    def __init__(self, params):
        """
        Initialize the tissue-omics encoder.

        Args:
            omics_count: Number of omics modalities.
            omics_input_dim: Input dimension of the omics data.
            vae_hidden_dims: Hidden dimensions of the VAE.
            vae_dropout: Dropout rate applied at each VAE layer.
            hid_dim: Output dimension of the encoder.
        """
        super().__init__()

        
        omics_count = params["omics_count"]
        omics_input_dim = params["omics_input_dim"]
        vae_hidden_dims = params["vae_hidden_dims"]
        vae_dropout = params["vae_dropout"]
        hid_dim = params["hid_dim"]

        
        self.omics_encoders = nn.ModuleList(nn.ModuleList([VAE(omics_input_dim, vae_hidden_dims, hid_dim, vae_dropout) for j in range(omics_count)]) for i in range(omics_count))
        self.hid_dim = hid_dim

        self.fc = nn.Linear(int(2 * hid_dim), hid_dim)  

    def forward(self, params):
        """
        Encode the list of omics inputs.

        Args:
            omics_data_list: List whose elements have shape [batch_size, omics_input_dim].
            is_training: Whether the model is in training mode and should sample random noise.
        """
        omics_data_list = params["omics_data_list"]
        is_training = params["is_training"]

        current_device = omics_data_list[0].device
        omics_count = len(omics_data_list)

        
        vae_z = [None for _ in range(omics_count * 2)]

        self_kl_loss = torch.tensor(0.0).to(current_device)
        cross_kl_loss = torch.tensor(0.0).to(current_device)
        
        for i in range(omics_count):
            others_mu_array = []
            others_log_var_array = []
            for j in range(omics_count):
                if i == j:  # self-VAE
                    # [batch_size, omics_input_dim] -> [batch_size, (mu, log_var)]
                    mu, log_var = self.omics_encoders[i][j](omics_data_list[j])
                    self_kl_loss += OmicsEncoder.kl_loss(mu, log_var)

                    # [batch_size, hid_dim] -> [batch_size, 1, hid_dim]
                    if is_training:
                        vae_z[i] = OmicsEncoder.re_parameterize(mu, log_var).unsqueeze(1)
                    else:
                        vae_z[i] = mu.unsqueeze(1)  

                else:  # cross-VAE
                    mu, log_var = self.omics_encoders[i][j](omics_data_list[j])  
                    others_mu_array.append(mu)
                    others_log_var_array.append(log_var)

            
            poe_mu, poe_log_var = OmicsEncoder.product_of_experts(others_mu_array, others_log_var_array)
            cross_kl_loss += OmicsEncoder.kl_loss(poe_mu, poe_log_var)

            if is_training:
                vae_z[omics_count + i] = OmicsEncoder.re_parameterize(poe_mu, poe_log_var).unsqueeze(1)
            else:
                vae_z[omics_count + i] = poe_mu.unsqueeze(1)

        
        vae_z = torch.cat(vae_z, dim=1)  # Tensor, [batch_size, omics_count*2, vae_dim]

        ###################################################################
        
        # vae_z = vae_z.mean(dim=1)  # [batch_size, hid_dim]
        # return vae_z, self_kl_loss, cross_kl_loss
        
        self_tokens = vae_z[:, :omics_count, :]  # [batch_size, 4, hid_dim]
        cross_tokens = vae_z[:, omics_count:, :]  # [batch_size, 4, hid_dim]

        
        self_emb = self_tokens.mean(dim=1)  # [batch_size, hid_dim]
        cross_emb = cross_tokens.mean(dim=1)  # [batch_size, hid_dim]

        
        final_emb = torch.cat([self_emb, cross_emb], dim=-1)  # [batch_size, 2*hid_dim]
        final_emb = self.fc(final_emb)

        return final_emb, self_kl_loss, cross_kl_loss
        ###################################################################

    @staticmethod
    def product_of_experts(mu_set_, log_var_set_):
        """
        Combine multiple Gaussian distributions using a product of experts.

        The joint mean is computed as a precision-weighted average, and the joint variance is derived from the combined precision. See page 16 of the TMO paper.

        Args:
            mu_set_: List of means from multiple distributions.
            log_var_set_: List of log variances from multiple distributions.

        Returns:
            poe_mu: Mean of the joint distribution.
            poe_log_var: Log variance of the joint distribution.
        """

        tmp = 0
        for i in range(len(mu_set_)):  
            tmp += torch.div(1, torch.exp(log_var_set_[i]))

        poe_var = torch.div(1.0, tmp)
        poe_log_var = torch.log(poe_var)  

        tmp = 0.0
        for i in range(len(mu_set_)):
            tmp += torch.div(1.0, torch.exp(log_var_set_[i])) * mu_set_[i]
        poe_mu = poe_var * tmp
        return poe_mu, poe_log_var

    @staticmethod
    def re_parameterize(mean, log_var):
        """
        Returns: z
        """

        
        log_var = torch.exp(log_var / 2)  # in log-space, square root is divide by two
        epsilon = torch.randn_like(log_var)  
        return epsilon * log_var + mean

    @staticmethod
    def kl_loss(mu, log_var, reduction="mean"):
        """
        Compute the Kullback-Leibler divergence loss.

        Args:
            mu: Mean of the latent distribution.
            log_var: Log variance of the latent distribution.
            reduction: Reduction method.

        Returns:
            The Kullback-Leibler divergence loss.
        """

        kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
        if reduction == "mean":
            return kl.mean()  
        elif reduction == "sum":
            return kl.sum(dim=1).mean()


class KGEncoder(nn.Module):
    """
    Knowledge graph encoder.
    """

    def __init__(self, params):
        """
        Initialize the knowledge graph encoder.

        Args:
            entity_vocab_size: Size of the complete KG entity vocabulary; 0 is <PAD> and 1 is <MASK>.
            hid_dim: Encoder output dimension, which is also the GCN output dimension.
            in_channels: Input dimension of the KG entity representations.
            hidden_channels: Hidden dimension of the GNN when multiple layers are used.
            gcn_layers: Number of GCN layers.
            graph_dropout: Graph dropout rate.
            num_relations: Number of relation types used by the RGCN.
            entity_init: Entity initialization method, either random or biobert.
            use_graph: Whether to use the RGCN.
            biobert_embeddings: Tensor or ndarray of shape [num_entities, 768], aligned with entity indices.
            pad_idx: Optional padding index; defaults to 0.
        """
        super().__init__()

        entity_vocab_size = params["entity_vocab_size"]
        hid_dim = params["hid_dim"]
        in_channels = params["in_channels"]
        hidden_channels = params["hidden_channels"]
        gcn_layers = params["gcn_layers"]
        graph_dropout = params["graph_dropout"]
        num_relations = params["num_relations"]

        entity_init = params.get("entity_init", "random")
        use_graph = params.get("use_graph", True)
        biobert_embeddings = params.get("biobert_embeddings", None)
        pad_idx = params.get("pad_idx", 0)

        self.entity_init = entity_init
        self.use_graph = use_graph
        self.gcn_layers = gcn_layers
        self.p_rel = params["p_rel"]
        self.kg_biobert_fusion_type =params["kg_biobert_fusion_type"]
        self.kg_node_type_emb_switch =params["kg_node_type_emb_switch"]

        
        
        
        

        
        self.num_node_types =  params["num_node_types"]

        self.type_emb_dim = params["kg_node_type_emb_dim"]
        self.type_emb = None
        if self.num_node_types is not None:
            self.type_emb = nn.Embedding(int(self.num_node_types), self.type_emb_dim)

        # --- Embedding / semantic init ---
        self.entity_encoder = None            # random init only
        self.entity_residual = None          # for biobert: trainable residual
        self.biobert_proj = None
        self.entity_ln = None

        if entity_init == "random":
            
            self.entity_encoder = nn.Embedding(entity_vocab_size, in_channels, padding_idx=pad_idx)

        elif entity_init == "biobert":
            if biobert_embeddings is None:
                raise ValueError("biobert_embeddings is required when entity_init='biobert'")
            if isinstance(biobert_embeddings, np.ndarray):
                biobert_embeddings = torch.from_numpy(biobert_embeddings)

            if biobert_embeddings.shape[0] != entity_vocab_size:
                raise ValueError(
                    f"biobert_embeddings row count {biobert_embeddings.shape[0]} "
                    f"does not match entity_vocab_size {entity_vocab_size}"
                )

            
            self.register_buffer("biobert_embeddings", biobert_embeddings.float())

            
            self.biobert_proj = nn.Linear(biobert_embeddings.shape[1], in_channels)

            
            
            self.entity_residual = nn.Embedding(entity_vocab_size, in_channels, padding_idx=pad_idx)
            nn.init.zeros_(self.entity_residual.weight)

            
            self.entity_ln = nn.LayerNorm(in_channels)

        else:
            raise ValueError(f"unsupported entity_init: {entity_init}")

        # --- Graph encoder ---
        self.gnn = None
        if use_graph:
            self.gnn = RGCN(in_channels, hidden_channels, hid_dim, num_relations, gcn_layers, graph_dropout)
            
        self.skip_proj = nn.Identity() if hid_dim == in_channels else nn.Linear(in_channels, hid_dim)
        
        self.concat_fusion_proj = None
        if self.kg_biobert_fusion_type == 1:
            self.concat_fusion_proj = nn.Linear(hid_dim * 2, hid_dim)
        
        
        gate_in_dim = hid_dim +  (self.type_emb_dim if self.kg_node_type_emb_switch and self.type_emb  is not None else 0)
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 1)
        )
        
        
        
        self.aux_gate_mean = None
        self.aux_gate_std = None
        self.aux_gate_q = None


    def _lookup_entity_embeddings(self, entity_idx):
        """
        entity_idx: shape [*] long
        returns:    shape [*, in_channels]
        """
        if self.entity_init == "random":
            return self.entity_encoder(entity_idx)

        if self.entity_init == "biobert":
            # semantic part
            x_sem = F.embedding(entity_idx, self.biobert_embeddings)   # [*, 768]
            x_sem = self.biobert_proj(x_sem)                           # [*, in_channels]

            # # residual part (trainable)
            # x_res = self.entity_residual(entity_idx)                   # [*, in_channels]

            # return self.entity_ln(x_sem + x_res)
            
            return self.entity_ln(x_sem) 

        raise ValueError(f"unsupported entity_init: {self.entity_init}")

    def forward(self, params):
        """
        Encode gene entities and their sampled knowledge graph context.

        Args:
            gene_entity: Entity indices of the genes, with shape [batch_size, 1], [batch_size, 2], or similar.
            kg_graph: Heterogeneous knowledge graph.
        """
        gene_entity = params["gene_entity"]
        kg_graph = params["kg_graph"]

        
        # if params.get("reset_aux", False):
        #     self.aux_sem_cons_loss = torch.zeros((), device=gene_entity.device)
        #     self.aux_gate_mean = None
        #     self.aux_gate_std = None
        #     self.aux_gate_q = None

        if not self.use_graph:
            return self._lookup_entity_embeddings(gene_entity)

        current_device = gene_entity.device

        batch_unique_entities = torch.unique(gene_entity, sorted=True)

        concatenated_edge_index = kg_graph.concatenated_edge_index
        concatenated_edge_type = kg_graph.concatenated_edge_type

        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=batch_unique_entities,
            num_hops=self.gcn_layers,
            edge_index=concatenated_edge_index,
            relabel_nodes=True,
            flow="source_to_target",
            directed=False,
        )

        sub_edge_type = concatenated_edge_type[edge_mask]

        subset = subset.to(current_device)
        sub_edge_index = sub_edge_index.to(current_device)
        sub_edge_type = sub_edge_type.to(current_device)
        
        if self.training and self.p_rel > 0:
            rel_ids = torch.unique(sub_edge_type)
            keep = torch.rand(rel_ids.size(0), device=sub_edge_type.device) > self.p_rel
            keep_rels = rel_ids[keep]
            edge_keep_mask = torch.isin(sub_edge_type, keep_rels)
            sub_edge_index = sub_edge_index[:, edge_keep_mask]
            sub_edge_type  = sub_edge_type[edge_keep_mask]


        x0 = self._lookup_entity_embeddings(subset)  # [subset_size, in_channels]
        x1 = self.gnn(x0, sub_edge_index, sub_edge_type)
        
        
        x0p = self.skip_proj(x0)                             # [N, hid_dim]
        
        
        # n = x0p.size(0)
        # deg = kg_graph.node_deg.to(current_device)[subset]          # [N]
        # deg_feat = torch.log1p(deg).float().unsqueeze(1)    # [N,1]


        g = None
        if self.kg_biobert_fusion_type == 0:
            x=x1
        elif self.kg_biobert_fusion_type == 1:
            x = torch.cat([x0p, x1], dim=1)
            x = self.concat_fusion_proj(x)
        elif self.kg_biobert_fusion_type == 2:
            x = x0p + x1
        elif self.kg_biobert_fusion_type == 3:
            
            if self.kg_node_type_emb_switch and (self.type_emb is not None) and hasattr(kg_graph, "node_type") and (kg_graph.node_type is not None):
                node_type = kg_graph.node_type.to(current_device)  # [num_entities]
                type_ids = node_type[subset].long()                # [N]
                type_feat = self.type_emb(type_ids)                # [N, type_emb_dim]
                gate_in = torch.cat([x0p, type_feat], dim=1)

            else:
                gate_in = torch.cat([x0p], dim=1)
            
            g = torch.sigmoid(self.gate_mlp(gate_in))                
            x = (1 - g) * x0p + g * x1
        else:
            raise ValueError(f"unsupported kg_biobert_fusion_type: {self.kg_biobert_fusion_type}")
        
        # g = torch.sigmoid(self.gnn_gate)                   # scalar in (0,1)
        # x = (1 - g) * self.skip_proj(x0) + g * x1       # [subset, hid_dim]

        
        
        # x = x[torch.searchsorted(subset, gene_entity, right=False)]
        
        
        

        
        # if self.training and self.semantic_cons_weight > 0.0:
        #     diff = x1 - x0p.detach()
        #     if self.cons_weight_by_gate:
        #         # w = g.detach()
        #         # cons = (w * (diff ** 2)).mean()
        #         w = (g.detach() * (1 - g.detach()))  # [N,1]
        #         cons = (w * (x1 - x0p.detach()).pow(2)).mean() 

        #     else:
        #         cons = (diff ** 2).mean()
        #     cons = cons * self.semantic_cons_weight

        #     if self.aux_sem_cons_loss is None:
        #         self.aux_sem_cons_loss = torch.zeros((), device=current_device)
        #     self.aux_sem_cons_loss = self.aux_sem_cons_loss + cons
            
        
        with torch.no_grad():
            if g is None:
                self.aux_gate_mean = None
                self.aux_gate_std = None
                self.aux_gate_q = None
            else:
                self.aux_gate_mean = g.mean().detach()
                self.aux_gate_std = g.std(unbiased=False).detach()
                
                self.aux_gate_q = torch.quantile(g.view(-1), torch.tensor([0.1, 0.5, 0.9], device=g.device)).detach()

        flat = gene_entity.reshape(-1)
        pos_in_unique = torch.searchsorted(batch_unique_entities, flat)  
        pos_in_subset = mapping.to(current_device)[pos_in_unique]        

        x = x[pos_in_subset].view(*gene_entity.shape, -1)
        return x


class SequenceEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        
        seq_embeddings = params["seq_embeddings"]
        if isinstance(seq_embeddings, np.ndarray):
            seq_embeddings = torch.from_numpy(seq_embeddings)
        
        
        self.register_buffer("seq_embeddings", seq_embeddings.float())
        
        input_dim = seq_embeddings.shape[1]      # e.g., 1280
        gene_final_dim = int(params["gene_final_dim"]) # e.g., 128 or 256
        hidden_dim = int(params.get("hidden_dim", 512)) 
        seq_encoder_mlp_dropout=params["seq_encoder_mlp_dropout"]

        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),  
            nn.Dropout(p=params.get("dropout", seq_encoder_mlp_dropout)),
            nn.Linear(hidden_dim, gene_final_dim),
            nn.LayerNorm(gene_final_dim)
        )

    def forward(self, params):
        gene_idx = params["gene_idx"]
        
        gene_idx = gene_idx.view(-1).to(self.seq_embeddings.device)
        
        
        x = self.seq_embeddings.index_select(0, gene_idx)
        
        
        x = self.mlp(x)
        return x


class KGSeqFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1, init_bias=-2.0):
        super().__init__()
        # MLP for gate mechanism
        self.gate = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # Concatenate the two modalities (kg + seq)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # Output a scalar gate value
            nn.Sigmoid()  # Gate between 0 and 1
        )
        # Optional: Layer Normalization on the fused output
        self.ln = nn.LayerNorm(input_dim)

        # To track gate statistics for debugging
        self.aux_gate_mean = None
        self.aux_gate_q = None

    def forward(self, kg_emb, seq_emb):
        # Concatenate KG and sequence embeddings
        fused_input = torch.cat([kg_emb, seq_emb], dim=-1)  # Shape: [B, kg_dim + seq_dim]
        gate_weight = self.gate(fused_input)  # Get the gate value between 0 and 1
        fused = (1 - gate_weight) * kg_emb + gate_weight * seq_emb  # Weighted fusion
        fused = self.ln(fused)  # Apply layer normalization for stability

        # Capture the gate statistics (optional for analysis)
        with torch.no_grad():
            self.aux_gate_mean = gate_weight.mean().detach()
            self.aux_gate_q = torch.quantile(gate_weight.view(-1), torch.tensor([0.1, 0.5, 0.9], device=gate_weight.device)).detach()

        return fused, gate_weight  # Return fused embedding and the gate weight
    

class LogitMoEGate(nn.Module):
    """
    Interpretable stacking module with a global alpha and temperature calibration.
    """
    def __init__(self, init_alpha=0.5, init_temp=1.0):
        super().__init__()
        init_alpha = float(init_alpha)
        if init_alpha <= 0.0 or init_alpha >= 1.0:
            raise ValueError("init_alpha must be in (0, 1)")
        alpha_logit = math.log(init_alpha / (1.0 - init_alpha))
        self.alpha_logit = nn.Parameter(torch.tensor(alpha_logit))
        init_log_temp = math.log(float(init_temp))
        self.log_t_kg = nn.Parameter(torch.tensor(init_log_temp))
        self.log_t_seq = nn.Parameter(torch.tensor(init_log_temp))

        self.aux_gate_mean = None
        self.aux_gate_q = None

    @staticmethod
    def _logit_from_prob(p):
        eps = 1e-6
        p = torch.clamp(p, eps, 1.0 - eps)
        return torch.log(p) - torch.log1p(-p)

    def forward(self, logit_kg, logit_seq, deg1=None, deg2=None):
        """
        Combine the calibrated expert logits.

        Args:
            logit_kg: KG-expert logits with shape [B, 1] or [B].
            logit_seq: Sequence-expert logits with shape [B, 1] or [B].
            deg1: Optional tensor with shape [B], retained for compatibility but not used.
            deg2: Optional tensor with shape [B], retained for compatibility but not used.
        """
        if logit_kg.dim() > 1:
            logit_kg = logit_kg.view(-1)
        if logit_seq.dim() > 1:
            logit_seq = logit_seq.view(-1)

        alpha = torch.sigmoid(self.alpha_logit)
        t_kg = torch.exp(self.log_t_kg)
        t_seq = torch.exp(self.log_t_seq)

        p_kg = torch.sigmoid(logit_kg / t_kg)
        p_seq = torch.sigmoid(logit_seq / t_seq)
        p = (1.0 - alpha) * p_kg + alpha * p_seq
        logit = self._logit_from_prob(p)

        with torch.no_grad():
            self.aux_gate_mean = alpha.detach()
            self.aux_gate_q = alpha.repeat(3).detach()

        return logit, alpha


class PredictiveGate(nn.Module):
    """
    Gate network that predicts expert preference from embeddings + logits.
    """

    def __init__(self, input_dim, hidden_dims=(64, 32), dropout=0.5):
        super().__init__()
        h1, h2 = hidden_dims

        
        self.bn_input = nn.BatchNorm1d(input_dim)

        # self.net = nn.Sequential(
        #     nn.Linear(input_dim, h1),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(h1),
        #     nn.Linear(h1, h2),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(h2, 1),
        # )
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

        self.aux_gate_mean = None
        self.aux_gate_q = None

    def forward(self, features):
        features = self.bn_input(features)         
        gate_logit = self.net(features).view(-1)
        alpha = torch.sigmoid(gate_logit)
        with torch.no_grad():
            self.aux_gate_mean = alpha.mean().detach()
            self.aux_gate_q = torch.quantile(alpha.view(-1), torch.tensor([0.1, 0.5, 0.9], device=alpha.device)).detach()
        return alpha, gate_logit


class FeatureEngineeredGate(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        
        
        
        
        
        
        input_dim = 11 
        
        
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim), 
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.aux_features = None

    def compute_entropy(self, p):
        # p is probability [0, 1]
        # handle epsilon to avoid log(0)
        eps = 1e-6
        p = torch.clamp(p, eps, 1.0 - eps)
        return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))

    def forward(self, emb_seq_g1, emb_seq_g2, emb_kg_g1, emb_kg_g2, logit_seq, logit_kg):
        """
        emb_*: [B, Dim]
        logit_*: [B, 1] or [B]
        """
        
        if logit_seq.dim() == 1: logit_seq = logit_seq.unsqueeze(1)
        if logit_kg.dim() == 1: logit_kg = logit_kg.unsqueeze(1)

        # ---------------------------
        
        # ---------------------------
        p_seq = torch.sigmoid(logit_seq)
        p_kg = torch.sigmoid(logit_kg)
        
        
        diff = torch.abs(p_seq - p_kg)
        
        
        
        ent_seq = self.compute_entropy(p_seq)
        ent_kg = self.compute_entropy(p_kg)

        # ---------------------------
        
        # ---------------------------
        
        
        
        
        
        s_g1 = emb_seq_g1.detach()
        s_g2 = emb_seq_g2.detach()
        k_g1 = emb_kg_g1.detach()
        k_g2 = emb_kg_g2.detach()

        
        
        cos_seq = F.cosine_similarity(s_g1, s_g2, dim=1).unsqueeze(1) 
        
        cos_kg = F.cosine_similarity(k_g1, k_g2, dim=1).unsqueeze(1)

        
        
        norm_s1 = torch.norm(s_g1, p=2, dim=1, keepdim=True)
        norm_s2 = torch.norm(s_g2, p=2, dim=1, keepdim=True)
        norm_k1 = torch.norm(k_g1, p=2, dim=1, keepdim=True)
        norm_k2 = torch.norm(k_g2, p=2, dim=1, keepdim=True)

        # ---------------------------
        
        # ---------------------------
        features = torch.cat([
            p_seq, p_kg,       
            diff,              
            ent_seq, ent_kg,   
            cos_seq, cos_kg,   
            norm_s1, norm_s2, norm_k1, norm_k2 
        ], dim=1)
        
        # features shape: [B, 11]
        gate_logit = self.net(features).view(-1)
        alpha = torch.sigmoid(gate_logit)
        with torch.no_grad():
            self.aux_gate_mean = alpha.mean().detach()
            self.aux_gate_q = torch.quantile(alpha.view(-1), torch.tensor([0.1, 0.5, 0.9], device=alpha.device)).detach()
            self.aux_features = features.detach()
        return alpha, gate_logit        


class SingleModalClassifier(nn.Module):
    def __init__(self, gene_final_dim, final_mlp_dropout): # TODO remove final_mlp_dropout
        super().__init__()
        # self.mlp = nn.Sequential(  # TODO MLP or fc?
        #     nn.Linear(int(2 * gene_final_dim), gene_final_dim),
        #     nn.ReLU(),
        #     nn.Dropout(final_mlp_dropout),
        #     nn.Linear(gene_final_dim, 1),
        # )
        self.fc = nn.Linear(int(2 * gene_final_dim), 1)

    def forward(self, gene_1_emb, gene_2_emb):
        combined_emb = torch.cat((gene_1_emb, gene_2_emb), dim=1)
        combined_emb = self.fc(combined_emb)
        return combined_emb


class UMTClassifier(nn.Module):
    def __init__(self, gene_final_dim, final_mlp_dropout):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(int(4 * gene_final_dim), gene_final_dim),
            nn.ReLU(),
            nn.Dropout(final_mlp_dropout),
            nn.Linear(gene_final_dim, 1),
        )

    def forward(self, gene_1_distill_seq_emb, gene_2_distill_seq_emb, gene_1_distill_kg_emb, gene_2_distill_kg_emb):
        combined_emb = torch.cat((gene_1_distill_seq_emb, gene_2_distill_seq_emb, gene_1_distill_kg_emb, gene_2_distill_kg_emb), dim=1)
        combined_emb = self.fc(combined_emb)  # logit
        return combined_emb


class UMTModel(nn.Module):
    def __init__(
        self,
        omics_encoder_params,
        kg_encoder_params,
        gene_final_dim,
        final_mlp_dropout,
        task_type,
        seq_encoder_params=None,
        gate_detach_experts=True,
        gate_lambda=1.0,
        gate_tau=0.15,
    ):
        super().__init__()
        self.pretrain_omics_encoder = OmicsEncoder(omics_encoder_params)
        self.distill_omics_encoder = OmicsEncoder(omics_encoder_params)
        self.pretrain_kg_encoder = KGEncoder(kg_encoder_params)
        self.distill_kg_encoder = KGEncoder(kg_encoder_params)
        
        self.omics_classifier = SingleModalClassifier(gene_final_dim, final_mlp_dropout)
        self.kg_classifier = SingleModalClassifier(gene_final_dim, final_mlp_dropout)
        self.umt_classifier = UMTClassifier(gene_final_dim, final_mlp_dropout)
        self.seq_encoder = None
        self.distill_seq_encoder = None
        self.seq_classifier = None
        self.kg_seq_fuser = None  # Define the fusion layer
        self.kg_seq_moe_gate = None
        self.gate_detach_experts = gate_detach_experts
        self.gate_lambda = float(gate_lambda)
        self.gate_tau = gate_tau


        if seq_encoder_params is not None:
            self.seq_encoder = SequenceEncoder(seq_encoder_params)
            if task_type in ('umt','umt_naive'):
                self.distill_seq_encoder = SequenceEncoder(seq_encoder_params)
            self.seq_classifier = SingleModalClassifier(gene_final_dim, final_mlp_dropout)
            self.kg_seq_fuser = KGSeqFusion(input_dim=gene_final_dim, hidden_dim=gene_final_dim//2)  # Initialize gate fusion
            gate_input_dim = int(4 * gene_final_dim + 2)
            self.kg_seq_moe_gate = FeatureEngineeredGate()


    def forward(self, task_type, gene_1_omics_params, gene_2_omics_params, gene_1_kg_params, gene_2_kg_params, gene_1_seq_params=None, gene_2_seq_params=None):
        if task_type == "only_omics":
            gene_1_emb, gene_1_self_kl_loss, gene_1_cross_kl_loss = self.pretrain_omics_encoder(gene_1_omics_params)
            gene_2_emb, gene_2_self_kl_loss, gene_2_cross_kl_loss = self.pretrain_omics_encoder(gene_2_omics_params)
            logit = self.omics_classifier(gene_1_emb, gene_2_emb)
            return logit, (gene_1_self_kl_loss + gene_2_self_kl_loss), (gene_1_cross_kl_loss + gene_2_cross_kl_loss)

        elif task_type == "only_kg":
            logit = self.kg_classifier(self.pretrain_kg_encoder(gene_1_kg_params), self.pretrain_kg_encoder(gene_2_kg_params))  
            return logit

        elif task_type == "only_seq":
            if self.seq_encoder is None or self.seq_classifier is None:
                raise ValueError("seq_encoder_params is required for task_type='only_seq'")
            gene_1_emb = self.seq_encoder(gene_1_seq_params)
            gene_2_emb = self.seq_encoder(gene_2_seq_params)
            logit = self.seq_classifier(gene_1_emb, gene_2_emb)
            return logit
        
        elif task_type == "kg_seq":  # New task type for KG + seq fusion
            
            kg1 = self.pretrain_kg_encoder(gene_1_kg_params)
            kg2 = self.pretrain_kg_encoder(gene_2_kg_params)

            
            seq1 = self.seq_encoder(gene_1_seq_params)
            seq2 = self.seq_encoder(gene_2_seq_params)

            
            fused1, _ = self.kg_seq_fuser(kg1, seq1)
            fused2, _ = self.kg_seq_fuser(kg2, seq2)

            
            logit = self.kg_classifier(fused1, fused2)
            return logit
        
        elif task_type in ("kg_seq_moe", "kg_seq_moe_ft"):
            if self.seq_encoder is None or self.seq_classifier is None or self.kg_seq_moe_gate is None:
                raise ValueError("seq_encoder_params is required for task_type='kg_seq_moe' or 'kg_seq_moe_ft'")

            # --- KG expert ---
            kg1 = self.pretrain_kg_encoder(gene_1_kg_params)
            kg2 = self.pretrain_kg_encoder(gene_2_kg_params)
            logit_kg = self.kg_classifier(kg1, kg2)  # [B,1] or [B]

            # --- Seq expert ---
            s1 = self.seq_encoder(gene_1_seq_params)
            s2 = self.seq_encoder(gene_2_seq_params)
            logit_seq = self.seq_classifier(s1, s2)

            logit_seq_flat = logit_seq.view(-1)
            logit_kg_flat = logit_kg.view(-1)

            if self.gate_detach_experts:
                kg1 = kg1.detach()
                kg2 = kg2.detach()
                s1 = s1.detach()
                s2 = s2.detach()
                logit_seq_feat = logit_seq_flat.detach()
                logit_kg_feat = logit_kg_flat.detach()
            else:
                logit_seq_feat = logit_seq_flat
                logit_kg_feat = logit_kg_flat

            alpha, gate_logit = self.kg_seq_moe_gate(s1,s2,kg1,kg2,logit_seq_feat.unsqueeze(1), logit_kg_feat.unsqueeze(1))

            p_seq = torch.sigmoid(logit_seq_flat)
            p_kg = torch.sigmoid(logit_kg_flat)
            if self.gate_tau is None or self.gate_tau < 0:
                gate_mask = 1.0
            else:
                gate_mask = (torch.abs(p_seq - p_kg) > self.gate_tau).float()
            alpha_final = 0.5 + self.gate_lambda * (alpha - 0.5) * gate_mask
            alpha_final = torch.clamp(alpha_final, 0.0, 1.0)
            p = alpha_final * p_seq + (1.0 - alpha_final) * p_kg
            logit = LogitMoEGate._logit_from_prob(p)
            return logit, logit_kg_flat, logit_seq_flat, alpha_final, gate_logit

        elif task_type == "umt":
            if self.seq_encoder is None or self.distill_seq_encoder is None:
                raise ValueError("seq_encoder_params is required for task_type='umt'")
            if gene_1_seq_params is None or gene_2_seq_params is None:
                raise ValueError("gene_1_seq_params and gene_2_seq_params are required for task_type='umt'")

            # pretrain (teacher)
            gene_1_pretrain_seq_emb = self.seq_encoder(gene_1_seq_params)
            gene_2_pretrain_seq_emb = self.seq_encoder(gene_2_seq_params)
            gene_1_pretrain_kg_emb = self.pretrain_kg_encoder(gene_1_kg_params)
            gene_2_pretrain_kg_emb = self.pretrain_kg_encoder(gene_2_kg_params)

            # scratch (student)
            gene_1_distill_seq_emb = self.distill_seq_encoder(gene_1_seq_params)
            gene_2_distill_seq_emb = self.distill_seq_encoder(gene_2_seq_params)
            gene_1_distill_kg_emb = self.distill_kg_encoder(gene_1_kg_params)
            gene_2_distill_kg_emb = self.distill_kg_encoder(gene_2_kg_params)

            # concat
            combined_pretrain_seq_emb = torch.cat((gene_1_pretrain_seq_emb, gene_2_pretrain_seq_emb), dim=1)
            combined_distill_seq_emb = torch.cat((gene_1_distill_seq_emb, gene_2_distill_seq_emb), dim=1)
            combined_pretrain_kg_emb = torch.cat((gene_1_pretrain_kg_emb, gene_2_pretrain_kg_emb), dim=1)
            combined_distill_kg_emb = torch.cat((gene_1_distill_kg_emb, gene_2_distill_kg_emb), dim=1)

            # logit
            logit = self.umt_classifier(gene_1_distill_seq_emb, gene_2_distill_seq_emb, gene_1_distill_kg_emb, gene_2_distill_kg_emb)
            zero_loss = logit.new_tensor(0.0)

            return logit, combined_pretrain_seq_emb, combined_distill_seq_emb, combined_pretrain_kg_emb, combined_distill_kg_emb, zero_loss, zero_loss

        elif task_type == "umt_naive":
            if self.distill_seq_encoder is None:
                raise ValueError("seq_encoder_params is required for task_type='umt_naive'")
            if gene_1_seq_params is None or gene_2_seq_params is None:
                raise ValueError("gene_1_seq_params and gene_2_seq_params are required for task_type='umt_naive'")

            s1 = self.distill_seq_encoder(gene_1_seq_params)
            s2 = self.distill_seq_encoder(gene_2_seq_params)
            k1 = self.distill_kg_encoder(gene_1_kg_params)
            k2 = self.distill_kg_encoder(gene_2_kg_params)
            logit = self.umt_classifier(s1, s2, k1, k2)
            return logit

        elif task_type == "ume":
            # gene_1_omics_emb, _, _ = self.pretrain_omics_encoder(gene_1_omics_params)
            # gene_2_omics_emb, _, _ = self.pretrain_omics_encoder(gene_2_omics_params)
            # logit_omics = self.omics_classifier(gene_1_omics_emb, gene_2_omics_emb)
            s1 = self.seq_encoder(gene_1_seq_params)
            s2 = self.seq_encoder(gene_2_seq_params)
            logit_seq = self.seq_classifier(s1, s2)
            logit_kg = self.kg_classifier(self.pretrain_kg_encoder(gene_1_kg_params), self.pretrain_kg_encoder(gene_2_kg_params))  
            return logit_seq, logit_kg

    @staticmethod
    def load_state_dicts(model, dicts):
        for module_name, state_dict in dicts.items():
            getattr(model, module_name).load_state_dict(state_dict)

    @staticmethod
    def get_target_module_names(task_type):
        module_names = None
        if task_type == "only_omics":
            module_names = ["pretrain_omics_encoder", "omics_classifier"]
        elif task_type == "only_kg":
            module_names = ["pretrain_kg_encoder", "kg_classifier"]
        elif task_type == "only_seq":
            module_names = ["seq_encoder", "seq_classifier"]
        elif task_type == "kg_seq":  # Add "kg_seq" for the new fusion task
            module_names = ["pretrain_kg_encoder", "seq_encoder", "kg_seq_fuser", "kg_classifier"]
        elif task_type in ("kg_seq_moe", "kg_seq_moe_ft"):
            module_names = ["pretrain_kg_encoder", "kg_classifier", "seq_encoder", "seq_classifier", "kg_seq_moe_gate"]
        elif task_type == "umt":
            module_names = ["seq_encoder", "distill_seq_encoder", "pretrain_kg_encoder", "distill_kg_encoder", "umt_classifier"]
        elif task_type == "umt_naive":
            module_names = ["distill_seq_encoder", "distill_kg_encoder", "umt_classifier"]
        elif task_type == "ume":
            module_names = []  
        return module_names

    @staticmethod
    def state_dicts(model, task_type, accelerator):
        model = accelerator.unwrap_model(model)
        module_names = UMTModel.get_target_module_names(task_type)
        dicts = {}
        for module_name in module_names:
            state_dict = getattr(model, module_name).state_dict()
            dicts[module_name] = state_dict
        return dicts


class MultiOptimizer:
    def __init__(self, model, lr, weight_decay):
        optimizers = {}
        for module_name, module in model.named_children():  
            optimizers[module_name] = MultiOptimizer.__create_optimizer(module, lr, weight_decay)
        self.optimizers = optimizers

    def load_state_dicts(self, dicts):  
        for module_name, state_dict in dicts.items():
            self.optimizers[module_name].load_state_dict(state_dict)

    def prepare(self, accelerator):
        for module_name, optimizer in self.optimizers.items():
            optimizer = accelerator.prepare(optimizer)
            self.optimizers[module_name] = optimizer

    def set_module_lrs(self, module_lrs):
        for module_name, lr in module_lrs.items():
            optimizer = self.optimizers.get(module_name)
            if optimizer is None:
                continue
            for group in optimizer.param_groups:
                group["lr"] = lr

    def zero_grad(self, task_type):
        module_names = UMTModel.get_target_module_names(task_type)
        for module_name in module_names:
            self.optimizers[module_name].zero_grad()

    def step(self, task_type):
        module_names = UMTModel.get_target_module_names(task_type)
        for module_name in module_names:
            self.optimizers[module_name].step()

    def state_dicts(self, task_type, accelerator):
        module_names = UMTModel.get_target_module_names(task_type)
        dicts = {}
        for module_name in module_names:
            state_dict = accelerator.unwrap_model(self.optimizers[module_name]).state_dict()
            dicts[module_name] = state_dict
        return dicts

    @staticmethod
    def __create_optimizer(
        module: nn.Module,
        # base
        lr_head: float = 1e-3,
        lr_semproj: float = 1e-3,
        lr_rgcn: float = 1e-3,          
        lr_residual: float = 0.0,       
        # weight decay
        wd_head: float = 0.0,
        wd_semproj: float = 0.0,
        wd_rgcn: float = 0.0, # TODO 1e-4
        wd_residual: float = 0.0, 
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        """
        Create AdamW parameter groups with separate learning rates and no Lookahead or RAdam initialization.

        Default policy, tuned primarily for CV3:
            - head and biobert_proj: learning rate 1e-3, weight decay 0.
            - rgcn: learning rate 2e-4, weight decay 1e-4.
            - residual: learning rate 0 during warm-up, weight decay 1e-3.
              After warm-up, the residual learning rate can be changed to approximately 2e-4.

        Compatible parameter names:
            - gnn.* or *.gnn.*: RGCN parameters.
            - biobert_proj.*: Semantic projection parameters.
            - entity_residual.*: Residual embedding parameters.
            - All other names: Head or base parameters.
        """

        def bucket(name: str) -> str:
            if name.startswith("gnn.") or ".gnn." in name:
                return "rgcn"
            if name.startswith("biobert_proj.") or ".biobert_proj." in name:
                return "semproj"
            if name.startswith("entity_residual.") or ".entity_residual." in name:
                return "residual"
            return "head"

        lr_map: Dict[str, float] = {
            "head": lr_head,
            "semproj": lr_semproj,
            "rgcn": lr_rgcn,
            "residual": lr_residual,
        }
        wd_map: Dict[str, float] = {
            "head": wd_head,
            "semproj": wd_semproj,
            "rgcn": wd_rgcn,
            "residual": wd_residual,
        }

        
        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "ln.weight", "norm.weight")

        
        groups: Dict[Tuple[str, str], List[torch.nn.Parameter]] = {}
        seen = set()

        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in seen:
                continue
            seen.add(id(p))

            b = bucket(name)
            decay_flag = "nodecay" if any(k in name for k in no_decay) else "decay"
            key = (b, decay_flag)
            groups.setdefault(key, []).append(p)

        param_groups = []
        for (b, decay_flag), params in groups.items():
            group_lr = lr_map[b]
            group_wd = 0.0 if decay_flag == "nodecay" else wd_map[b]
            param_groups.append(
                {
                    "params": params,
                    "lr": group_lr,
                    "weight_decay": group_wd,
                    
                    "group": b,
                }
            )

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=betas,
            eps=eps,
        )
        return optimizer


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.

    @From: MLEC_iSL
    @Modified: Jmpax
    """

    def __init__(self, accelerator, patience=7, verbose=False, reverse=False, delta=0, path="checkpoint.pt", auto_save_model=True):
        """
        Initialize early stopping.

        Args:
            accelerator: Accelerator process manager.
            patience (int): Number of epochs to wait after the last validation improvement. Early stopping cannot occur if this value exceeds the total number of epochs. Defaults to 7.
            verbose (bool): If True, print a message for each validation improvement. Defaults to False.
            reverse (bool): If True, smaller metric values are better; otherwise, larger values are better.
            delta (float): Minimum change required to qualify as an improvement. A value of 0 treats any decrease in validation loss as an improvement. Defaults to 0.
            path (str): Path at which to save the checkpoint. Defaults to "checkpoint.pt".
            trace_func (function): Trace-printing function; disabled in this implementation. Defaults to print.
            auto_save_model (bool): Whether to save the model every time the object is called. Defaults to True.
        """
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.reverse = reverse
        self.counter = 0  
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf  
        self.delta = delta
        self.path = path
        # self.trace_func = trace_func
        self.auto_save_model = auto_save_model

    def __call__(self, val_loss, model):
        if self.reverse:
            # loss
            score = -val_loss
        else:
            # AUC/AUPR
            score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            return True
        elif score <= self.best_score + self.delta:  
            self.counter += 1
            rank_main_print(self.accelerator, f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            return True

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation indicator better"""
        if self.verbose:
            rank_main_print(self.accelerator, f"The validation indicator is better ({self.val_loss_min:.6f} --> {val_loss:.6f}).")

        if self.auto_save_model:
            rank_main_print(self.accelerator, "Saving model ...")
            torch.save(model.state_dict(), self.path)

        self.val_loss_min = val_loss
