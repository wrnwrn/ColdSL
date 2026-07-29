
import numpy as np
import torch
from torch.utils.data import Dataset


from util.my import rank_print


class MultiModalDataset(Dataset):
    def __init__(self, accelerator, gene_path, omics_path_dict, kg_graph_path):
        """
        Initialize the dataset and load the multimodal data.

        Args:
            accelerator: Accelerator instance.
            gene_path: Path to the mapping from gene indices to gene names.
            omics_path_dict: Mapping from omics modality names to their data file paths.
            kg_graph_path: Path to the knowledge graph data.
        """

        
        rank_print(accelerator, f"reading gene data: {gene_path}")
        gene_idx_to_name = torch.load(gene_path, weights_only=False)
        self.gene_idx_to_name = gene_idx_to_name
        self.gene_name_to_idx = {gene: idx for idx, gene in gene_idx_to_name.items()}  

        
        self.omics_count = len(omics_path_dict)  
        omics_data = {}
        for omics_name, csv_path in omics_path_dict.items():
            rank_print(accelerator, f"reading {omics_name} data: {csv_path}")
            np_omics = np.load(csv_path)
            omics_data[omics_name] = np_omics
        self.omics_data = omics_data  

        
        rank_print(accelerator, f"reading kg data: {kg_graph_path}")
        kg_data_dict = torch.load(kg_graph_path, weights_only=False)  

        self.entity_to_index = kg_data_dict["entity_to_index"]
        self.index_to_entity = kg_data_dict["index_to_entity"]
        self.entity_idx_to_gene_name = kg_data_dict["entity_idx_to_gene_name"]
        self.gene_name_to_entity_idx = kg_data_dict["gene_name_to_entity_idx"]
        kg_graph = kg_data_dict["kg_graph"]  

        
        all_edge_indices = []  
        all_edge_types = []  
        type_index = 0
        for edge_type in kg_graph.edge_types:
            edge_index = kg_graph[edge_type].edge_index  
            all_edge_indices.append(edge_index)

            
            edge_index_length = edge_index.size(1)  
            
            edge_type_tensor = torch.full((edge_index_length,), type_index, dtype=torch.long)
            all_edge_types.append(edge_type_tensor)  
            type_index += 1  
        kg_graph.concatenated_edge_index = torch.cat(all_edge_indices, dim=1)  
        kg_graph.concatenated_edge_type = torch.cat(all_edge_types)  
        
        
        
        # edge = kg_graph.concatenated_edge_index  # [2, E]
        
        # deg = torch.bincount(edge[0], minlength=num_nodes) + torch.bincount(edge[1], minlength=num_nodes)
        # kg_graph.node_deg = deg  # LongTensor [num_nodes]
    
        kg_graph.node_type    =self.get_global_node_type_tensor(kg_graph)   # shape [entity_vocab_size], dtype long
        self.kg_graph = kg_graph
        
    def get_global_node_type_tensor(self, hetero_data):
        
        max_global_id = 0
        for node_type in hetero_data.node_types:
            if hasattr(hetero_data[node_type], 'node_id'):
                ids = hetero_data[node_type].node_id
                if ids.numel() > 0:
                    current_max = ids.max().item()
                    if current_max > max_global_id:
                        max_global_id = current_max
        
        
        
        global_type_tensor = torch.zeros(max_global_id + 1, dtype=torch.long)
        
        
        for i, node_type in enumerate(hetero_data.node_types):
            ids = hetero_data[node_type].node_id.long()
            
            
            
            
            global_type_tensor[ids] = i + 1
            
        return global_type_tensor
    
    def __len__(self):
        """
        Return the number of samples in the dataset.
        """

        return len(self.gene_idx_to_name)

    def __getitem__(self, idx):
        """
        Retrieve the multimodal data at the specified index.

        Args:
            idx: Index of the sample in the dataset.
        """

        
        omics_data_row_tensor = []
        for omics_name, np_omics in self.omics_data.items():
            omic_value = np_omics[idx]
            omics_data_row_tensor.append(torch.Tensor(omic_value))

        
        gene_name = self.gene_idx_to_name[idx]
        entity_idx = self.gene_name_to_entity_idx[gene_name]

        
        result = {
            
            "omics_data_list": omics_data_row_tensor,
            "gene_entity": torch.tensor(entity_idx, dtype=torch.long),
            "gene_idx": torch.tensor(idx, dtype=torch.long),
        }
        return result

    def get_omics_input_dim(self):
        """
        Return the sample dimension of the tissue-omics data.
        """

        first_key = list(self.omics_data.keys())[0]
        return self.omics_data[first_key].shape[1]

    def get_kg_relations_count(self):
        return len(self.kg_graph.edge_types)

    def get_entity_vocab_size(self):
        """
        Return the size of the complete knowledge graph entity vocabulary used for node embeddings.

        Index 0 is reserved for <PAD>, and index 1 is reserved for <MASK>.
        """

        return len(self.entity_to_index)

    def get_kg_graph(self):
        return self.kg_graph


class SLDataset(Dataset):
    def __init__(self, accelerator, sl_path, mm_dataset):
        """
        Initialize the dataset and load the synthetic lethality data.

        Args:
            accelerator: Accelerator instance.
            sl_path: Path to the synthetic lethality dataset.
            mm_dataset: Complete multimodal dataset.
        """

        rank_print(accelerator, f"reading SL data: {sl_path}")
        np_sl = np.load(sl_path)
        self.np_sl = np_sl
        self.mm_dataset = mm_dataset

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """

        return self.np_sl.shape[0]

    def __getitem__(self, idx):
        """
        Retrieve the data associated with the synthetic lethal pair at the specified index.

        All required fields are assembled here because __getitem__ does not support user-side batched indexing.

        Args:
            idx: Index of the synthetic lethal pair in the dataset.
        """

        sample = self.np_sl[idx]

        gene1_data = self.mm_dataset[sample[0]]
        gene2_data = self.mm_dataset[sample[1]]
        label = sample[2]

        gene1_data = {f"gene_1_{key}": value for key, value in gene1_data.items()}  
        gene2_data = {f"gene_2_{key}": value for key, value in gene2_data.items()}  

        
        result = gene1_data | gene2_data
        result["label"] = torch.tensor(label, dtype=torch.float)  
        result["sample_idx"] = torch.tensor(idx, dtype=torch.long)

        return result
