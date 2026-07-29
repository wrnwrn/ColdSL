
import argparse
import copy
from datetime import datetime
import json
import os
import time


from accelerate import Accelerator, DataLoaderConfiguration
from accelerate import DistributedDataParallelKwargs as DDPK
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


from dataset import MultiModalDataset, SLDataset
from executor import Tester, Trainer
from model import EarlyStopping, MultiOptimizer, UMTModel
from util.my import rank_main_print, rank_print, set_seed, time_elapsed
from visualization import plot_train_curve


def init_argparse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # program
    parser.add_argument("--device_id_for_debug", type=int, default=-1, help="a single GPU device id when debugging. MUST set -1 when using accelerate")
    parser.add_argument("--specify_result_saving_folder", type=str, default="", help="save current result to specified folder if the value is not empty")

    parser.add_argument("--task_type", type=str, default="kg_seq_moe", help="only_omics, only_kg, only_seq, umt, umt_naive, ume, kg_seq, kg_seq_moe, kg_seq_moe_ft") 
    parser.add_argument("--omics_ckpt_path", type=str, default=f"", help="seq teacher checkpoint data path") 
    parser.add_argument("--kg_ckpt_path", type=str, default=f"", help="kg teacher checkpoint data path") # ../result/4_omics_into_onlyKG_with_BioBERT/kg_biobert_pan_cv_3_fold_2_only_kg_kg_experiment_C/checkpoint.pth
    parser.add_argument("--gate_strategy", type=str, default="two_stage", choices=["two_stage", "end2end_aux"], help="two_stage or end2end_aux for kg_seq_moe")
    parser.add_argument("--gate_aux_weight", type=float, default=0.5, help="aux loss weight for gate supervision") 
    parser.add_argument("--gate_main_weight_stage1", type=float, default=1.0, help="main task loss weight during gate warmup") 
    parser.add_argument("--gate_warmup_epochs", type=int, default=5, help="epochs to train gate before unfreezing experts") 
    parser.add_argument("--gate_unfreeze_epoch", type=int, default=-1, help="epoch to unfreeze experts (-1 means never)")
    parser.add_argument("--gate_detach_experts", action="store_true", help="detach expert outputs when training gate")
    parser.add_argument("--no_gate_detach_experts", action="store_false", dest="gate_detach_experts", help="allow gate to backprop into experts")
    parser.set_defaults(gate_detach_experts=True) 
    parser.add_argument("--gate_lambda", type=float, default=1.0, help="shrinkage strength for gate blending (0=UME avg, 1=raw gate)")
    parser.add_argument("--gate_tau", type=float, default=-1, help="apply gate only when |p_seq - p_kg| > tau; set <0 to disable")
    parser.add_argument("--kg_seq_moe_ft_ckpt_path", type=str, default=f"", help="warm-start checkpoint for kg_seq_moe_ft")
    parser.add_argument("--expert_lr", type=float, default=1e-6, help="learning rate for experts during kg_seq_moe_ft")

    # model
    parser.add_argument("--cancer_type", type=str, default="pan", help="one cancer type")
    parser.add_argument("--metric", type=int, default=3, help="CV1, CV2, CV3")
    parser.add_argument("--train_fold", type=int, default=1, help="one of the folds in 5-fold cross validation")
    parser.add_argument("--epochs", type=int, default=30, help="number of maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=2048, help="batch size of training, validating and testing")
    parser.add_argument("--split_batches", type=bool, default=True, help="If True, the actual batch size of model is `batch_size`, which must be an integer multiple of GPUs count you used. If False, the actual batch size of model is `batch_size` * `number of GPUs`.")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="l2 regularized weight decay factor")
    parser.add_argument("--gradient_accumulation", type=int, default=1, help="gradient accumulation quantity")
    parser.add_argument("--patience", type=int, default=999, help="the number of epoch of early stop tolerance")

    # OmicsEncoder
    parser.add_argument("--hid_dim", type=int, default=128, help="hidden dim after Encoder")
    parser.add_argument("--omics_types", nargs="+", type=str, default=["cna", "exp", "mut"], help="name of omics category used")
    parser.add_argument("--vae_hidden_dims", nargs="+", type=int, default=[2048,1024,512, 256], help="hidden dims list of VAE")
    parser.add_argument("--vae_dropout", type=float, default=0.2, help="dropout rate of VAE layer")
    parser.add_argument("--self_kl_loss_weight", type=float, default=0.1)
    parser.add_argument("--cross_kl_loss_weight", type=float, default=0.5)

    # KGEncoder
    parser.add_argument("--in_channels", type=int, default=128, help="input dimension of RGCN")
    parser.add_argument("--hidden_channels", type=int, default=256, help="hidden dimension of RGCN")
    parser.add_argument("--gcn_layers", type=int, default=2, help="layers count of RGCN")
    parser.add_argument("--graph_dropout", type=float, default=0.5, help="dropout rate of the graph")
    parser.add_argument("--kg_experiment", type=str, default="C", choices=["A", "B", "C", "D"], help="A: random+RGCN, B: random+no-graph, C: BioBERT+RGCN, D: BioBERT+no-graph")
    parser.add_argument("--biobert_embedding_path", type=str, default="../data/pan/E_biobert.npy", help="path to BioBERT embedding .npy")
    parser.add_argument("--seq_embedding_path", type=str, default="../data/pan/E_esm2_600.npy", help="path to ESM2 embedding .npy")
    parser.add_argument("--p_rel", type=float, default=0.3, help="relation dropout prob for KG subgraph")
    parser.add_argument("--kg_biobert_fusion_type", type=int, default=3, choices=[0, 1, 2, 3], help="0: only use RGCN output, 1: concat fusion, 2: residual sum, 3: residual gate")
    parser.add_argument("--kg_node_type_emb_switch", type=bool, default=True, help="only useful when kg_biobert_fusion_type choose 3")
    parser.add_argument("--kg_node_type_emb_dim", type=int, default=32)

    # SeqEncoder
    parser.add_argument("--seq_encoder_mlp_dropout", type=float, default=0.3)

    # Classifier
    parser.add_argument("--gene_final_dim", type=int, default=128, help="gene final dim to input classifier")
    parser.add_argument("--final_mlp_dropout", type=float, default=0.5, help="the dropout of final dimensionality reduction MLP")
    parser.add_argument("--lambda_distill", type=float, default=1, help="weight of distillation loss")

    return parser.parse_args()


def load_checkpoint(ckpt_path, accelerator, model, multi_optimizer):
    checkpoint_dict = torch.load(ckpt_path, weights_only=True, map_location=accelerator.device)
    model_dicts = checkpoint_dict["model"]
    optimizer_dicts = checkpoint_dict["op"]
    UMTModel.load_state_dicts(model, model_dicts)
    multi_optimizer.load_state_dicts(optimizer_dicts)


def main():
    start_time_prepare = time.time()

    
    args = init_argparse()
    if args.task_type == "kg_seq_moe_ft":
        args.gate_detach_experts = True

    
    
    
    kwargs = DDPK(find_unused_parameters=True, broadcast_buffers=True)
    accelerator = Accelerator(
        kwargs_handlers=[kwargs],
        gradient_accumulation_steps=args.gradient_accumulation,
        
        
        
        dataloader_config=DataLoaderConfiguration(split_batches=args.split_batches),
    )
    rank_main_print(accelerator, f"param: {args}")

    
    
    if accelerator.is_main_process:  
        if args.device_id_for_debug != -1:
            torch.cuda.set_device(args.device_id_for_debug)

    
    set_seed(2025)

    ##################################################################### result related settings
    
    result_fold_save_path_prefix = "../result"  
    result_folder_name = args.specify_result_saving_folder
    if result_folder_name.strip() == "":
        current_time = datetime.now()  
        result_folder_name = current_time.strftime("%Y_%m_%d_%H_%M_%S")  
    result_fold_path = result_fold_save_path_prefix + "/" + result_folder_name

    if accelerator.is_main_process:  
        
        if not os.path.exists(result_fold_path):
            os.makedirs(result_fold_path)

        
        with open(result_fold_path + "/hyper_parameters.json", "w") as f:  
            json.dump(vars(args), f, indent=4)  
        print("success to save hyper parameters to:", result_fold_path + "/hyper_parameters.json")

    ##################################################################### create model
    # read dataset
    data_base_path = f"../data/{args.cancer_type}"
    mm_dataset = MultiModalDataset(
        accelerator,
        gene_path=os.path.join(data_base_path, "gene.pt"),
        omics_path_dict={omics_type: f"{data_base_path}/{omics_type}.npy" for omics_type in args.omics_types},
        kg_graph_path=os.path.join(data_base_path, "kg.pt"),
    )
    
    sl_train_dataset = SLDataset(accelerator, sl_path=f"../data/{args.cancer_type}/cv_{args.metric}_fold_{args.train_fold}/train_sl.npy", mm_dataset=mm_dataset)
    sl_val_dataset = SLDataset(accelerator, sl_path=f"../data/{args.cancer_type}/cv_{args.metric}_fold_{args.train_fold}/val_sl.npy", mm_dataset=mm_dataset)
    sl_test_dataset = SLDataset(accelerator, sl_path=f"../data/{args.cancer_type}/cv_{args.metric}_fold_{args.train_fold}/test_sl.npy", mm_dataset=mm_dataset)

    def sl_np_to_df(np_sl, gene_idx_to_name):
        gene1_names = [gene_idx_to_name[idx] for idx in np_sl[:, 0]]  
        gene2_names = [gene_idx_to_name[idx] for idx in np_sl[:, 1]]
        classes = np_sl[:, 2]
        df_SL = pd.DataFrame({"gene1": gene1_names, "gene2": gene2_names, "class": classes})  
        return df_SL

    def split_stats(df_train, df_val, df_test, gene_cols=("gene1", "gene2")):
        """
        Summarize detailed statistics for the training, validation, and test sets and quantify differences between validation and test data.

        Args:
            df_train: Training-set DataFrame.
            df_val: Validation-set DataFrame.
            df_test: Test-set DataFrame.
            gene_cols: Tuple containing the gene-name column names.

        Returns:
            A dictionary of statistics prefixed with val_*, test_*, or diff_*.
        """
        col1, col2 = gene_cols
        
        
        
        train_genes = set(df_train[col1]).union(set(df_train[col2]))
        
        
        def _get_single_set_stats(df, name):
            current_genes = set(df[col1]).union(set(df[col2]))
            
            
            n_samples = len(df)
            n_unique_genes = len(current_genes)
            pos_rate = df["class"].mean() if "class" in df.columns else np.nan
            
            
            unseen_genes = current_genes - train_genes
            unseen_gene_ratio = len(unseen_genes) / max(1, n_unique_genes)
            
            # Boolean masks
            mask_c1_unseen = ~df[col1].isin(train_genes)
            mask_c2_unseen = ~df[col2].isin(train_genes)
            
            
            
            pair_has_unseen = (mask_c1_unseen | mask_c2_unseen).mean()
            
            pair_both_unseen = (mask_c1_unseen & mask_c2_unseen).mean()

            return {
                f"{name}_n": n_samples,
                f"{name}_genes_count": n_unique_genes,
                f"{name}_pos_rate": float(pos_rate),
                f"{name}_unseen_gene_ratio": float(unseen_gene_ratio),
                f"{name}_pair_has_unseen_ratio": float(pair_has_unseen),
                f"{name}_pair_both_unseen_ratio": float(pair_both_unseen),
            }, current_genes

        
        stats_val, val_genes_set = _get_single_set_stats(df_val, "val")
        stats_test, test_genes_set = _get_single_set_stats(df_test, "test")
        
        
        results = {**stats_val, **stats_test}
        
        
        
        
        
        results["diff_pos_rate"] = results["test_pos_rate"] - results["val_pos_rate"]
        results["diff_unseen_gene_ratio"] = results["test_unseen_gene_ratio"] - results["val_unseen_gene_ratio"]
        
        
        
        overlap_genes = val_genes_set.intersection(test_genes_set)
        results["val_test_gene_overlap_ratio"] = len(overlap_genes) / max(1, len(test_genes_set))
        
        
        
        
        
        val_pairs = set(zip(df_val[col1], df_val[col2]))
        test_pairs = set(zip(df_test[col1], df_test[col2]))
        overlap_pairs = val_pairs.intersection(test_pairs)
        
        results["val_test_exact_pair_overlap_count"] = len(overlap_pairs) 
        
        return results


    def set_module_requires_grad(model, module_names, requires_grad):
        target_model = model.module if hasattr(model, "module") else model
        for module_name in module_names:
            module = getattr(target_model, module_name, None)
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = requires_grad

    
    # print( split_stats(
    #     sl_np_to_df(sl_train_dataset.np_sl,mm_dataset.gene_idx_to_name),
    #     sl_np_to_df(sl_val_dataset.np_sl,mm_dataset.gene_idx_to_name),
    #      sl_np_to_df(sl_test_dataset.np_sl,mm_dataset.gene_idx_to_name),
    #     ))

    task_type = args.task_type
    kg_experiment = args.kg_experiment.upper()
    use_graph = kg_experiment in {"A", "C"}
    entity_init = "random" if kg_experiment in {"A", "B"} else "biobert"
    biobert_embeddings = None

    if entity_init == "biobert":
        biobert_path = args.biobert_embedding_path
        rank_main_print(accelerator, f"reading BioBERT embeddings: {biobert_path}")
        biobert_embeddings = torch.from_numpy(np.load(biobert_path)).float()
        if biobert_embeddings.shape[0] != mm_dataset.get_entity_vocab_size():
            raise ValueError(
                f"BioBERT embeddings row count {biobert_embeddings.shape[0]} "
                f"does not match entity_vocab_size {mm_dataset.get_entity_vocab_size()}"
            )

    seq_embeddings = None
    if task_type in ["only_seq", "kg_seq", "kg_seq_moe", "kg_seq_moe_ft", "umt", "umt_naive", "ume"]:
        seq_path = args.seq_embedding_path
        rank_main_print(accelerator, f"reading ESM2 embeddings: {seq_path}")
        seq_embeddings = torch.from_numpy(np.load(seq_path)).float()
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
        "num_node_types":int(mm_dataset.kg_graph.node_type.max().item() + 1),
        "p_rel": args.p_rel,
        "kg_biobert_fusion_type": args.kg_biobert_fusion_type,
        "kg_node_type_emb_switch":args.kg_node_type_emb_switch,
        "kg_node_type_emb_dim":args.kg_node_type_emb_dim,
    }
    seq_encoder_params = None
    if seq_embeddings is not None:
        seq_encoder_params = {
            "seq_embeddings": seq_embeddings,
            "gene_final_dim": args.gene_final_dim,
            "seq_encoder_mlp_dropout":args.seq_encoder_mlp_dropout,
        }
    model = UMTModel(
        omics_encoder_params=omics_encoder_params,
        kg_encoder_params=kg_encoder_params,
        gene_final_dim=args.gene_final_dim,
        final_mlp_dropout=args.final_mlp_dropout,
        task_type = task_type,
        seq_encoder_params=seq_encoder_params,
        gate_detach_experts=args.gate_detach_experts,
        gate_lambda=args.gate_lambda,
        gate_tau=args.gate_tau,
    )

    
    early_stopping = EarlyStopping(accelerator, patience=args.patience, verbose=True, reverse=False, path="", auto_save_model=False)  

    
    multi_optimizer = MultiOptimizer(model, args.lr, args.weight_decay)

    
    if task_type == "umt" or task_type == "ume":
        load_checkpoint(args.omics_ckpt_path, accelerator, model, multi_optimizer)
        load_checkpoint(args.kg_ckpt_path, accelerator, model, multi_optimizer)

        
        if model.seq_encoder is not None:
            for param in model.seq_encoder.parameters():  
                param.requires_grad = False
        for param in model.pretrain_kg_encoder.parameters():
            param.requires_grad = False

    if task_type == "kg_seq_moe_ft":
        # load_checkpoint(args.kg_seq_moe_ft_ckpt_path, accelerator, model, multi_optimizer)
        checkpoint_dict = torch.load(args.kg_seq_moe_ft_ckpt_path, weights_only=True, map_location=accelerator.device)
        model_dicts = checkpoint_dict["model"]
        optimizer_dicts = checkpoint_dict["op"]
        
        single_model_dict={}
        single_optimizer_dict = {}
        single_model_dict['kg_seq_moe_gate'] = model_dicts['kg_seq_moe_gate']
        single_optimizer_dict['kg_seq_moe_gate'] = optimizer_dicts['kg_seq_moe_gate']
        UMTModel.load_state_dicts(model, single_model_dict)
        multi_optimizer.load_state_dicts(single_optimizer_dict)

        multi_optimizer.set_module_lrs(
            {
                "pretrain_kg_encoder": args.expert_lr,
                "seq_encoder": args.expert_lr,
                "kg_classifier": args.lr,
                "seq_classifier": args.lr,
                "kg_seq_moe_gate": args.lr,
            }
        )

    
    if task_type == "kg_seq_moe" and args.gate_strategy == "two_stage" :
        load_checkpoint(args.omics_ckpt_path, accelerator, model, multi_optimizer)
        load_checkpoint(args.kg_ckpt_path, accelerator, model, multi_optimizer)

    
    
    train_dataloader = DataLoader(sl_train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(sl_val_dataset, batch_size=args.batch_size)
    test_dataloader = DataLoader(sl_test_dataset, batch_size=args.batch_size)

    
    model, train_dataloader, val_dataloader, test_dataloader = accelerator.prepare(model, train_dataloader, val_dataloader, test_dataloader)
    multi_optimizer.prepare(accelerator)

    
    lambda_distill = args.lambda_distill
    trainer = Trainer(
        model,
        args.batch_size,
        multi_optimizer,
        lambda_distill=lambda_distill,
        self_kl_loss_weight=args.self_kl_loss_weight,
        cross_kl_loss_weight=args.cross_kl_loss_weight,
        gate_aux_weight=args.gate_aux_weight,
        gate_main_weight=1.0,
    )
    validator = Tester(
        model,
        args.batch_size,
        lambda_distill=lambda_distill,
        self_kl_loss_weight=args.self_kl_loss_weight,
        cross_kl_loss_weight=args.cross_kl_loss_weight,
        epochs=args.epochs,
        gate_aux_weight=args.gate_aux_weight,
    )  
    tester = Tester(
        model,
        args.batch_size,
        lambda_distill=lambda_distill,
        self_kl_loss_weight=args.self_kl_loss_weight,
        cross_kl_loss_weight=args.cross_kl_loss_weight,
        epochs=args.epochs,
        gate_aux_weight=args.gate_aux_weight,
    )  

    end_time_prepare = time.time()

    
    start_time_learn = time.time()

    kg_graph = mm_dataset.get_kg_graph()

    df_evaluate = pd.DataFrame()  
    checkpoint_epoch = 0
    checkpoint_model = None
    checkpoint_optimizer = None
    gate_freeze_modules = ["pretrain_kg_encoder", "kg_classifier", "seq_encoder", "seq_classifier"]

    
    for epoch in range(1, args.epochs + 1):
        start_time_epoch = time.time()
        rank_main_print(accelerator, f"\n##################################################################### training epoch: {epoch}")

        if task_type == "kg_seq_moe":
            if args.gate_strategy == "two_stage":
                if epoch <= args.gate_warmup_epochs:
                    set_module_requires_grad(model, gate_freeze_modules, False)
                    trainer.gate_main_weight = args.gate_main_weight_stage1
                    trainer.gate_aux_weight = args.gate_aux_weight
                else:
                    if args.gate_unfreeze_epoch >= 0 and epoch >= args.gate_unfreeze_epoch:
                        set_module_requires_grad(model, gate_freeze_modules, True)
                    trainer.gate_main_weight = 1.0
                    trainer.gate_aux_weight = args.gate_aux_weight
            elif args.gate_strategy == "end2end_aux":
                set_module_requires_grad(model, gate_freeze_modules, True)
                trainer.gate_main_weight = 1.0
                trainer.gate_aux_weight = args.gate_aux_weight
        elif task_type == "kg_seq_moe_ft":
            set_module_requires_grad(model, gate_freeze_modules, True)
            trainer.gate_main_weight = 1.0
            trainer.gate_aux_weight = args.gate_aux_weight

        
        rank_main_print(accelerator, "training...")
        if task_type == "ume":  
            evaluate_results_train = {"AUC": 0, "AUPR": 0, "F1": 0}
        else:
            evaluate_results_train = trainer.train(accelerator, train_dataloader, kg_graph, task_type)
        rank_main_print(accelerator, f"evaluate_results_train: {evaluate_results_train}")

        
        rank_main_print(accelerator, "validating...")
        evaluate_results_val = validator.test(accelerator, val_dataloader, kg_graph, task_type)
        val_auc = evaluate_results_val["AUC"]  
        rank_main_print(accelerator, f"evaluate_results_val: {evaluate_results_val}")

        rank_main_print(accelerator, "testing...")
        evaluate_results_test = tester.test(accelerator, test_dataloader, kg_graph, task_type)
        rank_main_print(accelerator, f"evaluate_results_test: {evaluate_results_test}")

        
        if accelerator.is_main_process:  
            
            evaluate_results_train = {f"train_{key}": value for key, value in evaluate_results_train.items()}
            evaluate_results_val = {f"val_{key}": value for key, value in evaluate_results_val.items()}
            evaluate_results_test = {f"test_{key}": value for key, value in evaluate_results_test.items()}
            evaluate_results_merge = evaluate_results_train | evaluate_results_val | evaluate_results_test
            evaluate_results_merge["epoch"] = epoch

            
            df_evaluate = pd.concat([df_evaluate, pd.DataFrame([evaluate_results_merge])], ignore_index=True)
            plot_train_curve(df_evaluate, result_fold_path + "/train_val_evaluate.png") 

        
        end_time_epoch = time.time()
        rank_main_print(accelerator, f"epoch {epoch} time use: {time_elapsed(start_time_epoch, end_time_epoch)}")

        
        is_result_better = early_stopping(val_auc, model=None)  
        if is_result_better:
            accelerator.wait_for_everyone()  
            if accelerator.is_main_process:  
                
                checkpoint_model = UMTModel.state_dicts(model, args.task_type, accelerator)
                checkpoint_optimizer = multi_optimizer.state_dicts(args.task_type, accelerator)
                
                checkpoint_model = copy.deepcopy(checkpoint_model)
                checkpoint_optimizer = copy.deepcopy(checkpoint_optimizer)
                checkpoint_epoch = epoch

        
        if task_type != "ume" and early_stopping.early_stop:
            rank_main_print(accelerator, "Early Stopping!!!")
            break

        if task_type == "ume":  
            break

    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:  
        
        if args.task_type != "ume":
            torch.save(
                {"epoch": checkpoint_epoch, "model": checkpoint_model, "op": checkpoint_optimizer},
                f"{result_fold_path}/checkpoint.pth",
            )
            print("success to save model checkpoint to:", f"{result_fold_path}/checkpoint.pth")

    
    if accelerator.is_main_process:  
        
        df_evaluate.to_csv(result_fold_path + "/train_val_evaluate.csv", index=False)
        print("success to save train and val evaluate data to:", result_fold_path + "/train_val_evaluate.csv")

        
        if task_type != "ume":  
            # plot_train_curve(df_evaluate, result_fold_path + "/train_val_evaluate.png")
            print("success to save evaluate curve to:", result_fold_path + "/train_val_evaluate.png")

    # show ending
    end_time_learn = time.time()
    ending_str = "Finished!!! " + time_elapsed(start_time_prepare, end_time_prepare, "prepare time use: ") + ", " + time_elapsed(start_time_learn, end_time_learn, "learn time use: ") + "."
    rank_print(accelerator, ending_str)


if __name__ == "__main__":
    main()
