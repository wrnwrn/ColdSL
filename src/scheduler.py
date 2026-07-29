
import argparse
import os
import queue
import random
import socket
import subprocess
import threading
import time


import torch


from util.my import cur_time_str


def init_argparse():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--conda_env_name", type=str, default="ME", help="conda's environment name")
    parser.add_argument("--single_card_memory", type=int, default=48, help="unit: GB")
    parser.add_argument("--card_groups", nargs="+", type=str, default=["0,1","2,3","4,5","6,7"], help="list of card groups. two in one group, separated by commas in each group")
    parser.add_argument("--worker_count_per_card_group", type=int, default=24)
    parser.add_argument("--base_port", type=int, default=33333, help="port for accelerate. step 1 within card groups and step 100 between")
    parser.add_argument("--task_folder_prex", type=str, default="test", help="the prefix of task folder to distinguish different configurations")
    parser.add_argument("--addition_unified_params", type=str, default="", help="effective for all datasets, eg. --hid_dim 128")

    return parser.parse_args()


def generate_tasks(task_folder_prex, addition_unified_params):
    
    custom_hyper_parameters = {  
        "BRCA": {
            "cv_1": {"epochs": 200},
            "cv_2": {"epochs": 200},
        },
        "CESC": {
            "cv_1": {"epochs": 200},
            "cv_2": {"epochs": 200},
        },
        "COAD": {
            "cv_1": {"epochs": 200},  # 150
            "cv_2": {"epochs": 200},
        },
        "KIRC": {
            "cv_1": {"epochs": 200},
            "cv_2": {"epochs": 200},
        },
        "LAML": {
            "cv_1": {"epochs": 200},  # 150
            "cv_2": {"epochs": 200},
        },
        "LUAD": {
            "cv_1": {"epochs": 200},
            "cv_2": {"epochs": 200},
        },
        "OV": {
            "cv_1": {"epochs": 200},
            "cv_2": {"epochs": 200},
        },
        "SKCM": {
            "cv_1": {"epochs": 200},
            "cv_2": {"epochs": 200},
        },
        "pan": {
            "cv_1": { "batch_size": 2048, "omics_types": "cna exp mut", "vae_hidden_dims": "2048 1024 512 256"},  # TODO 202603082027: 30 epoch
            "cv_2": { "batch_size": 2048, "omics_types": "cna exp mut", "vae_hidden_dims": "2048 1024 512 256"},
            "cv_3": { "batch_size": 2048, "omics_types": "cna exp mut", "vae_hidden_dims": "2048 1024 512 256"},
        },
        # "pan": {
        #     "cv_1": {"batch_size": 2048, "omics_types": "cna exp mut", "vae_hidden_dims": "2048 1024 512 256"},  # 20
        #     "cv_2": {"batch_size": 2048, "omics_types": "cna exp mut", "vae_hidden_dims": "2048 1024 512 256"},
        #     "cv_3": {"batch_size": 2048, "omics_types": "cna exp mut", "vae_hidden_dims": "2048 1024 512 256"},
        # },
    }

    
    mem_usage_dict = {
        "only_omics": {
            "single": 6,
            "pan": 14,
        },
        "only_kg": {
            "single": 20,
            "pan": 24, 
        },
        "only_seq":{
            "pan": 8
        },
        "umt": {
            "single": 20,
            "pan": 32,
        },
        "umt_naive": {
            "pan": 32,
        },
        "kg_seq_moe": {
            "pan": 16,
        },
        "ume": {
            "single": 8,  
            "pan": 16,
        },
    }

    experiments = {
        # Ablation / baselines
        # "random_no_graph": {
        #     "kg_experiment": "B",
        # },
        # "only_kg_random_with_graph": {
        #     "kg_experiment": "A",
        #     "p_rel": 0.0,
        #     "kg_biobert_fusion_type": 0,
        # },
        # "only_biobert_no_graph": {
        #     "kg_experiment": "D",
        # },

        # Fusion: BioBERT init + graph, vary fusion type
        # "biobert_graph_fusion_type_0_only_rgcn": {
        #     "kg_experiment": "C",
        #     "p_rel": 0.0,
        #     "kg_biobert_fusion_type": 0,
        # },
        # "biobert_graph_fusion_type_1_concat": {
        #     "kg_experiment": "C",
        #     "p_rel": 0.0,
        #     "kg_biobert_fusion_type": 1,
        # },
        # "biobert_graph_fusion_type_2_residual_sum": {
        #     "kg_experiment": "C",
        #     "p_rel": 0.0,
        #     "kg_biobert_fusion_type": 2,
        # },
        
        #     "kg_experiment": "C",
        #     "p_rel": 0.0,
        #     "kg_biobert_fusion_type": 3,
        # },

        # Sensitivity: node type embedding (only meaningful with fusion_type=3)
        # "node_type_emb_dim_4": {
        #     "kg_experiment": "C",
        #     "p_rel": 0.0,
        #     "kg_biobert_fusion_type": 3,
        #     "kg_node_type_emb_switch": True,
        #     "kg_node_type_emb_dim": 4,
        # },
        # "node_type_emb_dim_8": {
        #     "kg_experiment": "C",
        #     "p_rel": 0.0,
        #     "kg_biobert_fusion_type": 3,
        #     "kg_node_type_emb_switch": True,
        #     "kg_node_type_emb_dim": 8,
        # },
        # "node_type_emb_dim_16": {
        #     "kg_experiment": "C",
        #     "p_rel": 0.0,
        #     "kg_biobert_fusion_type": 3,
        #     "kg_node_type_emb_switch": True,
        #     "kg_node_type_emb_dim": 16,
        # },
        # "node_type_emb_dim_32": {
        #     "kg_experiment": "C",
        #     "p_rel": 0.0,
        #     "kg_biobert_fusion_type": 3,
        #     "kg_node_type_emb_switch": True,
        #     "kg_node_type_emb_dim": 32,
        # },
        # "node_type_emb_dim_64": {
        #     "kg_experiment": "C",
        #     "p_rel": 0.0,
        #     "kg_biobert_fusion_type": 3,
        #     "kg_node_type_emb_switch": True,
        #     "kg_node_type_emb_dim": 64,
        # },

        # Sensitivity: relation dropout p_rel (only meaningful with graph on)
        # "p_rel_0": {
        #     "kg_experiment": "C",
        #     "p_rel": 0.0,
        #     "kg_biobert_fusion_type": 3,
        # },
        # "p_rel_0_1": {
        #     "kg_experiment": "C",
        #     "p_rel": 0.1,
        #     "kg_biobert_fusion_type": 3,
        # },
        # "p_rel_0_3": {
        #     "kg_experiment": "C",
        #     "p_rel": 0.3,
        #     "kg_biobert_fusion_type": 3,
        # },
        # "p_rel_0_5": {
        #     "kg_experiment": "C",
        #     "p_rel": 0.5,
        #     "kg_biobert_fusion_type": 3,
        # },
        # "p_rel_0_7": {
        #     "kg_experiment": "C",
        #     "p_rel": 0.7,
        #     "kg_biobert_fusion_type": 3,
        # },

        
        # "node_type_emb_dim_8_p_rel_0_1": {
        #     "kg_node_type_emb_dim": 8,
        #     "p_rel": 0.1,
        # },
        # "node_type_emb_dim_8_p_rel_0_3": {
        #     "kg_node_type_emb_dim": 8,
        #     "p_rel": 0.3,
        # },
        # "node_type_emb_dim_8_p_rel_0_5": {
        #     "kg_node_type_emb_dim": 8,
        #     "p_rel": 0.5,
        # },
        # "node_type_emb_dim_16_p_rel_0_1": {
        #     "kg_node_type_emb_dim": 16,
        #     "p_rel": 0.1,
        # },
        # "node_type_emb_dim_16_p_rel_0_3": {
        #     "kg_node_type_emb_dim": 16,
        #     "p_rel": 0.3,
        # },
        # "node_type_emb_dim_16_p_rel_0_5": {
        #     "kg_node_type_emb_dim": 16,
        #     "p_rel": 0.5,
        # },
        # "node_type_emb_dim_32_p_rel_0_1": {
        #     "kg_node_type_emb_dim": 32,
        #     "p_rel": 0.1,
        # },
        # "node_type_emb_dim_32_p_rel_0_3": {
        #     "kg_node_type_emb_dim": 32,
        #     "p_rel": 0.3,
        # },
        # "node_type_emb_dim_32_p_rel_0_5": {
        #     "kg_node_type_emb_dim": 32,
        #     "p_rel": 0.5,
        # },

        # --------------------------------------------------------------------------------------seq
        # "mlp_gelu_dropout_0_5":{
        # }
        # "mlp_gelu_dropout_0_1":{
        #     "seq_encoder_mlp_dropout":0.1,
        # },
        # "mlp_gelu_dropout_0_3":{
        #     "seq_encoder_mlp_dropout":0.3,
        # },


        # --------------------------------------------------------------------------------------fusion
        # MOE
        # "kg_seq_moe_gate_aux_weight_1.0_gate_lambda_0.75": {
        #     "task_type": "kg_seq_moe",
        #     "gate_aux_weight": 1.0,
        #     "gate_lambda": 0.75,
        #     "epochs": 60,
        # },
        # "kg_seq_moe_gate_aux_weight_1.0_gate_lambda_1.0": {
        #     "task_type": "kg_seq_moe",
        #     "gate_aux_weight": 1.0,
        #     "gate_lambda": 1.0,
        #     "epochs": 60,
        # },
        # "kg_seq_moe_gate_aux_weight_0.25_gate_lambda_0.75": {
        #         "task_type": "kg_seq_moe",
        #         "gate_aux_weight": 0.25,
        #         "gate_lambda": 0.75,
        #         "epochs": 60,
        #     },
        #     "kg_seq_moe_gate_aux_weight_0.25_gate_lambda_1.0": {
        #         "task_type": "kg_seq_moe",
        #         "gate_aux_weight": 0.25,
        #         "gate_lambda": 1.0,
        #         "epochs": 60,
        #     },
        #     "kg_seq_moe_gate_aux_weight_0.5_gate_lambda_0.75": {
        #         "task_type": "kg_seq_moe",
        #         "gate_aux_weight": 0.5,
        #         "gate_lambda": 0.75,
        #         "epochs": 60,
        #     },
        #     "kg_seq_moe_gate_aux_weight_0.5_gate_lambda_1.0": {
        #         "task_type": "kg_seq_moe",
        #         "gate_aux_weight": 0.5,
        #         "gate_lambda": 1.0,
        #         "epochs": 60,
        #     },
        #     "kg_seq_moe_gate_aux_weight_0.75_gate_lambda_0.75": {
        #         "task_type": "kg_seq_moe",
        #         "gate_aux_weight": 0.75,
        #         "gate_lambda": 0.75,
        #         "epochs": 60,
        #     },
        #     "kg_seq_moe_gate_aux_weight_0.75_gate_lambda_1.0": {
        #         "task_type": "kg_seq_moe",
        #         "gate_aux_weight": 0.75,
        #         "gate_lambda": 1.0,
        #         "epochs": 60,
        #     },
            
        # # umt_naive
        # "umt_naive":{
        #     "task_type": "umt_naive",
        # },
        # # UMT
        # "kg_seq_umt_1":{
        #     "task_type": "umt",
        #     "lambda_distill": 1,
        # },
        # "kg_seq_umt_10":{
        #     "task_type": "umt",
        #     "lambda_distill": 10,
        # },
        # "kg_seq_umt_20":{
        #     "task_type": "umt",
        #     "lambda_distill": 20,
        # },
        # "kg_seq_umt_50":{
        #     "task_type": "umt",
        #     "lambda_distill": 50,
        # },
        # "kg_seq_umt_100":{
        #     "task_type": "umt",
        #     "lambda_distill": 100,
        # },
        
        
        "epoch60":{
            "epochs": 60,
        },
    }


    
    total_tasks = []
    for task_index, task_type in enumerate(["kg_seq_moe"]):  # TODO "only_omics", "only_kg", "umt", "ume" "only_omics", "only_kg", "umt", only_seq ,"kg_seq_moe"
        for cancer_type in ["pan"]:  # TODO "BRCA", "CESC", "COAD", "KIRC", "LAML" , "LUAD", "OV", "SKCM"
            for cv in range(3, 3 + 1): # TODO
                if cancer_type != "pan" and cv == 3:  
                    continue
                for exp_name, exp_params in experiments.items():
                    # if task_type != exp_params['task_type']:
                    #     continue

                    for fold in range(1, 5 + 1):
                    
                    # TODO
                    # isRun = False
                    # if cv ==1 and fold==2:
                    #     isRun = True
                    # if cv ==2 and fold==2:
                    #     isRun = True
                    # if cv ==3 and fold==2:
                    #     isRun = True
                    # if not isRun:
                    #     continue
                    
                        mem_usage = mem_usage_dict[task_type]["pan" if cancer_type == "pan" else "single"]
                        folder_name = f"{task_folder_prex}_{exp_name}_{task_type}_{cancer_type}_cv_{cv}_fold_{fold}"

                        specific_param_dict = {}
                        if cancer_type in custom_hyper_parameters and f"cv_{cv}" in custom_hyper_parameters[cancer_type]:
                            specific_param_dict = custom_hyper_parameters[cancer_type][f"cv_{cv}"]
                        specific_param_dict = {**specific_param_dict, **exp_params}
                        specific_cmd_args = " ".join(f"--{key} {value}" for key, value in specific_param_dict.items())  
                        if addition_unified_params.strip() != "":
                            specific_cmd_args = f"{specific_cmd_args} {addition_unified_params}"

                        
                        task_type_str = task_type
                        if task_type in ["umt","kg_seq_moe","umt_naive","ume"]:  
                            task_type_str = f"{task_type} --omics_ckpt_path ../result/{task_folder_prex}_{'single'}_{'only_seq'}_{cancer_type}_cv_{cv}_fold_{fold}/checkpoint.pth --kg_ckpt_path ../result/{task_folder_prex}_{'single'}_{'only_kg'}_{cancer_type}_cv_{cv}_fold_{fold}/checkpoint.pth"
                            # task_type_str = f"{task_type} --omics_ckpt_path ../result/7_seq_exp/seq_exp_mlp_gelu_dropout_0_3_pan_cv_{cv}_fold_{fold}/checkpoint.pth --kg_ckpt_path ../result/6_kg_exp/kg_exp_node_type_emb_dim_32_p_rel_0_3_pan_cv_{cv}_fold_{fold}/checkpoint.pth"
                            # task_type_str = f"{task_type} --omics_ckpt_path ../result/final_default_default_only_kg_pan_cv_{cv}_fold_{fold}/checkpoint.pth --kg_ckpt_path ../result/final_default_default_only_kg_pan_cv_{cv}_fold_{fold}/checkpoint.pth"

                        command = f"train.py --cancer_type {cancer_type} --metric {cv} --train_fold {fold} {specific_cmd_args} --task_type {task_type_str} --specify_result_saving_folder {folder_name} > ../result/{folder_name}/train.log 2>&1"

                        total_tasks.append(
                            {
                                "mem_usage": mem_usage,
                                "command": command,
                                "folder_name": folder_name,
                            }
                        )

    return total_tasks  


def get_conda_env_vars(conda_env_name):
    """
    Return all environment variables defined for the specified Conda environment.
    """

    command = f"conda run -n {conda_env_name} env"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    env_vars = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            env_vars[key] = value
    return env_vars


class PortManager:
    def __init__(self, base_port):
        self.port = base_port
        self.lock = threading.Lock()

    def _is_port_in_use(self, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) == 0

    def get_free_port(self):  
        with self.lock:
            while self._is_port_in_use(self.port):
                self.port += 1
            free_port = self.port
            self.port += 1
            return free_port


class Worker(threading.Thread):
    def __init__(self, worker_id, card_group, port_manager, env_vars, task_queue, wait_time_min=8.0, wait_time_max=16.0):
        """
        Initialize a task worker.

        Args:
            worker_id: Unique worker identifier.
            card_group: GPU group assigned to the worker.
            port_manager: Port manager.
            env_vars: Current environment variables.
            task_queue: Queue containing pending tasks.
            wait_time_min: Minimum retry delay in seconds after a failure.
            wait_time_max: Maximum retry delay in seconds after a failure.
        """
        super().__init__()
        self.daemon = True  

        self.worker_id = worker_id
        self.card_group = card_group
        self.port_manager = port_manager
        self.port = self.port_manager.get_free_port()  
        self.env_vars = env_vars
        self.task_queue = task_queue
        self.wait_time_min = wait_time_min
        self.wait_time_max = wait_time_max

    def run(self):  
        """
        Run the worker thread.
        """

        while True:  
            try:
                
                
                task = self.task_queue.get(block=False)
            except queue.Empty:
                break  
            self.work(task)  
            self.task_queue.task_done()  
        print(f"[{cur_time_str()}][Worker-{self.worker_id}] running completed.")

    def work(self, task):
        """
        Execute a single task.
        """

        mem_usage = task["mem_usage"]  
        command = task["command"]  
        folder_name = task["folder_name"]  

        while True:  
            if self.card_group.allocate_memory(mem_usage):
                self.execute_command(folder_name, command)
                self.card_group.release_memory(mem_usage)
                return  
            else:
                time.sleep(random.uniform(self.wait_time_min, self.wait_time_max))  

    def execute_command(self, folder_name, command):
        max_retries = 5
        for attempt in range(max_retries):
            gpu_ids = self.card_group.get_gpu_ids()
            gpu_count = len(gpu_ids)
            gpu_ids_str = ",".join(map(str, gpu_ids))
            commmand_prex = f"accelerate launch --main_process_port {self.port} --num_processes {gpu_count} --gpu_ids {gpu_ids_str}"
            full_command = f"{commmand_prex} {command}"

            try:
                print(f"{self.log_prex()} prepare to execute (attempt {attempt + 1}/{max_retries}): {full_command}")
                if attempt == 0:
                    subprocess.run(f"mkdir -p ../result/{folder_name}", shell=True, check=True, env=self.env_vars)

                subprocess.run(full_command, shell=True, check=True, env=self.env_vars)

                print(f"{self.log_prex()} finished to execute: {full_command}")
                return  

            except subprocess.CalledProcessError as e:
                print(f"{self.log_prex()}[err_code: {e.returncode}] command execution FAILED: {full_command}")
                if attempt < max_retries - 1:
                    old_port = self.port
                    self.port = self.port_manager.get_free_port()
                    print(f"[{cur_time_str()}][Worker-{self.worker_id}][Port-{old_port}] Retrying with new port {self.port}...")
                    time.sleep(random.uniform(1, 3))
                else:
                    print(f"[{cur_time_str()}][Worker-{self.worker_id}] All {max_retries} retries failed. Giving up on task.")

    def log_prex(self):
        return f"[{cur_time_str()}][Worker-{self.worker_id}][Port-{self.port}]"


class CardGroup:
    def __init__(self, single_card_memory, gpu_ids):
        """
        Initialize a GPU group containing one or more devices.

        Args:
            single_card_memory: Memory capacity of each GPU in GB; all devices are assumed to have the same capacity.
            gpu_ids: Device identifiers to use, for example [0, 1].
        """

        self.max_memory = single_card_memory * len(gpu_ids)
        self.used_memory = 0  
        self.lock = threading.Lock()  

        self.gpu_ids = gpu_ids

    def allocate_memory(self, memory):
        """
        Reserve GPU memory.

        Args:
            memory: Amount of memory to reserve.

        Returns:
            True if the reservation succeeds; otherwise, False.
        """

        with self.lock:  
            if self.used_memory + memory <= self.max_memory:
                self.used_memory += memory
                return True
            else:
                return False

    def release_memory(self, memory):
        """
        Release previously reserved GPU memory.

        Args:
            memory: Amount of memory to release.
        """

        with self.lock:  
            if memory <= self.used_memory:
                self.used_memory -= memory

    def get_used_memory(self):
        """
        Return the amount of GPU memory currently in use.
        """

        return self.used_memory

    def get_available_memory(self):
        """
        Return the amount of GPU memory currently available.
        """

        return self.max_memory - self.used_memory

    def get_gpu_ids(self):
        return self.gpu_ids


def main():
    args = init_argparse()
    print("parameters:", args)

    
    print("loading conda environment...")
    env_vars = os.environ.copy()  
    env_vars.update(get_conda_env_vars(args.conda_env_name))  

    
    task_queue = queue.Queue()
    print("generating tasks...")
    tasks = generate_tasks(args.task_folder_prex, args.addition_unified_params)
    print(f"generated done. total tasks: {len(tasks)}")
    torch.save(tasks, f"../result/task_list_{args.task_folder_prex}.pt")  
    print(f"success to save tasks list data to:", f"../result/task_list_{args.task_folder_prex}.pt")

    
    # tasks = [{ # "batch_size": 2048, "omics_types": "cna exp mut", "vae_hidden_dims": "2048 1024 512 256"
    #     "mem_usage": 35,
    #     "command": f'train.py --cancer_type pan --metric 1 --train_fold 1 --epochs 15 --batch_size 2048 --omics_types cna exp mut --vae_hidden_dims 2048 1024 512 256 --lambda_distill {param} --task_type umt --omics_ckpt_path ../result/default_pan_cv_1_fold_1_only_omics/checkpoint.pth --kg_ckpt_path ../result/default_pan_cv_1_fold_1_only_kg/checkpoint.pth --specify_result_saving_folder lambda_distill_{str(param)}_pan_cv_1_fold_1_umt > ../result/lambda_distill_{str(param)}_pan_cv_1_fold_1_umt/train.log 2>&1',
    #     "folder_name": f"lambda_distill_{str(param)}_pan_cv_1_fold_1_umt",
    # } for param in [1,2,5,10]]

    # tasks.extend( [{
    #     "mem_usage": 35,
    #     "command": f'train.py --cancer_type pan --metric 2 --train_fold 5 --epochs 15 --batch_size 2048 --omics_types cna exp mut --vae_hidden_dims 2048 1024 512 256 --lambda_distill {param} --task_type umt --omics_ckpt_path ../result/default_pan_cv_2_fold_5_only_omics/checkpoint.pth --kg_ckpt_path ../result/default_pan_cv_2_fold_5_only_kg/checkpoint.pth --specify_result_saving_folder lambda_distill_{str(param)}_pan_cv_2_fold_5_umt > ../result/lambda_distill_{str(param)}_pan_cv_2_fold_5_umt/train.log 2>&1',
    #     "folder_name": f"lambda_distill_{str(param)}_pan_cv_2_fold_5_umt",
    # } for param in [1,2,5,10]])

    # tasks.extend( [{
    #     "mem_usage": 35,
    #     "command": f'train.py --cancer_type pan --metric 3 --train_fold 1 --epochs 15 --batch_size 2048 --omics_types cna exp mut --vae_hidden_dims 2048 1024 512 256 --lambda_distill {param} --task_type umt --omics_ckpt_path ../result/default_pan_cv_3_fold_1_only_omics/checkpoint.pth --kg_ckpt_path ../result/default_pan_cv_3_fold_1_only_kg/checkpoint.pth --specify_result_saving_folder lambda_distill_{str(param)}_pan_cv_3_fold_1_umt > ../result/lambda_distill_{str(param)}_pan_cv_3_fold_1_umt/train.log 2>&1',
    #     "folder_name": f"lambda_distill_{str(param)}_pan_cv_3_fold_1_umt",
    # } for param in [1,2,5,10]])

    # tasks = [{
    #     "mem_usage": 20,
    #     "command": f"train.py --cancer_type CESC --metric 1 --train_fold {param} --epochs 150 --lambda_distill 100 --task_type umt --omics_ckpt_path ../result/default_CESC_cv_1_fold_{param}_only_omics/checkpoint.pth --kg_ckpt_path ../result/default_CESC_cv_1_fold_{param}_only_kg/checkpoint.pth --specify_result_saving_folder lambda_distill_100_CESC_cv_1_fold_{param}_umt > ../result/lambda_distill_100_CESC_cv_1_fold_{param}_umt/train.log 2>&1",
    #     "folder_name": f"lambda_distill_100_CESC_cv_1_fold_{param}_umt",
    # } for param in [1,3,4,5]]

    

    
    for t in tasks:
        task_queue.put(t)

    port_manager = PortManager(args.base_port)
    total_workers = []  
    worker_id_counter = 0
    for gpu_ids_str in args.card_groups:
        cur_workers = []
        gpu_ids = list(map(int, gpu_ids_str.split(",")))
        card_group = CardGroup(single_card_memory=args.single_card_memory, gpu_ids=gpu_ids)

        for i in range(args.worker_count_per_card_group):  
            worker_id_counter += 1
            cur_workers.append(Worker(worker_id_counter, card_group, port_manager, env_vars, task_queue))

        total_workers.append(cur_workers)

    workers = []  
    for i in range(args.worker_count_per_card_group):
        for j in range(len(args.card_groups)):
            workers.append(total_workers[j][i])

    
    for worker in workers:
        worker.start()
        time.sleep(1.5)  

    task_queue.join()  
    for worker in workers:  
        worker.join()

    print("all tasks running completed, program exit!")



if __name__ == "__main__":
    main()
