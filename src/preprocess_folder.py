
import argparse
import os
from queue import Queue
import random
import time


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch


from util.my import set_seed, time_elapsed

LUCKY_NUMBER = 2025
set_seed(LUCKY_NUMBER)
rng = np.random.default_rng(LUCKY_NUMBER)  

SLKG_KG_path = "../data_raw/SLKG2/raw_kg.tsv"
protein_sequence_path = "../data_raw/uniprot/uniprotkb_organism_id_9606_AND_reviewed_2024_10_26.fasta"
PAN_CN_KG = "TOTAL"


def init_argparse():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--ct",  
        nargs="+",  
        type=str,  
        default=["BRCA", "CESC", "COAD", "KIRC", "LAML", "LUAD", "OV", "SKCM", "pan"],
        help="one or more cancer types",
    )

    return parser.parse_args()


def balance_neg_pairs_cv2_3(positive_sl_pairs, negative_sl_pairs, one_genes, another_genes, total_pos_set, total_neg_set):
    """
    Balance the negative pairs according to the number of positive pairs.

    Args:
        positive_sl_pairs: List of positive synthetic lethal pairs.
        negative_sl_pairs: List of negative synthetic lethal pairs.
        one_genes: Allowed genes appearing in one position of the synthetic lethal pairs.
        another_genes: Allowed genes appearing in the other position of the synthetic lethal pairs.
        total_pos_set: Set of all positive synthetic lethal pairs.
        total_neg_set: Set of all negative synthetic lethal pairs.

    Returns:
        total_pos_set: Positive-pair set updated with the current pairs.
        total_neg_set: Negative-pair set updated with the current pairs.
        negative_sl_pairs: Adjusted list of negative pairs.
    """

    
    positive_samples_count = len(positive_sl_pairs)
    negative_samples_count = len(negative_sl_pairs)

    
    if positive_samples_count < negative_samples_count:  
        negative_sl_pairs = random.sample(negative_sl_pairs, positive_samples_count)

        
        pos_set = set([tuple(item[:2]) for item in positive_sl_pairs])  
        neg_set = set([tuple(item[:2]) for item in negative_sl_pairs])
        total_pos_set = total_pos_set | pos_set
        total_neg_set = total_neg_set | neg_set
        return total_pos_set, total_neg_set, negative_sl_pairs

    elif positive_samples_count >= negative_samples_count:  
        
        pos_set = set([tuple(item[:2]) for item in positive_sl_pairs])  
        neg_set = set([tuple(item[:2]) for item in negative_sl_pairs])
        total_pos_set = total_pos_set | pos_set
        total_neg_set = total_neg_set | neg_set

        
        end = positive_samples_count - negative_samples_count
        i = 0  

        while i < end:
            one_gene = str(rng.choice(one_genes, 1, replace=False)[0])
            another_gene = str(rng.choice(another_genes, 1, replace=False)[0])

            
            if ((one_gene, another_gene) in total_pos_set) or ((another_gene, one_gene) in total_pos_set) or ((one_gene, another_gene) in total_neg_set) or ((another_gene, one_gene) in total_neg_set):
                continue

            
            negative_sl_pairs.append([one_gene, another_gene, 0])
            total_neg_set.add((one_gene, another_gene))
            i = i + 1

        return total_pos_set, total_neg_set, negative_sl_pairs


def get_pairs(positive_sl_pairs, negative_sl_pairs, train_genes, test_genes, region_type):
    """
    Select positive and negative pairs for a CV2 or CV3 region.

    Args:
        positive_sl_pairs: Array of positive synthetic lethal pairs.
        negative_sl_pairs: Array of negative synthetic lethal pairs.
        train_genes: Genes assigned to the training set.
        test_genes: Genes outside the training set, assigned to validation or testing.
        region_type: Region identifier. CV2 and CV3 use three regions, as described in the PiLSL paper.

    Returns:
        pos_pairs_with_genes: Positive gene pairs assigned to the region.
        neg_pairs_with_genes: Negative gene pairs assigned to the region.
    """

    pos_pairs_with_genes = []  
    for pair in positive_sl_pairs:
        if region_type == 1:  
            if pair[0] in train_genes and pair[1] in train_genes:
                pos_pairs_with_genes.append(list(pair))
        elif region_type == 2:  
            if (pair[0] in test_genes and pair[1] in train_genes) or (pair[0] in train_genes and pair[1] in test_genes):
                pos_pairs_with_genes.append(list(pair))
        elif region_type == 3:  
            if pair[0] in test_genes and pair[1] in test_genes:
                pos_pairs_with_genes.append(list(pair))

    neg_pairs_with_genes = []  
    for pair in negative_sl_pairs:
        if region_type == 1:
            if pair[0] in train_genes and pair[1] in train_genes:
                neg_pairs_with_genes.append(list(pair))
        elif region_type == 2:
            if (pair[0] in test_genes and pair[1] in train_genes) or (pair[0] in train_genes and pair[1] in test_genes):
                neg_pairs_with_genes.append(list(pair))
        elif region_type == 3:
            if pair[0] in test_genes and pair[1] in test_genes:
                neg_pairs_with_genes.append(list(pair))

    return pos_pairs_with_genes, neg_pairs_with_genes


def _mfd_cv2_3_sub(train_index, test_index, positive_sl_pairs, negative_sl_pairs, genes, base_dir, fold_num, nz_cycle_time, gene_name_to_idx):
    """
    Generate one CV2 or CV3 split and report whether generation succeeded.
    """

    train_genes = genes[train_index]
    test_genes = genes[test_index]  

    
    kf_train_valid = KFold(n_splits=8, shuffle=True, random_state=LUCKY_NUMBER + nz_cycle_time + fold_num)
    is_generate_success = False  
    
    for train_train_index, train_valid_index in kf_train_valid.split(train_genes):
        train_train_genes = train_genes[train_train_index]  
        train_valid_genes = train_genes[train_valid_index]  

        
        total_pos_set = set()  
        total_neg_set = set()  
        # train
        train_train_pos_pairs, train_train_neg_pairs = get_pairs(positive_sl_pairs, negative_sl_pairs, train_train_genes, None, region_type=1)
        total_pos_set, total_neg_set, train_train_neg_pairs = balance_neg_pairs_cv2_3(train_train_pos_pairs, train_train_neg_pairs, train_train_genes, train_train_genes, total_pos_set, total_neg_set)
        # valid C2
        train_valid_c2_pos_pairs, train_valid_c2_neg_pairs = get_pairs(positive_sl_pairs, negative_sl_pairs, train_train_genes, train_valid_genes, region_type=2)
        total_pos_set, total_neg_set, train_valid_c2_neg_pairs = balance_neg_pairs_cv2_3(train_valid_c2_pos_pairs, train_valid_c2_neg_pairs, train_train_genes, train_valid_genes, total_pos_set, total_neg_set)
        # valid C3
        train_valid_c3_pos_pairs, train_valid_c3_neg_pairs = get_pairs(positive_sl_pairs, negative_sl_pairs, train_train_genes, train_valid_genes, region_type=3)
        total_pos_set, total_neg_set, train_valid_c3_neg_pairs = balance_neg_pairs_cv2_3(train_valid_c3_pos_pairs, train_valid_c3_neg_pairs, train_valid_genes, train_valid_genes, total_pos_set, total_neg_set)
        # test C2
        test_c2_pos_pairs, test_c2_neg_pairs = get_pairs(positive_sl_pairs, negative_sl_pairs, train_train_genes, test_genes, region_type=2)
        total_pos_set, total_neg_set, test_c2_neg_pairs = balance_neg_pairs_cv2_3(test_c2_pos_pairs, test_c2_neg_pairs, train_train_genes, test_genes, total_pos_set, total_neg_set)
        # test C3
        test_c3_pos_pairs, test_c3_neg_pairs = get_pairs(positive_sl_pairs, negative_sl_pairs, train_train_genes, test_genes, region_type=3)
        total_pos_set, total_neg_set, test_c3_neg_pairs = balance_neg_pairs_cv2_3(test_c3_pos_pairs, test_c3_neg_pairs, test_genes, test_genes, total_pos_set, total_neg_set)

        
        train_train_combined_pairs = train_train_pos_pairs + train_train_neg_pairs
        train_valid_c2_combined_pairs = train_valid_c2_pos_pairs + train_valid_c2_neg_pairs
        train_valid_c3_combined_pairs = train_valid_c3_pos_pairs + train_valid_c3_neg_pairs
        test_c2_combined_pairs = test_c2_pos_pairs + test_c2_neg_pairs
        test_c3_combined_pairs = test_c3_pos_pairs + test_c3_neg_pairs

        
        len_train = len(train_train_combined_pairs)
        len_valid_c2 = len(train_valid_c2_combined_pairs)
        len_valid_c3 = len(train_valid_c3_combined_pairs)
        len_test_c2 = len(test_c2_combined_pairs)
        len_test_c3 = len(test_c3_combined_pairs)
        if len_train == 0 or len_valid_c2 == 0 or len_valid_c3 == 0 or len_test_c2 == 0 or len_test_c3 == 0:
            print(f"Some dataset equals with 0. len_train={len_train}, " f"len_valid_c2={len_valid_c2}, len_test_c2={len_test_c2}, " f"len_valid_c3={len_valid_c3}, len_test_c3={len_test_c3}.")
            continue  
        else:
            print(f"CV2, train:val:test = {len_train}:{len_valid_c2}:{len_test_c2}")
            print(f"CV3, train:val:test = {len_train}:{len_valid_c3}:{len_test_c3}")

        df_train = pd.DataFrame(train_train_combined_pairs, columns=["gene1", "gene2", "class"])
        df_valid_c2 = pd.DataFrame(train_valid_c2_combined_pairs, columns=["gene1", "gene2", "class"])
        df_valid_c3 = pd.DataFrame(train_valid_c3_combined_pairs, columns=["gene1", "gene2", "class"])
        df_test_c2 = pd.DataFrame(test_c2_combined_pairs, columns=["gene1", "gene2", "class"])
        df_test_c3 = pd.DataFrame(test_c3_combined_pairs, columns=["gene1", "gene2", "class"])

        
        df_train = df_train.sort_values(by=["gene1", "gene2"])
        df_valid_c2 = df_valid_c2.sort_values(by=["gene1", "gene2"])
        df_valid_c3 = df_valid_c3.sort_values(by=["gene1", "gene2"])
        df_test_c2 = df_test_c2.sort_values(by=["gene1", "gene2"])
        df_test_c3 = df_test_c3.sort_values(by=["gene1", "gene2"])

        np_train = sl_df_to_np(df_train, gene_name_to_idx)
        np_valid_c2 = sl_df_to_np(df_valid_c2, gene_name_to_idx)
        np_valid_c3 = sl_df_to_np(df_valid_c3, gene_name_to_idx)
        np_test_c2 = sl_df_to_np(df_test_c2, gene_name_to_idx)
        np_test_c3 = sl_df_to_np(df_test_c3, gene_name_to_idx)

        
        cv2_fold_dir = os.path.join(base_dir, f"cv_2_fold_{fold_num}")
        cv3_fold_dir = os.path.join(base_dir, f"cv_3_fold_{fold_num}")
        if not os.path.exists(cv2_fold_dir):
            os.makedirs(cv2_fold_dir)
        if not os.path.exists(cv3_fold_dir):
            os.makedirs(cv3_fold_dir)

        np.save(os.path.join(cv2_fold_dir, "train_sl.npy"), np_train)
        np.save(os.path.join(cv2_fold_dir, "val_sl.npy"), np_valid_c2)
        np.save(os.path.join(cv2_fold_dir, "test_sl.npy"), np_test_c2)
        np.save(os.path.join(cv3_fold_dir, "train_sl.npy"), np_train)
        np.save(os.path.join(cv3_fold_dir, "val_sl.npy"), np_valid_c3)
        np.save(os.path.join(cv3_fold_dir, "test_sl.npy"), np_test_c3)

        print(f"Fold {fold_num} processed. Files saved to {cv2_fold_dir} and {cv3_fold_dir}")

        
        is_generate_success = True
        break

    return is_generate_success


def make_folder_cv2_3(cancer_types):
    """
    Create CV2 or CV3 dataset splits.

    Modified from PiLSL. Genes are divided in a 7:1:2 ratio, positive and negative pairs are then assigned to the splits, and negative pairs are balanced within each split.
    """

    for cancer_type in cancer_types:
        print(f"======================================== make cv2 and cv3 folder about {cancer_type} ========================================")

        
        gene_idx_to_name = torch.load(f"../data/{cancer_type}/gene.pt", weights_only=False)
        gene_name_to_idx = {value: key for key, value in gene_idx_to_name.items()}
        
        print("reading SL data...")
        np_sl = np.load(f"../data/{cancer_type}/sl.npy")

        
        df_SL = sl_np_to_df(np_sl, gene_idx_to_name)

        positive_sl_pairs = df_SL[df_SL["class"] == 1].to_numpy()
        negative_sl_pairs = df_SL[df_SL["class"] == 0].to_numpy()
        gene1 = set(df_SL["gene1"])
        gene2 = set(df_SL["gene2"])
        
        genes = np.array(sorted(list(gene1 | gene2)))  

        
        nz_cycle_time = 0
        nz_q = Queue()

        
        kf = KFold(n_splits=5, shuffle=True, random_state=LUCKY_NUMBER + nz_cycle_time)
        base_dir = f"../data/{cancer_type}"  
        for fold_num, (train_index, test_index) in enumerate(kf.split(genes), 1):
            
            is_generate_success = _mfd_cv2_3_sub(train_index, test_index, positive_sl_pairs, negative_sl_pairs, genes, base_dir, fold_num, nz_cycle_time, gene_name_to_idx)

            if not is_generate_success:
                nz_q.put(fold_num)
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Warning: {cancer_type} fold {fold_num} can not generate non zero datasets. It will be regenerated later.")

        
        while not nz_q.empty():
            nz_cycle_time = nz_cycle_time + 1
            fold_num = nz_q.get()
            print(f"regenerating fold {fold_num} with nz_cycle_time={nz_cycle_time}")
            nz_kf = KFold(n_splits=5, shuffle=True, random_state=LUCKY_NUMBER + nz_cycle_time)
            is_five_for_one_fold_success = False  
            for _, (train_index, test_index) in enumerate(nz_kf.split(genes), 1):
                is_generate_success = _mfd_cv2_3_sub(train_index, test_index, positive_sl_pairs, negative_sl_pairs, genes, base_dir, fold_num, nz_cycle_time, gene_name_to_idx)
                if is_generate_success:
                    is_five_for_one_fold_success = True
                    break

            if not is_five_for_one_fold_success:
                nz_q.put(fold_num)

            if nz_cycle_time == 100:
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Error: nz_cycle_time has >= 100, it is still not solved. Queue: {nz_q}")
                break


def str2num(string):
    """
    Convert a string to a base-128 integer by weighting and summing the ASCII value of each character.
    """

    result = 0
    for char in string:
        result = result * 128 + ord(char)  
    return result


def adjust_negative_samples_cv1(df_group, cancer_type, intersect_genes):
    """
    Adjust CV1 negative samples dynamically because DataFrame and Series sampling return identical results for the same random_state.
    """

    
    df_positive = df_group[df_group["class"] == 1]
    df_negative = df_group[df_group["class"] == 0]
    positive_samples_count = len(df_positive)
    negative_samples_count = len(df_negative)
    df_negative_adjusted = df_negative  

    
    hash_value = str2num(cancer_type)  
    hash_number = hash_value % (2**32)  
    random_number = (hash_number + LUCKY_NUMBER) % (2**32)  

    
    if positive_samples_count < negative_samples_count:  
        df_negative_adjusted = df_negative.sample(n=positive_samples_count, random_state=random_number)
    elif positive_samples_count > negative_samples_count:  
        
        end = positive_samples_count - negative_samples_count
        i = 0  
        j = 0  

        print(f"start to generate negative samples, total {end}: ")
        while i < end:
            if i % 500 == 0:
                print(i, end=",")
            random_genes = intersect_genes.sample(n=2, random_state=random_number + j)  
            j = j + 1  
            geneA = random_genes.iloc[0]
            geneB = random_genes.iloc[1]
            
            len1 = len(df_negative_adjusted[(df_negative_adjusted["gene1"] == geneA) & (df_negative_adjusted["gene2"] == geneB)])
            len2 = len(df_negative_adjusted[(df_negative_adjusted["gene1"] == geneB) & (df_negative_adjusted["gene2"] == geneA)])
            
            len_positive_1 = len(df_positive[(df_positive["gene1"] == geneA) & (df_positive["gene2"] == geneB)])
            len_positive_2 = len(df_positive[(df_positive["gene1"] == geneB) & (df_positive["gene2"] == geneA)])
            if len1 == 0 and len2 == 0 and len_positive_1 == 0 and len_positive_2 == 0:  
                
                df_negative_adjusted = pd.concat(
                    [
                        df_negative_adjusted,
                        pd.DataFrame({"gene1": geneA, "gene2": geneB, "class": 0}, index=[0]),  
                    ],
                    ignore_index=True,
                )
                i = i + 1
        print("finished!")

    print(f"positive_samples_count: {positive_samples_count} negative_samples_count_before: {negative_samples_count} after: {len(df_negative_adjusted)}")
    
    df_result = pd.concat([df_positive.copy(), df_negative_adjusted.copy()], ignore_index=True)
    return df_result


def sl_df_to_np(df_sl, gene_name_to_idx):
    df_sl = df_sl.copy()  

    
    df_sl["gene1"] = df_sl["gene1"].map(gene_name_to_idx)
    df_sl["gene2"] = df_sl["gene2"].map(gene_name_to_idx)

    
    np_sl = df_sl.to_numpy()
    return np_sl


def sl_np_to_df(np_sl, gene_idx_to_name):
    gene1_names = [gene_idx_to_name[idx] for idx in np_sl[:, 0]]  
    gene2_names = [gene_idx_to_name[idx] for idx in np_sl[:, 1]]
    classes = np_sl[:, 2]
    df_SL = pd.DataFrame({"gene1": gene1_names, "gene2": gene2_names, "class": classes})  
    return df_SL


def make_folder_cv1(cancer_types):
    """
    Create CV1 dataset splits based on gene pairs.

    Negative samples are balanced before the data are divided into training, validation, and test sets in a 7:1:2 ratio.
    """

    for cancer_type in cancer_types:
        print(f"======================================== make cv1 folder about {cancer_type} ========================================")

        
        gene_idx_to_name = torch.load(f"../data/{cancer_type}/gene.pt", weights_only=False)
        gene_names = pd.Series(gene_idx_to_name.values())  
        gene_name_to_idx = {value: key for key, value in gene_idx_to_name.items()}
        
        print("reading SL data...")
        np_sl = np.load(f"../data/{cancer_type}/sl.npy")

        
        df_SL = sl_np_to_df(np_sl, gene_idx_to_name)

        
        df_SL = adjust_negative_samples_cv1(df_SL, cancer_type, gene_names)

        kf = KFold(n_splits=5, shuffle=True, random_state=LUCKY_NUMBER)  
        base_dir = f"../data/{cancer_type}"  

        for fold_num, (train_val_index, test_index) in enumerate(kf.split(df_SL), 1):  
            
            fold_dir = os.path.join(base_dir, f"cv_1_fold_{fold_num}")
            if not os.path.exists(fold_dir):
                os.makedirs(fold_dir)

            
            train_val_data = df_SL.iloc[train_val_index]
            test_data = df_SL.iloc[test_index]

            
            train_data = None
            val_data = None
            kf_train_valid = KFold(n_splits=8, shuffle=True, random_state=LUCKY_NUMBER + fold_num)
            train_val_data_copy = train_val_data.copy()  
            train_val_data_copy = train_val_data_copy.reset_index().drop("index", axis=1)  
            
            for train_index, valid_index in kf_train_valid.split(train_val_data_copy):
                train_data = train_val_data_copy.iloc[train_index]
                val_data = train_val_data_copy.iloc[valid_index]
                break

            
            train_data = train_data.sort_values(by=["gene1", "gene2"])
            val_data = val_data.sort_values(by=["gene1", "gene2"])
            test_data = test_data.sort_values(by=["gene1", "gene2"])

            train_data = sl_df_to_np(train_data, gene_name_to_idx)
            val_data = sl_df_to_np(val_data, gene_name_to_idx)
            test_data = sl_df_to_np(test_data, gene_name_to_idx)

            
            np.save(os.path.join(fold_dir, "train_sl.npy"), train_data)
            np.save(os.path.join(fold_dir, "val_sl.npy"), val_data)
            np.save(os.path.join(fold_dir, "test_sl.npy"), test_data)

            print(f"Fold {fold_num} processed. Files saved to {fold_dir}")


def main():
    """
    Create the cross-validation dataset splits.
    """

    args = init_argparse()
    print("parameters:", args)
    cancer_types = args.ct

    start_time = time.time()  
    print("*********************make folder*********************")
    make_folder_cv1(cancer_types)
    make_folder_cv2_3(cancer_types)
    end_time = time.time()  
    print()
    print(time_elapsed(start_time, end_time, "All preprocess finished! Time used: "))


if __name__ == "__main__":
    main()
