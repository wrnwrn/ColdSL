import re
import numpy as np
import pandas as pd


def cal(cancer,cv,model_name,task_folder_prex):

    data=[]
    for fold in range(1,5+1):

        train_val_evaluate_path = f'./result/' \
        f'{task_folder_prex}_{model_name}_{cancer}_cv_{cv}_fold_{fold}/train_val_evaluate.csv'

        df = pd.read_csv(train_val_evaluate_path)

        
        #     if cancer =='pan':
        #         df =  df.iloc[[29]]
        #     else:
        #         df =  df.iloc[[50]]
          

        
        max_auc_row = df.loc[df['val_AUC'].idxmax()] 
        
        

        
        cur = [max_auc_row['test_AUC'],max_auc_row['test_AUPR'],max_auc_row['test_F1'],max_auc_row['epoch']]
        # print(f'fold {fold}: {cur}') # TODO
        data.append(cur)

    # Convert to numpy array
    data_array = np.array(data)

    # Calculate mean and variance for each column, rounded to 4 decimals
    means = np.mean(data_array, axis=0)
    stds = np.std(data_array, axis=0)

    print("mean:", ", ".join(f"{mean:.4f}" for mean in means),cancer,model_name,f'cv{cv}',task_folder_prex)
    print("std: ", ", ".join(f"{std:.4f}" for std in stds),cancer,model_name,f'cv{cv}',task_folder_prex)



if __name__ == '__main__':
    task_folder_prex = f'{"test"}_{"single"}' # lambda_distill_25
    # task_folder_prex = f'{"test"}_{"epoch60"}' # lambda_distill_25
    for cancer in ['pan']: # 'BRCA','CESC','COAD','KIRC','LAML','LUAD','OV','SKCM','pan'
        print(f'############################################### {cancer}')

        # cal(cancer,1,'only_omics',task_folder_prex)
        # cal(cancer,1,'only_kg',task_folder_prex)
        # cal(cancer,1,'umt',task_folder_prex)
        # cal(cancer,1,'ume',task_folder_prex)

        # cal(cancer,2,'only_omics',task_folder_prex)
        # cal(cancer,2,'only_kg',task_folder_prex)
        # cal(cancer,2,'umt',task_folder_prex)
        # cal(cancer,2,'ume',task_folder_prex)
        if cancer=='pan': 

            cal(cancer,3,'only_kg',task_folder_prex)
            cal(cancer,3,'only_seq',task_folder_prex)
            cal(cancer,3,'ume',task_folder_prex)
            # cal(cancer,3,'kg_seq_moe',task_folder_prex)
            # cal(cancer,1,'','final_default_fusion_kg_seq_moe')
            # cal(cancer,2,'','final_default_fusion_kg_seq_moe')
            # cal(cancer,3,'umt',task_folder_prex)
            # cal(cancer,3,'ume',task_folder_prex)
        
