import os
from torch.utils.data import Dataset
import csv
MAX_IC = 10.0

def read_csv_to_dict(file_path):
    result = {}
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            col_1, col_2, col_3，col_4= row[0], row[2],row[3],row[4]
            result[col_1] = (col_2,col_3，col_4)
    return result

def process_ft_mat(ft_mat_str):
    ft_mat_list = []
    ft_rows = ft_mat_str.split(',')
    for i in range(0, len(ft_rows), 7726):  # Assuming each ft_mat is of shape (1, 7726)
        ft_mat_list.append(list(map(float, ft_rows[i:i+7726])))
    return np.array(ft_mat_list)

def read_csv_to_list(file_path):
    result = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader)  # Get the header row

        for row in reader:
            vh_seq = row[-4]
            vl_seq = row[-3]
            for i in range(1, len(header) - 2):
                ic50 = row[i]
                if '>' in ic50:
                    ic50 = MAX_IC
                else:
                    try:
                        ic50 = min(MAX_IC,float(ic50))
                    except ValueError:
                        # Skip the data if ic50 is not a number
                        continue
#                 if header[i] != 'D614G':
#                     continue
                result.append({
                    'H': vh_seq,
                    'L': vl_seq,
                    'virus': header[i],
                    'batch_virus_idx':i-1,
                    'IC50': ic50,
                    'antibody_pssm':np.hstack(process_ft_mat(row[-4]),process_ft_mat(row[-3])),,
                    'antibody_kmer_whole':np.hstack(process_ft_mat(row[-2]),process_ft_mat(row[-1])),
#                     'H_CDR3':row[-4],
#                     'L_CDR3':row[-1]
                })
    return result

class DeepAAIDataset(Dataset):
    def __init__(self, spike_data = '/code/lyt/AbAgPRED/data/output.csv', ic50_data = '/code/lyt/AbAgPRED/data/processed_data_cdr.csv'):
        super().__init__()
        self.ic_50_data = read_csv_to_list(ic50_data)
        self.spike_dict = read_csv_to_dict(spike_data)

    def __len__(self):
        return len(self.ic_50_data)

    def __getitem__(self, idx):# {spike, VH, VL, ic50 }
        res = self.ic_50_data[idx]
        res['spike'], res['virus_pssm'],res['virus_kmer_whole'] = self.spike_dict[res['virus']]
        res['batch_antibody_idx'] = batch_antibody_idx
        return res

    
# import numpy as np

# def compute_ic50_mean_std(ic50_data):
#     ic50_values = [item['IC50'] for item in ic50_data]
#     mean = np.mean(ic50_values)
#     std = np.std(ic50_values)
#     return mean, std

# # 读取 IC50 数据
# ic50_data = read_csv_to_list('/code/AbAgPRED/data/processed_data_cdr.csv')

# # 计算 IC50 均值和标准差
# mean, std = compute_ic50_mean_std(ic50_data)

# print("Mean of IC50:", mean)
# print("Standard deviation of IC50:", std)
# # protein_dataset = IC50Dataset()

# sample = protein_dataset[0]
# print(sample)
# print(protein_dataset[2])
# print(len(protein_dataset))
