import os
from torch.utils.data import Dataset
import torch
import csv
MAX_IC = 10.0

def parse_weight_vector(weight_vector_str):
    return [1.0] + [float(weight) for weight in weight_vector_str[1:-1].split(',')] + [1.0]

def read_csv_to_dict(file_path):
    result = {}
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            col_1, col_2,col_3 = row[0], row[2],parse_weight_vector(row[3])
            result[col_1] = (col_2,torch.tensor(col_3))
    return result

def read_csv_to_list(file_path):
    result = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader)  # Get the header row

        for row in reader:
            vh_seq = row[-8]
            vl_seq = row[-7]
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
                    'IC50': ic50,
                    'H_CDR3':row[-4],
                    'L_CDR3':row[-1]
                })
    return result

class IC50Dataset(Dataset):
    def __init__(self, spike_data = '/code/lyt/S3AI/data/output2.csv', ic50_data = '/code/lyt/S3AI/data/processed_data_cdr.csv',pssm_data = '/code/lyt/S3AI/data/processed_data_cdr.csv'):
        super().__init__()
        self.ic_50_data = read_csv_to_list(ic50_data)
        self.spike_dict = read_csv_to_dict(spike_data)

    def __len__(self):
        return len(self.ic_50_data)

    def __getitem__(self, idx):# {spike, VH, VL, ic50 }
        res = self.ic_50_data[idx]
        res['spike'],res['weight'] = self.spike_dict[res['virus']]
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
