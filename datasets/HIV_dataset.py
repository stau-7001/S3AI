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

        for row in reader:
            Ab_seq = row[0]
            try:
                ic50 = min(MAX_IC,float(row[2]))
            except ValueError:
                continue
#                 if header[i] != 'D614G':
#                     continue
            result.append({
                    'H': Ab_seq,
                    'L': '',
                    'spike': row[1],
                    'IC50': ic50,
                })
    return result


class HIVRegDataset(Dataset):
    def __init__(self, ic50_data = './data/dataset_hiv_reg.csv'):
        super().__init__()
        self.ic_50_data = read_csv_to_list(ic50_data)

    def __len__(self):
        return len(self.ic_50_data)

    def __getitem__(self, idx):# {spike, VH, VL, ic50 }
        res = self.ic_50_data[idx]
        return res

class HIVClsDataset(Dataset):
    def __init__(self, n01_data = './data/dataset_hiv_cls.csv'):
        super().__init__()
        self.ic_50_data = read_csv_to_list(n01_data)

    def __len__(self):
        return len(self.ic_50_data)

    def __getitem__(self, idx):# {spike, VH, VL, ic50 }
        res = self.ic_50_data[idx]
        return res

