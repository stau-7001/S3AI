#!/usr/bin/env
# coding:utf-8
import sys, os
base_path = os.path.dirname(os.path.dirname(
                            os.path.abspath(__file__)))
sys.path.append(base_path)
from sklearn.feature_extraction.text import CountVectorizer
from dataset.feature_trans_content import amino_map_idx
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys
try:
    import cPickle as pickle
except ImportError:
    import pickle

import os.path as osp


# def k_mer_ft_generate(raw_protein_list, trans_type='prob', min_df=1):
#     trans_protein_list = split_protein_str(raw_protein_list)
#     vectorizer = CountVectorizer(ngram_range=(1, 3), token_pattern=r'\b\w+\b', min_df=min_df)
#     count_mat = vectorizer.fit_transform(trans_protein_list)
#     count_mat = count_mat.toarray()
#
#     # print(np.sum(count_mat, axis=0).shape)
#
#     if trans_type == 'std':
#     # 按列归一化
#         stand_scaler = StandardScaler()
#         stand_scaler.fit(count_mat)
#         count_mat = stand_scaler.transform(count_mat)
#
#     elif trans_type == 'prob':
#     # 直接转成概率
#         count_mat = count_mat / np.sum(count_mat, axis=0)
#
#     # print(count_mat.shape)
#     return count_mat
#
# # EVQLVESGGGVVQPGR ->  1 2 1 4 5 1 20
# def split_protein_str(raw_protein_list):
#     trans_protein_list = []
#     for raw_str in raw_protein_list:
#         trans_str = ''
#         for char in raw_str:
#             amino_idx = amino_map_idx[char]
#             trans_str = trans_str + ' ' + str(amino_idx)
#         trans_protein_list.append(trans_str)
#         # print(trans_str)
#     # print(len(trans_protein_list))
#     return trans_protein_list


class KmerTranslator(object):
    def __init__(self, trans_type='std', min_df=1, name=''):
        self.obj_name = name
        self.obj_file_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed_kmer_obj', self.obj_name + '.pkl')
        self.trans_type = trans_type
        self.min_df = min_df
        self.vectorizer = CountVectorizer(ngram_range=(1, 3), token_pattern=r'\b\w+\b', min_df=self.min_df)
        self.stand_scaler = StandardScaler()
        self.count_mat = None

    # EVQLVESGGGVVQPGR ->  1 2 1 4 5 1 20
    def split_protein_str(self, raw_protein_list):
        trans_protein_list = []
        for raw_str in raw_protein_list:
            trans_str = ''
            for char in raw_str:
                amino_idx = amino_map_idx[char]
                trans_str = trans_str + ' ' + str(amino_idx)
            trans_protein_list.append(trans_str)
        return trans_protein_list

    def fit(self, protein_char_list):
        # mat -> str list
        protein_list = self.split_protein_str(protein_char_list)
        print(protein_list)
        ft_mat = self.vectorizer.fit_transform(protein_list)
        print(ft_mat)
        ft_mat = ft_mat.toarray()
        self.stand_scaler.fit(ft_mat)
        self.count_mat = ft_mat / np.sum(ft_mat, axis=0)
        print(self.count_mat.shape)

    def transform(self, protein_char_list):
        protein_list = self.split_protein_str(protein_char_list)
        ft_mat = self.vectorizer.transform(protein_list)
        ft_mat = ft_mat.toarray()

        if self.trans_type == 'std':
            trans_ft_mat = self.stand_scaler.transform(ft_mat)
        elif self.trans_type == 'prob':
            trans_ft_mat = ft_mat / self.count_mat
        else:
            exit()

        return trans_ft_mat

    def fit_transform(self, protein_char_list):
        protein_list = self.split_protein_str(protein_char_list)
        ft_mat = self.vectorizer.fit_transform(protein_list)
        ft_mat = ft_mat.toarray()
        self.stand_scaler.fit(ft_mat)
        self.count_mat = ft_mat / np.sum(ft_mat, axis=0)

        ft_mat = self.vectorizer.transform(protein_list)
        ft_mat = ft_mat.toarray()

        if self.trans_type == 'std':
            trans_ft_mat = self.stand_scaler.transform(ft_mat)
        elif self.trans_type == 'prob':
            trans_ft_mat = ft_mat / self.count_mat
        else:
            exit()
        print(trans_ft_mat.shape)
        return trans_ft_mat


    def save(self):
        with open(self.obj_file_path, 'wb') as f:
            if sys.version_info > (3, 0):
                pickle.dump(self.__dict__, f)
            else:
                pickle.dump(self.__dict__, f)
            print('save kmer obj ', self.obj_file_path)

    def load(self):
        print('loading kmer obj ', self.obj_file_path)
        with open(self.obj_file_path, 'rb') as f:
            if sys.version_info > (3, 0):
                obj_dict = pickle.load(f)
            else:
                obj_dict = pickle.load(f)
        self.__dict__ = obj_dict

if __name__ == '__main__':

    # raw_list = ['EVQLVESGGGVVQPGRSLRLSCVASQFTFSGHGMHWLRQAPGKGLEWVASTSFAGTKSHYANSVRGRFTISRDNSKNTLYLQMNNLRAEDTALYYCARDSREYECELWTSDYYDFGKPQPCIDTRDVGGLFDMWGQGTMVTVSSQSVLTQPPSVSATPGQKVTISCSGSNSNIGTKYVSWYQHVPGTAPKLLIFESDRRPTGIPDRFSGSKSATSATLTITGLQTGDEAIYYCGTYGDSRTPGGLFGTGTKLTVL',
    #             'MRVTGIRKNYRHLWRWGTMLLGMLMICSAVGNLWVTVYYGVPVWREATTTLFCASDAKAYDTEVHNVWATHACVPTDPNPQEMFVENVTENFNMWKNDMVNQMHEDVISLWDQSLKPCVKLTPLCVTLECSNVNSSGDHNEAHQESMKEMKNCSFNATTVLRDKKQTVYALFYRLDIVPLTENNSSENSSDYYRLINCNTSAITQACPKVTFDPIPIHYCTPAGYAILKCNDKRFNGTGPCHNVSTVQCTHGIKPVVSTQLLLNGSIAEEEIIIRSENLTDNVKTIIVHLNQSVEITCTRPGNNTRKSIRIGPGQTFYATGDIIGDIRQAHCNISEGKWKETLQNVSRKLKEHFQNKTIKFAASSGGDLEITTHSFNCRGEFFYCNTSGLFNGTYNTSMSNGTNSNSTITIPCRIKQIINMWQEVGRAMYAPPIAGNITCKSNITGLLLVRDGGNTDSNTTETFRPGGGDMRNNWRSELYKYKVVEIKPLGIAPTAAKRRVVEREKRAVGIGAVFLGFLGAAGSTMGAASITLTVQARQLLSGIVQQQSNLLKAIEAQQHLLQLTVWGIKQLQTRVLAIERYLKDQQLLGIWGCSGKLICTTAVPWNSSWSNKSQKEIWDNMTWMQWDKEISNYTDTIYRLLEDSQNQQEKNEQDLLALDNWKNLWSWFDITNWLWYIKIFIMIVGGLIGLRIIFAVLSIVNRVRQGYSPLSFQTLTPNPGGPDRLGRIEEEGGEQDKDRSIRLVNGFLALAWDDLRNLCLFSYHRLRDFILVAARVVELLGRSSLKGLQRGWEALKYLGSLVQYWGQELKKSAINLIDTIAIAVAEGTDRIIELVQALCRA',
    #             'EVQLVESGGGVVQPGRSLRLSCVASQFTFSGHGMHWLRQAPGKGLEWVASTSFAGTKSHYANSVRGRFTISRDNSKNTLYLQMNNLRAEDTALYYCARDSREYECELWTSDYYDFGKPQPCIDTRDVGGLFDMWGQGTMVTVSSQSVLTQPPSVSATPGQKVTISCSGSNSNIGTKYVSWYQHVPGTAPKLLIFESDRRPTGIPDRFSGSKSATSATLTITGLQTGDEAIYYCGTYGDSRTPGGLFGTGTKLTVL',
    #             'MRVRKIKRNYHHLWRWGTMLLGLLMTCSVTGQLWVTVYYGVPVWKEATTTLFCASDAKSYEPEAHNVWATHACVPTDPNPQEIKLENVTENFNMWKNNMVEQMHEDIISLWDQSLKPCVKLTPLCVTLNCTEWNQNSTNANSTGRSNVTDDTGMRNCSFNITTEIRDKKKQVHALFYKLDVVQMDGSDNNSYRLINCNTSAITQACPKVSFEPIPIHYCAPAGFAILKCNDKKFNGTGPCKNVSTVQCTHGIKPVVSTQLLLNGSLAEEEIIIRSENITNNAKIIIVQFNES']
    raw_list = ["EEE"]
    kmer_trans = KmerTranslator(trans_type='std', min_df=0.1, name='test')
    kmer_trans.fit(raw_list)
    kmer_trans.save()
    kmer_trans.load()
    # print(len(raw_list[1]))
    print(kmer_trans.transform(raw_list))
    # kmer_trans.save()

