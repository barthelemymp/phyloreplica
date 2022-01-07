# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 16:40:06 2021

@author: bartm
"""
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import math 
import numpy as np
import pandas as pd

from torch._six import string_classes
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader


def read_fasta(fasta_path, alphabet='ACDEFGHIKLMNPQRSTVWY-', default_index=20):

    # read all the sequences into a dictionary
    seq_dict = {}
    with open(fasta_path, 'r') as file_handle:
        seq_id = None
        for line in file_handle:
            line = line.strip()
            if line.startswith(">"):
                seq_id = line
                seq_dict[seq_id] = ""
                continue
            assert seq_id is not None
            line = ''.join([c for c in line if c.isupper() or c == '-'])
            seq_dict[seq_id] += line

    aa_index = defaultdict(lambda: default_index, {alphabet[i]: i for i in range(len(alphabet))})

    seq_msa = []
    keys_list = []
    for k in seq_dict.keys():
        seq_msa.append([aa_index[s] for s in seq_dict[k]])
        keys_list.append(k)

    seq_msa = np.array(seq_msa, dtype=int)

    # reweighting sequences
    seq_weight = np.zeros(seq_msa.shape)
    for j in range(seq_msa.shape[1]):
        aa_type, aa_counts = np.unique(seq_msa[:, j], return_counts=True)
        num_type = len(aa_type)
        aa_dict = {}
        for a in aa_type:
            aa_dict[a] = aa_counts[list(aa_type).index(a)]
        for i in range(seq_msa.shape[0]):
            seq_weight[i, j] = (1.0 / num_type) * (1.0 / aa_dict[seq_msa[i, j]])
    tot_weight = np.sum(seq_weight)
    seq_weight = seq_weight.sum(1) / tot_weight

    return seq_msa, seq_weight,keys_list, len(alphabet)


class MSA(torch.utils.data.Dataset):
    def __init__(self, fastaPath,  mapstring = 'ACDEFGHIKLMNPQRSTVWY-', transform=None, device=None, get_fitness = None, flatten=False):
        """
        Args:
            fastaPath (string): Path to the fasta file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        seq_nat, w_nat, ks, q = read_fasta(fastaPath)
        self.q = q
        self.get_fitness = get_fitness
        if get_fitness !=None:
            self.fitness = torch.tensor(list(map(get_fitness, ks)))
        
            
        self.mapstring=mapstring
        self.SymbolMap=dict([(mapstring[i],i) for i in range(len(mapstring))])

        
        ##todo
#         self.inputsize = len(df.iloc[1][0].split(" "))+2
#         self.outputsize = len(df.iloc[1][1].split(" "))+2
        
        # read data
       
        self.nseq, self.len_protein = seq_nat.shape
        
#         seq_msa = torch.from_numpy(seq_msa)
#         train_msa = one_hot(seq_msa, num_classes=num_res_type).cuda()
#         train_msa = train_msa.view(train_msa.shape[0], -1).float()

        self.train_weight = torch.from_numpy(w_nat)
        self.train_weight = (self.train_weight / torch.sum(self.train_weight))
        self.gap = "-"
        train_msa = torch.nn.functional.one_hot(torch.from_numpy(seq_nat).long(), num_classes=self.q)
        if flatten:
            self.sequences = train_msa.view(train_msa.shape[0], -1).float()
        else:
            self.sequences = train_msa.float()
        
#         self.tensorIN=torch.zeros(self.inputsize,len(df), 25)
#         self.tensorOUT=torch.zeros(self.outputsize,len(df), 25)
        self.device = device
        self.transform = transform
#         self.batch_first = batch_first

#         if Unalign==False:
#             print("keeping the gap")
#             for i in range(len(df)):
#                 inp = [self.SymbolMap[self.init_token]]+[self.SymbolMap[k] for k in df[0][i].split(" ")]+[self.SymbolMap[self.eos_token]]
#                 out = [self.SymbolMap[self.init_token]]+[self.SymbolMap[k] for k in df[1][i].split(" ")]+[self.SymbolMap[self.eos_token]]
    
#                 self.tensorIN[:,i,:] = torch.nn.functional.one_hot(torch.tensor(inp), num_classes=25)
#                 self.tensorOUT[:,i,:] = torch.nn.functional.one_hot(torch.tensor(out), num_classes=25)
#         else:
#             print("Unaligning and Padding")
#             for i in range(len(df)):
#                 inp = [self.SymbolMap[self.init_token]]+[self.SymbolMap[k] for k in df[0][i].split(" ") if k!=self.gap]+[self.SymbolMap[self.eos_token]]
#                 out = [self.SymbolMap[self.init_token]]+[self.SymbolMap[k] for k in df[1][i].split(" ") if k!=self.gap]+[self.SymbolMap[self.eos_token]]
#                 inp += [self.SymbolMap[self.pad_token]]*(self.inputsize - len(inp))
#                 out += [self.SymbolMap[self.pad_token]]*(self.outputsize - len(out))
#                 self.tensorIN[:,i,:] = torch.nn.functional.one_hot(torch.tensor(inp), num_classes=25)
#                 self.tensorOUT[:,i,:] = torch.nn.functional.one_hot(torch.tensor(out), num_classes=25)
                
#         if batch_first:
#             self.tensorIN = torch.transpose(self.tensorIN, 0,1)
#             self.tensorOUT = torch.transpose(self.tensorOUT, 0,1)
            
        if device != None:
            self.train_weight= self.train_weight.to(device, non_blocking=True)
            self.sequences= self.sequences.to(device, non_blocking=True)
            if get_fitness !=None:
                self.fitness= self.fitness.to(device, non_blocking=True)

    def __len__(self):
        return self.sequences.shape[0]
#         if self.batch_first:
#         else:
#             return self.tensorIN.shape[1]

    def __getitem__(self, idx): # from the dataset, gives the data in the form it will be used by the NN
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.get_fitness !=None:
            return self.sequences[idx,:],self.train_weight[idx], self.fitness[idx]
        else:
            return self.sequences[idx,:], self.train_weight[idx]
#         if self.batch_first:
#         else:
#             return self.tensorIN[:,idx,:], self.tensorOUT[:,idx,:]

    def uniformWeights(self):
        self.train_weight = torch.ones(self.train_weight.shape)
        
    def to_(self, device):
        if device != None:
            self.train_weight= self.train_weight.to(device, non_blocking=True)
            self.sequences= self.sequences.to(device, non_blocking=True)
            if self.get_fitness !=None:
                self.fitness= self.fitness.to(device, non_blocking=True)
        
        
        


















class PhyloNodeDataset(torch.utils.data.Dataset):
    def __init__(self, list_of_dataset, mappingTree,weights=None):
        self.list_of_dataset = list_of_dataset
        if weights !=None:
            self.weights=weights
        else:
            self.weights = []
            self.totlen = 0
            for i in range(len(list_of_dataset)):
                self.totlen += len(list_of_dataset[i])
                self.weights.append(len(list_of_dataset[i]))
            self.weights /= self.totlen
            
        self.isNode = []
        self.mappingTree = mappingTree
        for i in range(len(list_of_dataset)):
            self.isNode.append(not isinstance(list_of_dataset[i], PhyloNodeDataset))
            
    def __len__(self):
        length = 0
        for i in range(len(list_of_dataset)):
            length+=len(self.list_of_dataset[i])
        return length
    
    def __getitem__(self, idx): # from the dataset, gives the data in the form it will be used by the NN
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        treeIdx = self.mappingTree[idx]
        return self.list_of_dataset[treeIdx].__getitem__(idx)
    
# class PhyloNodeDataset(torch.utils.data.Dataset):
#     def __init__(self, list_of_dataset, mappingTree,weights=None):
#         self.list_of_dataset = list_of_dataset
#         if weights !=None:
#             self.weights=weights
#         else:
#             self.weights = []
#             self.totlen = 0
#             for i in range(len(list_of_dataset)):
#                 self.totlen += len(list_of_dataset[i])
#                 self.weights.append(len(list_of_dataset[i]))
#             self.weights /= self.totlen
            
#         self.isNode = []
#         self.mappingTree = mappingTree
#         for i in range(len(list_of_dataset)):
#             self.isNode.append(not isinstance(list_of_dataset[i], PhyloNodeDataset))
            
#     def __len__(self):
#         length = 0
#         for i in range(len(list_of_dataset)):
#             length+=len(self.list_of_dataset[i])
#         return length
    
#     def __getitem__(self, idx): # from the dataset, gives the data in the form it will be used by the NN
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
            
#         treeIdx = self.mappingTree[idx]
#         return self.list_of_dataset[treeIdx].__getitem__(idx)
    
    