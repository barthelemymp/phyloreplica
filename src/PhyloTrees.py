# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 17:24:54 2021

@author: bartm
"""
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import math 
import numpy as np
import pandas as pd
import wandb 
from torch._six import string_classes
import collections
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans

class gammaManager_Independant(nn.Module):
    def __init__(self):
        super(gammaManager_Independant, self).__init__()
        self.gammaParents = torch.tensor(0.0)
        self.gammaChildren = torch.tensor(0.0)
        self.timestep = torch.tensor(0.0)
        
    def composeLoss(self, Node):
        return Node.loss + self.gammaParents * Node.coupling_loss_Parents + self.gammaChildren * Node.coupling_loss_Children
    
    def updateGamma(self,Node):
            self.timestep +=1
    def reinitGamma(self, Node):
        self.gammaParents = torch.tensor(0.0).to(self.gammaParents.device)
        self.gammaChildren = torch.tensor(0.0).to(self.gammaChildren.device)
    def to_(self,device):
        self.gammaParents = self.gammaParents.to(device)
        self.gammaChildren = self.gammaChildren.to(device)
        self.timestep = self.timestep.to(device)
        
        
class gammaManager_Constant(nn.Module):
    def __init__(self, gammaParents,gammaChildren, startingTime=0.0):
        super(gammaManager_Constant, self).__init__()
        self.gammaParents_0 = torch.tensor(gammaParents)
        self.gammaChildren_0 = torch.tensor(gammaChildren)
        self.gammaParents = torch.tensor(0.0)
        self.gammaChildren = torch.tensor(0.0)
        self.startingTime = torch.tensor(startingTime)
        self.timestep = torch.tensor(0.0)
        
    def composeLoss(self, Node):
        return Node.loss + self.gammaParents * Node.coupling_loss_Parents + self.gammaChildren * Node.coupling_loss_Children
    
    def updateGamma(self, Node):
        if self.timestep < self.startingTime:
            self.gammaParents = torch.tensor(0.0).to(self.gammaParents.device)
            self.gammaChildren = torch.tensor(0.0).to(self.gammaParents.device)
        else:
            self.gammaParents = self.gammaParents_0
            self.gammaChildren = self.gammaChildren_0
        self.timestep+=1
        return self.gammaParents, self.gammaChildren
    # def reinitGamma(self, Node):
    #     self.gammaParents = 0.0
    #     self.gammaChildren = 0.0
    def to_(self, device):
        self.gammaParents_0 = self.gammaParents_0.to(device)
        self.gammaChildren_0 = self.gammaChildren_0.to(device)
        self.gammaParents = self.gammaParents.to(device)
        self.gammaChildren = self.gammaChildren.to(device)
        self.startingTime = self.startingTime.to(device)
        self.timestep = self.timestep.to(device)
    
class gammaManager_exponential(nn.Module):
    def __init__(self, startingTime, maxiter):
        super(gammaManager_exponential, self).__init__()
        self.gammaParents_0 = torch.tensor(0.0)
        self.gammaParents_1 = torch.tensor(0.0)
        self.timestep = torch.tensor(0.0)
        self.gammaChildren_0 = torch.tensor(0.0)
        self.gammaChildren_1 = torch.tensor(0.0)
        self.startingTime = startingTime
        self.maxiter = maxiter
        
    def composeLoss(self, Node):
        return Node.loss + self.gammaParents * Node.coupling_loss_Parents + self.gammaChildren * Node.coupling_loss_Children
    
    def updateGamma(self, Node):
        return
    
    def reinitGamma(self, Node):
        self.gammaParents_0 = Node.loss
        self.gammaChildren = torch.tensor(0.0)
    
    
    
    
    
class gammaManager_Linear(nn.Module):
    def __init__(self, startingTime, maxiter, finalsplit):
        super(gammaManager_Linear, self).__init__()
        self.gammaParents_0 = torch.tensor(0.0)
        self.gammaParents = torch.tensor(0.0)
        self.timestep = torch.tensor(0.0)
        self.gammaChildren_0 = torch.tensor(0.0)
        self.gammaChildren = torch.tensor(0.0)
        self.startingTime = torch.tensor(startingTime)
        self.maxiter = torch.tensor(maxiter)
        self.finalsplit = torch.tensor(finalsplit)
        
    def composeLoss(self, Node):
        return Node.loss + self.gammaParents * Node.coupling_loss_Parents + self.gammaChildren * Node.coupling_loss_Children
    
    def updateGamma(self, Node):
        if self.timestep < self.startingTime:
            self.gammaParents = torch.tensor(0.0)
            self.gammaChildren = torch.tensor(0.0)
        elif self.timestep == self.startingTime:
            self.reinitGamma(Node)
        else:
            self.gammaParents = self.gammaParents_0 * (torch.min(self.timestep,self.maxiter) - self.startingTime)
            self.gammaChildren =  self.gammaChildren_0 * (torch.min(self.timestep,self.maxiter) - self.startingTime)
        self.timestep+=1
        return self.gammaParents, self.gammaChildren
    
    def reinitGamma(self, Node):
        
        if Node.isRoot:
            self.gammaChildren_0 = self.finalsplit* (Node.loss.clone().detach()/Node.coupling_loss_Children.clone().detach()) / (self.maxiter - self.startingTime)
        elif Node.isLeaf:
            self.gammaParents_0 =self.finalsplit* (Node.loss.clone().detach()/Node.coupling_loss_Parents.clone().detach()) / (self.maxiter - self.startingTime)
        else:
            self.gammaParents_0 =self.finalsplit* (Node.loss.clone().detach()/Node.coupling_loss_Parents.clone().detach()) / (self.maxiter - self.startingTime)
            self.gammaChildren_0 = self.finalsplit* (Node.loss.clone().detach()/Node.coupling_loss_Children.clone().detach()) / (self.maxiter - self.startingTime)
            
    def to_(self,device):
        self.gammaParents_0 = self.gammaParents_0.to(device)
        self.gammaParents = self.gammaParents.to(device)
        self.timestep = self.timestep.to(device)
        self.gammaChildren_0 = self.gammaChildren_0.to(device)
        self.gammaChildren = self.gammaChildren.to(device)
        self.startingTime = self.startingTime.to(device)
        self.maxiter = self.maxiter.to(device)
        self.finalsplit = self.finalsplit.to(device)

        
        
        
        
    
class Callback_SimpleLossSaver():
    def __init__(self):
        self.trainingloss = []
        self.testingloss = []
        self.validationloss = []
        
    def updatetrain(self,Node):
        self.trainingloss.append(Node.loss.item())
    def updatetest(self,Node):
        self.testingloss.append(Node.loss.item())
    def updateval(self,Node):
        self.validationloss.append(Node.loss.item())
        
class Callback_WandBSimpleLossSaver():
    def __init__(self, project):
        import wandb
        print("wandb imported")
        wandb.login()
        print("wandb login")
        wandb.init(project=project, entity="barthelemymp")
        self.config_dict = {
        }
        
        self.trainingloss = []
        self.testingloss = []
        self.validationloss = []
        
    def pushConfig(self, config=None):
        if config==None:
            wandb.config.update(self.config_dict)
        else:
            self.config_dict = config
            wandb.config.update(self.config_dict) 
            
    def updateConfig(self, key, value):
        self.config_dict[key] = value
        self.pushConfig()

    def updatetrain(self,Node, recursive=True):
        wandb.log({"Train loss"+Node.Name: Node.loss.item(), 
                   "epoch":Node.gammaManager.timestep, 
                   "gamma parents"+Node.Name:Node.gammaManager.gammaParents, 
                   "gamma children"+Node.Name:Node.gammaManager.gammaChildren,
                   "distance Loss"+Node.Name:Node.coupling_loss_Parents
                   })
        if Node.isLeaf==False:
            if recursive:
                for i in range(len(Node.children)):
                    child = Node.children[i]
                    self.updatetrain(child)

    def updatetest(self, Node, recursive=True):
        wandb.log({"Test loss"+Node.Name: Node.loss.item(), "epoch":Node.gammaManager.timestep})
        if Node.isLeaf==False:
            if recursive:
                for i in range(len(Node.children)):
                    child = Node.children[i]
                    self.updatetest(child)
                
    def updateval(self, Node, recursive=True):
        wandb.log({"Val loss"+Node.Name: Node.loss.item(), "epoch":Node.gammaManager.timestep})
        if Node.isLeaf==False:
            if recursive:
                for i in range(len(Node.children)):
                    child = Node.children[i]
                    self.updateval(child)
                    
    def updatevalonChildren(self, Node, recursive=True):
        for i in range(len(Node.children)):
            loss = Node.LossFunction(Node.model, Node.children[i].batch)
            wandb.log({"Val loss"+Node.Name+" on "+Node.children[i].Name: Node.loss.item(), "epoch":Node.gammaManager.timestep})
        if recursive:
            for i in range(len(Node.children)):
                if self.children[i].isLeaf==False:
                    self.updatevalonChildren(Node.children[i],recursive=True)
        
        
# class gammaManager_Selflearning(nn.Module):
#     def __init__(self, startingTime, maxiter):
#         super(gammaManager_selflearning, self).__init__()
 
#     def composeLoss(self, Node):
#         return Node.loss + self.gammaParents * Node.coupling_loss_Parents + self.gammaChildren * Node.coupling_loss_Children
    
#     def updateGamma(self):
#         for center_parameters, replica_parameters in zip(self.model.parameters(), self.parent.model.parameters()):
#             self.coupling_loss_Parents += loss_fn_elastic(center_parameters, replica_parameters)
#         return self.gammaParents, self.gammaChildren
#     def reinitGamma(self, Node, finalsplit):

    

class PhyloNode():#nn.Module
    def __init__(self, 
                 model, 
                 optimizer, 
                 LossFunction, 
                 parent=None, 
                 children=[], 
                 dataset = None, 
                 tuplesize=1, 
                 batch_size=32, 
                 gammaManager = gammaManager_Independant(),
                 # callback=Callback_SimpleLossSaver(), 
                 Name="Root"
                 ):
        # super(PhyloNode, self).__init__()
        self.model = model
        self.Name = Name
        self.optimizer = optimizer
        self.parent = parent
        self.children = children
        self.isLeaf = len(children)==0
        self.isRoot = parent==None
        self.dataset = dataset
        self.batch_size = batch_size
        if dataset is not None:
            if isinstance(dataset, list):
                self.train_set = dataset[0]
                self.test_set = dataset[1]
                self.val_set = dataset[2]
            else:
                trainL = int(0.8 * len(dataset))
                testL = int(0.1 * len(dataset))
                valL = len(dataset) - trainL -testL
                self.train_set, self.test_set,  self.val_set = torch.utils.data.random_split(dataset, [trainL, testL, valL])
            self.train_iterator = iter(DataLoader(self.train_set, batch_size=batch_size, shuffle=True))
            self.test_iterator = iter(DataLoader(self.test_set, batch_size=batch_size, shuffle=True))
            self.val_iterator = iter(DataLoader(self.val_set, batch_size=batch_size, shuffle=True))
        self.gammaManager = gammaManager

        self.batch = None
        self.tuplesize = tuplesize
        self.LossFunction = LossFunction
        
        self.isAttractedBychildren = False
        self.isAttractedByParent = False
        
        self.coupling_loss_Parents = torch.tensor(0.0)
        self.coupling_loss_Children = torch.tensor(0.0)
        self.loss = torch.tensor(0.0)
        
    def kmeansSplit(self,K):
        assert self.dataset !=None
        kmeans = KMeans(n_clusters=K, random_state=0).fit(self.dataset.sequences)
        
        self.train_set = torch.utils.data.Subset(self.dataset, kmeans.labels_<=K-3)
        self.test_set = torch.utils.data.Subset(self.dataset, kmeans.labels_==K-2)
        self.val_set = torch.utils.data.Subset(self.dataset, kmeans.labels_==K-1)

        
    def getTrainLength(self): 
        if self.isLeaf:
            return int(0.8 * len(self.dataset))#### error if kmeans TO DO
        else: 
            l = 0
            for i in range(len(self.children)):
                l+=self.children[i].getTrainLength()
            return l
        
    def getTestLength(self): 
        if self.isLeaf:
            return int(0.1 * len(self.dataset))
        else: 
            l = 0
            for i in range(len(self.children)):
                l+=self.children[i].getTestLength()
            return l
        
    def getValLength(self): 
        if self.isLeaf:
            trainL = int(0.8 * len(self.dataset))
            testL = int(0.1 * len(self.dataset))
            valL = len(self.dataset) - trainL -testL
            return valL
        else: 
            l = 0
            for i in range(len(self.children)):
                l+=self.children[i].getValLength()
            return l
    
    

    def getNewTrainBatch(self, fullBatch=False):
        if self.isLeaf:
            if fullBatch == False:
                try:
                    self.batch = next(self.train_iterator)
                except StopIteration:
                    # StopIteration is thrown if dataset ends
                    # reinitialize data loader 
                    self.train_iterator = iter(DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True))
                    self.batch = next(self.train_iterator)
            else:
                self.batch = self.train_set[:]
        else:
            batchlist = []
            for j in range(self.tuplesize):
                batchlist.append([])
            for i in range(len(self.children)):
                child = self.children[i]
                ba = child.getNewTrainBatch(fullBatch=fullBatch)
                for j in range(self.tuplesize):
                    batchlist[j].append(ba[j])
            
            seqs = torch.cat(batchlist[0], dim=0)
            self.batch = (seqs,)
            for j in range(1,self.tuplesize):
                self.batch += (torch.cat(batchlist[j], dim=0),)
        # print(self.Name, self.batch[0].shape, self.batch[1].shape)
        return self.batch
    
    
    def getNewTestBatch(self, fullBatch=False):
        if self.isLeaf:
            if fullBatch == False:
                try:
                    self.batch = next(self.test_iterator)
                except StopIteration:
                    # StopIteration is thrown if dataset ends
                    # reinitialize data loader 
                    self.test_iterator = iter(DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True))
                    self.batch = next(self.test_iterator)
            else:
                self.batch = self.test_set[:]
        else:
            batchlist = []
            for j in range(self.tuplesize):
                batchlist.append([])
            for i in range(len(self.children)):
                child = self.children[i]
                ba = child.getNewTestBatch(fullBatch=fullBatch)
                for j in range(self.tuplesize):
                    batchlist[j].append(ba[j])
            
            seqs = torch.cat(batchlist[0], dim=0)
            self.batch = (seqs,)
            for j in range(1,self.tuplesize):
                self.batch += (torch.cat(batchlist[j], dim=0),)
        return self.batch
    
    def getNewValBatch(self, fullBatch=False):
        if self.isLeaf:
            if fullBatch == False:
                try:
                    self.batch = next(self.val_iterator)
                except StopIteration:
                    # StopIteration is thrown if dataset ends
                    # reinitialize data loader 
                    self.val_iterator = iter(DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True))
                    self.batch = next(self.val_iterator)
                self.batch = next(self.val_iterator)
            else:
                self.batch = self.val_set[:]
        else:
            batchlist = []
            for j in range(self.tuplesize):
                batchlist.append([])
            for i in range(len(self.children)):
                child = self.children[i]
                ba = child.getNewValBatch(fullBatch=fullBatch)
                for j in range(self.tuplesize):
                    batchlist[j].append(ba[j])
            
            seqs = torch.cat(batchlist[0], dim=0)
            self.batch = (seqs,)
            for j in range(1,self.tuplesize):
                self.batch += (torch.cat(batchlist[j], dim=0),)
        return self.batch
                
    def computeLoss(self, recursive=True):
        self.loss = self.LossFunction(self.model, self.batch)
        if recursive:
            for i in range(len(self.children)):
                self.children[i].computeLoss(recursive=True)
        
    def evalmode(self,recursive=True):
        self.model.eval()
        if recursive:
            for i in range(len(self.children)):
                self.children[i].evalmode(recursive=recursive)
                
    def trainmode(self,recursive=True):
        self.model.train()
        if recursive:
            for i in range(len(self.children)):
                self.children[i].trainmode(recursive=recursive)
        
    def zero_grad(self,recursive=True):
        self.optimizer.zero_grad()
        if recursive:
            for i in range(len(self.children)):
                self.children[i].optimizer.zero_grad()
                
    def set_isAttractedBychildren(self, newValue, recursive=True):
        
        if self.isLeaf:
            self.isAttractedBychildren = False
        else:
            self.isAttractedBychildren = newValue
            if recursive:
                for i in range(len(self.children)):
                    self.children[i].set_isAttractedBychildren(newValue, recursive=True)

    def set_isAttractedByParent(self, newValue, recursive=True):
        
        if self.isRoot:
            self.isAttractedByParent = False
        else:
            self.isAttractedByParent = newValue
            if recursive:
                for i in range(len(self.children)):
                    self.children[i].set_isAttractedByParent(newValue, recursive=True)
    
    def computeCouplingLossParent(self, recursive=True):
        loss_fn_elastic = torch.nn.MSELoss(reduction='sum')
        self.coupling_loss_Parents = torch.zeros_like(self.coupling_loss_Parents)
        if self.isRoot==False:
            for center_parameters, replica_parameters in zip(self.model.parameters(), self.parent.model.parameters()):
                self.coupling_loss_Parents += loss_fn_elastic(center_parameters, replica_parameters.clone().detach())
        if recursive:
            for i in range(len(self.children)):
                self.children[i].computeCouplingLossParent(recursive=True)
        return self.coupling_loss_Parents

    
    def computeCouplingLossChildren(self, recursive=True):
        
        loss_fn_elastic = torch.nn.MSELoss(reduction='sum')
        self.coupling_loss_Children = torch.zeros_like(self.coupling_loss_Children)
        if self.isLeaf==False:
            for i in range(len(self.children)):
                child = self.children[i]
                for center_parameters, replica_parameters in zip(self.model.parameters(), child.model.parameters()):
                    self.coupling_loss_Children += loss_fn_elastic(center_parameters, replica_parameters.clone().detach())
        if recursive:
            for i in range(len(self.children)):
                self.children[i].computeCouplingLossChildren(recursive=True)
        return self.coupling_loss_Children
    
    def addChildren(self, child):
        child.isRoot = False
        self.isLeaf = False
        self.children.append(child)
        child.parent = self
        
    def addParent(self, parent):
        assert (self.isRoot),"Already has a parent"
        self.parent = parent
        self.isRoot =False
        parent.isLeaf = False
        parent.children.append(self)
        
    def trainingStep(self, recursive=True):
        # self.getnewTrainBatch()
        # self.trainmode(recursive=recursive)
        # self.computeLoss(recursive=recursive)
        # self.coupling_loss_Children(recursive=recursive)
        # self.coupling_loss_Parents(recursive=recursive)
        self.gammaManager.updateGamma(self)
        self.zero_grad(recursive=False)
        Totalloss = self.gammaManager.composeLoss(self)
        # print("totaloss", Totalloss)
        Totalloss.backward()
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
        # print(self.Name)
        # self.optimizer.step()
        # self.callback.updatetrain(self)
        
        if recursive:
            for i in range(len(self.children)):
                self.children[i].trainingStep(recursive=recursive)
    def optimizerstep(self,recursive=True):
        self.optimizer.step()
        if recursive:
            for i in range(len(self.children)):
                self.children[i].optimizerstep(recursive=recursive)
    def to_(self, device, recursive=True):
        self.model = self.model.to(device)
        if self.dataset is not None:
            self.train_set.dataset.to_(device)
            self.test_set.dataset.to_(device)
            self.val_set.dataset.to_(device)
            self.train_iterator = iter(DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True))
            self.test_iterator = iter(DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True))
            self.val_iterator = iter(DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True))
        self.gammaManager.to_(device)
        self.coupling_loss_Parents = self.coupling_loss_Parents.to(device)
        self.coupling_loss_Children = self.coupling_loss_Children.to(device)
        self.loss = self.loss.to(device)
        if recursive:
            for i in range(len(self.children)):
                self.children[i].to_(device, recursive=recursive)
            
        
    # def testingStep(self, recursive=True):
    #     # self.getnewTrainBatch()
    #     # self.trainmode(recursive=recursive)
    #     # self.computeLoss(recursive=recursive)
    #     # self.coupling_loss_Children(recursive=recursive)
    #     # self.coupling_loss_Parents(recursive=recursive)
    #     # self.gammaManager.updateGamma()
    #     # self.gammaManager.timestep += 1
    #     # self.zero_grad(self,recursive=True)
    #     # Totalloss = self.gammaManager.composeLoss(self)
    #     # Totalloss.backward()
    #     # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1) 
    #     # self.optimizer.step()
    #     self.callback.updatetest(self)
    #     if recursive:
    #         for i in range(len(self.children)):
    #             self.children[i].trainingStep(recursive=recursive)

        
        
        
