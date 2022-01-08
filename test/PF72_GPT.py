# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 13:27:09 2022

@author: bartm
"""

import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))#os.getcwd()
sys.path.append('/home/bart/Documents/phyloreplica/src')
from PhyloDataset import *
from PhyloTrees import *
from GProT import *
import torch
import wandb
import copy
torch.autograd.set_detect_anomaly(True)


config = GPTConfig(21, 112)
co = GPT1Config(21, 112)
co.n_layer = 3
co.n_head = 7
co.n_embd = 50*21

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

datapath1 = "../data/PF00072/PF00072_rp15_has_PF00196.faa"
datapath2 = "../data/PF00072/PF00072_rp15_has_PF00486.faa"
datapath3 = "../data/PF00072/PF00072_rp15_has_PF00512.faa"
datapath4 = "data/PF00072/PF00072_rp15_has_PF00158.faa"
datapath5 = "data/PF00072/PF00072_rp15_has_PF00990.faa"
datapath6 = "data/PF00072/PF00072_rp15_has_PF01339.faa"
datapath7 = "data/PF00072/PF00072_rp15_has_PF04397.faa"
datapath8 = "data/PF00072/PF00072_rp15_has_PF12833.faa"

lossfn = GPT_loss

dataset1 = MSA(datapath1,onehot=False)
dataset2 = MSA(datapath2,onehot=False)
dataset3 = MSA(datapath3,onehot=False)
maxbs = 64
lt = len(dataset1) + len(dataset2) + len(dataset1)
l1 = int(32*len(dataset1)/lt) 
l2 = int(32*len(dataset2)/lt) 
l3 = 32 - l1 -l2



GPT10 = model = GPT(co, device)
optimizer10 = optim.Adam(GPT10.parameters())
# gammaManager1 = gammaManager_Independant()
gammaManager1 = gammaManager_Linear(500, 1500, 0)
Node1O = PhyloNode(GPT10,
          optimizer10, 
          lossfn,
          parent=None, 
          children=[], 
          dataset = dataset1, 
          tuplesize=2, 
          batch_size=l1, 
          gammaManager = gammaManager1,
          Name = "196"
    )
Node1O.kmeansSplit(6)

GPT20 = GPT(co, device)
optimizer20 = optim.Adam(GPT20.parameters())
# gammaManager2 = gammaManager_Independant()
gammaManager2 = gammaManager_Linear(500, 1500, 0)
Node2O = PhyloNode(GPT20,
          optimizer20, 
          lossfn,
          parent=None, 
          children=[], 
          dataset = dataset2, 
          tuplesize=2, 
          batch_size=l2, 
          gammaManager = gammaManager2,
          Name="486"
    )
Node2O.kmeansSplit(6)

GPT30 = GPT(co, device)
optimizer30 = optim.Adam(GPT30.parameters())
# gammaManager3 = gammaManager_Independant()
gammaManager3 = gammaManager_Linear(500, 1500,0)
Node3O = PhyloNode(GPT30,
          optimizer30, 
          lossfn,
          parent=None, 
          children=[], 
          dataset = dataset3, 
          tuplesize=2, 
          batch_size=l3, 
          gammaManager = gammaManager3,
          Name="512"
    )
Node3O.kmeansSplit(6)

GPTR0 =  GPT(co, device)
optimizerR0 = optim.Adam(GPTR0.parameters())
# gammaManagerR = gammaManager_Independant()
gammaManagerR = gammaManager_Linear(500, 1500, 0)
NodeRO = PhyloNode(GPTR0,
          optimizerR0, 
          lossfn,
          parent=None, 
          children=[], 
          tuplesize=2, 
          batch_size=32, 
          gammaManager = gammaManagerR,
          Name="Root"
    )


for gammaP in [0.0, 0.001, 0.01, 0.1]:
    for gammC in [0.0, 0.001, 0.01, 0.1]:
        gammaManager1 = gammaManager_Constant(gammaP,gammC)
        gammaManager2 = gammaManager_Constant(gammaP,gammC)
        gammaManager3 = gammaManager_Constant(gammaP,gammC)
        GPT1 = copy.deepcopy(GPT10)
        optimizer1 = optim.Adam(GPT1.parameters())
        Node1 = PhyloNode(GPT1,
                  optimizer1, 
                  lossfn,
                  parent=None, 
                  children=[], 
                  dataset = [Node1O.train_set, Node1O.test_set, Node1O.val_set], 
                  tuplesize=2, 
                  batch_size=l1, 
                  gammaManager = gammaManager1,
                  Name = "196"
            )
        GPT2 = copy.deepcopy(GPT20)
        optimizer2 = optim.Adam(GPT2.parameters())
        Node2 = PhyloNode(GPT2,
                  optimizer2, 
                  lossfn,
                  parent=None, 
                  children=[], 
                  dataset =  [Node2O.train_set, Node2O.test_set, Node2O.val_set], 
                  tuplesize=2, 
                  batch_size=l2, 
                  gammaManager = gammaManager2,
                  Name="486"
            )
        GPT3 = copy.deepcopy(GPT30)
        optimizer3 = optim.Adam(GPT3.parameters())
        Node3 = PhyloNode(GPT3,
                  optimizer3, 
                  lossfn,
                  parent=None, 
                  children=[], 
                  dataset = [Node3O.train_set, Node3O.test_set, Node3O.val_set], 
                  tuplesize=2, 
                  batch_size=l3, 
                  gammaManager = gammaManager3,
                  Name="512"
            )
        GPTR = copy.deepcopy(GPTR0)
        optimizerR = optim.Adam(GPTR.parameters())
        gammaManagerRoot = gammaManager_Constant(gammaP, gammC)
        NodeR = PhyloNode(GPTR,
                  optimizerR, 
                  lossfn,
                  parent=None, 
                  children=[], 
                  tuplesize=2, 
                  batch_size=32, 
                  gammaManager = gammaManagerRoot,
                  Name="Root"
            )
        
       
    
        NodeR.addChildren(Node1)
        NodeR.addChildren(Node2)
        NodeR.addChildren(Node3)
        NodeR.to_(device, recursive=True)
        
        
        callback = Callback_WandBSimpleLossSaver("pf72 phylotree GPT")
        callback.updateConfig("gamma manager", "Constant")
        callback.updateConfig("familly", "pf72(196 486 512)")
        callback.updateConfig("n layers",co.n_layer )
        callback.updateConfig("n_head = 7", co.n_head)
        callback.updateConfig("internal dim", co.n_embd)
        callback.updateConfig("batch size", "Full batch")
        # callback.updateConfig("weight_decay",Wdecay)
        callback.updateConfig("scheduler", "No scheduler")
        Nstep = 2000
        for step in range(Nstep):
            recursive = True
            NodeR.getNewTrainBatch(fullBatch=True)
            NodeR.trainmode(recursive=recursive)
            NodeR.computeLoss(recursive=recursive)
            NodeR.computeCouplingLossChildren(recursive=recursive)
            print("children loss done")
            NodeR.computeCouplingLossParent(recursive=recursive)
            NodeR.trainingStep(recursive=recursive)
            NodeR.optimizerstep(recursive=recursive)
            callback.updatetrain(NodeR, recursive=True)
            if step%1==0:
                NodeR.getNewTestBatch(fullBatch=True)
                NodeR.evalmode(recursive=recursive)
                NodeR.computeLoss(recursive=recursive)
                callback.updatetest(NodeR, recursive=True)
        wandb.finish()
    
    
    
    # class mycallback():
    #     def __init__(self,a=5):
    #         self.a=a
    #     def updatetrain(self,Node, recursive=True):
    #         print(Node.Name, Node.gammaManager.gammaParents, Node.gammaManager.gammaChildren)
    #         for i in range(len(Node.children)):
    #             print("children time of callback", i,Node.children[i].Name )
    #             self.updatetrain(Node.children[i])
    #     def updatetest(self,Node, recursive=True):
    #         print(Node.Name, Node.gammaManager.gammaParents, Node.gammaManager.gammaChildren)
    #         for i in range(len(Node.children)):
    #             self.updatetest(Node.children[i])
    # callback = mycallback()
    
