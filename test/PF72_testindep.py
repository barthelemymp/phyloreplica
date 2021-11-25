# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 17:30:33 2021

@author: bartm
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from PhyloDataset import *
from PhyloTrees import *
from vae import *
import torch
import wandb
import copy
torch.autograd.set_detect_anomaly(True)

datapath1 = "../data/PF00072/PF00072_rp15_has_PF00196.faa"
datapath2 = "../data/PF00072/PF00072_rp15_has_PF00486.faa"
datapath3 = "../data/PF00072/PF00072_rp15_has_PF00512.faa"
datapath4 = "data/PF00072/PF00072_rp15_has_PF00158.faa"
datapath5 = "data/PF00072/PF00072_rp15_has_PF00990.faa"
datapath6 = "data/PF00072/PF00072_rp15_has_PF01339.faa"
datapath7 = "data/PF00072/PF00072_rp15_has_PF04397.faa"
datapath8 = "data/PF00072/PF00072_rp15_has_PF12833.faa"

lossfn = vae_loss

dataset1 = MSA(datapath1)
dataset2 = MSA(datapath2)
dataset3 = MSA(datapath3)
lt = len(dataset1) + len(dataset2) + len(dataset1)
l1 = int(32*len(dataset1)/lt) 
l2 = int(32*len(dataset2)/lt) 
l3 = 32 - l1 -l2



vae1 = VAE(21, 5, dataset1.len_protein * dataset1.q, [512, 256, 128])
optimizer1 = optim.Adam(vae1.parameters())
# gammaManager1 = gammaManager_Independant()
gammaManager1 = gammaManager_Linear(500, 1500, 0)
Node1O = PhyloNode(vae1,
          optimizer1, 
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

vae2 = VAE(21, 5, dataset2.len_protein * dataset2.q, [512, 256, 128])
optimizer2 = optim.Adam(vae2.parameters())
# gammaManager2 = gammaManager_Independant()
gammaManager2 = gammaManager_Linear(500, 1500, 0)
Node2O = PhyloNode(vae2,
          optimizer2, 
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

vae3 = VAE(21, 5, dataset3.len_protein * dataset3.q, [512,256, 128])
optimizer3 = optim.Adam(vae3.parameters())
# gammaManager3 = gammaManager_Independant()
gammaManager3 = gammaManager_Linear(500, 1500,0)
Node3O = PhyloNode(vae3,
          optimizer3, 
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

vaeR =  VAE(21, 5, dataset3.len_protein * dataset3.q, [512, 256, 128])
optimizerR = optim.Adam(vaeR.parameters())
# gammaManagerR = gammaManager_Independant()
gammaManagerR = gammaManager_Linear(500, 1500, 0)
NodeRO = PhyloNode(vaeR,
          optimizerR, 
          lossfn,
          parent=None, 
          children=[], 
          dataset = dataset3, 
          tuplesize=2, 
          batch_size=32, 
          gammaManager = gammaManagerR,
          Name="Root"
    )

for fsplit in [0.0, 0.01, 0.1, 0.2, 0.5, 1.0]:
    gammaManager1 = gammaManager_Linear(500, 1500, fsplit)
    gammaManager2 = gammaManager_Linear(500, 1500, fsplit)
    gammaManager3 = gammaManager_Linear(500, 1500, fsplit)
    gammaManagerRoot = gammaManager_Linear(500, 1500, fsplit)
    Node1 = copy.deepcopy(Node1O)
    Node1.gammaManager = gammaManager1
    Node2 = copy.deepcopy(Node2O)
    Node2.gammaManager = gammaManager2
    Node3 = copy.deepcopy(Node3O)
    Node3.gammaManager = gammaManager3
    NodeR = copy.deepcopy(NodeRO)
    NodeR.gammaManager = gammaManagerRoot
    
    
    NodeR.addChildren(Node1)
    NodeR.addChildren(Node2)
    NodeR.addChildren(Node3)
    callback = Callback_WandBSimpleLossSaver("pf72 phylotree")
    callback.updateConfig("gamma manager", "Linear")
    callback.updateConfig("final split", fsplit)
    callback.updateConfig("familly", "pf72(196 486 512)")
    callback.updateConfig("layers", "256 128")
    callback.updateConfig("batch size", "Full batch")
    callback.updateConfig("weight_decay", 0.0)
    callback.updateConfig("scheduler", "No scheduler")
    Nstep = 3000
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
    