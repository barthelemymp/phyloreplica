# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 17:30:33 2021

@author: bartm
"""
import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))#os.getcwd()
sys.path.append('/home/bart/Documents/phyloreplica/src')
from PhyloDataset import *
from PhyloTrees import *
from vae import *
import torch
import wandb
import copy
torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

datapath1 = "../data/PF00072/PF00072_rp15_has_PF00196.faa"
datapath2 = "../data/PF00072/PF00072_rp15_has_PF00486.faa"
datapath3 = "../data/PF00072/PF00072_rp15_has_PF00512.faa"
datapath4 = "data/PF00072/PF00072_rp15_has_PF00158.faa"
datapath5 = "data/PF00072/PF00072_rp15_has_PF00990.faa"
datapath6 = "data/PF00072/PF00072_rp15_has_PF01339.faa"
datapath7 = "data/PF00072/PF00072_rp15_has_PF04397.faa"
datapath8 = "data/PF00072/PF00072_rp15_has_PF12833.faa"

lossfn = vae_loss

dataset1 = MSA(datapath1, device=device)
dataset2 = MSA(datapath2, device=device)
dataset3 = MSA(datapath3, device=device)
lt = len(dataset1) + len(dataset2) + len(dataset1)
l1 = int(32*len(dataset1)/lt) 
l2 = int(32*len(dataset2)/lt) 
l3 = 32 - l1 -l2



vae10 = VAE(21, 5, dataset1.len_protein * dataset1.q, [256, 128]).to(device)
optimizer10 = optim.Adam(vae10.parameters())
# gammaManager1 = gammaManager_Independant()
gammaManager1 = gammaManager_Linear(500, 1500, 0).to(device)
Node1O = PhyloNode(vae10,
          optimizer10, 
          lossfn,
          parent=None, 
          children=[], 
          dataset = dataset1, 
          tuplesize=2, 
          batch_size=l1, 
          gammaManager = gammaManager1,
          Name = "196"
    ).to(device)
Node1O.kmeansSplit(6)

vae20 = VAE(21, 5, dataset2.len_protein * dataset2.q, [256, 128]).to(device)
optimizer20 = optim.Adam(vae20.parameters())
# gammaManager2 = gammaManager_Independant()
gammaManager2 = gammaManager_Linear(500, 1500, 0).to(device)
Node2O = PhyloNode(vae20,
          optimizer20, 
          lossfn,
          parent=None, 
          children=[], 
          dataset = dataset2, 
          tuplesize=2, 
          batch_size=l2, 
          gammaManager = gammaManager2,
          Name="486"
    ).to(device)
Node2O.kmeansSplit(6)

vae30 = VAE(21, 5, dataset3.len_protein * dataset3.q, [256, 128]).to(device)
optimizer30 = optim.Adam(vae30.parameters())
# gammaManager3 = gammaManager_Independant()
gammaManager3 = gammaManager_Linear(500, 1500,0).to(device)
Node3O = PhyloNode(vae30,
          optimizer30, 
          lossfn,
          parent=None, 
          children=[], 
          dataset = dataset3, 
          tuplesize=2, 
          batch_size=l3, 
          gammaManager = gammaManager3,
          Name="512"
    ).to(device)
Node3O.kmeansSplit(6)

vaeR0 =  VAE(21, 5, dataset3.len_protein * dataset3.q, [256, 128]).to(device)
optimizerR0 = optim.Adam(vaeR0.parameters())
# gammaManagerR = gammaManager_Independant()
gammaManagerR = gammaManager_Linear(500, 1500, 0).to(device)
NodeRO = PhyloNode(vaeR0,
          optimizerR0, 
          lossfn,
          parent=None, 
          children=[], 
          tuplesize=2, 
          batch_size=32, 
          gammaManager = gammaManagerR,
          Name="Root"
    ).to(device)

for Wdecay in [0.0, 0.001, 0.01]:
    for gammaP in [0.0, 0.001, 0.005, 0.01, 0.1, 0.5]:
        for gammC in [0.0, 0.001, 0.005, 0.01, 0.1, 0.5]:
        gammaManager1 = gammaManager_Constant(gammaP,gammC)
        gammaManager2 = gammaManager_Constant(gammaP,gammC)
        gammaManager3 = gammaManager_Constant(gammaP,gammC)
        vae1 = copy.deepcopy(vae10).to(device)
        optimizer1 = optim.Adam(vae1.parameters(),weight_decay=Wdecay)
        Node1 = PhyloNode(vae1,
                  optimizer1, 
                  lossfn,
                  parent=None, 
                  children=[], 
                  dataset = [Node1O.train_set, Node1O.test_set, Node1O.val_set], 
                  tuplesize=2, 
                  batch_size=l1, 
                  gammaManager = gammaManager1,
                  Name = "196"
            ).to(device)
        vae2 = copy.deepcopy(vae20).to(device)
        optimizer2 = optim.Adam(vae2.parameters(),weight_decay=Wdecay)
        Node2 = PhyloNode(vae2,
                  optimizer2, 
                  lossfn,
                  parent=None, 
                  children=[], 
                  dataset =  [Node2O.train_set, Node2O.test_set, Node2O.val_set], 
                  tuplesize=2, 
                  batch_size=l2, 
                  gammaManager = gammaManager2,
                  Name="486"
            ).to(device)
        vae3 = copy.deepcopy(vae30).to(device)
        optimizer3 = optim.Adam(vae3.parameters(),weight_decay=Wdecay)
        Node3 = PhyloNode(vae3,
                  optimizer3, 
                  lossfn,
                  parent=None, 
                  children=[], 
                  dataset = [Node3O.train_set, Node3O.test_set, Node3O.val_set], 
                  tuplesize=2, 
                  batch_size=l3, 
                  gammaManager = gammaManager3,
                  Name="512"
            ).to(device)
        vaeR = copy.deepcopy(vaeR0).to(device)
        optimizerR = optim.Adam(vaeR.parameters(),weight_decay=Wdecay)
        gammaManagerRoot = gammaManager_Constant(gammaP, gammC)
        NodeR = PhyloNode(vaeR,
                  optimizerR, 
                  lossfn,
                  parent=None, 
                  children=[], 
                  tuplesize=2, 
                  batch_size=32, 
                  gammaManager = gammaManagerRoot,
                  Name="Root"
            ).to(device)
       
    
        NodeR.addChildren(Node1)
        NodeR.addChildren(Node2)
        NodeR.addChildren(Node3)
        
        callback = Callback_WandBSimpleLossSaver("pf72 phylotree")
        callback.updateConfig("gamma manager", "Linear")
        callback.updateConfig("final split", fsplit)
        callback.updateConfig("familly", "pf72(196 486 512)")
        callback.updateConfig("layers", "512 256 128")
        callback.updateConfig("batch size", "Full batch")
        callback.updateConfig("weight_decay",Wdecay)
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
    
