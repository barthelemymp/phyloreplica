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

vae1 = VAE(21, 5, dataset1.len_protein * dataset1.q, [256, 128])
optimizer1 = optim.Adam(vae1.parameters(),weight_decay=0.01)
# gammaManager1 = gammaManager_Independant()
gammaManager1 = gammaManager_Linear(1, 1500, 0.1)

Node1 = PhyloNode(vae1,
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


vae2 = VAE(21, 5, dataset2.len_protein * dataset2.q, [256, 128])
optimizer2 = optim.Adam(vae2.parameters(),weight_decay=0.01)
# gammaManager2 = gammaManager_Independant()
gammaManager2 = gammaManager_Linear(500, 1500, 0.1)
Node2 = PhyloNode(vae2,
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


vae3 = VAE(21, 5, dataset3.len_protein * dataset3.q, [256, 128])
optimizer3 = optim.Adam(vae3.parameters(),weight_decay=0.01)
# gammaManager3 = gammaManager_Independant()
gammaManager3 = gammaManager_Linear(500, 1500, 0.1)
Node3 = PhyloNode(vae3,
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

vaeR =  VAE(21, 5, dataset3.len_protein * dataset3.q, [256, 128])
optimizerR = optim.Adam(vaeR.parameters(),weight_decay=0.01)
# gammaManagerR = gammaManager_Independant()
gammaManagerR = gammaManager_Linear(500, 1500, 0.1)
NodeR = PhyloNode(vaeR,
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

NodeR.addChildren(Node1)
NodeR.addChildren(Node2)
NodeR.addChildren(Node3)
callback = Callback_WandBSimpleLossSaver("pf72 phylotree")


Nstep = 5000
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
        
        
    