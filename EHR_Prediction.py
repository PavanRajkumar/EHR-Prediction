#!/usr/bin/env python
# coding: utf-8

# ## Predicting medical diagnoses with RNNs using Fast AI API 
# implementation Doctor AI paper using Electronic Health Records
# Author : Pavan Rajkumar

import pickle
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


# This study will utilize the [MIMIC III](https://mimic.physionet.org/) electronic health record (EHR) dataset

seqs = np.array(pickle.load(open('data/Jan19.seqs','rb')))


# The data pre-processed datasets will be loaded and split into a train, test and validation set at a `75%:15%:10%` ratio.


def load_data(seqFile, labelFile, test_frac=0.15, valid_frac=0.10):
    sequences = np.array(pickle.load(open(seqFile,'rb')))
    labels = np.array(pickle.load(open(labelFile,'rb')))
    
    dataSize = len(labels)
    idx = np.random.permutation(dataSize)
    nTest = int(np.ceil(test_frac * dataSize))
    nValid = int(np.ceil(valid_frac * dataSize))

    test_idx = idx[:nTest]
    valid_idx = idx[nTest:nTest+nValid]
    train_idx = idx[nTest+nValid:]

    train_x = sequences[train_idx]
    train_y = labels[train_idx]
    test_x = sequences[test_idx]
    test_y = labels[test_idx]
    valid_x = sequences[valid_idx]
    valid_y = labels[valid_idx]

    train_x = [sorted(seq) for seq in train_x]
    train_y = [sorted(seq) for seq in train_y]
    valid_x = [sorted(seq) for seq in valid_x]
    valid_y = [sorted(seq) for seq in valid_y]
    test_x = [sorted(seq) for seq in test_x]
    test_y = [sorted(seq) for seq in test_y]

    train = (train_x, train_y)
    test = (test_x, test_y)
    valid = (valid_x, valid_y)
    return (train, test, valid)


# ### Padding sequences: to address variable length sequences

artificalData_seqs = np.array(pickle.load(open('../GRU_EHR/ArtificialEHR_Data.encodedDxs','rb')))




artificalData_seqs


# In[10]:


## Remove array 
artificalData_seqs = [sorted(seq) for seq in artificalData_seqs]
artificalData_seqs


# <img src="img/Patients.png" style="height:200px">

# In[11]:


def padding(seqs, labels, inputDimSize, numClass):
    lengths = np.array([len(seq) for seq in seqs]) - 1
    maxlen = np.max(lengths)
    num_samples = len(seqs)
    

    x = torch.zeros(maxlen, num_samples, inputDimSize)
    y = torch.zeros(maxlen, num_samples, numClass)
    mask = torch.zeros(maxlen, num_samples)

    for idx, (seq, label) in enumerate(zip(seqs, labels)):
        for xvec, subseq in zip(x[:, idx, :], seq[:-1]):
            xvec[subseq] = 1.
        for yvec, subseq in zip(y[:, idx, :], label[1:]):
            yvec[subseq] = 1. 

        mask[:lengths[idx], idx] = 1.

    lengths = torch.LongTensor(lengths)
    return x, y, mask, lengths



lenghts_artificial = np.array([len(seq) for seq in artificalData_seqs]) - 1
lenghts_artificial # here we can see that the final sequences for each patient was reduced by 1



x_artifical, y_artifical, mask_artifical, lenghts_artifical = padding(artificalData_seqs, artificalData_seqs, 11, 11)

x_artifical

x_artifical.shape 

y_artifical

lenghts_artificial

mask_artifical

class Dataset():
    def __init__(self, x, y):
        self.x, self.y = x, y
    def __len__(self): return len(self.x)
    def __getitem__(self, i ): return self.x[i], self.y[i]


class Sampler():
    def __init__(self, ds, bs, shuffle=False):
        self.n,self.bs,self.shuffle = (len(ds)//bs)*bs,bs,shuffle  
        # Note: self.n = (len(ds)//bs) keeps the exact amount of samples needed for your desired batchSize
        
    def __iter__(self):
        self.idxs = torch.randperm(self.n) if self.shuffle else torch.arange(self.n)
        for i in range(0, self.n, self.bs): yield self.idxs[i:i+self.bs]


# The `DataLoader` class combinds the dataset and the data sampler which iterates over the dataset and grabs batches.


def collate(batch_pairs):
    x,y = zip(*batch_pairs)
    return (x,y)

class DataLoader():
    def __init__(self, ds, sampler, collate_fn=collate):
        self.ds,self.sampler,self.collate_fn = ds,sampler,collate_fn
    def __len__(self): return len(self.ds)
    def __iter__(self):
        for s in self.sampler: yield self.collate_fn([self.ds[i] for i in s])


# ### Embedding Layer


class Custom_Embedding(nn.Module):
    def __init__ (self, inputDimSize, embSize):
        super(Custom_Embedding, self).__init__()
        self.inputDimSize = inputDimSize
        self.embSize = embSize
        
        self.W_emb = nn.Parameter(torch.randn(self.inputDimSize, self.embSize) * 0.01)
        self.b_emb = nn.Parameter(torch.zeros(self.embSize) * 0.01) 
       
    def forward(self, x):
        return torch.tanh(x@self.W_emb + self.b_emb)


# ### Dropout Layer



def dropout_mask(x, sz, p):
    return x.new(*sz).bernoulli_(1-p).div_(1-p)




dropout_mask = dropout_mask(x_artifical, (x_artifical.size(0),1,x_artifical.size(2)), 0.3)
dropout_mask





(x_artifical * dropout_mask)


# In[26]:


(x_artifical * dropout_mask).std(),x_artifical.std() 



class EHR_GRU(Custom_Embedding):
    def __init__(self, inputDimSize, hiddenDimSize, embSize, numClass, numLayers):
        super().__init__(inputDimSize, embSize)
        
        self.numClass = numClass
        self.numLayers = numLayers
        self.hiddenDimSize = hiddenDimSize
        self.emb = Custom_Embedding(inputDimSize, embSize)
        
        self.W_r = nn.Parameter(torch.randn(embSize, hiddenDimSize)* 0.01)
        self.W_z = nn.Parameter(torch.randn(embSize, hiddenDimSize)* 0.01)
        self.W_h = nn.Parameter(torch.randn(embSize, hiddenDimSize)* 0.01)
        
        self.U_r = nn.Parameter(torch.randn(hiddenDimSize, hiddenDimSize)* 0.01)
        self.U_z = nn.Parameter(torch.randn(hiddenDimSize, hiddenDimSize)* 0.01)
        self.U_h = nn.Parameter(torch.randn(hiddenDimSize, hiddenDimSize)* 0.01)
        
        self.b_r = nn.Parameter(torch.randn(hiddenDimSize))
        self.b_z = nn.Parameter(torch.randn(hiddenDimSize))
        self.b_h = nn.Parameter(torch.randn(hiddenDimSize))
        
        self.W_output = nn.Parameter(torch.randn(embSize, numClass))
        self.b_output = nn.Parameter(torch.randn(numClass))
        
    def forward(self, emb, mask):
        h = self.init_hidden(emb.size(1))
        
        z = torch.sigmoid(emb@self.W_z + h@self.U_z + self.b_z) 
        r = torch.sigmoid(emb@self.W_r + h@self.U_r + self.b_r)
        h_tilde = torch.tanh(emb@self.W_h + (r * h)@self.U_h + self.b_h)
        h_new = z * h + ((1. - z) * h_tilde)
        h_new = mask[:, :, None] * h_new + (1. - mask)[:, :, None] * h
       
        return h_new
    
    def init_hidden(self, batchSize):
        return Variable(torch.zeros(1, batchSize, hiddenDimSize))


# ### GRU Layer:



class build_EHR_GRU(EHR_GRU):
    def __init__(self, GRUCell, *kwargs):
        super().__init__(inputDimSize, hiddenDimSize, embSize, numClass, numLayers)
        self.cell = GRUCell(*kwargs)
        self.emb = Custom_Embedding(inputDimSize, embSize)

    def forward(self, x, mask):
        inputVector = self.emb(x)
        for i in range(numLayers):
            memories = self.cell(inputVector, mask)
            drop_out = dropout_mask(inputVector, (inputVector.size(0), 1, inputVector.size(2)), 0.5)
            inputVector = memories * drop_out
        
        y_linear = inputVector@self.W_output + self.b_output
        output = F.softmax(y_linear, dim=1)
        output = output * mask[:,:,None]
        return output, inputVector


# ### Loss Function:


class cost_function():
    def __init__(self, yhat, y, L_2=0.001, logEps=1e-8):
        self.yhat = yhat
        self.y = y
       
        self.logEps = logEps
        self.L_2 = L_2
        
        self.W_out = nn.Parameter(torch.randn(hiddenDimSize, numClass)*0.01)
        
    def cross_entropy(self):
        return  -(self.y * torch.log(self.yhat + self.logEps) + (1. - self.y) * torch.log(1. - self.yhat + self.logEps))
    
    def prediction_loss(self):
        return  (torch.sum(torch.sum(self.cross_entropy(), dim=0),dim=1)).float()/  lengths.float()
    
    def cost(self):
        return torch.mean(self.prediction_loss()) + self.L_2 * (self.W_out ** 2).sum() # regularize
    


# ### Model Parameters:



numClass = 4894
inputDimSize = 4894
embSize = 200
hiddenDimSize = 200
batchSize = 100
numLayers = 2


# ### Load Data:



train, valid, test = load_data('data/Jan19.seqs', 'data/Jan19.seqs')




train_ds= Dataset(train[0], train[1])
train_samp = Sampler(train_ds, batchSize, shuffle=True)
train_dl = DataLoader(train_ds, sampler=train_samp, collate_fn=collate)





valid_ds= Dataset(valid[0], valid[1])
valid_samp = Sampler(valid_ds, batchSize, shuffle=False)
valid_dl = DataLoader(valid_ds, sampler=valid_samp, collate_fn=collate)


# ### Instantiate model




model = build_EHR_GRU(EHR_GRU, inputDimSize, hiddenDimSize, embSize, numClass, numLayers)


# ### Training and validation loop



optimizer = torch.optim.Adadelta(model.parameters(), lr = 0.01, rho=0.95)
epochs = 10

counter = 0
for e in range(epochs):
    for x, y in train_dl:
        x, y , mask, lengths = padding(x, y, inputDimSize, numClass)
        
        output, h = model(x, mask)
        
        loss = cost_function(output, y).cost()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        optimizer.zero_grad()
    
    with torch.no_grad():
            model.eval()
            val_loss = []
            for x_valid, y_valid in valid_dl:
                    x_val, y_val, mask, lengths = padding(x_valid, y_valid, inputDimSize, numClass)
                    outputs_val, hidden_val = model(x_val,  mask)
                    loss = cost_function(outputs_val, y_val).cost()
                    val_loss.append(loss.item())
            model.train()

            print("Epoch: {}/{}...".format(e+1, epochs),
                                  "Step: {}...".format(counter),
                                  "Training Loss: {:.4f}...".format(loss.item()),
                                  "Val Loss: {:.4f}".format(torch.mean(torch.tensor(val_loss))))



