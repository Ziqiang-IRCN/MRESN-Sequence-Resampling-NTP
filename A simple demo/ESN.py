# -*- coding: utf-8 -*-
import torch
import numpy as np
import scipy as sc
import random
from torch.nn.modules import Module
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import sys
def input_state_concatenation(input,state):
    temp_input = torch.cat((input,state),dim=1)
    return temp_input

def state_concatenation_np(input,state):
    temp_input = np.concatenate((input,state),axis=1)
    return temp_input

def state_transform(state,flag = 1,gpu = False):
    if flag == 1:
        state_dim = state[0].shape[1]
        _state = torch.Tensor(len(state),state_dim)
        if gpu == True:
            _state = _state.cuda()
        for col_idx in range(len(state)):
            _state[col_idx,:] = state[col_idx]
    else:
        state_dim = state[0].shape[0]
        _state = torch.Tensor(len(state),state_dim)
        if gpu == True:
            _state = _state.cuda()
        for col_idx in range(len(state)):
            _state[col_idx,:] = state[col_idx]
    return _state


def MAPE(label,predicted):
    result = np.mean(np.abs((label - predicted) / label))
    return result

def MAAPE(label,predicted):
    result = np.mean(np.arctan(np.abs((label - predicted) / label)))
class ESN(Module):

    def __init__(self,input,reservoir,sr,scale_in,leaking_rate,density,Nepochs,eta,mu,sigma,threshold,
                    Win_assign='Uniform',W_assign = 'Uniform',gpu=False):
        super(ESN, self).__init__()
        self.input = int(input)
        self.reservoir = int(reservoir)
        self.spectral_radius = sr
        self.density = density
        self.scale_in = scale_in
        self.leaking_rate = leaking_rate
        self.W = None
        self.Win = None
        self.Win_assigned = Win_assign
        self.W_assigned = W_assign
        self.state_list = []
        self.state = None
        self.Nepochs = Nepochs
        self.threshold = threshold
        self.gain = None
        self.bias = None
        self.eta = eta
        self.mu = mu
        self.sigma = sigma
        self.gpu = gpu
        self.reset_parameters()
        
    def reset_parameters(self):
        key_words_for_Win = ['Xvaier','Uniform','Guassian']
        key_words_for_W = ['Xvaier','Uniform','Guassian']
        try:
            key_words_for_Win.index(self.Win_assigned)
        except ValueError:
            raise ValueError("Only Xvaier,Uniform and Guassian types for assignment of Win")
        try:
            key_words_for_W.index(self.W_assigned)
        except ValueError:
            raise ValueError("Only Xvaier,Uniform and Guassian types for assignment of W")
        #np.random.seed(0)
        if self.Win_assigned == 'Uniform':
            Win_np = np.random.uniform(-self.scale_in,self.scale_in,size=(self.input+1,self.reservoir))
        elif self.Win_assigned =='Xvaier':
            Win_np = (np.random.randn(self.input+1,self.reservoir)/np.sqrt(self.input+1))*self.scale_in
        elif self.Win_assigned == 'Guassian':
            Win_np = np.random.randn(self.input+1,self.reservoir)*self.scale_in
        #np.random.seed(0)
        if self.W_assigned == 'Uniform':
            if self.density<1:
                W_np = np.zeros((self.reservoir,self.reservoir))
                for row in range(self.reservoir):
                    number_row_elements = round(self.density * self.reservoir)
                    #random.seed(0)
                    row_elements = random.sample(range(self.reservoir),int(number_row_elements))
                    #np.random.seed(0)
                    W_np[row,row_elements] = np.random.uniform(-1,+1,size=(1,int(number_row_elements)))
            else:
                #np.random.seed(0)
                W_np = np.random.uniform(-1,+1,size=(self.reservoir,self.reservoir))

        elif self.W_assigned == 'Guassian':
            if self.density<1:
                W_np = np.zeros((self.reservoir,self.reservoir))
                for row in range(self.reservoir):
                    number_row_elements = round(self.density * self.reservoir)
                    row_elements = random.sample(range(self.reservoir),int(number_row_elements))
                    W_np[row,row_elements] = np.random.randn(1,int(number_row_elements))
            else:
                W_np = np.random.randn(self.reservoir,self.reservoir)
        elif self.W_assigned == 'Xvaier':
            if self.density<1:
                W_np = np.zeros((self.reservoir,self.reservoir))
                for row in range(self.reservoir):
                    number_row_elements = round(self.density * self.reservoir)
                    row_elements = random.sample(range(self.reservoir),int(number_row_elements))
                    W_np[row,row_elements] = np.random.randn(1,int(number_row_elements))/np.sqrt(self.reservoir)
            else:
                W_np = np.random.randn(self.reservoir,self.reservoir)/np.sqrt(self.reservoir)


        
        Ws = (1-self.leaking_rate) * np.eye(W_np.shape[0], W_np.shape[1]) + self.leaking_rate * W_np
        eig_values = np.linalg.eigvals(Ws)
        actual_sr = np.max(np.absolute(eig_values))
        Ws = (Ws*self.spectral_radius)/actual_sr
        W_np = (Ws-(1.-self.leaking_rate)*np.eye(W_np.shape[0],W_np.shape[1]))/self.leaking_rate
        Gain_np = np.ones((1,self.reservoir))
        Bias_np = np.zeros((1,self.reservoir))

        self.Win = torch.Tensor(Win_np)
        self.W = torch.Tensor(W_np)
        self.gain = torch.Tensor(Gain_np)
        self.bias = torch.Tensor(Bias_np)
        if self.gpu == True:
            self.Win = torch.Tensor(Win_np).cuda()
            self.W = torch.Tensor(W_np).cuda()
            self.gain = torch.Tensor(Gain_np).cuda()
            self.bias = torch.Tensor(Bias_np).cuda()




    
    def forward(self,input,h_0=None,useIP = True, IPmode = 'training'):

        if useIP == True and IPmode =='training':
            self.computeIntrinsicPlasticity(input,useIP = useIP,IPmode = IPmode)
        if useIP == True and IPmode == 'testing':
            return self.computeState(input,h_0 = h_0,useIP = useIP,IPmode = IPmode)
        if useIP == False:
            return self.computeState(input,h_0=h_0,useIP = useIP,IPmode=IPmode)

    def reset_state(self):
        self.state_list = []

    def computeIntrinsicPlasticity(self,input,useIP,IPmode):
        for epoch in range(self.Nepochs):
            Gain_epoch = self.gain
            Bias_epoch = self.bias
            sys.stdout.write("IP training {} epoch.\n".format(epoch+1))
            self.computeState(input,h_0 = None, useIP = True, IPmode = 'training')

            if (torch.norm(self.gain-Gain_epoch,2) < self.threshold) and (torch.norm(self.bias-Bias_epoch,2)< self.threshold):
                sys.stdout.write("IP training training over.\n")
                sys.stdout.flush()
                break
            if epoch+1 == self.Nepochs:
                sys.stdout.write("total IP training epochs are over.\n".format(epoch+1))
                sys.stdout.write('.')
                sys.stdout.flush()

    
    
    def computeState(self,input,h_0 = None, useIP = False, IPmode = 'testing'):
        back_state_list = []
        if h_0 is None:
            h_0 = torch.zeros(1,self.reservoir)
            if self.gpu == True:
                h_0 = h_0.cuda()
        state = h_0
        input_bias = torch.ones(input.shape[0],1)
        if self.gpu == True:
            input_bias = input_bias.cuda()
        input = torch.cat((input,input_bias),dim=1)
        

        if useIP ==True and IPmode =='training':
            state_before_activated = torch.zeros(1,self.reservoir)
            for col_idx in range(input.shape[0]):
                if col_idx == 0:
                    e = input[col_idx,:].unsqueeze(0)
                    input_part = torch.mm(e,self.Win)
                    inner_part = torch.mm(state,self.W)
                    state_before_activated = input_part+inner_part
                    state = (1.0-self.leaking_rate)*state+(self.leaking_rate)*torch.tanh(self.gain*state_before_activated+self.bias)
                else:
                    e = input[col_idx,:].unsqueeze(0)
                    input_part = torch.mm(e,self.Win)
                    inner_part = torch.mm(state,self.W)
                    state_before_activated = input_part+inner_part
                    state = (1.0-self.leaking_rate)*state+(self.leaking_rate)*torch.tanh(self.gain * state_before_activated+self.bias)
                    eta = self.eta
                    mu = self.mu
                    sigma2 = self.sigma**2
                    deltaBias = -eta*((-mu/sigma2)+state*(2*sigma2+1-(state**2)+mu*state)/sigma2)
                    deltaGain = eta/self.gain + deltaBias*state_before_activated
                    self.gain = self.gain+deltaGain
                    self.bias = self.bias+deltaBias
                self.state_list.append(state)
            back_state_list = self.state_list
            self.reset_state()
        if useIP == True and IPmode == 'testing':
            state_before_activated = torch.zeros(1,self.reservoir)
            for col_idx in range(input.shape[0]):
                e = input[col_idx,:].unsqueeze(0)
                input_part = torch.mm(e,self.Win)
                inner_part = torch.mm(state,self.W)
                state_before_activated = input_part+inner_part
                state = (1.0-self.leaking_rate)*state+(self.leaking_rate)*torch.tanh(self.gain * state_before_activated+self.bias)      
                self.state_list.append(state)
            back_state_list = self.state_list
            self.reset_state()
        if useIP == False:
            for col_idx in range(input.shape[0]):
                e = input[col_idx,:].unsqueeze(0)
                input_part = torch.mm(e,self.Win)
                inner_part = torch.mm(state,self.W)
                state_before_activated = input_part+inner_part
                state = (1.0-self.leaking_rate)*state+(self.leaking_rate)*torch.tanh(state_before_activated)
                self.state_list.append(state)
            back_state_list = self.state_list
            self.reset_state()
        return back_state_list







    
    
    


    






    


    






        



        

        
        
        

    