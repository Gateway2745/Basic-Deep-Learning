#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 16:34:51 2019

@author: rohit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset=pd.read_csv("bank.csv")

X=dataset.iloc[:,3:-1].values
y=dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
le2=LabelEncoder()
X[:,1]=le.fit_transform(X[:,1])
X[:,2]=le2.fit_transform(X[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:] 
  
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0) 

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

class ANN():
    global m
    global y_pred
    def __init__(self, n_output, n_features , no_hidden=128 ,epochs=1000, eta=0.3 ,lb = 0.001):
        
        self.n_inp = n_features
        self.n_out = n_output
        self.no_hidden = no_hidden
        self.epochs = epochs
        self.eta = eta
        self.lb = lb
        self.w1 ,self.w2 ,self.b1 , self.b2 = self.initialize_weights_and_bias()
        self.output=[]
        self.grads=[]
        
    def initialize_weights_and_bias(self):
        w1 = np.random.rand(self.no_hidden , self.n_inp)
        w2 = np.random.rand(self.n_out, self.no_hidden)
        b1 = np.random.rand(self.no_hidden, 1)
        b2 = np.random.rand(self.n_out, 1)
        return w1 , w2 , b1 , b2
    
    def sigmoid_(self,z):
        return 1.0/(1.0+np.exp(-z))
    
    def feed_forward(self , X , w1 , w2):
        a1 = X
        z2 = np.dot(w1,a1) + self.b1
        a2 = self.sigmoid_(z2)
        z3 = np.dot(w2 , a2) +self.b2
        a3 = self.sigmoid_(z3)
        self.output.append(a3)
        return a1 , z2 , a2 , z3  , a3
        
    def L2_reg(self , w1 , w2 , lb):
        s1 = np.square(w1) 
        s2 = np.square(w2)
        return (np.sum(s1)+ np.sum(s2)) * lb /(2 * m) 
    
    def get_cost(self , y ,a3 ,w1 ,w2):
        c1 = - (y * np.log(a3))
        c2 = -(1-y)*np.log(1-a3)
        c = (np.sum(c1+c2)) / m
        #reg = self.L2_reg(w1 ,w2 , self.lb)
        return c 
    
    def sigmoid_grad(self , z):
        a = self.sigmoid_(z)
        return a * (1-a)
    
    def gradient(self , a1 , a2 , a3 , z2 , y , w1 , w2):
        del3 = a3 - y
        del2 = np.dot(w2.T , del3) * self.sigmoid_grad(z2)
        grad_w2 = np.dot(del3 , a2.T) / m
        grad_w1 = np.dot(del2 , a1.T) / m
        
        grad_b2 = np.sum(del3 , axis=1 , keepdims=True) / m
        grad_b1 = np.sum(del2 , axis=1 ,keepdims=True) / m
        #grad1[:,1:] += (w1[:,1:] * self.lb)
        #grad2[:,1:] += (w2[:,1:] * self.lb)
        
        return grad_w1 , grad_w2 , grad_b1 , grad_b2
        
    def predict(self , X):  
        
        X = X.T
        a1 , z2 , a2 , z3 , a3 = self.feed_forward(X=X,w1=self.w1,w2=self.w2)
        a3=np.squeeze(a3)
        for i in range(len(a3)):
            if(a3[i] >= 0.5):
                y_pred_val = 1
            else:
                y_pred_val = 0      
            y_pred.append(y_pred_val)
            
        return y_pred
    
    def fit(self,X,y):  
        X = X.T
        y.reshape(y.shape[0],1)
        y = y.T
        self.cost_=[] 
        for i in range(self.epochs):
            print("in epoch " + str(i) + '\n')
            a1,z2,a2,z3,a3=self.feed_forward(X=X,w1=self.w1,w2=self.w2)
            cost = self.get_cost(y=y,a3=a3,w1=self.w1,w2=self.w2 )
            self.cost_.append(cost)
            gradw1,gradw2,gradb1,gradb2 = self.gradient(a1,a2,a3,z2,y,self.w1,self.w2)
            del_w1 , del_w2 , del_b1 ,del_b2 = self.eta * gradw1 , self.eta *gradw2 ,self.eta *gradb1 , self.eta *gradb2
            self.w1 -= del_w1
            self.w2 -= del_w2
            self.b1 -= del_b1
            self.b2 -= del_b2
            print('cost after epoch ' + str(i) + ' ' + str(cost))
        return self
    

classifier = ANN(n_output = 1, n_features =11 ,no_hidden = 28)

y_pred=[]
m = X_train.shape[0]
classifier.fit(X_train,y_train)   
y_pred = classifier.predict(X_test) 
     
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)


        
        
        

        
        
        






