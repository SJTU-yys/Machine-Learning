# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 19:17:54 2015

@author: Administrator
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import random

class SVM:
    def GaussianKernel(x1, x2, sigma = 5):
        return math.exp(-np.sum((x1-x2)**2)/(2*sigma**2))
        
    def MultinomialKernel(x1, x2, R, d):
        return (x1*x2+R)**d
        
    def LinearKernel(x1,x2):
        return np.sum(x1*x2)
        
    def __init__(self, x, y, C = 10, toler = 0.01, kernel=LinearKernel):
        #self.w = w
        self.toler = toler        # termination condition for iteration 
        self.b = 0                # intercept b
        self.C = C                # slack variable 
        self.kernel = kernel      # kernel function
        self.x = x                # training set
        self.y = y                # training labels
        self.sp_size = x.shape[1] # the size of training set
        self.E = [0 for i in range(x.shape[1])] # Error Cache
        # Most value of alpha[i] equal to zero because only the alpha values
        # corresponding to those support vectors don't equal to zero, so there
        # is no need to store the value of w
        self.alpha = np.zeros((1,x.shape[1])).flatten(1)
        self.supportVector = []
        #self.u = w*x+b
    
    def fitKKT(self,i):
        if ((self.y[i]*self.E[i]<-self.toler) and (self.alpha[i]<self.C)) or \
        (((self.y[i]*self.E[i]>self.toler)) and (self.alpha[i]>0)):
            return False
        return True
        
    def getE(self, i):
        self.E[i] = 0
        for t in range(self.sp_size):
        #for t in range(self.M):
            self.E[i]+=self.alpha[t]*self.y[t]*self.kernel(self.x[:,i],self.x[:,t])
            self.E[i]+=self.b-self.y[i]
        return self.E[i]
    
    def updateE(self, i):
        self.E[i] = 0
        for t in range(self.sp_size):
        #for t in range(self.M):
            self.E[i]+=self.alpha[t]*self.y[t]*self.kernel(self.x[:,i],self.x[:,t])
            self.E[i]+=self.b-self.y[i]
    
    def findJ(self,i):
        support_index = np.nonzero(self.alpha > 0)[0]
        E_i = self.getE(i)
        max_diff = -1
        max_j = -1
        # find the alpha with max iterative step  
        if len(support_index) >= 1:
            for j in support_index:
                if j == i: continue
                t = abs(E_i - self.getE(j))
                if t > max_diff:
                    max_diff = t
                    max_j = j
            if max_j == -1:  
                max_j = random.sample(range(self.sp_size), 1)[0]
                while max_j == i:  
                    max_j = random.sample(range(self.sp_size), 1)[0]
        # for the first iteration, all alphas are zero, select j randomly
        else:
            for j in range(self.sp_size):
                if j == i: continue
                t = abs(E_i - self.getE(j))
                if t > max_diff:
                    max_diff = t
                    max_j = j
            if max_j == -1: 
                max_j = random.sample(range(self.sp_size), 1)[0]
                while max_j == i:  
                    max_j = random.sample(range(self.sp_size), 1)[0]
        return max_j
        
    def update_pair(self, i, j, kernel, threshold):
        old_alpha1 = self.alpha[i]
        old_alpha2 = self.alpha[j]
        # Calculate the upper bound and lower bound
        if self.y[i] != self.y[j]:
            L = max(0, old_alpha2 - old_alpha1)
            H = min(self.C, self.C + old_alpha2 - old_alpha1)
        else:
            L = max(0, old_alpha2 + old_alpha1 - self.C)
            H = min(self.C, old_alpha2 + old_alpha1)
         
        # Calculate eta
        Kij = kernel(self.x[:,i], self.x[:,j])
        Kii = kernel(self.x[:,i],self.x[:,i])
        Kjj = kernel(self.x[:,j], self.x[:,j])
        eta = 2*Kij - Kii - Kjj
        
        if eta>=0:  
            return 1
        
        # Update Ls
        new_alpha2 = old_alpha2 - self.y[j] * (self.getE(i) - self.getE(j))/eta
        #print "new_alpha2:",new_alpha2
        # Clip alpha2
        if new_alpha2 >= H:
            new_alpha2 = H
        elif new_alpha2 <= L:
            new_alpha2 = L
        
        # If alpha2 not moving enough, just return 
        if np.abs(new_alpha2-old_alpha2)<threshold:  
            #print np.abs(a2_new-self.alpha[j])  
            return 1
        
        # Update alpha1
        new_alpha1 = old_alpha1 + self.y[i] * self.y[j]*(old_alpha2 - new_alpha2)
        self.alpha[i] = new_alpha1
        self.alpha[j] = new_alpha2
        #print "a1,a2:", self.alpha[i],self.alpha[j]
        # Update b
        new_b1=self.b-self.E[i]-self.y[i]*Kii*(self.alpha[i]-old_alpha1)-self.y[j]*Kij*\
        (self.alpha[j]-old_alpha2)  
                
        new_b2=self.b-self.E[j]-self.y[i]*Kij*(self.alpha[i]-old_alpha1)-self.y[j]*Kjj*\
        (self.alpha[j]-old_alpha2)
        
        if new_alpha1 > 0 and new_alpha1 < self.C:
            self.b = new_b1
        elif new_alpha2 > 0 and new_alpha2 < self.C:
            self.b = new_b2
        else:
            self.b = (new_b1 + new_b2)/2
            
        self.updateE(j)  
        self.updateE(i)  
        return 0
        
            
    def train(self, max_iter = 10000, threshold=0.000001):
        iters=0  
        flag=False  
        for i in range(self.sp_size):  
            self.updateE(i)  
        while (iters<max_iter) and (not flag):  
            flag=True  
            support_index=np.nonzero((self.alpha>0))[0] 
            print "temp_supportVec:",support_index
            iters+=1  
            for i in support_index:  
                self.updateE(i)  
                if not self.fitKKT(i):  
                    j = self.findJ(i)
                    print i,j
                    flag=flag and self.update_pair(i, j, self.kernel, threshold) 
                #if not flag:break  
            if (flag):  
                for i in range(self.sp_size):  
                    self.updateE(i)  
                    if (not self.fitKKT(i)):  
                        print "aaaaaaa"
                        j = self.findJ(i)
                        print i,j
                        flag = flag and self.update_pair(i, j, self.kernel, threshold)  
                        if not flag:break        
            print "the %d-th iter is running" % iters  
        self.supportVector=np.nonzero((self.alpha>0))[0] 
              
    def predict(self, x):
        w=0  
        for t in self.supportVector:  
            w+=self.alpha[t]*self.y[t]*self.kernel(self.x[:,t],x).flatten(1)  
        w+=self.b  
        return sign(w)
        
    def pred(self, x):
        test_X=np.array(x)  
        y=[]  
        for i in range(test_X.shape[1]):  
            y.append(self.predict(test_X[:,i]))  
        return y  
        
    def error(self, x, y):
        py=np.array(self.pred(np.array(x))).flatten(1)  
        #print y,py  
        print "the #error_case is  ",np.sum(py!=np.array(y))  
        
        
    def prints_test_linear(self):  
        w=0 
        for t in self.supportVector:  
            print "support"
            w+=self.alpha[t]*self.y[t]*self.x[:,t].flatten(1)  
        w=w.reshape(1,w.size)  
        #print np.sum(sigmoid(np.dot(w,self.x)+self.b).flatten(1)!=self.y),"errrr"  
        #print w,self.b  
        x1=0  
        y1=-self.b/w[0][1]  
        y2=0  
        x2=-self.b/w[0][0]  
        plt.plot([x1+x1-x2,x2],[y1+y1-y2,y2])  
        #plt.plot([x1+x1-x2,x2],[y1+y1-y2-1,y2-1])  
        plt.axis([0,30,0,30])  
  
        for i in range(self.x.shape[1]):  
            if  self.y[i]==-1:  
                plt.plot(self.x[0,i],self.x[1,i],'or')  
            elif  self.y[i]==1:  
                plt.plot(self.x[0,i],self.x[1,i],'ob')  
        for i in self.supportVector:  
            plt.plot(self.x[0,i],self.x[1,i],'oy')  
        plt.show()  
        
def sign(x):
	#print x,'========='
	#print "======================="
	q=np.zeros(np.array(x).shape)
	q[x>=0]=1
	q[x<0]=-1
	return q