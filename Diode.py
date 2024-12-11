import os
import time
import argparse


from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import mean_squared_error
from pyrcn.echo_state_network import ESNRegressor


parser = argparse.ArgumentParser()
parser.add_argument('-epochs', type=int, default=20)
parser.add_argument('-interpolations', type=int, default=2)
parser.add_argument('-reservoir_size', type=int, default=500)
parser.add_argument('-sparsity', type=float, default=0.5)
parser.add_argument('-savepath', type=str, default="res/Optimized.dat")
parser.add_argument('-plots', type=bool, default=True)
parser.add_argument('-train', type=bool, default=False)
opt = parser.parse_args()

inter = opt.interpolations+1 #interpolation amount
epochs = opt.epochs #number of epochs
r_size = opt.reservoir_size
spars = opt.sparsity
datafile_path = opt.savepath
plot = opt.plots
train = opt.train

length=201 #datapoints per training example

if (train==True):

    dataArrayx = [] 
    dataArrayy = [] 
    
    files= 0
    for filename in os.listdir("L_06/Train"): #Read training data
            fname = "L_06/Train/" + filename
            with open(fname) as f:
                files+=1
                tempArrayx = [] 
                tempArrayy = [] 
                for line in f: 
                        temp = [float(x) for x in line.split()]
                        xvals =[temp[0],temp[1]]
                        yvals = [temp[2]]
                        tempArrayx.append(xvals)
                        tempArrayy.append(yvals)
                dataArrayx.append(tempArrayx)
                dataArrayy.append(tempArrayy)
            
    
    
    dataArrayx = np.array(dataArrayx)
    dataArrayy = np.array(dataArrayy)
    
    
    
        
    newtrainy = [] #Interpolate new values
    newtrainx = []
    for i in range(0,files):
        interp_func1 = interp1d(dataArrayx[i,:,0] , dataArrayy[i,:,0],bounds_error=False,  fill_value="extrapolate" )
        interp_func3 = interp1d(dataArrayx[i,:,0] , dataArrayx[i,:,1],bounds_error=False,  fill_value="extrapolate" )
        temparr = []
        temparrx = []
        for l in range(0,len(dataArrayy[i])):
            newarr = []
            newarr3 = []
            temptrainx = []
            if(l+1<len(dataArrayy[i])):
                if(30<l<50 or 160<l<180):   #Interpolate only in the relevant areas
                    newarr = interp_func1(np.arange(dataArrayx[i,l,0], dataArrayx[i,l+1,0]-0.0000000000000001,(1/inter)*(dataArrayx[0,1,0] - dataArrayx[0,0,0]))) 
                    newarr3 = interp_func3(np.arange(dataArrayx[i,l,0], dataArrayx[i,l+1,0]-0.0000000000000001,(1/inter)*(dataArrayx[0,1,0] - dataArrayx[0,0,0]))) 
                    temptrainx= np.vstack((np.arange(dataArrayx[i,l,0], dataArrayx[i,l+1,0]-0.0000000000000001,(1/inter)*(dataArrayx[0,1,0] - dataArrayx[0,0,0])), newarr3)).T
                else:
                    newarr.append(dataArrayy[i,l,0])
                    newarr3.append(dataArrayx[i,l,1])
                    temptrainx= np.vstack((dataArrayx[i,l,0], newarr3)).T
            else:
                newarr.append(dataArrayy[i,l,0])
                newarr3.append(dataArrayx[i,l,1])
                temptrainx= np.vstack((dataArrayx[i,l,0], newarr3)).T
    
            for k in range(0,len(newarr)):
                temparr.append([newarr[k]])
                temparrx.append([temptrainx[k,0],temptrainx[k,1]])
        newtrainy.append(temparr)
        newtrainx.append(temparrx)
    
    plt.figure(0) #Plot figure and compute MSE
    
    
    
    
    
    dataArrayx2 = [] 
    dataArrayy2 = [] 
    
    files= 0
    for filename in os.listdir("L_06/Test"): #Read testing data
            fname = "L_06/Test/" + filename
            with open(fname) as f:
                files+=1
                tempArrayx = [] 
                tempArrayy = [] 
                for line in f: # read rest of lines
                        temp = [float(x) for x in line.split()]
                        xvals =[temp[0],temp[1]]
                        yvals = [temp[2]]
                        tempArrayx.append(xvals)
                        tempArrayy.append(yvals)
                dataArrayx2.append(tempArrayx)
                dataArrayy2.append(tempArrayy)
            
    
    
    dataArrayx2 = np.array(dataArrayx2)
    dataArrayy2 = np.array(dataArrayy2)
    
    
    
    
        
    
    
    newtesty = [] #Interpolate new values
    newtestx = []
    for i in range(0,files):
        interp_func1 = interp1d(dataArrayx2[i,0:length,0] , dataArrayy2[i,0:length,0],bounds_error=False,  fill_value="extrapolate" )
        interp_func3 = interp1d(dataArrayx2[i,0:length,0] , dataArrayx2[i,0:length,1],bounds_error=False,  fill_value="extrapolate" )
        temparr = []
        temparrx = []
        for l in range(0,len(dataArrayy[i])):
            newarr = []
            newarr3 = []
            temptrainx = []
            if(l+1<len(dataArrayy[i])):
                if(30<l<50 or 160<l<180):   #Interpolate only in the relevant areas
                    newarr = interp_func1(np.arange(dataArrayx2[i,l,0], dataArrayx2[i,l+1,0]-0.0000000000000001,(1/inter)*(dataArrayx2[0,1,0] - dataArrayx2[0,0,0])))     
                    newarr3 = interp_func3(np.arange(dataArrayx2[i,l,0], dataArrayx2[i,l+1,0]-0.0000000000000001,(1/inter)*(dataArrayx2[0,1,0] - dataArrayx2[0,0,0]))) 
                    temptrainx= np.vstack((np.arange(dataArrayx2[i,l,0], dataArrayx2[i,l+1,0]-0.0000000000000001,(1/inter)*(dataArrayx2[0,1,0] - dataArrayx2[0,0,0])), newarr3)).T
                else:
                    newarr.append(dataArrayy2[i,l,0])
                    newarr3.append(dataArrayx2[i,l,1])
                    temptrainx= np.vstack((dataArrayx2[i,l,0], newarr3)).T
            else:
                newarr.append(dataArrayy2[i,l,0])
                newarr3.append(dataArrayx2[i,l,1])
                temptrainx= np.vstack((dataArrayx2[i,l,0], newarr3)).T
    
            for k in range(0,len(newarr)):
                temparr.append([newarr[k]])
                temparrx.append([temptrainx[k,0],temptrainx[k,1]])
        newtesty.append(temparr)
        newtestx.append(temparrx)
    
    
    plt.plot(np.array(newtestx[0])[:,0],np.array(newtesty[0])[:,0], color='red')
    plt.legend(["Ground truth"], loc ="lower left") 
     
    k=0
    
    
    initArrx = []
    initArry = []
    for i in range(0,len(newtrainx)): #Generate initial point model data
        initArrx.append(newtrainx[i][0:10])
        initArry.append(newtrainy[i][0:10])
        
    initArrx = np.array(initArrx)
    initArry = np.array(initArry)
    
    
    reg = ESNRegressor(spectral_radius = 0.99, sparsity = spars, hidden_layer_size = r_size, feedback = True) #Main model
    reg2 = ESNRegressor(spectral_radius = 0.99, sparsity = spars, hidden_layer_size = 200, feedback = True) #Initial point model
    
    start = time.time()
    
    
    
    for k in range(0,epochs):
        print("Epoch: " + str(k+1))
        for i in range(0,len(newtrainy)): #train main model
            xdata = np.array(newtrainx[i][:])
            ydata= np.array(newtrainy[i])[:]
            reg.fit(X=xdata, y=ydata)
    
    
    for k in range(0,epochs):  
        for i in range(0,len(initArrx)): #train initial point model
            reg2.fit(X=initArrx[i,:], y=initArry[i,:])
    
    
    print(ESNRegressor)
    end = time.time()
        
    
    
    y_pred = reg.predict(np.array(newtestx[0][:]))  #Make predictions
    
    y_pred2 = reg2.predict(np.array(newtestx[0][0:30]))
    
    for i in range(0,30): #Replace initial point preeditions
            y_pred[i]=y_pred2[i]
    
    
    #Plot figure and compute MSE
    
    mse = mean_squared_error(np.array(newtesty[0])[:,0],y_pred[:]) 
    print("MSE for dataset "": " + str(mse) + " Training time:" + str(end - start) + " seconds")
    
    
    fileprint = []
    
    k=0
    for i in y_pred:
        num = y_pred[k]
        num2= num[0]
        fileprint.append([np.array(newtestx[0])[k,0],num2])
        k+=1
    

    np.savetxt(datafile_path , fileprint, fmt=['%10.7f','%10.7f'])
        


if(plot == True):

    plotArray = []
    fname = "res/true.dat"
    with open(fname) as f:
        tempArray = [] 
        for line in f: 
                temp = [float(x) for x in line.split()]
                xvals =[temp[0],temp[2]]
                tempArray.append(xvals)

        
    plotArray = np.array(tempArray)
    
    xt = plotArray[:,0]   # position [um]
    rhot = plotArray[:,1] # electron density [um^(-3)]
    
    plotArray = []
    fname = "res/GRU.dat"
    with open(fname) as f:
        tempArray = [] 
        for line in f: 
                temp = [float(x) for x in line.split()]
                xvals =[temp[0],temp[1]]
                tempArray.append(xvals)

    
    plotArray = np.array(tempArray)
    
    xGRU = plotArray[:,0]   #position [um]
    rhoGRU = plotArray[:,1] # electron density [um^(-3)]
    
    # ESN model, not optimized
    plotArray = []
    fname = "res/Unoptimized.dat"
    with open(fname) as f:
        tempArray = [] 
        for line in f: 
                temp = [float(x) for x in line.split()]
                xvals =[temp[0],temp[1]]#,temp[2]
                tempArray.append(xvals)

    
    plotArray = np.array(tempArray)
    
    xESNu = plotArray[:,0]   # position [um]
    rhoESNu = plotArray[:,1] # electron density [um^(-3)]
    
    # ESN model, optimized
    plotArray = []
    fname = "res/Optimized.dat"
    with open(fname) as f:
        tempArray = [] 
        for line in f: 
                temp = [float(x) for x in line.split()]
                xvals =[temp[0],temp[1]]#,temp[2]
                tempArray.append(xvals)

    
    plotArray = np.array(tempArray)
    

    
    xESNo = plotArray[:,0]   # position [um]
    rhoESNo = plotArray[:,1] #electron density [um^(-3)]
    
    
    plt.figure(0)
    plt.plot(xt,rhot, linewidth=2, color = (1,0,0))
    plt.plot(xGRU,rhoGRU, linestyle = 'dashed', linewidth = 2, color= (0,0.7,0))
    plt.plot(xESNu,rhoESNu, linestyle = 'dashed',  linewidth=2,  color=(1,0.6,0))
    plt.plot(xESNo,rhoESNo, linestyle = 'dashed', linewidth=2, color = (0,0,1))

    plt.legend(['Ground-truth', 'Predicted - GRU',
      'Predicted - ESN unoptimized', 'Predicted - ESN optimized'], loc = "upper center")
    plt.xlim([-0.02,0.62])
    plt.ylim([-0.02e6,1.1e6])

    plt.xlabel('x [\mum]');
    plt.ylabel('Electron density [\mum^{-3}]');


    



    ############## 
