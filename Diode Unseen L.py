import os
import time

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import mean_squared_error


from pyrcn.echo_state_network import ESNRegressor




inter = 1 #interpolation amount
length=201 #datapoints per training example
usecurrent = 0 #Wether to use the current as an added output
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
                    xvals =[temp[0],temp[1],0.6]
                    yvals = [temp[2],temp[3]]
                    tempArrayx.append(xvals)
                    tempArrayy.append(yvals)
            dataArrayx.append(tempArrayx)
            dataArrayy.append(tempArrayy)
        

for filename in os.listdir("L_04"): #Read training data
        fname = "L_04/" + filename
        with open(fname) as f:
            files+=1
            tempArrayx = [] 
            tempArrayy = [] 
            for line in f: 
                    temp = [float(x) for x in line.split()]
                    xvals =[temp[0],temp[1],0.4]
                    yvals = [temp[2],temp[3]]
                    tempArrayx.append(xvals)
                    tempArrayy.append(yvals)
            dataArrayx.append(tempArrayx)
            dataArrayy.append(tempArrayy)


for filename in os.listdir("L_1"): #Read training data
        fname = "L_1/" + filename
        with open(fname) as f:
            files+=1
            tempArrayx = [] 
            tempArrayy = [] 
            for line in f: 
                    temp = [float(x) for x in line.split()]
                    xvals =[temp[0],temp[1],1]
                    yvals = [temp[2],temp[3]]
                    tempArrayx.append(xvals)
                    tempArrayy.append(yvals)
            dataArrayx.append(tempArrayx)
            dataArrayy.append(tempArrayy)


dataArrayx = np.array(dataArrayx)
dataArrayy = np.array(dataArrayy)





    
newtrainy = [] #Interpolate new values
newtrainx = []
for i in range(0,files):

    lval = dataArrayx[i,0,2]
    interp_func1 = interp1d(dataArrayx[i,0:length,0] , dataArrayy[i,0:length,0],bounds_error=False,  fill_value="extrapolate" )
    interp_func2 = interp1d(dataArrayx[i,0:length,0] , dataArrayy[i,0:length,1],bounds_error=False,  fill_value="extrapolate" )
    interp_func3 = interp1d(dataArrayx[i,0:length,0] , dataArrayx[i,0:length,1],bounds_error=False,  fill_value="extrapolate" )
    temparr = []
    temparrx = []
    for l in range(0,len(dataArrayy[i])):
        newarr = []
        newarr2 = []
        newarr3 = []
        temptrainx = []
        if(l+1<len(dataArrayy[i])):
            if(30<l<50 or 160<l<180):   #Interpolate only in the relevant areas
                newarr = interp_func1(np.arange(dataArrayx[i,l,0], dataArrayx[i,l+1,0]-0.000000000001,(1/inter)*(dataArrayx[0,1,0] - dataArrayx[0,0,0]))) 
                newarr2 = interp_func2(np.arange(dataArrayx[i,l,0], dataArrayx[i,l+1,0]-0.000000000001,(1/inter)*(dataArrayx[0,1,0] - dataArrayx[0,0,0])))  
                newarr3 = interp_func3(np.arange(dataArrayx[i,l,0], dataArrayx[i,l+1,0]-0.000000000001,(1/inter)*(dataArrayx[0,1,0] - dataArrayx[0,0,0]))) 
                temptrainx= np.vstack((np.arange(dataArrayx[i,l,0], dataArrayx[i,l+1,0]-0.000000000001,(1/inter)*(dataArrayx[0,1,0] - dataArrayx[0,0,0])), newarr3)).T
            else:
                newarr.append(dataArrayy[i,l,0])
                newarr2.append(dataArrayy[i,l,1])
                newarr3.append(dataArrayx[i,l,1])
                temptrainx= np.vstack((dataArrayx[i,l,0], newarr3)).T
        else:
            newarr.append(dataArrayy[i,l,0])
            newarr2.append(dataArrayy[i,l,1])
            newarr3.append(dataArrayx[i,l,1])
            temptrainx= np.vstack((dataArrayx[i,l,0], newarr3)).T

        for k in range(0,len(newarr)):
            temparr.append([newarr[k],newarr2[k]])
            temparrx.append([temptrainx[k,0],temptrainx[k,1],lval])
    newtrainy.append(temparr)
    newtrainx.append(temparrx)




dataArrayx2 = [] 
dataArrayy2 = [] 

files= 0
for filename in os.listdir("L_08/Test"): #Read testing data
        fname = "L_08/Test/" + filename
        with open(fname) as f:
            files+=1
            tempArrayx = [] 
            tempArrayy = [] 
            for line in f: # read rest of lines
                    temp = [float(x) for x in line.split()]
                    xvals =[temp[0],temp[1],0.8]
                    yvals = [temp[2],temp[3]]
                    tempArrayx.append(xvals)
                    tempArrayy.append(yvals)
            dataArrayx2.append(tempArrayx)
            dataArrayy2.append(tempArrayy)


dataArrayx2 = np.array(dataArrayx2)
dataArrayy2 = np.array(dataArrayy2)






newtesty = [] #Interpolate new values
newtestx = []
for i in range(0,files):
    lval = dataArrayx2[i,0,2]
    interp_func1 = interp1d(dataArrayx2[i,0:length,0] , dataArrayy2[i,0:length,0],bounds_error=False,  fill_value="extrapolate" )
    interp_func2 = interp1d(dataArrayx2[i,0:length,0] , dataArrayy2[i,0:length,1],bounds_error=False,  fill_value="extrapolate" )
    interp_func3 = interp1d(dataArrayx2[i,0:length,0] , dataArrayx2[i,0:length,1],bounds_error=False,  fill_value="extrapolate" )
    temparr = []
    temparrx = []
    for l in range(0,len(dataArrayy[i])):
        newarr = []
        newarr2 = []
        newarr3 = []
        temptrainx = []
        if(l+1<len(dataArrayy[i])):
            if(30<l<50 or 160<l<180):   #Interpolate only in the relevant areas
                newarr = interp_func1(np.arange(dataArrayx2[i,l,0], dataArrayx2[i,l+1,0]-0.000000000001,(1/inter)*(dataArrayx2[0,1,0] - dataArrayx2[0,0,0])))     
                newarr2 = interp_func2(np.arange(dataArrayx2[i,l,0], dataArrayx2[i,l+1,0]-0.000000000001,(1/inter)*(dataArrayx2[0,1,0] - dataArrayx2[0,0,0])))  
                newarr3 = interp_func3(np.arange(dataArrayx2[i,l,0], dataArrayx2[i,l+1,0]-0.000000000001,(1/inter)*(dataArrayx2[0,1,0] - dataArrayx2[0,0,0]))) 
                temptrainx= np.vstack((np.arange(dataArrayx2[i,l,0], dataArrayx2[i,l+1,0]-0.000000000001,(1/inter)*(dataArrayx2[0,1,0] - dataArrayx2[0,0,0])), newarr3)).T
            else:
                newarr.append(dataArrayy2[i,l,0])
                newarr2.append(dataArrayy2[i,l,1])
                newarr3.append(dataArrayx2[i,l,1])
                temptrainx= np.vstack((dataArrayx2[i,l,0], newarr3)).T
        else:
            newarr.append(dataArrayy2[i,l,0])
            newarr2.append(dataArrayy2[i,l,1])
            newarr3.append(dataArrayx2[i,l,1])
            temptrainx= np.vstack((dataArrayx2[i,l,0], newarr3)).T

        for k in range(0,len(newarr)):
            temparr.append([newarr[k],newarr2[k]])
            temparrx.append([temptrainx[k,0],temptrainx[k,1],lval])
    newtesty.append(temparr)
    newtestx.append(temparrx)




initArrx = []
initArry = []
for i in range(0,len(newtrainx)): #Generate initial point model data
    initArrx.append(newtrainx[i][0:30])
    initArry.append(newtrainy[i][0:30])
    
initArrx = np.array(initArrx)
initArry = np.array(initArry)


reg = ESNRegressor(spectral_radius = 0.99, sparsity = 0.3, n_reservoir= 3500, hidden_size=5) #Main model
reg2 = ESNRegressor(spectral_radius = 0.99, sparsity = 0.3, n_reservoir= 200) #Initial point model

start = time.time()


if usecurrent == 0:
    for i in range(0,len(newtrainy)): #train main model
        xdata = np.array(newtrainx[i][:])
        ydata= np.array(newtrainy[i])[:,0]
        reg.fit(X=xdata, y=ydata)
        print (i)

    
    for i in range(0,len(initArrx)): #train initial point model
        reg2.fit(X=initArrx[i,:], y=initArry[i,:,0])
    
else:
    for i in range(0,len(newtrainy)): #train main model
        xdata = np.array(newtrainx[i][:])
        ydata= np.array(newtrainy[i])[:]
        reg.fit(X=xdata, y=ydata)
        print (i)
        


    
    for i in range(0,len(initArrx)): #train initial point model
        reg2.fit(X=initArrx[i,:], y=initArry[i,:])
    



end = time.time()
    


y_pred = reg.predict(np.array(newtestx[0][:]))  #Make predictions

y_pred2 = reg2.predict(np.array(newtestx[0][0:30]))

for i in range(0,30): #Replace initial point preeditions
    if usecurrent == 0:
        y_pred[i]=y_pred2[i]
    else:
        y_pred[i,0]=y_pred2[i,0]

        
plt.figure(0) #Plot figure and compute MSE

if usecurrent == 0:
    plt.plot(np.array(newtestx[0])[:,0],y_pred[:], color = 'blue' ) 
    plt.plot(np.array(newtestx[0])[:,0],np.array(newtesty[0])[:,0], color='red')
    plt.legend(["ESN prediction","Ground truth"], loc ="upper center") 
 
    mse = mean_squared_error(np.array(newtesty[0])[:,0],y_pred[:]) 

else:
    plt.plot(np.array(newtestx[0])[:,0],y_pred[:,0], color = 'blue' ) 
    plt.plot(np.array(newtestx[0])[:,0],np.array(newtesty[0])[:,0], color='red')
    plt.legend(["ESN prediction","Ground truth"], loc ="upper center") 
 
    mse = mean_squared_error(np.array(newtesty[0])[:,0],y_pred[:,0])        
        


print("MSE for dataset "": " + str(mse) + " Training time: " + str(end - start) + " seconds")

datafile_path = "res/" + "Optimized.dat" #Save values in .dat file

if usecurrent == 0:
    np.savetxt(datafile_path , y_pred, fmt=['%10.7f'])
else:
    np.savetxt(datafile_path , y_pred, fmt=['%10.7f','%10.7f'])


    ############## 
