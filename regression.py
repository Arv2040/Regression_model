import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy as cp
import math

data = pd.read_csv("C:\\Users\\A J\\Desktop\\archive (1)\\Student_Performance.csv")
hours = float(input("Enter hours"))
prev = float(input("Enter the amount of papers solved"))
sleep = float(input("Enter the hours of sleep"))
sampleQ = float(input("Enter the number of sample questions"))

x_train = np.array(data.iloc[:,0:-1:])
y_train = np.array(data.iloc[:,4])
m = x_train.shape[0]

# predict function to predict the values

def predict(x,w,b):
    f = np.dot(x,w) +b
    return f


# to calculate cost

def cost(x,y,w,b):
    cost = 0
    for i in range(m):
        y_hat = (np.dot(x[i],w) + b)
        cost += (y_hat - y[i])**2
    cost = cost/(2*m)
    return cost

def derivative(x,y,w,b):
    m,n = x_train.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.
    for i in range(m):
        err = (np.dot(x[i],w) + b)-y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] +  err * x[i,j]
        dj_db += err

    dj_dw = dj_dw/m
    dj_db = dj_db/m
    return dj_dw,dj_db

def gradient_descent(w_in,b_in,x,y,derivative,alpha,num):
    w = cp.deepcopy(w_in)
    b = b_in
    for i in range(num):
        dj_dw,dj_db = derivative(x,y,w,b)
        
        w = w - (alpha*dj_dw)
        b = b - (alpha*dj_db)


    return w,b

initial_w = [0.,0.,0.,0.]
initial_b = 0
alpha = 0.00001
num = 1500

w,b = gradient_descent(initial_w,initial_b,x_train,y_train,derivative,alpha,num)


y_hat = predict([hours,	prev,	sleep,	sampleQ],w,b)
print(y_hat)



