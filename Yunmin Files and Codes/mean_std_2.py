import numpy as np
from numpy import loadtxt
from statistics import stdev

Data1=loadtxt("stripper_data_1b.txt")
Data2=loadtxt("stripper_data_2b.txt")
Data3=loadtxt("stripper_data_3b.txt")
Data4=loadtxt("stripper_data_4b.txt")
Data5=loadtxt("stripper_data_5b.txt")
Data6=loadtxt("stripper_data_6b.txt")
Data7=loadtxt("stripper_data_7b.txt")
Data8=loadtxt("stripper_data_8b.txt")
Data9=loadtxt("stripper_data_9b.txt")
Data10=loadtxt("stripper_data_10b.txt")
Data11=loadtxt("stripper_data_11b.txt")
Data12=loadtxt("stripper_data_12b.txt")
Data13=loadtxt("stripper_data_13b.txt")

m=len(Data1[0]) # number of variables, 0th row

MTotal = np.zeros((m, 13))
STDTotal = np.zeros((m, 13))

for i in range(13):
    if i==0:
        Data=Data1
    elif i==1:
        Data=Data2
    elif i==2:
        Data=Data3
    elif i==3:
        Data=Data4
    elif i==4:
        Data=Data5
    elif i==5:
        Data=Data6
    elif i==6:
        Data=Data7
    elif i==7:
        Data=Data8
    elif i==8:
        Data=Data9
    elif i==9:
        Data=Data10
    elif i==10:
        Data=Data11
    elif i==11:
        Data=Data12
    elif i==12:
        Data=Data13

    for j in range(m):
        Mi=np.mean(Data[:,j]) # mean
        STDi=stdev(Data[:,j]) # std. dev.
        MTotal[j,i] = Mi
        STDTotal[j,i] = STDi