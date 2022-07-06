import numpy as np
from numpy import loadtxt
from statistics import stdev
import matplotlib.pyplot as plt
 

# import training data into the BigX matrix
BigX=loadtxt("process_data_training.txt")
# get number of variables and measurements in the data matrix
m=len(BigX[0])
n=len(BigX)
# prepare the shell of the auto scaled data
AutoX=np.zeros((n,m))

# calculate the mean and standard dev. for all columns (measurements) considering all rows
# notice that the for loop runs to cover the different columns
for i in range(m):
    Mi=np.mean(BigX[:,i])
    STDi=stdev(BigX[:,i])
    # As we calculate the mean and standard dev. we do the autoscaling for each column
    Ci=(BigX[:,i]-Mi)/STDi
    # We stack autoscaled columns next to each other
    # to form the full autoscaled training matrix for PCA
    AutoX[:,i]=Ci

# calculate the covariance matrix of the autoscaled training data
CovX=AutoX.transpose().dot(AutoX)/(len(AutoX)-1)
# do the singular value decomposition (SVD) on the covariance matrix
u, s, vh = np.linalg.svd(CovX, full_matrices=True)
# columns of the u matrix from the SVD are the principal components
# we decide to use 3 principal components to model the data
PC=u[:,0:4]

# import test data into the TestX matrix
TestX=loadtxt("process_data_testing.txt")
#TestX=loadtxt("process_data_training.txt")
# prepare the shell of the auto scaled test data
m=len(TestX[0])
n=len(TestX)
Test_AutoX=np.zeros((n,m))

# note that we need to scale the test data according to the scaling of training data!
for j in range(m):
    Mi_t=np.mean(BigX[:,j])
    STDi_t=stdev(BigX[:,j])
    Ci_t=(TestX[:,j]-Mi_t)/STDi_t
    Test_AutoX[:,j]=Ci_t

# Construct the projection matrix for Q values
Q_Projection=np.identity(m)-PC.dot(PC.transpose())

# Calculate Q values for every sample in the test data

n=len(TestX)
Q=np.zeros(n)
for i in range(n):
    Q[i]=Test_AutoX[i,:].dot(Q_Projection.dot(Test_AutoX[i,:].transpose()));
    
plt.plot(Q)
plt.show()


