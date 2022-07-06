# import library?
import numpy as np
from numpy import loadtxt # has to be loaded separately?
from statistics import stdev # np std divide by n, not n-1
import matplotlib.pyplot as plt


# import training data into the BigX matrix
BigX=loadtxt("stripper_data_3b.txt")
# get number of variables and measurements in the data matrix
m=len(BigX[0]) # number of variables, 0th row
n=len(BigX) # number of measurements
# prepare the shell of the auto scaled data
AutoX=np.zeros((n,m)) # zero matrix n * m

# calculate the mean and standard dev. for all columns (measurements) considering all rows
# notice that the for loop runs to cover the different columns
# for var in range() loop
for i in range(m): # loop for m (number of vars) times
    Mi=np.mean(BigX[:,i]) # mean
    STDi=stdev(BigX[:,i]) # std. dev.

    # As we calculate the mean and standard dev. we do the autoscaling for each column
    Ci=(BigX[:,i]-Mi)/STDi # (x-mean)/std eqn
    # We stack autoscaled columns next to each other
    # to form the full autoscaled training matrix for PCA
    AutoX[:,i]=Ci # for loop take column by column, add column to matrix

# calculate the covariance matrix of the autoscaled training data
CovX=AutoX.transpose().dot(AutoX)/(len(AutoX)-1) # covar eqn, .dot() for dot products fo array
# do the singular value decomposition (SVD) on the covariance matrix
u, s, vh = np.linalg.svd(CovX, full_matrices=True) # what does this function do :(
# columns of the u matrix from the SVD are the principal components
# we decide to use 3 principal components to model the data
PC=u[:,0:12] # wouldn't this use 5 PC? but shows only 4 on console? 0:4 does 0 to 3 columns
# think of it as less than
# Strong influence on PC, important sensor, plot individually

# import test data into the TestX matrix
TestX=loadtxt("stripper_data_13b.txt")
#TestX=loadtxt("process_data_training.txt")
# prepare the shell of the auto scaled test data
m=len(TestX[0]) # number of vars
n=len(TestX) # number of measurements
Test_AutoX=np.zeros((n,m)) # zero matrix

print(np.isfinite(TestX).all(),np.isfinite(TestX).all())
# note that we need to scale the test data according to the scaling of training data!
for j in range(m): # for loop, declare mean and std. dev again for this loop?
    # Scale with training data
    Mi_t=np.mean(TestX[:,j]) # mean
    STDi_t=stdev(TestX[:,j]) # std. dev
    print(STDi_t)
    
    Ci_t=(TestX[:,j]-Mi_t)/STDi_t # eqn for autoscale, note that x is TestX not BigX
    Test_AutoX[:,j]=Ci_t

# Construct the projection matrix for Q values
Q_Projection=np.identity(m)-PC.dot(PC.transpose()) # projection matrix?
# np.identity(m): identity matrix m * m
# calculate distance of a point to a plane, pc*transpose gives the "plane"

# Calculate Q values for every sample in the test data

n=len(TestX) # maybe repetitive?
Q=np.zeros(n) # zero 1 row array
for i in range(n): # for loop
    Q[i]=Test_AutoX[i,:].dot(Q_Projection.dot(Test_AutoX[i,:].transpose())); # eqn for Q val?
    # Q_Projection.dot(Test_AutoX[i,:].transpose())
    # cotributon of 6 senor readings for point 15 of the calculation of Q val
    
plt.plot(Q) # plot and show Q

# np.argmax(Q) for max index
# np.argwhere(Q>x) for all index of Q val greather than x

# Q_Projection.dot(Test_AutoX[14,:].transpose()) type this into console
# determine which measurements were contributing how much to the Q value
# 14: index of the row of the anomaly, which is actually 15 rows, search for max
# Plot everything over certain number, use if statement, record into array

# Plot Q vals as bar charts?
# plt.bar
plt.show()