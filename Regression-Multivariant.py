import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly

path='week_2-ex_1.txt'
data=pd.read_csv(path,header=None,names=['Size','Bedrooms','Price'])

# show data
print('data = ')
print(data.head(10))
print()
print('data.describe = ')
print(data.describe())

# data is not in the same scale
# should be rescaling

# rescaling data (-1 : 1) -> x-mean/std for each feature
data=(data - data.mean()) / data.std()

print()
print('data after normalization = ')
print(data.head(10))

# add ones column
data.insert(0,'Ones',1)

# separate X (training data) from y (target variable)
cols=data.shape[1]
X=data.iloc[:,0:cols - 1]
y=data.iloc[:,cols - 1:cols]

print('**************************************')
print('X2 data = \n',X.head(10))
print('y2 data = \n',y.head(10))
print('**************************************')

# convert to matrices and initialize theta
X=np.matrix(X.values)
y=np.matrix(y.values)

# nums of theta is (nums of features + 1)
# y = theta0+ theta1.x1 + theta2. x2
theta=np.matrix(np.array([0,0,0]))

# cost function

def computeCost(X,y,theta):
    inner=np.power(((X * theta.T) - y),2)
    return np.sum(inner) / (2 * len(X))


def gradientDescent(X,y,theta,alpha,iters):
    temp=np.matrix(np.zeros(theta.shape))
    parameters=int(theta.ravel().shape[1])
    cost=np.zeros(iters)

    for i in range(iters):
        error=(X * theta.T) - y

        for j in range(parameters):
            term=np.multiply(error,X[:,j])
            temp[0,j]=theta[0,j] - ((alpha / len(X)) * np.sum(term))

        theta=temp
        cost[i]=computeCost(X,y,theta)

    return theta,cost


# initialize variables for learning rate and iterations
alpha=0.1
iters=1000

# perform linear regression on the data set
g,cost=gradientDescent(X,y,theta,alpha,iters)

# get the cost (error) of the model
thiscost=computeCost(X,y,g)

print('g2 = ',g)
print('cost2  = ',cost[0:50])
print('computeCost = ',thiscost)
print('**************************************')

# get best fit line for Size vs. Price

x=np.linspace(data.Size.min(),data.Size.max(),100)
print('x \n',x)
print('g \n',g)

f=g[0,0] + (g[0,1] * x)
print('f \n',f)

# draw the line for Size vs. Price

fig,ax=plt.subplots(figsize=(5,5))
ax.plot(x,f,'r',label='Prediction')
ax.scatter(data.Size,data.Price,label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Size')
ax.set_ylabel('Price')
ax.set_title('Size vs. Price')

# get best fit line for Bedrooms vs. Price

x=np.linspace(data.Bedrooms.min(),data.Bedrooms.max(),100)
print('x \n',x)
print('g \n',g)

f=g[0,0] + (g[0,1] * x)
print('f \n',f)

# draw the line  for Bedrooms vs. Price

fig,ax=plt.subplots(figsize=(5,5))
ax.plot(x,f,'r',label='Prediction')
ax.scatter(data.Bedrooms,data.Price,label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Bedrooms')
ax.set_ylabel('Price')
ax.set_title('Size vs. Price')

# draw error graph

fig,ax=plt.subplots(figsize=(5,5))
ax.plot(np.arange(iters),cost,'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')

plt.show()

# 3d scatter
trace1 = go.Scatter3d(
    x=(X[:,1:2]).flatten().tolist()[0],
    y=X[:,2:].flatten().tolist()[0],
    z=y.flatten().tolist()[0],
    mode='markers',
    name='markers'
)
trace2 = go.Scatter3d(
    x=(X[:,1:2]).flatten().tolist()[0],
    y=X[:,2:].flatten().tolist()[0],
    z=f.tolist(),
    mode='markers',
    name='markers'
)
fig = go.Figure(data=[trace1, trace2])
plotly.offline.iplot(fig, filename='simple-3d-scatter')