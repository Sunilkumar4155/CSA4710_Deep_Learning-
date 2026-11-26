import numpy as np
img=np.array([[0,1,2],[3,4,5]],dtype=float)
x=img.flatten()
W=np.arange(1,7,dtype=float).reshape(3,2)/10
b=np.array([0.1,-0.2])
# demonstration with 6 input values and 2 output neurons
W=np.arange(1,13,dtype=float).reshape(6,2)/10
z=x@W+b
print("Image matrix:\n",img)
print("Flattened vector:",x)
print("Weights shape:",W.shape)
print("Fully connected output:",z)
print("ReLU output:",np.maximum(0,z))