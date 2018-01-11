import numpy as np


training_inputs = np.array([[0,0,0,0,0,0,0,0,0,0],
                            [0,1,0,0,1,1,0,1,0,0],
                            [1,0,1,1,0,1,1,1,1,1],
                            [1,0,1,0,1,0,1,1,0,0],
                            [1,0,0,0,1,1,0,0,1,1],
                            [1,0,0,1,0,1,0,1,1,1],
                            [0,1,1,0,1,1,0,1,0,0],
                            [1,0,1,0,0,1,0,0,0,1],
                            [1,1,0,1,0,0,1,1,0,0]])


training_outputs = np.array([[0,0,1,1,0,0,1,1,0]]).T

print (training_inputs)
print (training_outputs)


def nonlin(x, deriv=False):
    
    if deriv==True:
        return x*(1-x)

    return 1/(1+ np.exp(-x))


np.random.seed(1)


synaptic_weight=2*np.random.random((10,1))-1



for iter in range(20000):
    

    gettrain=nonlin(np.dot(training_inputs,synaptic_weight))

    training_error = training_outputs - gettrain

    #multiply how much we missed with the slope of the sigmoid(nonlinear)

    train_delta= training_error * nonlin(gettrain,True)

    synaptic_weight += np.dot(training_inputs.T,train_delta)
    

test_array= np.array([1,1,0,1,1,1,1,1,1,1])

print (nonlin(np.dot(test_array,synaptic_weight)))


