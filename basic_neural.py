import numpy as np

np.random.seed(17)

def sigmoid(X):
    return 1/(1 + np.exp(-X))


def sigmoid_backwards(X):
    return np.exp(X)/(1 + np.exp(X))


def initialize_parameters(X, Y):
    '''
        Input
        ---
        X: training examples [n_x, m]
        Y: labels [1, m]
        
        Output
        ---
        params: A dict containing the params of two layer 
                neural network

    '''
    params = {}
    n_x = X.shape[0]
    n_h = 5
    n_y = Y.shape[0]

    params['W1'] = np.random.randn(n_h, n_x)
    params['b1'] = np.zeros((n_h, 1))

    params['W2'] = np.random.randn(n_y, n_h)
    params['b2'] = np.zeros((n_y, 1))

    print(params['W2'].shape)

    return params


def forward_propagation(X, params):
    '''
        Input
        ---
        X: training examples [n_x, m]
        params: A dict containing params of two layer 
                neural net
        
        Output
        ---
        A2: final output of feed forward layer [1, m]
        cache: dict containing A1, Z1, A2, Z2
    '''
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {
            'A1': A1,
            'A2': A2, 
            'Z1': Z1,
            'Z2': Z2
        }
    
    return A2, cache


def compute_loss(A2, Y):
    '''
        Input
        ---
        A2: Ouptut of feed forward layer
        Y: Labels [1, m]

        Output
        ---
        J: Loss of the network 
    '''
    m = Y.shape[1]

    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1-A2), 1-Y)
    cost = -np.sum(logprobs)/m

    return np.squeeze(cost)


def backward_prop(X, Y, cache, params):
    '''
        Input
        ---
        X: training examples [n_x, m]
        Y: labels [1, m]
        cache: dict containing A1, Z1, A2, Z2
        params: A dict containing the params of two layer 
            neural network
        
        Output
        ---
        grads: dict containing dW1, dW2, db1, db2

    '''
    m = X.shape[1]

    A1 = cache['A1']
    A2 = cache['A2']
    W1 = params['W1']
    W2 = params['W2']
    Z1 = cache['Z1']
    Z2 = cache['Z2']

    dZ2 = A2 - Y
    dW2 = (1/m)*np.dot(dZ2, A1.T)
    db2 = (1/m)*np.sum(dZ2, keepdims=True, axis=1)

    dZ1 = np.multiply(np.dot(W2.T, dZ2), sigmoid_backwards(Z1))
    dW1 = (1/m)*np.dot(dZ1, X.T)
    db1 = (1/m)*np.sum(dZ1, keepdims=True, axis=1)

    grads = {
            'dW1': dW1,
            'dW2': dW2,
            'db2': db2,
            'db1': db1
	}

    return grads 

def update_parameters(grads, params, learning_rate = 0.2):
    '''
        Input
        ---
        cache: dict containing A1, Z1, A2, Z2
        params: A dict containing the params of two layer 
            neural network
        
        Output
        ---
        params: updated dict containing parameters

    '''

    params['W1'] -= learning_rate * grads['dW1']
    params['W2'] -= learning_rate * grads['dW2']
    params['b1'] -= learning_rate * grads['db1']
    params['b2'] -= learning_rate * grads['db2']

    return params


def training(X, Y, epoch = 5000):
    '''
        Input
        ---
        X: training examples [n_x, m]
        Y: labels [1, m]
        epoch: Number of iteration through the layer
        Output
        ---
        params: A dict containing the params of two layer 
            neural network

    '''
    params = initialize_parameters(X, Y)

    for i in range(epoch):
        A2, cache = forward_propagation(X, params)
        J = compute_loss(A2, Y)
        grads = backward_prop(X, Y, cache, params)
        params = update_parameters(grads, params)

        if i%100 == 0:
            print('Cost :'+ str(J))
    return params


def test(X, params):
    A2, cache = forward_propagation(X, params)
    return A2

if __name__ == '__main__':

    X = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]])
    Y = np.array([[0], [0], [1], [1]])

    params = training(X.T, Y.T)

    print(test(X.T, params))
    