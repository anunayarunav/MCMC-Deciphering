import numpy as np

from nn.layers import *
from nn.rnn_layers import *

class SequenceClassifier(object):
    
    """
    This class takes sequences as input and learns to classify them.
    """
    
    def __init__(self, input_dims, embedding_dims=50, lstm_layers=2, 
                 hidden_dims=[100, 100], label_count=2, reg=0.01, dtype=np.float32):
        """
        Arguments:

        input_dims: number of possible values each member of a sequence can take, i.e. vocabulary size
        embedding_dims: number of dimensions the input data to be embedded in
        lstm_layers: number of lstm cells to be stacked
        hidden_dims: dimension of each lstm cells
        label_count: number of possible labels

        """
        self.params = {}
        self.lstm_layers = lstm_layers
        self.V = input_dims
        self.embedding_dims = embedding_dims
        self.hidden_dims = hidden_dims
        self.reg = reg
        self.dtype = dtype
        
        i = 1
        h1 = embedding_dims
        weight_scale = 0.01
        dims_mul = 4
        
        self.params['W_embed'] = np.random.randn(input_dims, embedding_dims)
        
        for h in hidden_dims:
            self.params['Wx%d'%i] = np.random.randn(h1, dims_mul*h)*weight_scale
            self.params['Wh%d'%i] = np.random.randn(h, dims_mul*h)*weight_scale
            self.params['b%d'%i] = np.zeros(dims_mul*h)
            
            i += 1
            h1 = h
            
        #affine output layer
        self.params['Wo'] = np.random.randn(h1, label_count)*weight_scale
        self.params['bo'] = np.zeros(label_count)
        
        # Cast parameters to correct dtype
        for k, v in self.params.iteritems():
          self.params[k] = v.astype(self.dtype)
    
    def loss(self, X, label=None):
        """
        Arguments:

        X: An nd-array of shape (N, T), where N is number of sequences, T is length of each sequence,
           and each element in X is between [0,V), i.e. 0 <= X[i, j] < V

        label: true labels for each element, a numpy array of size N
        """
        loss, grads = 0.0, {}
        reg = self.reg
        N, T = X.shape
        
        #forward pass
        W_embed = self.params['W_embed']
        
        x, embedding_cache = word_embedding_forward(X, W_embed)

        cache = {}
        for j in xrange(self.lstm_layers):
            i = j + 1
            Wx = self.params['Wx%d'%i]
            Wh = self.params['Wh%d'%i]
            b = self.params['b%d'%i]
            h0 = np.zeros((N, self.hidden_dims[j]))
            
            x, cache['%d'%i] = lstm_forward(x, h0, Wx, Wh, b)
        
        ht = x[:,-1,:]
        Wo = self.params['Wo']
        bo = self.params['bo']
        
        scores, affine_cache = affine_forward(ht, Wo, bo)
        
        if label is None:
            return scores
        
        loss, dx = softmax_loss(scores, label)
        
        #backward pass
        dx, grads['Wo'], grads['bo'] = affine_backward(dx, affine_cache)
        
        loss += 0.5*reg*np.sum(Wo*Wo)
        grads['Wo'] += reg*Wo
        
        #reshape dx
        dh = np.zeros((N, T, self.hidden_dims[-1]))
        dh[:,-1,:] = dx
        dx = dh
        
        for i in xrange(self.lstm_layers, 0, -1):
            dx, _, grads['Wx%d'%i], grads['Wh%d'%i], grads['b%d'%i] = lstm_backward(dx, cache['%d'%i])
            
            Wx = self.params['Wx%d'%i]
            Wh = self.params['Wh%d'%i]
            
            loss += 0.5*reg*np.sum(Wx*Wx)
            loss += 0.5*reg*np.sum(Wh*Wh)
            
            grads['Wx%d'%i] += reg*Wx
            grads['Wh%d'%i] += reg*Wh
        
        grads['W_embed'] = word_embedding_backward(dx, embedding_cache)
        
        return loss, grads
    
    def log_entropy_wrt_true(self, X):
        scores = self.loss(X)
        labels = np.ones(X.shape[0], dtype='int')
        loss, _ = softmax_loss(scores, labels)
        return loss
    
    def predict(self, X):
        scores = self.loss(X)
        return np.argmax(scores, axis=1)