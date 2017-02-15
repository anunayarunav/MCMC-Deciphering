import numpy as np

from nn.layers import *
from nn.rnn_layers import *

class CharNN(object):
    
    """
    This class takes characters as input and learn to predicts next character sequence as output
    using LSTM model as an input
    """
    def __init__(self, char_to_idx, hidden_dim=100, dtype=np.float32):
        """
        Construct a new instance of charnn model
        """
        
        V = len(char_to_idx)
        self.V = V
        self.H = hidden_dim
        self.dtype = dtype
        
        self.params = {}
        dim_mul = 4
        self.params['Wh'] = np.random.randn(hidden_dim, dim_mul*hidden_dim)*0.01
        self.params['Wx'] = np.random.randn(V, dim_mul*hidden_dim)*0.01
        self.params['b'] = np.zeros(dim_mul*hidden_dim)
        self.params['Wo'] = np.random.randn(hidden_dim, V)*0.01
        self.params['bo'] = np.zeros(V)
        
        # Cast parameters to correct dtype
        for k, v in self.params.iteritems():
          self.params[k] = v.astype(self.dtype)
    
    def loss(self, sequence, h0=None):
        """
        Input: sequence = sequence of characters of dim (N, T), with,  0 <= sequence[i, j] < V
        """
        loss, grads = 0.0, {}
        
        N, T = sequence.shape
        T -= 1
        mask = np.ones((N, T))
        V = self.V
        H = self.H
        Wh = self.params['Wh']
        Wx = self.params['Wx']
        b = self.params['b']
        Wo = self.params['Wo']
        bo = self.params['bo']
        
        x = np.zeros((N*T, V))
        x[np.arange(N*T), sequence[:, :-1].reshape(-1, 1)] = 1
        x = x.reshape((N, T, V))
        
        if h0 is None:
            h0 = np.zeros((N, H))
        
        h, cache_lstm = lstm_forward(x, h0, Wx, Wh, b)
        scores, cache_affine = temporal_affine_forward(h, Wo, bo)
        loss, dout = temporal_softmax_loss(scores, sequence[:,1:], mask)
        
        dout, grads['Wo'], grads['bo'] = temporal_affine_backward(dout, cache_affine)
        dx, dh0, grads['Wx'], grads['Wh'], grads['b'] = lstm_backward(dout, cache_lstm)
        
        return loss, grads, h[:,T-1,:]
    
    def compute_log_probability(self, sequence, h0=None):
        
        N = 1
        
        V = self.V
        H = self.H
        Wh = self.params['Wh']
        Wx = self.params['Wx']
        b = self.params['b']
        Wo = self.params['Wo']
        bo = self.params['bo']
        
        x = np.zeros((N, V))
        x[0, sequence[0][0]] = 1
        
        if h0 is None:
            h = np.zeros((N, H))
        else:
            h = h0
        
        c = np.zeros((N, H))
        
        k = 1
        
        p = 0.
        
        while k < len(sequence[0]):
            x[0, sequence[0][k]] = 1
            h, c, _ = lstm_step_forward(x, h, c, Wx, Wh, b)
            scores, _ = affine_forward(h, Wo, bo)
            prob = np.exp(scores)/np.sum(np.exp(scores))
            p += np.log(prob[0, sequence[0][k]])
            x = np.zeros((N, V))
            k += 1
        
        return p
        
        
    def sample(self, first_char, h0=None, max_length=30):
        loss, grads = 0.0, {}
        
        N = 1
        
        V = self.V
        H = self.H
        Wh = self.params['Wh']
        Wx = self.params['Wx']
        b = self.params['b']
        Wo = self.params['Wo']
        bo = self.params['bo']
        
        x = np.zeros((N, V))
        x[np.arange(N), first_char] = 1
        
        if h0 is None:
            h = np.zeros((N, H))
        else:
            h = h0
        
        c = np.zeros((N, H))
        
        k = 0
        
        ixes = []
        
        while k < max_length:
            h, c, _ = lstm_step_forward(x, h, c, Wx, Wh, b)
            scores, _ = affine_forward(h, Wo, bo)
            p = np.exp(scores)/np.sum(np.exp(scores))
            ix = np.random.choice(range(V), p=p.ravel())
            x = np.zeros((N, V))
            x[0, ix] = 1
            ixes.append(ix)
            k += 1
        
        return ixes
        
            