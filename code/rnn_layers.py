from __future__ import division
import numpy as np

def rnn_step_forward(x, prev_h, Wx, Wh, b):

	"""
	Inputs:
	- x: Input data for this timestep, of shape (N, D).
	- prev_h: Hidden state from previous timestep, of shape (N, H)
	- Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
	- Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
	- b: Biases of shape (H,)

	Returns a tuple of:
	- next_h: Next hidden state, of shape (N, H)
	- cache: Tuple of values needed for the backward pass.
	"""
	next_h, cache = None, None
	next_h = np.tanh(np.dot(prev_h, Wh) + np.dot(x, Wx) + b)
	cache = next_h, x, prev_h, Wx, Wh, b
	return next_h, cache

def rnn_step_backward(dnext_h, cache):
	"""
	Backward pass for a single timestep of a vanilla RNN.

	Inputs:
	- dnext_h: Gradient of loss with respect to next hidden state
	- cache: Cache object from the forward pass

	Returns a tuple of:
	- dx: Gradients of input data, of shape (N, D)
	- dprev_h: Gradients of previous hidden state, of shape (N, H)
	- dWx: Gradients of input-to-hidden weights, of shape (N, H)
	- dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
	- db: Gradients of bias vector, of shape (H,)
	"""
	dx, dprev_h, dWx, dWh, db = None, None, None, None, None
	next_h, x, prev_h, Wx, Wh, b = cache
	dL_dnexth = dnext_h * tanh_prime(next_h)
	dx = np.dot(dL_dnexth, Wx.T)
	dprev_h = np.dot(dL_dnexth, Wh.T)
	dWx = np.dot(x.T, dL_dnexth)
	dWh = np.dot(prev_h.T, dL_dnexth)
	db  = dL_dnexth.sum(axis=0)
	return dx, dprev_h, dWx, dWh, db

def rnn_forward(x, h0, Wx, Wh, b):
    """
    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)
    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, []
    _, H = h0.shape
    N, T, D = x.shape
    h = np.zeros((N, T, H))
    prev_h = h0
    for i in range(T):
        th, tcache = rnn_step_forward(x[:, i, :], prev_h, Wx, Wh, b)
        
        prev_h = th
        h[:, i, :] = th
        cache.append(tcache)
    return h, cache

def rnn_backward(dh, cache):
	"""
	Compute the backward pass for a vanilla RNN over an entire sequence of data.

	Inputs:
	- dh: Upstream gradients of all hidden states, of shape (N, T, H)

	Returns a tuple of:
	- dx: Gradient of inputs, of shape (N, T, D)
	- dh0: Gradient of initial hidden state, of shape (N, H)
	- dWx: Gradient of input-to-hidden weights, of shape (D, H)
	- dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
	- db: Gradient of biases, of shape (H,)
	"""

	dx, dh0, dWx, dWh, db = None, None, None, None, None
	N, T, H = dh.shape 
	D = cache[0][1].shape[1]
	dx = np.zeros((N, T, D))
	dh0 = np.zeros((N, H))
	dWx = np.zeros((D, H))
	dWh = np.zeros((H, H))
	db = np.zeros((H,))

	dht = np.zeros((N, H))
	for i in reversed(range(T-1)):
		dxt, dht, dWxt, dWht, dbt = rnn_step_backward(dh[:,i,:]+dht, cache[i])
		dx[:, i, :] = dxt
		dWx += dWxt
		dWh += dWht
		db += dbt
	dh0 = dht
		
	return dx, dh0, dWx, dWh, db

def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.
    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)
    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    _, H = prev_c.shape

    a = np.dot(x, Wx) + np.dot(prev_h, Wh) + b #(N, 4H)
    #divide into 4 vectors ai, af, ao, ag
    ai = a[:, :H]
    af = a[:, H:2*H]
    ao = a[:, 2*H:3*H]
    ag = a[:, 3*H:4*H]
    # print "ai: ", ai.shape

    #input gate
    i = sigmoid(ai)
    #forget gate
    f = sigmoid(af)
    #output gate
    o = sigmoid(ao)
    #control gate
    g = np.tanh(ag)

    #next cell state
    next_c = f * prev_c + i * g
    #next hidden state
    next_h = o * np.tanh(next_c)
    cache = (x, prev_h, prev_c, Wx, Wh, b, i, f, o, g, next_c)

    return next_h, next_c, cache

def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.
    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass
    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
    dprev_h = None
    x, prev_h, prev_c, Wx, Wh, b, i, f, o, g, next_c = cache
    tanh_nextc = np.tanh(next_c)
    dprev_c = (dnext_c * f) + ((o * (1- np.power(tanh_nextc, 2)) * f) * dnext_h)
    di = (dnext_c * g) + ((o * (1- np.power(tanh_nextc, 2)) * g) * dnext_h)
    df = (dnext_c * prev_c) + ((o * (1- np.power(tanh_nextc, 2)) * prev_c) * dnext_h)
    dg = (dnext_c * i) + ((o * (1- np.power(tanh_nextc, 2)) * i) * dnext_h)
    do = dnext_h * tanh_nextc
    
    di = (di * i) * (1 - i)
    df = (df * f) * (1 - f)
    do = (do * o) * (1 - o)
    dg = dg * (1 - np.power(g, 2))

    di_forgate = np.hstack((di, df, do, dg))

    dWh = np.dot(prev_h.T, di_forgate)
    dWx = np.dot(x.T, di_forgate)
    db = di_forgate.sum(axis=0)

    dx = np.dot(di_forgate, Wx.T)
    dprev_h = np.dot(di_forgate, Wh.T)
    dx = np.dot(di_forgate, Wx.T)
    dprev_h = np.dot(di_forgate, Wh.T)
    return dx, dprev_h, dprev_c, dWx, dWh, db

def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. 
    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)
    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    _, H = h0.shape
    N, T, D = x.shape
    h = np.zeros((N, T, H))
    prev_h = h0
    prev_c = np.zeros(h0.shape)
    cache = []
    for i in range(T):
        h_t, c_t, cache_t = lstm_step_forward(x[:, i, :], prev_h, prev_c, Wx, Wh, b)
        prev_h = h_t
        prev_c = c_t
        h[:, i, :] = h_t
        cache.append(cache_t)
    
    return h, cache

def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]
    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass
    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    N, T, H = dh.shape
    D = cache[0][0].shape[1]
    dx = np.zeros((N, T, D))
    dh0 = np.zeros((N, H))
    dWx = np.zeros((D, 4 * H))
    dWh = np.zeros((H, 4 * H))
    db = np.zeros((4 * H))
    dprev_h_t = np.zeros((N, H))
    dprev_c_t = np.zeros((N, H))
    for i in range(T - 1, -1, -1):
        dxt, dprev_h_t,dprev_c_t, dWxt, dWht, dbt = lstm_step_backward(dh[:,i,:]+dprev_h_t,dprev_c_t, cache[i])
        dx[:, i, :] = dxt
        dWx += dWxt
        dWh += dWht
        db += dbt
    dh0 = dprev_h_t
    
    return dx, dh0, dWx, dWh, db

	
def word_embedding_forward(x, W):
	"""
	Forward pass for word embeddings. We operate on minibatches of size N where
	each sequence has length T. 

	Inputs:
	- x: Integer array of shape (N, T) giving indices of words. Each element idx
	of x muxt be in the range 0 <= idx < V.
	- W: Weight matrix of shape (V, D) giving word vectors for all words.

	Returns a tuple of:
	- out: Array of shape (N, T, D) giving word vectors for all input words.
	- cache: Values needed for the backward pass
	"""
	out, cache = None, []
	N = x.shape[0]
	T = x.shape[1]
	D = W.shape[1]
	V = W.shape[0]
	out = np.zeros((N, T, D))
	x_t = np.zeros((N, T, V))
	for n in range(N):
		for t in range(T):
			element = np.zeros((1,V))
			element[0,x[n,t]] = 1
			x_t[n, t] = element
	out = np.dot(x_t, W)
	cache = out, x_t, W
	return out, cache

def word_embedding_backward(dout, cache):
  """
  Backward pass for word embeddings.
  
  Inputs:
  - dout: Upstream gradients of shape (N, T, D)
  - cache: Values from the forward pass
  
  Returns:
  - dW: Gradient of word embedding matrix, of shape (V, D).
  """
  _, x, W = cache
  V, D = W.shape
  dW = np.zeros((V, D))
  for i in range(dout.shape[1]):
  	dW += np.dot(x[:,i,:].T, dout[:,i,:])
  return dW


def temporal_affine_forward(x, w, b):
	"""
	Forward pass for a temporal affine layer. 

	Inputs:
	- x: Input data of shape (N, T, D)
	- w: Weights of shape (D, M)
	- b: Biases of shape (M,)

	Returns a tuple of:
	- out: Output data of shape (N, T, M)
	- cache: Values needed for the backward pass
	"""
	N, T, D = x.shape
	M = w.shape[1]
	out = np.zeros((N, T, M))
	for i in range(x.shape[1]):
		out[:,i,:] = np.dot(x[:,i,:],w[:,:]) + b
	cache = x, w, b

	return out, cache

def temporal_affine_backward(dout, cache):
	"""
	Backward pass for temporal affine layer.

	Input:
	- dout: Upstream gradients of shape (N, T, M)
	- cache: Values from forward pass

	Returns a tuple of:
	- dx: Gradient of input, of shape (N, T, D)
	- dw: Gradient of weights, of shape (D, M)
	- db: Gradient of biases, of shape (M,)
	"""
	N, T, M = dout.shape
	D = cache[1].shape[0]
	M = cache[1].shape[1]
	dx = np.zeros((N, T, D))
	dw = np.zeros((D, M))
	db0 = np.zeros((N, M))
	db = np.zeros((M))
	x, w, b = cache

	for i in range(T):
		dx[:,i,:] = np.dot(dout[:,i,:], w[:,:].T)
		dw += np.dot( x[:,i,:].T, dout[:,i,:])
	db = np.sum(dout, axis=(0,1))
	print "dx.shape: ", dx.shape
	print "dw.shape: ", dw.shape
	print "db.shape: ",db.shape
	return dx, dw, db

def temporal_softmax_loss(x, y, mask, verbose=False):
	"""
	A temporal version of softmax loss for use in RNNs. 
	Inputs:
	- x: Input scores, of shape (N, T, V)
	- y: Ground-truth indices, of shape (N, T) where each element is in the range
	   0 <= y[i, t] < V
	- mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
	the scores at x[i, t] should contribute to the loss.

	Returns a tuple of:
	- loss: Scalar giving loss
	- dx: Gradient of loss with respect to scores x.
	"""

	N, T, V = x.shape

	x_flat = x.reshape(N * T, V)
	y_flat = y.reshape(N * T)
	mask_flat = mask.reshape(N * T)

	probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
	probs /= np.sum(probs, axis=1, keepdims=True)
	loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
	dx_flat = probs.copy()
	dx_flat[np.arange(N * T), y_flat] -= 1
	dx_flat /= N
	dx_flat *= mask_flat[:, None]

	if verbose:
	    print 'dx_flat: ', dx_flat.shape

	dx = dx_flat.reshape(N, T, V)

	return loss, dx
		


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def softmax_prime(z):
	return softmax(z)* np.subtract(1,softmax(z))

def softmax(z):
	N, M = z.shape
	out = np.zeros((N, M))
	for i in range(N):
		out[i, :] = np.exp(z[i,:])/np.sum(np.exp(z[:,:]))
	print out.shape
	return out

def rel_error(x, y):
	""" returns relative error """
	return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def tanh(a):
	return np.tanh(a)

def tanh_prime(a):
	return np.subtract(1, np.power(np.tanh(a),2))

