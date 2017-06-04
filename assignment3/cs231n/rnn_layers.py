import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
  """
  Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
  activation function.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

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
  ##############################################################################
  # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
  # hidden state and any values you need for the backward pass in the next_h   #
  # and cache variables respectively.                                          #
  ##############################################################################
  next_h = np.tanh(np.dot(x, Wx) + np.dot(prev_h, Wh) + b)
  cache = x, prev_h, Wx, Wh, b, next_h
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
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
  ##############################################################################
  # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
  #                                                                            #
  # HINT: For the tanh function, you can compute the local derivative in terms #
  # of the output value from tanh.                                             #
  ##############################################################################
  x, prev_h, Wx, Wh, b, next_h = cache 
  dtanh = (1 - next_h * next_h)*dnext_h

  #print('dtanh.shape', dtanh.shape)
  #print('Wx.T.shape', Wx.T.shape)
  dx = np.dot(dtanh, Wx.T)

  dWx = np.dot(x.T, dtanh)
  dprev_h = np.dot(dtanh, Wh.T)
  dWh = np.dot(prev_h.T, dtanh)
  db = np.sum(dtanh, axis=0)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
  """
  Run a vanilla RNN forward on an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The RNN uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the RNN forward, we return the hidden states for all timesteps.
  
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
  h, cache = None, None
  ##############################################################################
  # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
  # input data. You should use the rnn_step_forward function that you defined  #
  # above.                                                                     #
  ##############################################################################
  N, T, D = x.shape
  N, H = h0.shape
  h = np.zeros((T, N, H))
  x_t = x.transpose(1, 0, 2)
  #cache = {}
  h_temp = h0
  for i in range(T):
        h_temp, cache_temp = rnn_step_forward(x_t[i], h_temp, Wx, Wh, b)
        h[i] += h_temp
        #cache[str(i)] = cache_temp
  h = h.transpose(1,0,2)     
  cache = x, h, h0, Wx, Wh, b  
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
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
  ##############################################################################
  # TODO: Implement the backward pass for a vanilla RNN running an entire      #
  # sequence of data. You should use the rnn_step_backward function that you   #
  # defined above.                                                             #
  ##############################################################################
 
  x, h, h0, Wx, Wh, b = cache
  N, T, D = x.shape
  N, H = h0.shape
  #print('x.shape before', x.shape)
  dx = np.zeros(x.shape)
  #print('dx.shape before transpose', dx.shape)
  dx = dx.transpose(1, 0, 2)
  dh0 = np.zeros(h0.shape)
  dWx = np.zeros(Wx.shape)
  #print('dWx.shape', dWx.shape)
  dWh = np.zeros(Wh.shape)
  db = np.zeros(b.shape)
    
  #dh = np.zeros(T, N, H)
  x = x.transpose(1, 0, 2)
  dh_prev = np.zeros(dh0.shape)
  #print('x.shape', x.shape)
  #print('dx.shape', dx.shape)
  dh = dh.transpose(1, 0, 2)
  h = h.transpose(1, 0, 2)
  for i in reversed(range(T)):
        dh_current = dh[i] + dh_prev #np.dot(dh_prev, Wh.T)
        if(i==0):
            h_prev = h0
        else:
            h_prev = h[i-1]
        cache_step = x[i], h_prev, Wx, Wh, b, h[i]
        #print('x[i] cache.shape', x[i].shape)
        #print('dx_current.shape',dh_current.shape)
        dx_i, dh_prev, dWx_i, dWh_i, db_i = rnn_step_backward(dh_current, cache_step)
        #print('dx[i].shape', dx[i].shape)
        #print('dx_i.shape', dx_i.shape)
        dx[i] += dx_i
        dWx += dWx_i
        dWh += dWh_i
        db += db_i
  
  dh0 = dh_prev
  dx = dx.transpose(1, 0, 2)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
  """
  Forward pass for word embeddings. We operate on minibatches of size N where
  each sequence has length T. We assume a vocabulary of V words, assigning each
  to a vector of dimension D.
  
  Inputs:
  - x: Integer array of shape (N, T) giving indices of words. Each element idx
    of x muxt be in the range 0 <= idx < V.
  - W: Weight matrix of shape (V, D) giving word vectors for all words.
  
  Returns a tuple of:
  - out: Array of shape (N, T, D) giving word vectors for all input words.
  - cache: Values needed for the backward pass
  """
  out, cache = None, None
  ##############################################################################
  # TODO: Implement the forward pass for word embeddings.                      #
  #                                                                            #
  # HINT: This should be very simple.                                          #
  ##############################################################################

  N, T = x.shape
  V, D = W.shape
  X = np.zeros((N * T, V))
  y = x.reshape(N*T)
  X[range(N*T), y[range(N*T)]] = 1
    
  out = np.dot(X, W).reshape(N, T, -1)
  cache = X, x, W 
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return out, cache


def word_embedding_backward(dout, cache):
  """
  Backward pass for word embeddings. We cannot back-propagate into the words
  since they are integers, so we only return gradient for the word embedding
  matrix.
  
  HINT: Look up the function np.add.at
  
  Inputs:
  - dout: Upstream gradients of shape (N, T, D)
  - cache: Values from the forward pass
  
  Returns:
  - dW: Gradient of word embedding matrix, of shape (V, D).
  """
  dW = None
  ##############################################################################
  # TODO: Implement the backward pass for word embeddings.                     #
  #                                                                            #
  # HINT: Look up the function np.add.at                                       #
  ##############################################################################
  X, x, W = cache
  N, T, D = dout.shape
  V, D = W.shape
    
  dout_temp = dout.reshape(N*T, D)
  dW = np.dot(X.T, dout_temp)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dW


def sigmoid(x):
  """
  A numerically stable version of the logistic sigmoid function.
  """
  pos_mask = (x >= 0)
  neg_mask = (x < 0)
  z = np.zeros_like(x)
  z[pos_mask] = np.exp(-x[pos_mask])
  z[neg_mask] = np.exp(x[neg_mask])
  top = np.ones_like(x)
  top[neg_mask] = z[neg_mask]
  return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
  """
  Forward pass for a single timestep of an LSTM.
  
  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.
  
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
  #############################################################################
  # TODO: Implement the forward pass for a single timestep of an LSTM.        #
  # You may want to use the numerically stable sigmoid implementation above.  #
  #############################################################################
  N, H = prev_h.shape
  input_temp = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
  gate_i = sigmoid(input_temp[:, range(H)])
  gate_f = sigmoid(input_temp[:, range(H, 2*H)])
  gate_o = sigmoid(input_temp[:, range(2*H, 3*H)])
  input_g = np.tanh(input_temp[:, range(3*H, 4*H)])

  next_c = gate_f*prev_c + gate_i * input_g
  next_h = gate_o*np.tanh(next_c)

  cache = x, prev_h, prev_c, Wx, Wh, b, gate_i, gate_f, gate_o, input_g, input_temp, next_c, next_h
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
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
  #############################################################################
  # TODO: Implement the backward pass for a single timestep of an LSTM.       #
  #                                                                           #
  # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
  # the output value from the nonlinearity.                                   #
  #############################################################################
  x, prev_h, prev_c, Wx, Wh, b, gate_i, gate_f, gate_o, input_g, input_temp, next_c, next_h = cache
  N, H = prev_h.shape
    
  dgate_o = np.tanh(next_c)*dnext_h

  dnext_c += gate_o*(1-np.tanh(next_c)**2)*dnext_h
    
  dgate_f = prev_c * dnext_c
  dprev_c = gate_f * dnext_c
  dgate_i = input_g * dnext_c
  dinput_g = gate_i * dnext_c
  dinput_temp_1 =  sigmoid(input_temp[:, range(H)])*(1-sigmoid(input_temp[:, range(H)])) * dgate_i
  dinput_temp_2 =  sigmoid(input_temp[:, range(H, 2*H)])*(1-sigmoid(input_temp[:, range(H, 2*H)])) * dgate_f
  dinput_temp_3 =  sigmoid(input_temp[:, range(2*H, 3*H)])*(1-sigmoid(input_temp[:, range(2*H, 3*H)])) * dgate_o
  dinput_temp_4 = (1- np.tanh(input_temp[:, range(3*H, 4*H)])**2) * dinput_g

  
  #print('dinput_temp_1.shape', dinput_temp_1.shape)
  #print('dinput_temp_2.shape', dinput_temp_2.shape)
  #print('dinput_temp_3.shape', dinput_temp_3.shape)
  #print('dinput_temp_4.shape', dinput_temp_4.shape)
  dinput_temp = np.concatenate((dinput_temp_1, dinput_temp_2, dinput_temp_3, dinput_temp_4), axis=1)
    
  #print('dinput_temp.shape', dinput_temp.shape)

  dx = np.dot(dinput_temp, Wx.T)
  dWx = np.dot(x.T, dinput_temp)
  dprev_h = np.dot(dinput_temp, Wh.T)
  dWh = np.dot(prev_h.T, dinput_temp)
  db = np.sum(dinput_temp, axis=0)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
  """
  Forward pass for an LSTM over an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the LSTM forward, we return the hidden states for all timesteps.
  
  Note that the initial cell state is passed as input, but the initial cell
  state is set to zero. Also note that the cell state is not returned; it is
  an internal variable to the LSTM and is not accessed from outside.
  
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
  #############################################################################
  # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
  # You should use the lstm_step_forward function that you just defined.      #
  #############################################################################
  N, T, D = x.shape
  N, H = h0.shape
  x_inter = x.transpose(1, 0, 2)
  prev_h = h0
  prev_c = np.zeros(h0.shape)

  #c = np.zeros((T, N, H))
  h = np.zeros((T, N, H))

  cache_inter={}
  for i in range(T):
        prev_h, prev_c, cache_inter[str(i)] = lstm_step_forward(x_inter[i], prev_h, prev_c, Wx, Wh, b)
        h[i] += prev_h
        #c[i] += next_c
  h = h.transpose(1, 0, 2)
  cache = x, h0, Wx, Wh, b, cache_inter
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

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
  #############################################################################
  # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
  # You should use the lstm_step_backward function that you just defined.     #
  #############################################################################
  x, h0, Wx, Wh, b, cache_inter = cache
  N, T, H = x.shape
  N, H = h0.shape
  dx = np.zeros_like(x)
  dx = dx.transpose(1, 0, 2)
  #print('dh.shape', dh.shape)
  dh = dh.transpose(1, 0, 2)
    
  dh0 = np.zeros_like(h0)
  dWx = np.zeros_like(Wx)
  dWh = np.zeros_like(Wh)
  db = np.zeros_like(b)
  
  dnext_c = np.zeros_like(h0)
  dprev_h = np.zeros_like(h0)
  for i in reversed(range(T)):
        dprev_h = dprev_h + dh[i]
        dx_i, dprev_h, dnext_c, dWx_i, dWh_i, db_i = lstm_step_backward(dprev_h, dnext_c, cache_inter[str(i)])
        dx[i] += dx_i
        dWx += dWx_i
        dWh += dWh_i
        db += db_i
        
  dx = dx.transpose(1, 0, 2)
  dh0 = dprev_h
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
  """
  Forward pass for a temporal affine layer. The input is a set of D-dimensional
  vectors arranged into a minibatch of N timeseries, each of length T. We use
  an affine function to transform each of those vectors into a new vector of
  dimension M.

  Inputs:
  - x: Input data of shape (N, T, D)
  - w: Weights of shape (D, M)
  - b: Biases of shape (M,)
  
  Returns a tuple of:
  - out: Output data of shape (N, T, M)
  - cache: Values needed for the backward pass
  """
  N, T, D = x.shape
  M = b.shape[0]
  out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
  cache = x, w, b, out
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
  x, w, b, out = cache
  N, T, D = x.shape
  M = b.shape[0]

  dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
  dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
  db = dout.sum(axis=(0, 1))

  return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
  """
  A temporal version of softmax loss for use in RNNs. We assume that we are
  making predictions over a vocabulary of size V for each timestep of a
  timeseries of length T, over a minibatch of size N. The input x gives scores
  for all vocabulary elements at all timesteps, and y gives the indices of the
  ground-truth element at each timestep. We use a cross-entropy loss at each
  timestep, summing the loss over all timesteps and averaging across the
  minibatch.

  As an additional complication, we may want to ignore the model output at some
  timesteps, since sequences of different length may have been combined into a
  minibatch and padded with NULL tokens. The optional mask argument tells us
  which elements should contribute to the loss.

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
  
  if verbose: print('dx_flat: '), dx_flat.shape
  
  dx = dx_flat.reshape(N, T, V)
  
  return loss, dx
