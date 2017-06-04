import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    stab_factor = 1e-10
    for i in range(num_train):
        scores = X[i].dot(W)
        #scores -=np.max(scores)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores)
        
        logprobs = -np.log(probs[y[i]]+stab_factor)
        loss += logprobs
        
        dscore = probs
        dscore[y[i]] -= 1
        #dscore /= num_train
        
        dW += np.outer(X[i], dscore)
    
    loss /= num_train
    loss += 0.5*reg*np.sum(W*W)
    dW /= num_train
    dW += reg*W
    
    return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
    """
      Softmax loss function, vectorized version.

      Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train=X.shape[0]

    stab_factor = 1e-10
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    scores = np.dot(X, W)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    reg_loss = 0.5*reg*np.sum(W*W)
    
    logprobs = -np.log(probs[range(num_train), y] + stab_factor)
    data_loss = np.sum(logprobs)/num_train
    loss = data_loss + reg_loss
    
    dscores = probs
    dscores[range(num_train),y]-= 1
    dscores = dscores / num_train
    
    dW = np.dot(X.T, dscores)
    dW = dW + reg*W
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW