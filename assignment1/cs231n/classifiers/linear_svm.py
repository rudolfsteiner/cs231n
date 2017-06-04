import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero
  
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
    
  for i in range(num_train):#mini-batch circle
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]

    dscores = np.zeros(num_classes)

    minus_count=0
    for j in range(num_classes):#loss computing circle

      if j == y[i]:
        continue

      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dscores[j] = 1
        minus_count+=1

    dscores[y[i]] = 0-minus_count
    dW=dW+np.outer(X[i], dscores)
    dW+=reg*W
    
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW = dW/num_train
    
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW



def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.(3073*10)
  - X: A numpy array of shape (N, D) containing a minibatch of data. (500*3073)
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means(500,)
  that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]

  scores = X.dot(W)
  dscores= np.zeros(scores.shape)
  rx=range(X.shape[0])

  rscore=range(scores.shape[0])
  yscore=scores[rscore, y[rscore]]
  loss_scores = np.maximum(0, (scores.transpose()-yscore).transpose()+1)

  loss_scores[rx, y[rx]]=0
  loss = np.sum(loss_scores)/num_train

  loss += 0.5*reg*np.sum(W*W)
  dscore=(loss_scores>0)*1
  dscore[rx, y[rx]]=0-np.sum(dscore, axis=1)
  dscore = dscore / num_train

  dW = np.dot(X.T, dscore)
  dW = dW + reg * W

  #print("reg*W", reg * W)

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW

