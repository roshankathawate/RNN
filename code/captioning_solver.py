import numpy as np
import optim
from coco_utils import sample_coco_minibatch

#Code refernce: http://cs231n.stanford.edu/
class CaptioningSolver(object):
  def __init__(self, model, data, **more_args):
    self.model = model
    self.data = data
    
    # Unpack keyword arguments
    self.update_rule = more_args.pop('update_rule', 'sgd')
    self.optim_config = more_args.pop('optim_config', {})
    self.lr_decay = more_args.pop('lr_decay', 1.0)
    self.batch_size = more_args.pop('batch_size', 100)
    self.num_epochs = more_args.pop('num_epochs', 10)

    self.print_every = more_args.pop('print_every', 10)
    self.verbose = more_args.pop('verbose', True)

    # Throw an error if there are extra keyword arguments
    if len(more_args) > 0:
      extra = ', '.join('"%s"' % k for k in more_args.keys())
      raise ValueError('Unrecognized arguments %s' % extra)

    # Make sure the update rule exists, then replace the string
    # name with the actual function
    if not hasattr(optim, self.update_rule):
      raise ValueError('Invalid update_rule "%s"' % self.update_rule)
    self.update_rule = getattr(optim, self.update_rule)

    self.maintainance()


  def maintainance(self):
   
    # Set up some variables for book-keeping
    self.epoch = 0
    self.best_val_acc = 0
    self.best_params = {}
    self.loss_history = []
    self.train_acc_history = []
    self.val_acc_history = []

    # Make a deep copy of the optim_config for each parameter
    self.optim_configs = {}
    for p in self.model.params:
      d = {k: v for k, v in self.optim_config.iteritems()}
      self.optim_configs[p] = d


  def iterate(self):
    """
    Make a single gradient update.
    """
    # Make a minibatch of training data
    minibatch = sample_coco_minibatch(self.data,
                  batch_size=self.batch_size,
                  split='train')
    captions, features, urls = minibatch

    # Compute loss and gradient
    loss, grads = self.model.loss(features, captions)
    self.loss_history.append(loss)

    # Perform a parameter update
    for p, w in self.model.params.iteritems():
      dw = grads[p]
      config = self.optim_configs[p]
      next_w, next_config = self.update_rule(w, dw, config)
      self.model.params[p] = next_w
      self.optim_configs[p] = next_config

  def check_accuracy(self, X, y, num_samples=None, batch_size=100):
    
    # Maybe subsample the data
    N = X.shape[0]
    if num_samples is not None and N > num_samples:
      mask = np.random.choice(N, num_samples)
      N = num_samples
      X = X[mask]
      y = y[mask]

    # Compute predictions in batches
    num_batches = N / batch_size
    if N % batch_size != 0:
      num_batches += 1
    y_pred = []
    for i in xrange(num_batches):
      start = i * batch_size
      end = (i + 1) * batch_size
      scores = self.model.loss(X[start:end])
      y_pred.append(np.argmax(scores, axis=1))
    y_pred = np.hstack(y_pred)
    acc = np.mean(y_pred == y)

    return acc


  def train(self):
    """
    Run optimization to train the model.
    """
    total_train_ex = self.data['train_captions'].shape[0]
    num_iter_per_epoch = max(total_train_ex / self.batch_size, 1)
    num_iterations = self.num_epochs * num_iter_per_epoch

    for t in xrange(num_iterations):
      self.iterate()

      # Maybe print training loss
      if self.verbose and t % self.print_every == 0:
        print '(Iteration %d / %d) loss: %f' % (
               t + 1, num_iterations, self.loss_history[-1])

      # At the end of every epoch, increment the epoch counter and decay the
      # learning rate.
      epoch_end = (t + 1) % num_iter_per_epoch == 0
      if epoch_end:
        self.epoch += 1
        for k in self.optim_configs:
          self.optim_configs[k]['learning_rate'] *= self.lr_decay

      