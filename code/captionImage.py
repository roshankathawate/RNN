import numpy as np
from layers import *
from rnn_layers import *
#Code refernce: http://cs231n.stanford.edu/
class CaptioningRNN(object):

    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
                 hidden_dim=128, cell_type='rnn', dtype=np.float32):
        if cell_type not in {'rnn', 'lstm'}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
        self.params = {}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)

        # Initialize word vectors
        self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim)
        self.params['W_embed'] /= 100

        # Initialize CNN -> hidden state projection parameters
        self.params['W_proj'] = np.random.randn(input_dim, hidden_dim)
        self.params['W_proj'] /= np.sqrt(input_dim)
        self.params['b_proj'] = np.zeros(hidden_dim)

        # Initialize parameters for the RNN
        dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
        self.params['Wx'] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params['Wx'] /= np.sqrt(wordvec_dim)
        self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
        self.params['Wh'] /= np.sqrt(hidden_dim)
        self.params['b'] = np.zeros(dim_mul * hidden_dim)

        # Initialize output to vocab weights
        self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)
        self.params['W_vocab'] /= np.sqrt(hidden_dim)
        self.params['b_vocab'] = np.zeros(vocab_size)

        # Cast parameters to correct dtype
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(self.dtype)

    def loss(self, features, captions):
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        # You'll need this
        mask = (captions_out != self._null)

        # Weight and bias for the affine transform from image features to initial
        # hidden state
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']

        # Word embedding matrix
        W_embed = self.params['W_embed']

        # Input-to-hidden, hidden-to-hidden, and biases for the RNN
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

        # Weight and bias for the hidden-to-vocab transformation.
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        loss, grads = 0.0, {}
        
        h0, h0_cache = affine_forward(features, W_proj, b_proj)

        wembed, wembed_cache = word_embedding_forward(captions_in, W_embed)

        if self.cell_type == "rnn":        
            rnn_h, rnn_h_cache = rnn_forward(wembed, h0, Wx, Wh, b)

        elif self.cell_type =="lstm":
            rnn_h, rnn_h_cache = lstm_forward(wembed, h0, Wx, Wh, b)
        else:
            raise NotImplementedError("Cell type not implemented")

        rnn_out, rnn_out_cache = temporal_affine_forward(
            rnn_h, W_vocab, b_vocab)
        loss, dout = temporal_softmax_loss(rnn_out, captions_out, mask)

        # gradient computation
        dhid, dW_vocab, db_vocab = temporal_affine_backward(
            dout, rnn_out_cache)
        if self.cell_type == "rnn":
            dembed, dh0, dWx, dWh, db = rnn_backward(dhid, rnn_h_cache)
        else:
            dembed, dh0, dWx, dWh, db = lstm_backward(dhid, rnn_h_cache)

        dwembed = word_embedding_backward(dembed, wembed_cache)
        _, dWproj, dbproj = affine_backward(dh0, h0_cache)

        # Initialize word vectors
        grads['W_embed'] = dwembed

        # Initialize CNN -> hidden state projection parameters
        grads['W_proj'] = dWproj
        grads['b_proj'] = dbproj

        # Initialize parameters for the RNN
        grads['Wx'] = dWx
        grads['Wh'] = dWh
        grads['b'] = db

        # Initialize output to vocab weights
        grads['W_vocab'] = dW_vocab
        grads['b_vocab'] = db_vocab

       
        return loss, grads

    def sample(self, features, max_length=30):
        N = features.shape[0]
        captions = self._null * np.ones((N, max_length), dtype=np.int32)

        # Unpack parameters
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
        W_embed = self.params['W_embed']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']
        h0 = features
        if self.cell_type == "lstm":
            c0 = np.zeros(h0.shape)
        
        h0, h0_cache = affine_forward(h0, W_proj, b_proj)
        for i in range(max_length):
            word = np.ones((N, 1), dtype=np.int32) * self._start
            wembed, wembed_cache = word_embedding_forward(word, W_embed)
            # rnn_h, rnn_h_cache = rnn_forward(wembed, h0, Wx, Wh, b) #actually we can use rnn_forward
            if self.cell_type == "rnn":
                rnn_h, rnn_h_cache = rnn_step_forward(np.squeeze(wembed), h0, Wx, Wh, b)
            elif self.cell_type == "lstm":
                rnn_h, rnn_c, rnn_h_cache = lstm_step_forward(np.squeeze(wembed), h0, c0, Wx, Wh, b)

            rnn_out, rnn_out_cache = temporal_affine_forward(
                rnn_h.reshape((N,1,-1)), W_vocab, b_vocab)

            h0 = rnn_h
            if self.cell_type == "lstm":
                c0 = rnn_c
            predicted_word = np.argmax(np.squeeze(rnn_out), axis=1)
            word = predicted_word
            captions[:, i] = word
        return captions