import logging

import torch
import torchvision
import torch.nn as nn
import numpy as np


logger = logging.getLogger()


class CNNEncoder(nn.Module):
    """
    Encoder.
    See here: https://pytorch.org/docs/stable/torchvision/models.html
    """

    def __init__(self, proj_dim=None, feat_layer="conv", fine_tune=False):
        super(CNNEncoder, self).__init__()
        resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

        if feat_layer == "conv":
            self.encoded_image_size = 14
            # Resize image to fixed size to allow input images of variable size
            self.adaptive_pool = nn.AdaptiveAvgPool2d((self.encoded_image_size, self.encoded_image_size))
            # Remove linear and pool layers
            modules = list(resnet.children())[:-2]
        else:
            self.encoded_image_size = 1
            self.adaptive_pool = None
            # Remove last output layer
            modules = list(resnet.children())[:-1]

        self.cnn_output_size = 2048
        self.output_size = proj_dim if proj_dim is not None else self.cnn_output_size

        self.resnet = nn.Sequential(*modules)

        if proj_dim is not None:
            self.lin_proj = nn.Linear(self.cnn_output_size, proj_dim, True)
        else:
            self.lin_proj = None

        self.fine_tune(fine_tune)

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        batch_size = images.size(0)
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        if self.adaptive_pool is not None:
            out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
            out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
            out = out.view(batch_size, -1, self.cnn_output_size)  # (batch_size, encoded_image_size^2, 2048)
        else:
            out = out.permute(0, 2, 3, 1)
            out = out.squeeze(2).squeeze(1)  # (batch_size, 2048)

        if self.lin_proj is not None:  # Only called when no attention is used
            out = self.lin_proj(out)  # (batch_size, proj_dim)

        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class AttentionMechanism(nn.Module):
    """
    Attention mechanism. Computes attention over the location representations of an image, 
    which are the response of a convolutional layer in a CNN.
    """

    def __init__(self, encode_dim, decode_dim):
        """
        :param encode_dim: Dimension of encoder vectors (this is the size of each location representation).
        :param decode_dim: Dimension of decoder hidden state.

        Attention Mechanism based on "https://www.aclweb.org/anthology/D15-1166" 
        """
        super(AttentionMechanism, self).__init__()

        self.encode_dim = encode_dim
        self.decode_dim = decode_dim

        ##############################################################################
        # TODO: Create the weights of the attention mechanism.                       #
        ##############################################################################
        self.Wa = nn.Parameter(torch.FloatTensor(torch.linspace(-0.1, 0.9, steps= decode_dim * encode_dim).view(decode_dim, encode_dim)))
        self.Wc = nn.Parameter(torch.FloatTensor(torch.linspace(-0.3, 0.7, steps= decode_dim * (encode_dim + decode_dim)).view(encode_dim + decode_dim, decode_dim)))
        self.bc = nn.Parameter(torch.FloatTensor(torch.linspace(-0.2, 0.4, steps= decode_dim)))

        self.reset_parameters()   # Uncomment this for TestAttention

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.Wa)
        nn.init.xavier_uniform_(self.Wc)
        self.bc.data.fill_(0.01)

    def forward(self, decoder_hidden, all_encoder_hidden):
        """
        The attention mechanism forward on an entire sequence of decoder hidden states.
        We assume an input sequence composed of T vectors, each of dimension H. H is the
        hidden state size of the decoder, T is the number of decoder steps, and N is the
        batch size. To reiterate, the attention mechanism takes in the (N x T x H) batch
        of decoder hidden state sequences and the batch of  encoder outputs, which is a
        (N x L x C) set of location vectors where L is the number of locations and C is
        the size of each location vector.
        """
        output = None  # Context vectors
        norm_attn_scores = None # Normalized attention scores

        # h_t = tanh(W_c[c_t;h_t])
        # h_t^T * -h_s
        # h_t^T * W_a * -h_s
        # W_a[h_t;-h_s]
        # target hidden state h_t 
        # context vector c_t
        inner = decoder_hidden @ self.Wa
        s = inner @ torch.transpose(all_encoder_hidden, 2, 1)
        norm_attn_scores = torch.softmax(s, dim=2)
        context = norm_attn_scores @ all_encoder_hidden
        concat_layer = torch.cat((context, decoder_hidden), dim=2)
        output = torch.tanh(concat_layer @ self.Wc)
        return output.data, norm_attn_scores.data

class RNNDecoderSingle(nn.Module):
    """
    RNN decoder for a single timestep.
    """

    def __init__(self,
                 embed_size: int, #Dimension of Word Vectors/Embeddings = D
                 hidden_size: int #Dimension of hidden state = H
                 ):
        """
        :param embed_size: Size of input embeddings.
        :param hidden_size: Size of hidden state.
        """
        super(RNNDecoderSingle, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        # The weights of a vanilla rnn.
        self.Wx = torch.FloatTensor(torch.linspace(-0.1, 0.9, steps= embed_size * hidden_size).view(embed_size, hidden_size))
        self.Wh = torch.FloatTensor(torch.linspace(-0.3, 0.7, steps= hidden_size * hidden_size).view(hidden_size, hidden_size))
        self.b = torch.FloatTensor(torch.linspace(-0.2, 0.4, steps= hidden_size))

    def forward(self, input_embeddings, prev_decoder_state):
        """
        The forward pass for a single timestep of a vanilla RNN that uses a tanh
        activation function. The input data has dimension D, the hidden state has dimension H, and
        we use a minibatch size of N.

        Inputs:
            :param input_embeddings: Input data for a single step of the timeseries of shape (N x D)
            :param prev_decoder_state: Initial hidden state of shape (N x H)

        Returns:
            :return next_h: Next hidden state of shape (N x H)
        """
        # ht = tanh(xt−1 * Wx + ht−1 * Wh + b)
        inner1 = input_embeddings @ self.Wx 
        inner2=  prev_decoder_state @ self.Wh

        product = inner1 + inner2 + self.b
        next_h = torch.tanh(product)
        return next_h


class RNNDecoder(nn.Module):
    """
    Full RNN decoder that operates on an entire input sequence.
    """

    def __init__(self,
                 vocab_size: int,
                 embed_size: int,
                 encoder_state_size: int,
                 hidden_size: int,
                 decoder_out_dropout_prob: int = 0.0,
                 use_attention: bool = False
                 ):
        """
        :param vocab_size: Size of the output vocabulary.
        :param embed_size: Size of input embeddings.
        :param encoder_state_size: Size of vectors coming from the encoder.
        :param hidden_size: Size of hidden state.
        :param decoder_out_dropout_prob: Dropout probability. Not required to use, but is useful for regularization.
        :param use_attention: Whether or not to use attention.
        """
        super(RNNDecoder, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.encoder_state_size = encoder_state_size
        self.hidden_size = hidden_size
        self.use_attention = use_attention
        self.attention = None

        # The weights of a vanilla rnn.
        self.Wx = nn.Parameter(torch.FloatTensor(torch.linspace(-0.2, 0.4, steps= embed_size * hidden_size).view(embed_size, hidden_size)))
        self.Wh = nn.Parameter(torch.FloatTensor(torch.linspace(-0.4, 0.1, steps= hidden_size * hidden_size).view(hidden_size, hidden_size)))
        self.b = nn.Parameter(torch.FloatTensor(torch.linspace(-0.7, 0.1, steps= hidden_size)))

        self.Wo = nn.Parameter(torch.zeros(hidden_size,vocab_size))  # Output layer weight
        self.bo = nn.Parameter(torch.zeros(vocab_size))  # Output layer bias 

        if use_attention:
            self.attention = AttentionMechanism(self.encoder_state_size, self.hidden_size)

        self.reset_parameters()   # Uncomment this for TestAttention


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.Wx)
        nn.init.xavier_uniform_(self.Wh)
        self.b.data.fill_(0.01)

        nn.init.xavier_uniform_(self.Wo)
        self.bo.data.fill_(0.01)

    def forward(self, input_embeddings, init_decoder_state, all_encoder_outputs=None):
        """
        Run a vanilla RNN forward on an entire sequence of data. We assume an input
        sequence composed of T vectors, each of dimension D. The RNN uses a hidden
        size of H, and we work over a minibatch containing N sequences. When using
        attention, the attention mechanism takes in the (N x T x H) decoder hidden
        state and the encoder outputs, which is a (N x L x C) set of location
        vectors where L is the number of locations and C is the size of each
        location vector.

        Inputs:
            :param input_embeddings: Input data for the entire timeseries, of shape (N x T x D)
            :param init_decoder_state: Initial hidden state, of shape (N x 1 x H)
            :param all_encoder_hidden: Set of encoder outputs for attention mechanism, of shape (N x L x C)
        """
        outputs = None
        final_h = None
        attn = None
        N, T, D = input_embeddings.shape
        _, _, H = init_decoder_state.shape
        outputs = torch.zeros(N,T,H)
        prev_h = init_decoder_state.view(N, H)
    
        outputs[:,0,:] = prev_h

        for t in range(T):
            inner1 = input_embeddings[:, t, :] @ self.Wx 
            inner2 =  prev_h @ self.Wh  
            product = inner1 + inner2 + self.b
            next_h = torch.tanh(product)
            prev_h = next_h
            outputs[:, t, :] = next_h

        final_h = next_h.view(N, 1, H)

        if self.use_attention:
            outputs, attn = self.attention(outputs, all_encoder_outputs)

        outputs = outputs @ self.Wo + self.bo
        return outputs.data, final_h, attn

class LSTMDecoderSingle(nn.Module):
    """
    LSTM decoder for a single timestep.
    """
    def __init__(self,
                 embed_size: int,
                 hidden_size: int
                 ):
        """
        :param embed_size: Size of input embeddings.
        :param hidden_size: Size of hidden state (and cell state).
        """
        super(LSTMDecoderSingle, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        # The weights of an LSTM.
        self.Wx = nn.Parameter(torch.FloatTensor(torch.linspace(-2.1, 1.3, steps= 4 * embed_size * hidden_size).view(embed_size, 4 * hidden_size)))
        self.Wh = nn.Parameter(torch.FloatTensor(torch.linspace(-0.7, 2.2, steps= 4 * hidden_size * hidden_size).view(hidden_size, 4 * hidden_size)))
        self.b = nn.Parameter(torch.FloatTensor(torch.linspace(0.3, 0.7, steps= 4 * hidden_size)))

    def forward(self, input_embeddings, prev_decoder_state):
        """
        Forward pass for a single timestep of an LSTM. The input data has dimension D,
        the hidden state has dimension H, and we use a minibatch size of N.

        Inputs:
            :param input_embeddings: Input data for a single step of the timeseries of shape (N x D)
            :param prev_decoder_state: Tuple of initial hidden state of shape (N x H) and
                                        initial cell state of shape (N x H)

        Returns:
            :return next_h_c: Tuple of next hidden state of shape (N x H) and
                                next cell state of shape (N x H)
        """
        next_h_c = None
        #a^(t) = Wx * xt + Wh* ht−1 + b
        inner1 = input_embeddings @ self.Wx 
        inner2=  prev_decoder_state[0] @ self.Wh 

        activationVector = inner1 + inner2 + self.b
        H = self.hidden_size

        sigmoid = nn.Sigmoid()
        i_t = sigmoid(activationVector[:, 0*H:1*H])
        f_t = sigmoid(activationVector[:, 1*H:2*H])
        o_t = sigmoid(activationVector[:, 2*H:3*H])
        g_t = torch.tanh(activationVector[:, 3*H:4*H])

        # c_t = f_t * c_(t−1) + i_t * g_t 
        # h_t = o_t * tanh(c_t) 
        prev_c = prev_decoder_state[1]
        next_c = f_t * prev_c + i_t * g_t 
        next_h = o_t * torch.tanh(next_c)

        next_h_c = (next_h.data, next_c.data)
        return next_h_c


class LSTMDecoder(nn.Module):
    """
    Full LSTM decoder that operates on an entire input sequence.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_size: int,
                 encoder_state_size: int,
                 hidden_size: int,
                 decoder_out_dropout_prob: int = 0.0,
                 use_attention: bool = False
                 ):
        """
        :param vocab_size: Size of the output vocabulary.
        :param embed_size: Size of input embeddings.
        :param encoder_state_size: Size of vectors coming from the encoder.
        :param hidden_size: Size of hidden state (and cell state).
        :param decoder_out_dropout_prob: Dropout probability. Not required to use, but is useful for regularization.
        :param use_attention: Whether or not to use attention.
        """
        super(LSTMDecoder, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.encoder_state_size = encoder_state_size
        self.hidden_size = hidden_size
        self.use_attention = use_attention
        self.attention = None

        # The weights of a LSTM. 
        self.Wx = nn.Parameter(torch.FloatTensor(torch.linspace(-0.2, 0.9, steps= 4 * embed_size * hidden_size).view(embed_size, 4 * hidden_size)))
        self.Wh = nn.Parameter(torch.FloatTensor(torch.linspace(-0.3, 0.6, steps= 4* hidden_size * hidden_size).view(hidden_size, 4 * hidden_size)))
        self.b  = nn.Parameter(torch.FloatTensor(torch.linspace(0.2, 0.7, steps= 4 * hidden_size)))

        self.Wo = nn.Parameter(torch.zeros(self.hidden_size, self.vocab_size)) # Output layer weight
        self.bo = nn.Parameter(torch.zeros(self.vocab_size))  # Output layer bias


        if use_attention:
            self.attention = AttentionMechanism(self.encoder_state_size, self.hidden_size)

        self.reset_parameters()   # Uncomment this for TestAttention


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.Wx)
        nn.init.xavier_uniform_(self.Wh)
        self.b.data.fill_(0.01)

        nn.init.xavier_uniform_(self.Wo)
        self.bo.data.fill_(0.01)

    def forward(self, input_embeddings, init_decoder_state, all_encoder_hidden=None):
        """
        Forward pass for an LSTM over an entire sequence of data. We assume an input
        sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
        size of H, and we work over a minibatch containing N sequences. When using
        attention, the attention mechanism takes in the (N x T x H) decoder hidden
        state and the encoder outputs, which is a (N x L x C) set of location
        vectors where L is the number of locations and C is the size of each
        location vector.

        Inputs:
            :param input_embeddings: Input data for the entire timeseries, of shape (N x T x D)
            :param init_decoder_state: Initial hidden state, of shape (N x 1 x H)
            :param all_encoder_hidden: Set of encoder outputs for attention mechanism, of shape (N x L x C)
        """
        outputs = None
        final_h_c = None
        attn = None
        N, T, D = input_embeddings.shape
        _, _, H = init_decoder_state[0].shape

        outputs = torch.zeros(N, T, H)

        prev_h = init_decoder_state[0].view(N, H)
        prev_c = init_decoder_state[1].view(N, H) 
        prev_h_c = (prev_h, prev_c)

        for t in range(T):

            inner1 = input_embeddings[:, t, :] @ self.Wx  
            inner2=  prev_h_c[0] @ self.Wh 

            activationVector = inner1 + inner2 + self.b

            sigmoid = nn.Sigmoid()
            i_t = sigmoid(activationVector[:, 0*H:1*H])
            f_t = sigmoid(activationVector[:, 1*H:2*H])
            o_t = sigmoid(activationVector[:, 2*H:3*H])
            g_t = torch.tanh(activationVector[:, 3*H:4*H])

            prev_c = prev_h_c[1]
            next_c = f_t * prev_c + i_t * g_t 
            next_h = o_t * torch.tanh(next_c)
            prev_h_c = (next_h, next_c)

            outputs[:, t, :] = prev_h_c[0].data
        final_h_c = (prev_h_c[0].view(N,1,H), prev_h_c[1].view(N,1,H))

        if self.use_attention:
            outputs, attn = self.attention(outputs, all_encoder_hidden)  

        outputs = outputs @ self.Wo + self.bo
        return outputs.data, final_h_c, attn
