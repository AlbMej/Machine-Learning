import torch
import numpy as np
import unittest

from src.layers import RNNDecoderSingle, RNNDecoder, LSTMDecoderSingle, LSTMDecoder, AttentionMechanism


def rel_error(x, y):
    """ returns relative error """
    x = x.numpy()
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

class TestRNN(unittest.TestCase):

    def test_rnn_single(self):
        """
        N is the batch size
        D is the word embedding dimension
        H is the hidden state size
        """
        N, D, H = 3, 10, 4

        x = torch.FloatTensor(np.linspace(-0.4, 0.7, num=N * D).reshape(N, D))
        prev_h = torch.FloatTensor(np.linspace(-0.2, 0.5, num=N * H).reshape(N, H))

        expected_next_h = np.asarray([
            [-0.58172089, -0.50182032, -0.41232771, -0.31410098],
            [0.66854692, 0.79562378, 0.87755553, 0.92795967],
            [0.97934501, 0.99144213, 0.99646691, 0.99854353]])

        dec = RNNDecoderSingle(D, H)
        next_h = dec(x, prev_h)
        error = rel_error(next_h, expected_next_h)

        print("Single step vanilla RNN error: ", error)
        self.assertTrue(error >= 10**-8 and error <= 10**-7)


    def test_rnn_multi(self):
        """
        N is the batch size
        T is the max sentence length (all sequences in the batch are padded to this dimension)
        D is the word embedding dimension
        H is the hidden state size
        V is the vocabulary size
        C is the dimension of each location vector in the convolutional layer
        """
        N, T, D, H = 2, 3, 4, 5
        V, C = 10, 6  # V and C are here purely for understanding purposes and are not used in this test

        x = torch.FloatTensor(np.linspace(-0.1, 0.3, num=N * T * D).reshape(N, T, D))
        h0 = torch.FloatTensor(np.linspace(-0.3, 0.1, num=N * H).reshape(N, H)).unsqueeze(1)  # N x 1 x H
        encoder_states = None

        expected_all_h = np.asarray([
            [
                [-0.42070749, -0.27279261, -0.11074945, 0.05740409, 0.22236251],
                [-0.39525808, -0.22554661, -0.0409454, 0.14649412, 0.32397316],
                [-0.42305111, -0.24223728, -0.04287027, 0.15997045, 0.35014525],
            ],
            [
                [-0.55857474, -0.39065825, -0.19198182, 0.02378408, 0.23735671],
                [-0.27150199, -0.07088804, 0.13562939, 0.33099728, 0.50158768],
                [-0.51014825, -0.30524429, -0.06755202, 0.17806392, 0.40333043]
            ]
        ])

        dec = RNNDecoder(V, D, C, H, use_attention=False)
        all_h, _, _ = dec(x, h0, encoder_states)
        error = rel_error(all_h, expected_all_h)

        print("Multi-step vanilla RNN error: ", error)
        self.assertTrue(error >= 10**-8 and error < 7**-7)

class TestLSTM(unittest.TestCase):

    def test_lstm_single(self):
        """
        N is the batch size
        D is the word embedding dimension
        H is the hidden state size
        """
        N, D, H = 3, 4, 5
        x = torch.FloatTensor(np.linspace(-0.4, 1.2, num=N * D).reshape(N, D))
        prev_h = torch.FloatTensor(np.linspace(-0.3, 0.7, num=N * H).reshape(N, H))
        prev_c = torch.FloatTensor(np.linspace(-0.4, 0.9, num=N * H).reshape(N, H))

        expected_next_h = np.asarray([
            [0.24635157, 0.28610883, 0.32240467, 0.35525807, 0.38474904],
            [0.49223563, 0.55611431, 0.61507696, 0.66844003, 0.7159181],
            [0.56735664, 0.66310127, 0.74419266, 0.80889665, 0.858299]])
        expected_next_c = np.asarray([
            [0.32986176, 0.39145139, 0.451556, 0.51014116, 0.56717407],
            [0.66382255, 0.76674007, 0.87195994, 0.97902709, 1.08751345],
            [0.74192008, 0.90592151, 1.07717006, 1.25120233, 1.42395676]])

        dec = LSTMDecoderSingle(D, H)
        (next_h, next_c) = dec(x, (prev_h, prev_c))

        next_h_error = rel_error(next_h, expected_next_h)
        next_c_error = rel_error(next_c, expected_next_c)

        print('Single step LSTM next_h error: ', next_h_error)
        print('Single step LSTM next_c error: ', next_c_error)

        self.assertTrue(next_h_error >= 10**-8 and next_h_error <= 10**-7)
        self.assertTrue(next_c_error > 10**-9 and next_c_error < 10**-6)


    def test_lstm_multi(self):
        """
        N is the batch size
        T is the max sentene length (all sequences in the batch are padded to this dimension)
        D is the word embedding dimension
        H is the hidden state size
        V is the vocabulary size
        C is the dimension of each location vector in the convolutional layer
        """
        N, T, D, H = 2, 3, 5, 4
        V, C = 10, 6  # V and C are here purely for understanding purposes and are not used in this test

        x = torch.FloatTensor(np.linspace(-0.4, 0.6, num=N * T * D).reshape(N, T, D))
        h0 = torch.FloatTensor(np.linspace(-0.4, 0.8, num=N * H).reshape(N, H)).unsqueeze(1)
        c0 = torch.FloatTensor(np.zeros((N, H), dtype=np.float)).unsqueeze(1)
        encoder_states = None

        expected_h = np.asarray([
            [
                [0.01764008, 0.01823233, 0.01882671, 0.0194232],
                [0.11287491, 0.12146228, 0.13018446, 0.13902939],
                [0.31358768, 0.33338627, 0.35304453, 0.37250975]
            ],
            [
                [0.45767879, 0.4761092, 0.4936887, 0.51041945],
                [0.6704845, 0.69350089, 0.71486014, 0.7346449],
                [0.81733511, 0.83677871, 0.85403753, 0.86935314]
            ]
        ])

        dec = LSTMDecoder(V, D, C, H, use_attention=False)
        h, _, _ = dec(x, (h0, c0), encoder_states)

        print('Multi-step LSTM h error: ', rel_error(h, expected_h))

class TestAttention(unittest.TestCase):

    def test_attention(self):
        """
        N is the batch size
        T is the max sentence length (all sequences in the batch are padded to this dimension)
        C is the convolutional location vector dimension
        H is the hidden state size
        L is the number of locations from the convolutional layer, which is 196 (14x14 feature map)
        """
        N, C, H, L = 2, 3, 4, 5
        encoder_hidden = torch.FloatTensor(np.linspace(-0.4, 1.2, num=N * C * L).reshape(N, L, C))
        h = torch.FloatTensor(np.linspace(-0.3, 0.7, num=N * H).reshape(N, H)).unsqueeze(1)  # N x 1 x H, so T=1

        expected_attn_h = np.asarray([
            [[-0.00157962, -0.01475676, -0.02792878, -0.04109112]],
            [[0.37780291, 0.51978892, 0.63788921, 0.73206514]]
        ])

        expected_alpha = np.asarray([
            [[0.18871407, 0.19419549, 0.19983614, 0.20564061, 0.21161369]],
            [[0.06100287, 0.09884408, 0.16015883, 0.25950822, 0.42048603]]
        ])

        attn_mec = AttentionMechanism(C, H)
        attn_h, alpha_t = attn_mec(h, encoder_hidden)

        print("Attention context vector error: ", rel_error(attn_h, expected_attn_h))
        print("Attention scores error: ", rel_error(alpha_t, expected_alpha))


if __name__ == "__main__":
    with torch.no_grad():
        TestRNN().test_rnn_single()
        TestRNN().test_rnn_multi()
        TestLSTM().test_lstm_single()
        TestLSTM().test_lstm_multi()
        TestAttention().test_attention()
