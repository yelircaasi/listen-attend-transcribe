""" Define the network architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import numpy as np


class EncoderRNN(nn.Module):
    """
    A bidirectional RNN. It takes FBANK features and outputs the output state vectors of every time step.
    """

    def __init__(self, input_size, hidden_size, num_layers, drop_p):
        """
        Args:
            hidden_size (integer): Size of GRU cells.
            num_layers (integer): Number of GRU layers.
            drop_p (float): Probability to drop elements at Dropout layers.
        """
        super(EncoderRNN, self).__init__()
        self.embed = nn.Linear(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size,
                          hidden_size,
                          batch_first=True,
                          bidirectional=True,
                          num_layers=num_layers,
                          dropout=drop_p)
        # The initial state is a trainable vector.
        self.init_state = torch.nn.Parameter(
            torch.randn([2 * num_layers, 1, hidden_size]))

    def forward(self, xs, xlens):
        """
        We pack the padded sequences because it is especially important for bidirectional RNN to work properly. The RNN 
        in opposite direction can ignore the first few <PAD> tokens after packing.
        Args:
            xs (torch.FloatTensor, [batch_size, seq_length, dim_features]): A mini-batch of FBANK features.
            xlens (torch.LongTensor, [batch_size]): Sequence lengths before padding.
        Returns:
            outputs (PackedSequence): The packed output states.
        """
        batch_size = xs.shape[0]
        xs = self.embed(xs)
        xs = rnn_utils.pack_padded_sequence(xs,
                                            xlens,
                                            batch_first=True,
                                            enforce_sorted=False)
        outputs, _ = self.rnn(xs, self.init_state.repeat([1, batch_size, 1]))
        return outputs
