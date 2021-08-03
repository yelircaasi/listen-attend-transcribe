""" Define the network architecture.

For convenience, I annotate the dimensions of each objectː
    bː batch size
    hː hidden state dimension
    eː padded length of encoder states
    dː padded length of decoder states
    oː output size
    lː number of layers
    m: max length




#TODO: https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/blob/master/decode_beam.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import numpy as np

from mlgru import MultiLayerGRUCell

gpu = torch.cuda.is_available()


class PhoneDecoderRNN(nn.Module):
    """
    A decoder network which applies Luong attention (https://arxiv.org/abs/1508.04025).
    """

    def __init__(self, target_size, hidden_size, num_layers, drop_p,):
        """
        Args:
            target_size (integer): Size of the target vocabulary.
            hidden_size (integer): Size of GRU cells.
            num_layers (integer): Number of GRU layers.
            drop_p (float): Probability to drop elements at Dropout layers.
        """
        super(PhoneDecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.embed = nn.Embedding(target_size, hidden_size)
        self.cell = MultiLayerGRUCell(2 * hidden_size,
                                      hidden_size,
                                      num_layers=num_layers,
                                      drop_p=drop_p)
        # The initial states are trainable vectors.
        self.init_h = torch.nn.Parameter(
            torch.randn([num_layers, 1, hidden_size]))
        self.init_y = torch.nn.Parameter(torch.randn([1, hidden_size]))

        self.attn_W = nn.Linear(2 * hidden_size, hidden_size)
        self.attn_U = nn.Linear(hidden_size, hidden_size)
        self.attn_v = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(3 * hidden_size, hidden_size)
        self.drop = nn.Dropout(drop_p)
        self.classifier = nn.Linear(hidden_size, target_size)

    def forward(self, encoder_states, ground_truths=None):
        """
        The forwarding behavior depends on whether ground-truths are provided.
        Args:
            encoder_states (PackedSequence): 
                Packed output state vectors from the EncoderRNN.
            ground_truths (torch.LongTensor, [b, d]): 
                Padded ground-truths.
        Returns:
            * When ground-truths are provided, it returns cross-entropy loss. Otherwise it returns predicted word IDs
            and the attention weights.
            loss (float): 
                The cross-entropy loss to maximizing the probability of generating ground-truths.
            predictions (torch.FloatTensor, [b, m]): 
                The sentence generated by Greedy Search.
            all_attn_weights (torch.FloatTensor, [b, m, e]): 
                A list containing attention alignment weights for the predictions.
        """
        states, states_lengths = rnn_utils.pad_packed_sequence(
            encoder_states, batch_first=True)                # [b, e, 2h], [b]
        batch_size = states.shape[0]
        h = self.init_h.repeat([1, batch_size, 1])           # [l, b, h]
        y = self.init_y.repeat([batch_size, 1])              # [b, h]

        if ground_truths is None:
            all_attn_weights = []
            # The first predicted word is always <s> (ID=3).
            if gpu:
                predictions = [torch.full([batch_size], 3, dtype=torch.int64).cuda()]
            else:
                predictions = [torch.full([batch_size], 3, dtype=torch.int64)]
            
            # Unrolling the forward pass
            for time_step in range(100):                     # Empirically set max_length=100
                x = predictions[-1]                          # [b]
                x = self.embed(x)                            # [b, h]
                h = self.cell(torch.cat([y, x], dim=-1), h)  # [l, b, h]
                attns, attn_weights = self.apply_attn(
                    states, states_lengths, h[-1])           # [b, 2h], [b, e]
                y = torch.cat([attns, h[-1]], dim=-1)        # [b, 3h]
                y = F.relu(self.fc(y))                      # [b, h]
                all_attn_weights.append(attn_weights)
                
                # Output
                logits = self.classifier(y)                  # [b, o]
                # TODO: Beam Search to replace Greedy Search
                samples = torch.argmax(logits, dim=-1)       # [b]
                predictions.append(samples)
            all_attn_weights = torch.stack(all_attn_weights, dim=1)  # [b, p, T]
            predictions = torch.stack(predictions, dim=-1)   # [b, p]
            return predictions, all_attn_weights
        else:
            xs = self.embed(ground_truths[:, :-1])           # [b, d, h]
            outputs = []
            
            # Unrolling the forward pass
            for time_step in range(xs.shape[1]):
                h = self.cell(torch.cat([y, xs[:, time_step]], dim=-1), h)    # [n, b, h]
                attns, _ = self.apply_attn(states, states_lengths, h[-1])     # [b, 2h]
                y = torch.cat([attns, h[-1]], dim=-1)        # [b, 3h]
                y = F.relu(self.fc(y))                       # [b, h]
                outputs.append(y)

            # Output
            outputs = torch.stack(outputs, dim=1)            # [b, d, h]
            outputs = self.drop(outputs)                     # [b, d, h]   
            outputs = self.classifier(outputs)               # [b, d, o]

            # Compute loss
            mask = ground_truths[:, 1:].gt(0)                # [b, pd]
            loss = nn.CrossEntropyLoss()(outputs[mask], ground_truths[:, 1:][mask])    # [b, d]
            return loss

    def apply_attn(self, source_states, source_lengths, target_state):
        """
        Apply attention.
        Args:
            source_states (torch.FloatTensor, [b, e, 2h]):
                The padded encoder output states.
            source_lengths (torch.LongTensor, [b]): 
                The length of encoder output states before padding.
            target_state (torch.FloatTensor, [b, h]): 
                The decoder output state (of previous time step).
        Returns:
            attns (torch.FloatTensor, [batch_size, hidden_size]):
                The attention result (weighted sum of Encoder output states).
            attn_weights (torch.FloatTensor, [batch_size, padded_length_of_encoder_states]): 
                The attention alignment weights.
        """
        # A two-layer network used to project target state with each source state
        attns = self.attn_W(source_states) \
              + self.attn_U(target_state).unsqueeze(1)                 # [b, e, h]
        attns = self.attn_v(F.relu(attns)).squeeze(2)                  # [b, e]

        # Create a mask to ignore the encoder states with <PAD> tokens.
        mask = torch.arange(attns.shape[1]).unsqueeze(0).repeat(
            [attns.shape[0], 1]).ge(source_lengths.unsqueeze(1))       # [b, e]
        if gpu:
            attns = attns.masked_fill_(mask.cuda(), -float('inf'))     # [b, e]
        else:
            attns = attns.masked_fill_(mask, -float('inf'))            # [b, e]
        attns = F.softmax(attns, dim=-1)                               # [b, e]
        attn_weights = attns.clone()
        # weighted sum of source states
        attns = torch.sum(source_states * attns.unsqueeze(-1), dim=1)  # [b, 2h]
        return attns, attn_weights
    
    def beam_search(self):
        pass


class BinFeatDecoderRNN(PhoneDecoderRNN):
    """
    A decoder network which applies Luong attention (https://arxiv.org/abs/1508.04025).
    """

    def __init__(self, target_size, hidden_size, num_layers, drop_p):
        """
        Args:
            target_size (integer): Size of the target vocabulary.
            hidden_size (integer): Size of GRU cells.
            num_layers (integer): Number of GRU layers.
            drop_p (float): Probability to drop elements at Dropout layers.
        """
        super(BinFeatDecoderRNN, self).__init__(target_size, hidden_size, 
                                                num_layers, drop_p)

        self.hidden_size = hidden_size
        self.embed = nn.Embedding(target_size, hidden_size)
        self.cell = MultiLayerGRUCell(2 * hidden_size,
                                      hidden_size,
                                      num_layers=num_layers,
                                      drop_p=drop_p)
        # The initial states are trainable vectors.
        self.init_h = torch.nn.Parameter(
            torch.randn([num_layers, 1, hidden_size]))
        self.init_y = torch.nn.Parameter(torch.randn([1, hidden_size]))

        self.attn_W = nn.Linear(2 * hidden_size, hidden_size)
        self.attn_U = nn.Linear(hidden_size, hidden_size)
        self.attn_v = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(3 * hidden_size, hidden_size)
        self.drop = nn.Dropout(drop_p)
        self.classifier = nn.Linear(hidden_size, target_size)

    def forward(self, encoder_states, ground_truths=None):
        """
        The forwarding behavior depends on if ground-truths are provided.
        Args:
            encoder_states (PackedSequence): Packed output state vectors from the EncoderRNN.
            ground_truths (torch.LongTensor, [batch_size, padded_len_tgt]): Padded ground-truths.
        Returns:
            * When ground-truths are provided, it returns cross-entropy loss. Otherwise it returns predicted word IDs
            and the attention weights.
            loss (float): The cross-entropy loss to maximizing the probability of generating ground-truths.
            predictions (torch.FloatTensor, [batch_size, max_length]): The sentence generated by Greedy Search.
            all_attn_weights (torch.FloatTensor, [batch_size, max_length, length_of_encoder_states]): A list contains
                attention alignment weights for the predictions.
        """
        states, states_lengths = rnn_utils.pad_packed_sequence(
            encoder_states, batch_first=True)   # [batch_size, padded_len_src, 2 * hidden_size], [batch_size]
        batch_size = states.shape[0]
        # [num_layers, batch_size, hidden_size]
        h = self.init_h.repeat([1, batch_size, 1])
        # [batch_size, hidden_size]
        y = self.init_y.repeat([batch_size, 1])

        if ground_truths is None:
            all_attn_weights = []
            if gpu:
                # The first predicted word is always <s> (ID=3).
                predictions = [torch.full(
                    [batch_size], 3, dtype=torch.int64).cuda()]
            else:
                # The first predicted word is always <s> (ID=3).
                predictions = [torch.full([batch_size], 3, dtype=torch.int64)]
            # Unrolling the forward pass
            for time_step in range(100):   # Empirically set max_length=100
                x = predictions[-1]                           # [batch_size]
                # [batch_size, hidden_size]
                x = self.embed(x)
                # [num_layers, batch_size, hidden_size]
                h = self. cell(torch.cat([y, x], dim=-1), h)
                attns, attn_weights = self.apply_attn(
                    states, states_lengths, h[-1])            # [batch_size, 2 * hidden_size], [batch_size, length_of_encoder_states]
                # [batch_size, 3 * hidden_size]
                y = torch.cat([attns, h[-1]], dim=-1)
                # [batch_size, hidden_size]
                y = F.relu(self.fc(y))

                all_attn_weights.append(attn_weights)
                # Output
                # [batch_size, target_size]
                logits = self.classifier(y)
                # TODO: Adopt Beam Search to replace Greedy Search
                samples = torch.argmax(logits, dim=-1)        # [batch_size]
                predictions.append(samples)
            # [batch_size, max_length, length_of_encoder_states]
            all_attn_weights = torch.stack(all_attn_weights, dim=1)
            # [batch_size, max_length]
            predictions = torch.stack(predictions, dim=-1)
            return predictions, all_attn_weights
        else:
            # [batch_size, padded_len_tgt, hidden_size]
            xs = self.embed(ground_truths[:, :-1])
            outputs = []
            # Unrolling the forward pass
            for time_step in range(xs.shape[1]):
                # [num_layers, batch_size, hidden_size]
                h = self.cell(torch.cat([y, xs[:, time_step]], dim=-1), h)
                # [batch_size, 2 * hidden_size]
                attns, _ = self.apply_attn(states, states_lengths, h[-1])
                # [batch_size, 3 * hidden_size]
                y = torch.cat([attns, h[-1]], dim=-1)
                # [batch_size, hidden_size]
                y = F.relu(self.fc(y))
                outputs.append(y)

            # Output
            # [batch_size, padded_len_tgt, hidden_size]
            outputs = torch.stack(outputs, dim=1)
            outputs = self.drop(outputs)
            # [batch_size, padded_len_tgt, target_size]
            outputs = self.classifier(outputs)

            # Compute loss
            # [batch_size, padded_len_tgt]
            mask = ground_truths[:, 1:].gt(0)
            loss = nn.CrossEntropyLoss()(
                outputs[mask], ground_truths[:, 1:][mask])
            return loss

    def apply_attn(self, source_states, source_lengths, target_state):
        """
        Apply attention.
        Args:
            source_states (torch.FloatTensor, [batch_size, padded_length_of_encoder_states, 2 * hidden_size]):
                The padded encoder output states.
            source_lengths (torch.LongTensor, [batch_size]): The length of encoder output states before padding.
            target_state (torch.FloatTensor, [batch_size, hidden_size]): The decoder output state (of previous time step).
        Returns:
            attns (torch.FloatTensor, [batch_size, hidden_size]):
                The attention result (weighted sum of Encoder output states).
            attn_weights (torch.FloatTensor, [batch_size, padded_length_of_encoder_states]): The attention alignment weights.
        """
        # A two-layer network used to project every pair of [source_state, target_state].
        attns = self.attn_W(source_states) + \
            self.attn_U(target_state).unsqueeze(1)
        # [batch_size, padded_len_src, hidden_size]
        attns = self.attn_v(F.relu(attns)).squeeze(
            2)            # [batch_size, padded_len_src]

        # Create a mask with shape [batch_size, padded_len_src] to ignore the encoder states with <PAD> tokens.
        mask = torch.arange(attns.shape[1]).unsqueeze(0).repeat(
            [attns.shape[0], 1]).ge(source_lengths.unsqueeze(1))
        if gpu:
            # [batch_size, padded_len_src]
            attns = attns.masked_fill_(mask.cuda(), -float('inf'))
        else:
            # [batch_size, padded_len_src]
            attns = attns.masked_fill_(mask, -float('inf'))
        # [batch_size, padded_len_src]
        attns = F.softmax(attns, dim=-1)
        attn_weights = attns.clone()
        attns = torch.sum(source_states * attns.unsqueeze(-1),
                          dim=1)   # [batch_size, 2 * hidden_size]
        return attns, attn_weights


class ContFeatDecoderRNN(PhoneDecoderRNN):
    """
    A decoder network which applies Luong attention (https://arxiv.org/abs/1508.04025).
    """

    def __init__(self, target_size, hidden_size, num_layers, drop_p):
        """
        Args:
            target_size (integer): Size of the target vocabulary.
            hidden_size (integer): Size of GRU cells.
            num_layers (integer): Number of GRU layers.
            drop_p (float): Probability to drop elements at Dropout layers.
        """
        super(ContFeatDecoderRNN, self).__init__(target_size, hidden_size, 
                                                num_layers, drop_p)

        self.hidden_size = hidden_size
        self.embed = nn.Embedding(target_size, hidden_size)
        self.cell = MultiLayerGRUCell(2 * hidden_size,
                                      hidden_size,
                                      num_layers=num_layers,
                                      drop_p=drop_p)
        # The initial states are trainable vectors.
        self.init_h = torch.nn.Parameter(
            torch.randn([num_layers, 1, hidden_size]))
        self.init_y = torch.nn.Parameter(torch.randn([1, hidden_size]))

        self.attn_W = nn.Linear(2 * hidden_size, hidden_size)
        self.attn_U = nn.Linear(hidden_size, hidden_size)
        self.attn_v = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(3 * hidden_size, hidden_size)
        self.drop = nn.Dropout(drop_p)
        self.classifier = nn.Linear(hidden_size, target_size)

    def forward(self, encoder_states, ground_truths=None):
        """
        The forwarding behavior depends on if ground-truths are provided.
        Args:
            encoder_states (PackedSequence): Packed output state vectors from the EncoderRNN.
            ground_truths (torch.LongTensor, [batch_size, padded_len_tgt]): Padded ground-truths.
        Returns:
            * When ground-truths are provided, it returns cross-entropy loss. Otherwise it returns predicted word IDs
            and the attention weights.
            loss (float): The cross-entropy loss to maximizing the probability of generating ground-truths.
            predictions (torch.FloatTensor, [batch_size, max_length]): The sentence generated by Greedy Search.
            all_attn_weights (torch.FloatTensor, [batch_size, max_length, length_of_encoder_states]): A list contains
                attention alignment weights for the predictions.
        """
        states, states_lengths = rnn_utils.pad_packed_sequence(
            encoder_states, batch_first=True)   # [batch_size, padded_len_src, 2 * hidden_size], [batch_size]
        batch_size = states.shape[0]
        # [num_layers, batch_size, hidden_size]
        h = self.init_h.repeat([1, batch_size, 1])
        # [batch_size, hidden_size]
        y = self.init_y.repeat([batch_size, 1])

        if ground_truths is None:
            all_attn_weights = []
            if gpu:
                # The first predicted word is always <s> (ID=3).
                predictions = [torch.full(
                    [batch_size], 3, dtype=torch.int64).cuda()]
            else:
                # The first predicted word is always <s> (ID=3).
                predictions = [torch.full([batch_size], 3, dtype=torch.int64)]
            # Unrolling the forward pass
            for time_step in range(100):   # Empirically set max_length=100
                x = predictions[-1]                           # [batch_size]
                # [batch_size, hidden_size]
                x = self.embed(x)
                # [num_layers, batch_size, hidden_size]
                h = self. cell(torch.cat([y, x], dim=-1), h)
                attns, attn_weights = self.apply_attn(
                    states, states_lengths, h[-1])            # [batch_size, 2 * hidden_size], [batch_size, length_of_encoder_states]
                # [batch_size, 3 * hidden_size]
                y = torch.cat([attns, h[-1]], dim=-1)
                # [batch_size, hidden_size]
                y = F.relu(self.fc(y))

                all_attn_weights.append(attn_weights)
                # Output
                # [batch_size, target_size]
                logits = self.classifier(y)
                # TODO: Adopt Beam Search to replace Greedy Search
                samples = torch.argmax(logits, dim=-1)        # [batch_size]
                predictions.append(samples)
            # [batch_size, max_length, length_of_encoder_states]
            all_attn_weights = torch.stack(all_attn_weights, dim=1)
            # [batch_size, max_length]
            predictions = torch.stack(predictions, dim=-1)
            return predictions, all_attn_weights
        else:
            # [batch_size, padded_len_tgt, hidden_size]
            xs = self.embed(ground_truths[:, :-1])
            outputs = []
            # Unrolling the forward pass
            for time_step in range(xs.shape[1]):
                # [num_layers, batch_size, hidden_size]
                h = self.cell(torch.cat([y, xs[:, time_step]], dim=-1), h)
                # [batch_size, 2 * hidden_size]
                attns, _ = self.apply_attn(states, states_lengths, h[-1])
                # [batch_size, 3 * hidden_size]
                y = torch.cat([attns, h[-1]], dim=-1)
                # [batch_size, hidden_size]
                y = F.relu(self.fc(y))
                outputs.append(y)

            # Output
            # [batch_size, padded_len_tgt, hidden_size]
            outputs = torch.stack(outputs, dim=1)
            outputs = self.drop(outputs)
            # [batch_size, padded_len_tgt, target_size]
            outputs = self.classifier(outputs)

            # Compute loss
            # [batch_size, padded_len_tgt]
            mask = ground_truths[:, 1:].gt(0)
            loss = nn.MSELoss()(
                outputs[mask], ground_truths[:, 1:][mask])
            return loss

    def apply_attn(self, source_states, source_lengths, target_state):
        """
        Apply attention.
        Args:
            source_states (torch.FloatTensor, [batch_size, padded_length_of_encoder_states, 2 * hidden_size]):
                The padded encoder output states.
            source_lengths (torch.LongTensor, [batch_size]): The length of encoder output states before padding.
            target_state (torch.FloatTensor, [batch_size, hidden_size]): The decoder output state (of previous time step).
        Returns:
            attns (torch.FloatTensor, [batch_size, hidden_size]):
                The attention result (weighted sum of Encoder output states).
            attn_weights (torch.FloatTensor, [batch_size, padded_length_of_encoder_states]): The attention alignment weights.
        """
        # A two-layer network used to project every pair of [source_state, target_state].
        attns = self.attn_W(source_states) + \
            self.attn_U(target_state).unsqueeze(1)
        # [batch_size, padded_len_src, hidden_size]
        attns = self.attn_v(F.relu(attns)).squeeze(
            2)            # [batch_size, padded_len_src]

        # Create a mask with shape [batch_size, padded_len_src] to ignore the encoder states with <PAD> tokens.
        mask = torch.arange(attns.shape[1]).unsqueeze(0).repeat(
            [attns.shape[0], 1]).ge(source_lengths.unsqueeze(1))
        if gpu:
            # [batch_size, padded_len_src]
            attns = attns.masked_fill_(mask.cuda(), -float('inf'))
        else:
            # [batch_size, padded_len_src]
            attns = attns.masked_fill_(mask, -float('inf'))
        # [batch_size, padded_len_src]
        attns = F.softmax(attns, dim=-1)
        attn_weights = attns.clone()
        attns = torch.sum(source_states * attns.unsqueeze(-1),
                          dim=1)   # [batch_size, 2 * hidden_size]
        return attns, attn_weights
