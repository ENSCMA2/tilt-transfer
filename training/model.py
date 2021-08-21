import torch
import torch.nn as nn

from .embed_regularize import embedded_dropout
from .locked_dropout import LockedDropout
from .weight_drop import WeightDrop

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False, stack = False):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        if rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        elif rnn_type == 'QRNN':
            from torchqrnn import QRNNLayer
            self.rnns = [QRNNLayer(input_size=ninp if l == 0 else nhid, hidden_size=nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in range(nlayers)]
            for rnn in self.rnns:
                rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=wdrop)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights
        self.stack = stack
        self.W_y = nn.Linear(self.nhid, ninp)
        self.W_a = nn.Linear(nhid, 2)
        self.W_n = nn.Linear(nhid, 5)
        self.W_sh = nn.Linear (5, nhid)
        self.new_elt = None
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid ()
    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, mem = None, return_h=False):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            if not self.stack:
                raw_output, new_h = rnn(raw_output, hidden[l])
            else:
                h, c = hidden[l]
                wsh = self.W_sh (mem[0])
                wshview = wsh.view(1, 1, -1)
                hidden0_bar = wshview + h
                raw_output, new_h = rnn(raw_output, (hidden0_bar, c))
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden
        output = self.lockdrop(raw_output, self.dropout)
        if self.stack:
            output = self.sigmoid(self.W_y(output)).view(-1, self.ninp)
        print("output size:")
        print(output.size())
        print("raw output size:")
        print(raw_output.size())
        if self.stack:
            self.action_weights = self.softmax (self.W_a (output)).view(-1)
            ne1 = self.W_n(output)
            ne2 = self.sigmoid (ne1)
            self.new_elt = ne2.view(ne2.size()[1] * ne2.size()[0], 5)
            push_side = torch.cat ((self.new_elt, mem[:-1]), dim=0)
            pop_side = torch.cat ((mem[1:], torch.zeros(104, 5).to(device = "cuda:0")), dim=0)
            mem = self.action_weights [0] * push_side + self.action_weights [1] * pop_side
        outputs.append(output)

        result = output.view(output.size(0)*output.size(1), output.size(2))
        if return_h:
            if self.stack:
                return result, hidden, mem, raw_outputs, outputs
            return result, hidden, raw_outputs, outputs
        if self.stack:
            return result, hidden, mem
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
                    weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()
                    for l in range(self.nlayers)]

class SLSTM_Softmax (nn.Module):
    def __init__(self, hidden_dim, output_size, vocab_size, n_layers=1, memory_size=104, memory_dim = 5):
        super(SLSTM_Softmax, self).__init__()
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        
        self.lstm = nn.LSTM(self.vocab_size, self.hidden_dim, self.n_layers)

        self.W_y = nn.Linear(self.hidden_dim, output_size)
        self.W_n = nn.Linear(self.hidden_dim, self.memory_dim)
        self.W_a = nn.Linear(self.hidden_dim, 2)
        self.W_sh = nn.Linear (self.memory_dim, self.hidden_dim)
        
        # Actions -- push : 0 and pop: 1
        self.softmax = nn.Softmax(dim=2) 
        self.sigmoid = nn.Sigmoid ()
    
    def init_hidden (self):
        return (torch.zeros (self.n_layers, 1, self.hidden_dim).to(device = "cuda:0"),
                torch.zeros (self.n_layers, 1, self.hidden_dim).to(device = "cuda:0"))
    
    def forward(self, input, hidden0, stack, temperature=1.):
        h0, c0 = hidden0
        hidden0_bar = self.W_sh (stack[0]).view(1, 1, -1) + h0
        ht, hidden = self.lstm(input, (hidden0_bar, c0))
        output = self.sigmoid(self.W_y(ht)).view(-1, self.output_size)
        self.action_weights = self.softmax (self.W_a (ht)).view(-1)
        self.new_elt = self.sigmoid (self.W_n(ht)).view(1, self.memory_dim)
        push_side = torch.cat ((self.new_elt, stack[:-1]), dim=0)
        pop_side = torch.cat ((stack[1:], torch.zeros(1, self.memory_dim).to(device = "cuda:0")), dim=0)
        stack = self.action_weights [0] * push_side + self.action_weights [1] * pop_side
        return output, hidden, stack
