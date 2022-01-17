# coding=utf-8
from __future__ import print_function, division

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import torch.nn.init as init
import torch.nn.utils.rnn as rnn_utils


class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        self.seq_model = SeqModel(args.input_size, args.seq_hidden_size, args.output_size, args.score_mode)
        self.dropout_layer = nn.Dropout(p=0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, fea, time, hidden=None):
        output, hidden = self.seq_model(fea, hidden)
        hidden = self.dropout_layer(hidden)
        output = self.softmax(output)
        return output, hidden


class SeqModel(nn.Module):

    def __init__(self, feature_size, seq_hidden_size, output_size, score_mode, num_layers=1):
        super(SeqModel, self).__init__()
        self.feature_size = feature_size
        self.seq_hidden_size = seq_hidden_size
        self.num_layers = num_layers
        self.score_mode = score_mode
        self.dropout_layer = nn.Dropout(p=0.2)
        self.rnn = nn.GRU(feature_size, seq_hidden_size, num_layers)
        self.hidden2out = nn.Linear(seq_hidden_size + feature_size, output_size)

    def forward(self, v, h):
        if h is None:
            h = self.default_hidden()
        pred_v = torch.cat([v, h.view(-1)])
        pred_v = self.hidden2out(pred_v.view(1, -1))
        _, h = self.rnn(v.view(1, 1, -1), h)
        return pred_v, h

    def default_hidden(self):
        return Variable(torch.zeros(self.num_layers, 1, self.seq_hidden_size))


class LSTMM(nn.Module):
    def __init__(self, args):
        super(LSTMM, self).__init__()
        self.args = args
        self.seq_model = SeqScoreModel(args.input_size, args.seq_hidden_size, args.output_size)
        self.dropout_layer = nn.Dropout(p=0.2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, feature, beta, time, hidden=None):
        output, hidden = self.seq_model(feature, hidden, beta)
        hidden = self.dropout_layer(hidden)
        output = output.mul(beta)
        output = self.softmax(output)
        return output, hidden


class SeqScoreModel(nn.Module):

    def __init__(self, feature_size, seq_hidden_size, output_size, num_layers=1):
        super(SeqScoreModel, self).__init__()
        self.feature_size = feature_size
        self.seq_hidden_size = seq_hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_layer = nn.Dropout(p=0.1)
        self.rnn = nn.GRU(feature_size + output_size, seq_hidden_size, num_layers)
        self.hidden2out = nn.Linear(seq_hidden_size + feature_size, output_size)

    def forward(self, fea, h, beta):
        if h is None:
            h = self.default_hidden()
        pred_v = torch.cat([fea, h.view(-1)])
        pred_v = self.hidden2out(pred_v.view(1, -1))
        x = torch.cat([fea, beta])
        _, h = self.rnn(x.view(1, 1, -1), h)
        return pred_v, h

    def default_hidden(self):
        return Variable(torch.zeros(self.num_layers, 1, self.seq_hidden_size))


class LSTMA(nn.Module):

    def __init__(self, args):
        super(LSTMA, self).__init__()
        self.args = args
        self.seq_model = AttnSeqModel(args.input_size, args.seq_hidden_size, args.output_size, args.k, args.score_mode, args.with_last)
        self.fs = None
        self.hs = None
        self.dropout_layer = nn.Dropout(p=0.2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, feature, time, hidden=None):
        output, h = self.seq_model(feature, hidden)
        h = self.dropout_layer(h)
        if hidden is None:
            hidden = h, feature, h
        else:
            _, fs, hs = hidden
            fs = torch.cat([fs, feature])
            hs = torch.cat([hs, h])

            hidden = h, fs, hs

        output = self.softmax(output)
        return output, hidden


class AttnSeqModel(nn.Module):

    def __init__(self, input_size, seq_hidden_size, output_size, k, score_mode, with_last, num_layers=1):
        super(AttnSeqModel, self).__init__()
        self.input_size = input_size
        self.seq_hidden_size = seq_hidden_size
        self.num_layers = num_layers
        self.score_mode = score_mode
        self.rnn = nn.GRU(input_size, seq_hidden_size, num_layers)
        self.with_last = with_last
        h_size = seq_hidden_size * 2 + 1 if with_last else seq_hidden_size
        self.hidden2out = nn.Linear(input_size + h_size, output_size)
        self.initial_h = nn.Parameter(torch.zeros(self.num_layers * self.seq_hidden_size))
        self.initial_h.data.uniform_(-1., 1.)
        self.k = k

    def forward(self, fea, hidden):  # feature_t, hidden_t-1
        if hidden is None:
            h = self.initial_h.view(self.num_layers, 1, self.seq_hidden_size)
            attn_h = self.initial_h
            length = Variable(torch.FloatTensor([0.]))
        else:
            h, fs, hs = hidden
            # print('start')
            # print("fs.size()=", fs.size())
            fs = fs.view(-1, fea.size(0))
            # print("fea.size()=", fea.size())
            # print("fea.view(-1, 1).size()=", fea.view(-1, 1).size())
            # print(torch.mm(fs, fea.view(-1, 1)).size())

            # print(hs)

            # calculate alpha using dot product
            alpha = torch.mm(fs, fea.view(-1, 1)).view(-1)
            # print("alpha.size()=", alpha.size())
            # print('end')
            # print(alpha.size())
            alpha, idx = alpha.topk(min(len(alpha), self.k), sorted=False)
            alpha = nn.functional.softmax(alpha.view(1, -1), dim=-1)

            length = Variable(torch.FloatTensor([alpha.size()[1]]))

            # flatten each h
            hs = hs.view(-1, self.num_layers * self.seq_hidden_size)
            attn_h = torch.mm(alpha, torch.index_select(hs, 0, idx)).view(-1)

        if self.with_last:
            pred_v = torch.cat([fea, attn_h, h.view(-1), length]).view(1, -1)
        else:
            pred_v = torch.cat([fea, attn_h]).view(1, -1)
        pred_v = self.hidden2out(pred_v.view(1, -1))

        x = fea

        _, h = self.rnn(x.view(1, 1, -1), h)
        return pred_v, h

    def default_hidden(self):
        return Variable(torch.zeros(self.num_layers, 1, self.seq_hidden_size))


class RADecay(nn.Module):

    def __init__(self, args):
        super(RADecay, self).__init__()
        self.args = args
        self.seq_model = AttnSeqTimeDecayModel(args.input_size, args.seq_hidden_size, args.output_size, args.k, args.with_last, args.exp_decay)
        self.fs = None
        self.hs = None
        self.dropout_layer = nn.Dropout(p=0.2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, feature, time, hidden=None):
        # topic rnn part size: (seq_len, bz) -> (bz, topic_size)
        # h = self.default_hidden(1)
        time = torch.tensor([time])
        output, h = self.seq_model(feature, time, hidden)
        h = self.dropout_layer(h)
        if hidden is None:
            hidden = feature, h, time
        else:
            fs, hs, ts = hidden
            fs = torch.cat([fs, feature])
            hs = torch.cat([hs, h])
            ts = torch.cat([ts, time])
            hidden = fs, hs, ts

        output = self.softmax(output)
        return output, hidden


class AttnSeqTimeDecayModel(nn.Module):
    """
    同AttnSeqModel，但增加了依照考试时间远近调整alpha
    """

    def __init__(self, input_size, seq_hidden_size, output_size, k, with_last, exp, num_layers=1):
        super(AttnSeqTimeDecayModel, self).__init__()
        self.input_size = input_size
        self.seq_hidden_size = seq_hidden_size
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size, seq_hidden_size, num_layers)
        self.with_last = with_last
        h_size = seq_hidden_size * 2 + 1 if with_last else seq_hidden_size
        self.hidden2out = nn.Linear(input_size + h_size, output_size)
        self.k = k
        self.exp = exp
        self.initial_h = Variable(torch.zeros(self.num_layers * self.seq_hidden_size), requires_grad=True)

    def forward(self, fea, time, hidden):
        if hidden is None:
            h = self.default_hidden()
            attn_h = self.initial_h
            length = Variable(torch.FloatTensor([0.]))
        else:
            fs, hs, ts = hidden
            h = hs[-1:]
            ts = time.expand_as(ts) - ts
            # calculate alpha using dot product
            fs = fs.view(-1, fea.size(0))
            alpha = torch.mm(fs, fea.view(-1, 1)).view(-1)
            alpha, idx = alpha.topk(min(len(alpha), self.k), sorted=False)
            alpha = alpha * (self.exp ** torch.index_select(ts, 0, idx))
            alpha = nn.functional.softmax(alpha.view(1, -1), dim=-1)

            length = Variable(torch.FloatTensor([alpha.size()[1]]))

            # flatten each h
            hs = hs.view(-1, self.num_layers * self.seq_hidden_size)
            attn_h = torch.mm(alpha, torch.index_select(hs, 0, idx)).view(-1)

        if self.with_last:
            pred_v = torch.cat([fea, attn_h, h.view(-1), length]).view(1, -1)
        else:
            pred_v = torch.cat([fea, attn_h]).view(1, -1)
        pred_v = self.hidden2out(pred_v)

        x = fea

        _, h = self.rnn(x.view(1, 1, -1), h)
        return pred_v, h

    def default_hidden(self):
        return Variable(torch.zeros(self.num_layers, 1, self.seq_hidden_size))


class LSTMMAD(nn.Module):

    def __init__(self, args):
        super(LSTMMAD, self).__init__()
        self.args = args
        self.seq_model = SeqScoreAttnDecayModel(args.input_size, args.seq_hidden_size, args.output_size, args.k, args.with_last, args.exp_decay)
        self.fs = None
        self.hs = None
        self.dropout_layer = nn.Dropout(p=0.2)
        self.softmax = nn.LogSoftmax(dim=1)
        # self.softmax = nn.Softmax(dim=1)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, feature, beta, time, hidden=None, father=None):
        # topic rnn part size: (seq_len, bz) -> (bz, topic_size)
        # h = self.default_hidden(1)
        time = torch.tensor([time]).to(self.device)
        if father is None:
            father = torch.tensor([0]).to(self.device)
        output, h = self.seq_model(feature, time, hidden, beta, father)
        h = self.dropout_layer(h)
        if hidden is None:
            hidden = feature, h, time
        else:
            fs, hs, ts = hidden
            fs = torch.cat([fs, feature])
            hs = torch.cat([hs, h])
            ts = torch.cat([ts, time])
            hidden = fs, hs, ts

        output = output.mul(beta)
        output = self.softmax(output)

        return output, hidden


class SeqScoreAttnDecayModel(nn.Module):

    def __init__(self, input_size, seq_hidden_size, output_size, k, with_last, exp, num_layers=1):
        super(SeqScoreAttnDecayModel, self).__init__()
        self.input_size = input_size
        self.seq_hidden_size = seq_hidden_size
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size + output_size + 1, seq_hidden_size, num_layers)
        # self.rnn = nn.GRU(input_size + output_size, seq_hidden_size, num_layers)
        self.with_last = with_last
        h_size = seq_hidden_size * 2 + 1 if with_last else seq_hidden_size
        self.hidden2out = nn.Linear(input_size + h_size, output_size)
        self.k = k
        self.exp = exp
        self.initial_h = Variable(torch.zeros(self.num_layers * self.seq_hidden_size), requires_grad=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, fea, time, hidden, beta, father):
        if hidden is None:
            h = self.default_hidden()
            attn_h = self.initial_h.to(self.device)
            length = Variable(torch.FloatTensor([0.])).to(self.device)
        else:
            fs, hs, ts = hidden
            h = hs[-1:]
            ts = time.expand_as(ts) - ts
            # calculate alpha using dot product
            fs = fs.view(-1, fea.size(0))
            alpha = torch.mm(fs, fea.view(-1, 1)).view(-1)
            alpha, idx = alpha.topk(min(len(alpha), self.k), sorted=False)
            alpha = alpha * (self.exp ** torch.index_select(ts, 0, idx))
            alpha = nn.functional.softmax(alpha.view(1, -1), dim=-1)

            length = Variable(torch.FloatTensor([alpha.size()[1]])).to(self.device)

            # flatten each h
            hs = hs.view(-1, self.num_layers * self.seq_hidden_size)
            attn_h = torch.mm(alpha, torch.index_select(hs, 0, idx)).view(-1)

        if self.with_last:
            pred_v = torch.cat([fea, attn_h, h.view(-1), length]).view(1, -1)
        else:
            pred_v = torch.cat([fea, attn_h]).view(1, -1)
        pred_v = self.hidden2out(pred_v)

        x = torch.cat([fea, beta, father])
        # x = torch.cat([fea, beta])

        _, h = self.rnn(x.view(1, 1, -1), h)
        return pred_v, h

    def default_hidden(self):
        return Variable(torch.zeros(self.num_layers, 1, self.seq_hidden_size)).to(self.device)


