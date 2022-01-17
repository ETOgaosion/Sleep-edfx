# coding=utf-8

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import torch.nn.init as init
import torch.nn.utils.rnn as rnn_utils


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.hidden2out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)
        self.dropout_layer = nn.Dropout(p=0.2)

    def forward(self, fea):
        # fea_pad, fea_len = rnn_utils.pad_packed_sequence(fea, batch_first=True)
        packed_output, packed_len = self.lstm(fea)
        output, output_len = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)   # [16,2883,64]
        output = self.dropout_layer(output)
        output = self.hidden2out(output)   # [16,2883,5]
        output = self.softmax(output)
        return output, output_len



        # fea_len = fea_len - 1
        # for fea_idx in range(len(fea_len)):
        #     output = outputs[fea_idx, 0:fea_len[fea_idx], :]
        #     output = self.dropout_layer(output)
        #     output = self.hidden2out(output)
        #     # output = self.softmax(output, 3)
        #     pred.append(output)
        # return pred

    # def init_state(self, batch_size=1):
    #     return(
    #         autograd.Variable(torch.zeros(1, batch_size, self.hidden_size)),
    #         autograd.Variable(torch.zeros(1, batch_size, self.hidden_size))
    #     )