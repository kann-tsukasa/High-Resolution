import torch.nn as nn


class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        # print(recurrent.size())
        # print(len(_))
        # print(_[0].size())
        # print(_[1].size())
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output


# if __name__ == '__main__':
#     import torch
#     model = BidirectionalLSTM(input_size=100, hidden_size=50, output_size=20)
#     x = torch.randn(5, 10, 100)
#     print(model(x).size())
