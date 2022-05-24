from .model_asr import ResidualCNN, BidirectionalGRU
from .model_kws import Attention

import torch.nn as nn
import torch.nn.functional as F


class MultitaskModel(nn.Module):
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
        super(MultitaskModel, self).__init__()
        num_kws_classes = 2
        n_feats = n_feats // 2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3 // 2)  # cnn for extracting heirachal features

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats)
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats * 32, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i == 0 else rnn_dim * 2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i == 0)
            for i in range(n_rnn_layers)
        ])
        self.classifier_asr = nn.Sequential(
            nn.Linear(rnn_dim * 2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )
        self.attention = Attention(rnn_dim)
        self.classifier_kws = nn.Linear(rnn_dim, num_kws_classes)

    def forward_body(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2)  # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        return x

    def forward_kws_head(self, x):
        x = self.forward_body(x)
        x = self.attention(x)
        x = self.classifier_kws(x)
        return x

    def forward_asr_head(self, x):
        x = self.forward_body(x)
        x = self.classifier_asr(x)
        return x
