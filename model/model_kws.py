from utils.utils import *
from torch.quantization import QuantStub, DeQuantStub


class Attention(nn.Module):

    def __init__(self, hidden_size: int):
        super().__init__()

        self.energy = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input):
        energy = self.energy(input)
        alpha = torch.softmax(energy, dim=-2)
        return (input * alpha).sum(dim=-2)


class CRNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=config.cnn_out_channels,
                kernel_size=config.kernel_size, stride=config.stride
            ),
            nn.Flatten(start_dim=1, end_dim=2),
        )

        self.conv_out_frequency = (config.n_mels - config.kernel_size[0]) // \
                                  config.stride[0] + 1

        self.gru = nn.GRU(
            input_size=self.conv_out_frequency * config.cnn_out_channels,
            hidden_size=config.hidden_size,
            num_layers=config.gru_num_layers,
            dropout=config.dropout,
            bidirectional=config.bidirectional,
            batch_first=True
        )
        hidden_size = config.hidden_size * 2 if config.bidirectional else config.hidden_size
        self.attention = Attention(hidden_size)
        self.classifier = nn.Linear(hidden_size, config.num_classes)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, input_):
        input_ = self.quant(input_)
        input_ = input_.unsqueeze(dim=1)
        conv_output = self.conv(input_).transpose(-1, -2)
        conv_output = self.dequant(conv_output)

        gru_output, _ = self.gru(conv_output)
        context_vector = self.attention(gru_output)

        context_vector = self.quant(context_vector)
        output = self.classifier(context_vector)
        output = self.dequant(output)
        return output

    def fuse_model(self):
        for m in self.modules():
            #if type(m) == nn.Sequential:
            #    print(m)
            #    torch.quantization.fuse_modules(m, ['0', '1'], inplace=False)
            #if type(m) == Attention:
            #    torch.quantization.fuse_modules(m.energy, ['0', '1', '2'], inplace=True)
            if type(m) == nn.Linear:
                print(m)
                torch.quantization.fuse_modules(m, ['Linear'], inplace=True)
