import torch
import torch.nn as nn
import torch.nn.functional as F

from librosa.filters import mel as librosa_mel_fn
from audio_processing import dynamic_range_compression
from audio_processing import dynamic_range_decompression
from stft import STFT

from pytorch_revgrad import RevGrad


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert(torch.min(y.data) >= -1)
        assert(torch.max(y.data) <= 1)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output

class ReferenceEncoder(nn.Module):
    '''
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, E]              Reference embedding
    '''
    def __init__(self, hp):

        super(ReferenceEncoder, self).__init__()
        K = len(hp.ref_enc_filters)
        filters = [1] + hp.ref_enc_filters
        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=hp.ref_enc_filters[i]) for i in range(K)])

        out_channels = self.calculate_channels(hp.n_mel_channels, 3, 2, 1, K)
        self.gru = nn.GRU(input_size=hp.ref_enc_filters[-1] * out_channels,
                          hidden_size=hp.feature_embedding_dim // 2,
                          batch_first=True)

        self.fc1 = nn.Linear(in_features=hp.feature_embedding_dim // 2, out_features=hp.feature_embedding_dim)
        self.fc2 = nn.Linear(in_features=hp.feature_embedding_dim, out_features=hp.feature_embedding_dim)
        self.fc3 = nn.Linear(in_features=hp.feature_embedding_dim, out_features=hp.feature_embedding_dim)

        self.hp = hp

    def forward(self, inputs):
        N = inputs.size(0)
        out = inputs.reshape(N, 1, -1, self.hp.n_mel_channels)  # [N, 1, Ty, n_mels]
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        self.gru.flatten_parameters()
        _, out = self.gru(out)  # out --- [1, N, E//2]

        out = out.squeeze(0) # out --- [N, E//2]
        out = F.relu(self.fc1(out)) # [N, E]
        out = F.relu(self.fc2(out)) # [N, E]
        out = torch.tanh(self.fc3(out)) # [N, E]
        # out = F.relu(self.fc3(out))

        return out

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L

class EDM(nn.Module):
    '''
    input --- [N, Ty, n_mels]
    output --- [N, E] intonation_embedding [N, hp.n_gender] [N, hp.n_intonation] intonation/gender classifier [N, hp.n_intonation] 
    '''
    def __init__(self, hp):
        super(EDM, self).__init__()
        self.style1_encoder = ReferenceEncoder(hp)
        self.style2_encoder = ReferenceEncoder(hp)

        self.style1_classifier = nn.Linear(hp.feature_embedding_dim, hp.n_gender)
        self.style2_classifier = nn.Linear(hp.feature_embedding_dim, hp.n_intonation)
        self.style2_adv = nn.Sequential(
                            RevGrad(),
                            nn.Linear(hp.feature_embedding_dim, hp.n_intonation)    
                            # self.style2_classifier           
                            )
    
    def forward(self, input):
        style1_emb = self.style1_encoder(input)
        style2_emb = self.style2_encoder(input)

        pred_style1 = self.style1_classifier(style1_emb)
        pred_style2 = self.style2_classifier(style2_emb)

        pred_style2_2 = self.style2_adv(style1_emb)

        return style1_emb, style2_emb, pred_style1, pred_style2, pred_style2_2

class ReferenceEncoder_64(nn.Module):
    '''
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, 32]              Reference embedding
    '''
    def __init__(self, hp):

        super(ReferenceEncoder_64, self).__init__()
        K = len(hp.ref_enc_filters)
        self.feature_embedding_dim = 64

        filters = [1] + hp.ref_enc_filters
        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=hp.ref_enc_filters[i]) for i in range(K)])

        out_channels = self.calculate_channels(hp.n_mel_channels, 3, 2, 1, K)
        self.gru = nn.GRU(input_size=hp.ref_enc_filters[-1] * out_channels,
                          hidden_size=self.feature_embedding_dim // 2,
                          batch_first=True)

        self.fc1 = nn.Linear(in_features=self.feature_embedding_dim // 2, out_features=self.feature_embedding_dim)
        self.fc2 = nn.Linear(in_features=self.feature_embedding_dim, out_features=self.feature_embedding_dim)
        self.fc3 = nn.Linear(in_features=self.feature_embedding_dim, out_features=self.feature_embedding_dim)

        self.hp = hp

    def forward(self, inputs):
        N = inputs.size(0)
        out = inputs.reshape(N, 1, -1, self.hp.n_mel_channels)  # [N, 1, Ty, n_mels]
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        self.gru.flatten_parameters()
        _, out = self.gru(out)  # out --- [1, N, E//2]

        out = out.squeeze(0) # out --- [N, E//2]
        out = F.relu(self.fc1(out)) # [N, E]
        out = F.relu(self.fc2(out)) # [N, E]
        out = torch.sigmoid(self.fc3(out)) # [N, 32]
        # out = F.relu(self.fc3(out))

        return out

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L

class EDM_64(nn.Module):
    '''
    input --- [N, Ty, n_mels]
    output --- [N, E] intonation_embedding [N, hp.n_gender] [N, hp.n_intonation] intonation/gender classifier [N, hp.n_intonation] 
    '''
    def __init__(self, hp):
        super(EDM_64, self).__init__()
        self.style1_encoder = ReferenceEncoder_64(hp)
        self.style2_encoder = ReferenceEncoder_64(hp)

        self.style1_classifier = nn.Linear(64, hp.n_gender)
        self.style2_classifier = nn.Linear(64, hp.n_intonation)
        # self.style2_adv = nn.Sequential(
        #                     RevGrad(),
        #                     nn.Linear(hp.feature_embedding_dim, hp.n_intonation)    
        #                     # self.style2_classifier           
        #                     )
    
    def forward(self, input):
        style1_emb = self.style1_encoder(input)
        style2_emb = self.style2_encoder(input)

        pred_style1 = self.style1_classifier(style1_emb)
        pred_style2 = self.style2_classifier(style2_emb)

        # pred_style2_2 = self.style2_adv(style1_emb)

        return style1_emb, style2_emb, pred_style1, pred_style2

class PreNet_E(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.5, fixed=False):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.p = dropout
        self.fixed = fixed

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, self.p, training=self.training or self.fixed)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, self.p, training=self.training or self.fixed)
        return x

class CBHG(nn.Module):
    def __init__(self, K, input_channels, channels, projection_channels, n_highways, highway_size, rnn_size,):
        super().__init__()

        self.conv_bank = nn.ModuleList(
            [
                BatchNormConv(input_channels, channels, kernel_size)
                for kernel_size in range(1, K + 1)
            ]
        )
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        self.conv_projections = nn.Sequential(
            BatchNormConv(K * channels, projection_channels, 3),
            BatchNormConv(projection_channels, input_channels, 3, relu=False),
        )

        self.project = (
            nn.Linear(input_channels, highway_size, bias=False)
            if input_channels != highway_size
            else None
        )

        self.highway = nn.Sequential(
            *[HighwayNetwork(highway_size) for _ in range(n_highways)]
        )

        self.rnn = nn.GRU(highway_size, rnn_size, batch_first=True, bidirectional=True)

    def forward(self, x):
        T = x.size(-1)
        residual = x

        x = [conv(x)[:, :, :T] for conv in self.conv_bank]
        x = torch.cat(x, dim=1)

        x = self.max_pool(x)

        x = self.conv_projections(x[:, :, :T])

        x = x + residual
        x = x.transpose(1, 2)

        if self.project is not None:
            x = self.project(x)

        x = self.highway(x)
        x, _ = self.rnn(x)
        return x

class BatchNormConv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, relu=True):
        super().__init__()
        self.conv = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bnorm = nn.BatchNorm1d(output_channels)
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x) if self.relu is True else x
        return self.bnorm(x)


class HighwayNetwork(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear1 = nn.Linear(size, size)
        self.linear2 = nn.Linear(size, size)
        nn.init.zeros_(self.linear1.bias)

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear2(x)
        g = torch.sigmoid(x2)
        return g * F.relu(x1) + (1.0 - g) * x

if __name__ == '__main__':
    import torch
    from hparams import hougen_hparams as hp
    
    sample = torch.rand([64,80,1089])

    model = EDM(hp)
    model(sample)
