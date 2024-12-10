import torch
import torch.nn as nn
from torchvision import models

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetFeatureExtractor, self).__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        # Remove the final fully connected layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
    
    def forward(self, x):
        features = self.resnet(x)
        features = features.view(features.size(0), -1)  # Flatten
        return features

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
    
    def forward(self, hidden, encoder_outputs, mask):
        # hidden: [batch_size, hidden_size]
        # encoder_outputs: [batch_size, seq_len, hidden_size]
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, hidden_size]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch, seq, hidden]
        energy = energy.transpose(2, 1)  # [batch, hidden, seq]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [batch, 1, hidden]
        energy = torch.bmm(v, energy).squeeze(1)  # [batch, seq]
        energy = energy.masked_fill(mask == 0, -1e10)
        return torch.softmax(energy, dim=1)  # [batch, seq]

class LSTMAttentionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMAttentionClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x, mask):
        # x: [batch, seq_len, input_size]
        # mask: [batch, seq_len]
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_size*2]
        # Use the last hidden state as the query
        hidden = lstm_out[:, -1, :]  # [batch, hidden_size*2]
        attn_weights = self.attention(hidden, lstm_out, mask)  # [batch, seq_len]
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)  # [batch, hidden_size*2]
        out = self.fc(context)  # [batch, num_classes]
        return out
