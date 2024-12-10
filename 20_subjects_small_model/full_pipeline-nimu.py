import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
import seaborn as sns

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
    
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    # batch: list of tuples (sequence, mask, label)
    sequences = [torch.tensor(item['features']) for item in batch]
    labels = torch.tensor([item['label'] for item in batch])
    masks = [torch.tensor(item['mask']) for item in batch]
    
    # Pad sequences
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    padded_masks = pad_sequence(masks, batch_first=True, padding_value=0)
    
    return padded_sequences, padded_masks, labels

from torch.utils.data import DataLoader

dataset = "./text_data/. "#YourCustomDataset
dataloader = DataLoader(
    dataset,
    batch_size=1,  # Start with 1 as per your setup
    shuffle=True,
    collate_fn=collate_fn
)

# Example using Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def generate_grad_cam(model, image, target_layer):
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=True)
    grayscale_cam = cam(input_tensor=image, targets=None)
    visualization = show_cam_on_image(image.cpu().numpy()[0], grayscale_cam[0], use_rgb=True)
    plt.imshow(visualization)
    plt.show()

def plot_attention_weights(attention_weights):
    plt.figure(figsize=(10, 6))
    sns.heatmap(attention_weights.detach().cpu().numpy(), annot=True, fmt=".2f")
    plt.xlabel('Visits')
    plt.ylabel('Batch')
    plt.show()
