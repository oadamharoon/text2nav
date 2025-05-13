import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MultiInputSingleOutputLSTM(nn.Module):
    def __init__(self, hidden_size=256, num_layers=4):
        super(MultiInputSingleOutputLSTM, self).__init__()
        
        # Pretrained CNN backbone (e.g., ResNet)
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # Remove FC layers
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = resnet.fc.in_features  # Get feature dimension
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # LSTM for sequence modeling
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True, dropout=0.2)
        self.lstm.flatten_parameters()

        # Fully connected layer for bbox prediction
        self.fc = nn.Linear(hidden_size, 4)  # 4 for bbox coordinates

    def forward(self, x):
        batch_size, seq_len, _, _, _ = x.size()

        # Extract features for each image in the sequence
        x = x.view(batch_size * seq_len, 3, x.size(-2), x.size(-1))  # Flatten sequences
        features = self.feature_extractor(x)  # Feature maps
        features = self.pool(features).view(batch_size, seq_len, -1)  # (B, T, feature_dim)

        # Pass through LSTM
        lstm_out, _ = self.lstm(features)  # (B, T, hidden_size)
        

        # Get output for the last time step
        final_out = lstm_out[:, -1, :]  # (B, hidden_size)

        # Predict bounding box
        bbox = self.fc(final_out)  # (B, 4)
        return bbox