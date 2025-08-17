# models/audio_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioCNN(nn.Module):
    """
    Simple, robust CNN for log-mel inputs shaped (B, 1, 128, T).
    Uses adaptive pooling to guarantee a 2048-dim feature vector before the head,
    so the classifier is always Linear(2048, n_classes), regardless of T.
    """
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (B,32,128,T)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),            # (B,32,64,T/2)

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # (B,64,64,T/2)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),            # (B,64,32,T/4)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),# (B,128,32,T/4)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Force (freq_dim, time_dim) -> (1, 16) no matter the original T
        self.adapt = nn.AdaptiveAvgPool2d(output_size=(1, 16))  # (B,128,1,16)

        self.classifier = nn.Sequential(
            nn.Flatten(),              # (B, 128*1*16) = (B, 2048)
            nn.Dropout(p=0.3),
            nn.Linear(2048, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, 128, T) log-mel spectrogram
        returns: (B, n_classes) logits
        """
        x = self.features(x)
        x = self.adapt(x)     # -> (B,128,1,16)
        x = self.classifier(x)
        return x

