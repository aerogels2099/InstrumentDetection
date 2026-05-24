import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, pool_size=(2, 2)):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, pool_size)
        return x


class InstrumentCNN14(nn.Module):
    """PANNs CNN14 backbone fine-tuned for multi-label instrument detection.

    Input: (B, 1, time_frames, mel_bins) — PANNs format.
    Pre-trained weights: https://zenodo.org/records/3987831 (CNN14_mAP=0.431.pth)
    """

    def __init__(self, num_classes=13, pretrained_path=None):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc = nn.Linear(2048, num_classes, bias=True)

        self._init_weights()
        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _load_pretrained(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)

        own_state = self.state_dict()
        loaded, skipped = {}, []
        for k, v in state_dict.items():
            if k in ("fc_audioset.weight", "fc_audioset.bias"):
                continue  # 527-class AudioSet head — skip
            if k in own_state and own_state[k].shape == v.shape:
                loaded[k] = v
            else:
                skipped.append(k)

        own_state.update(loaded)
        self.load_state_dict(own_state)
        print(f"Loaded {len(loaded)} / {len(own_state)} layers from PANNs checkpoint "
              f"({len(skipped)} skipped).")

    def forward(self, x):
        # x: (B, 1, time_frames, mel_bins)
        x = x.transpose(1, 3)   # (B, mel_bins, time_frames, 1)
        x = self.bn0(x)
        x = x.transpose(1, 3)   # (B, 1, time_frames, mel_bins)

        x = self.conv_block1(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1))
        x = F.dropout(x, p=0.2, training=self.training)

        # Average over the remaining mel dimension → (B, 2048, T')
        x = torch.mean(x, dim=3)

        # Global avg + max pooling over time → (B, 2048)
        x1 = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        x2 = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.fc(x)
