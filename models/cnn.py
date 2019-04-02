import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, config, weights):
        super(CNN, self).__init__()
        self.config = config

        V = config.EMBED_NUM
        D = config.EMBED_DIM
        C = config.CLASS_NUM
        Ci = 1
        Co = config.KERNEL_NUM
        Ks = config.CNN_KERNEL_SIZES

        self.embedding = nn.Embedding(V, D)
        self.embedding = self.embedding.from_pretrained(weights) if config.PRETRAINED else self.embedding

        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])

        self.drop_out = nn.Dropout(config.DROP_OUT)
        self.fc = nn.Linear(len(Ks) * Co, C)

    def forward(self, x):
        x = self.embedding(x)   # N, W, D

        x = x.unsqueeze(1)  # N, Ci=1, W, D
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]     # [(N, Co, W)] * k
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]      # [(N, Co] * k
        x = torch.cat(x, 1)

        x = self.drop_out(x)
        log = self.fc(x)
        return log
