import torch.nn as nn

class AutoencoderModel(nn.Module):
    def __init__(self, sizes):
        super().__init__()

        assert len(sizes) == 7, "Sizes must contain exactly 7 elements for the autoencoder architecture."
        assert sizes[0] == 784, "The first size must be 784."
        assert sizes[6] == 784, "The last size must be 784."

        if sizes is None:
            sizes = [784, 256, 32, 10, 32, 256, 784]

        self.encoder = nn.Sequential(
            nn.Linear(sizes[0], sizes[1]),
            nn.ReLU(),
            nn.Linear(sizes[1], sizes[2]),
            nn.ReLU(),
            nn.Linear(sizes[2], sizes[3]),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(sizes[3], sizes[4]),
            nn.ReLU(),
            nn.Linear(sizes[4], sizes[5]),
            nn.ReLU(),
            nn.Linear(sizes[5], sizes[6]),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encoder_forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.encoder(x)
        return x

    def decoder_forward(self, x):
        x = self.decoder(x)
        return x


