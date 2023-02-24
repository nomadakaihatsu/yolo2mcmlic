import torch.nn as nn

class MultiChannelMultiLabelImageCrassifer(nn.Module):
    def __init__(self,num_input_channels,num_output_labels):
        super(MultiChannelMultiLabelImageCrassifer, self).__init__()

        self.ConvLayer1 = nn.Sequential(
            # ref(H_out & W_out): https://pytorch.org/docs/stable/nn.html#conv2d
            nn.Conv2d(num_input_channels, 32, kernel_size=3,padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

        self.ConvLayer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3,padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

        self.ConvLayer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3,padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

        self.ConvLayer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3,padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            # nn.Dropout(0.2, inplace=False),
        )

        self.Linear1 = nn.Linear(256 * 14 * 14, 2048)
        self.Linear2 = nn.Linear(2048, 1024)
        self.Linear3 = nn.Linear(1024, 512)
        self.Linear4 = nn.Linear(512, num_output_labels)

    def forward(self, x):
        x = self.ConvLayer1(x)
        x = self.ConvLayer2(x)
        x = self.ConvLayer3(x)
        x = self.ConvLayer4(x)
        x = x.view(-1, 256 * 14 * 14)
        x = self.Linear1(x)
        x = self.Linear2(x)
        x = self.Linear3(x)
        x = self.Linear4(x)
        return x
