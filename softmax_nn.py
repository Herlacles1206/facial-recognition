import torch
import torch.nn as nn
import torch.optim as optim

class SoftMax(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(SoftMax, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = nn.Sequential(
            nn.Linear(input_shape[0], 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

    def get_loss(self):
        return nn.CrossEntropyLoss()

    def get_optimizer(self):
        initial_learning_rate = 0.001
        # lr_schedule = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
        optimizer = optim.Adam(self.model.parameters(), lr=initial_learning_rate, betas=(0.9, 0.999), eps=1e-07, amsgrad=False)
        return optimizer