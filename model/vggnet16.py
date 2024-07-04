import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils, models


class VggNetModel(nn.Module):
    def __init__(self, model):
        super(VggNetModel, self).__init__()
        new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        model.classifier = new_classifier
        self.vggNet16_layer = model
        self.layer2 = nn.Sequential(
            nn.Linear(4096, 2048, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, 17, bias=True),
        )

    def forward(self, x):
        feature = self.vggNet16_layer(x).view(x.size(0), -1)
        output = self.layer2(feature)
        return feature, output
