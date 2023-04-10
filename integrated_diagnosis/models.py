import torch
import torch.nn as nn
import torchvision.models as models
from constants import temperature
import numpy as np


def logit_normalization(logits, temperature):
    norms = torch.norm(logits, p=2, dim=-1, keepdim=True) + 1e-7  # adding constant for stability
    return torch.div(logits, norms) / temperature  # normalized logits


class EfficientNetB2_and_LR(nn.Module):
    def __init__(self, pretrained=True, requires_grad=False):
        super(EfficientNetB2_and_LR, self).__init__()

        # Define the baseline model
        self.efficientNetB2 = models.efficientnet_b2(progress=True, pretrained=pretrained)

        if not requires_grad:
            for param in self.efficientNetB2.parameters():
                param.requires_grad = False
        elif requires_grad:
            for param in self.efficientNetB2.parameters():
                param.requires_grad = True

        num_features = self.efficientNetB2.classifier[1].in_features
        self.efficientNetB2.classifier[1] = nn.Linear(num_features, 2)

        self.efficientNetB2.load_state_dict(torch.load(f"C:\\Users\\AntonioM\\Desktop\\efficientNetB2\\es_model.pth"))

        # Define the LogisticRegression model
        self.logisticRegression = LogisticRegression(n_classes=2, n_features=10)
        self.logisticRegression.load_state_dict(torch.load('C:\\Users\\AntonioM\\Desktop\\LogisticRegression_EfficientNetB2_without_n_lesions\\es_model.pth'))

        for param in self.logisticRegression.parameters():
            param.requires_grad = False

        self.combinedOutput = nn.Linear(4, 2)
        self.combinedOutput.weight.data.uniform_(-0.001, 0.001)
        self.combinedOutput.bias.data.fill_(0.0)
        """
        self.combinedOutput.weight.data.fill_(0.001)  # set initial weights equal to 1 (same importance in both ways)
        #self.combinedOutput.weight.data.uniform_(-1e-4, 1e-4)
        #self.combinedOutput.weight.data.normal_(0.0, 1/np.sqrt(self.combinedOutput.in_features))
        #self.combinedOutput.bias.data.fill_(0.0001)  # ... no bias
        #self.combinedOutput.bias.data.uniform_(-1e-5, 1e-5)
        self.combinedOutput.bias.data.fill_(0.0)
        """

        for param in self.combinedOutput.parameters():
            param.requires_grad = True

    def forward(self, image_input, embedding_input):

        resnet_output = self.efficientNetB2(image_input)
        resnet_logits_normalized = logit_normalization(resnet_output, temperature)

        logistic_regression_output = self.logisticRegression(embedding_input)
        logistic_regression_logits_normalized = logit_normalization(logistic_regression_output, temperature)

        combined_input = torch.cat((resnet_logits_normalized, logistic_regression_logits_normalized), dim=1)
        combined_output = self.combinedOutput(combined_input)

        return resnet_output, logistic_regression_output, combined_output


def resnet18(pretrained=True, requires_grad=True):
    """
    :param pretrained: initial weights from the trained network
    :param requires_grad: (=True) means fine-tunning all network
    :return: model
    """
    model = models.resnet18(progress=True, pretrained=pretrained)

    if not requires_grad:
        for param in model.parameters():
            param.requires_grad = False
    elif requires_grad:
        for param in model.parameters():
            param.requires_grad = True

    # make the classification layer learnable
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 2)
    )

    # print(model)
    return model


def efficient_net_b2(pretrained=True, requires_grad=True):
    """
    :param pretrained: initial weights from the trained network
    :param requires_grad: (=True) means fine-tunning all network
    :return: model
    """
    model = models.efficientnet_b2(progress=True, pretrained=pretrained)

    if not requires_grad:
        for param in model.parameters():
            param.requires_grad = False
    elif requires_grad:
        for param in model.parameters():
            param.requires_grad = True

    # make the classification layer learnable
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 2)

    print(model)
    return model


class LogisticRegression(nn.Module):

    def __init__(self, n_classes, n_features, **kwargs):
        super().__init__()
        self.layer = nn.Linear(n_features, n_classes)

    def forward(self, x, **kwargs):
        outputs = self.layer(x)
        return outputs
