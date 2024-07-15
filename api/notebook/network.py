# Importing necessary libraries
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        # Define the feature extractor part of the network.
        self.features_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # First convolutional layer
            nn.ELU(),  # Activation function
            nn.MaxPool2d(kernel_size=2),  # Max pooling layer
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Second convolutional layer
            nn.ELU(),  # Activation function
            nn.MaxPool2d(kernel_size=2),  # Max pooling layer
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Third convolutional layer
            nn.ELU(),  # Activation function
            nn.MaxPool2d(kernel_size=2),  # Max pooling layer
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # Fourth convolutional layer
            nn.ELU(),  # Activation function
            nn.MaxPool2d(kernel_size=2),  # Max pooling layer
            nn.Flatten()  # Flatten the features to feed into the classifier
        )
        # Define the classifier part of the network.
        self.classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 1024),  # Fully connected layer
            nn.ELU(),  # Activation function
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(1024, 512),  # Fully connected layer
            nn.ELU(),  # Activation function
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(512, num_classes)  # Output layer
        )

    def forward(self, x):
        x = self.features_extractor(x)  # Pass input through feature extractor
        x = x.view(x.size(0), -1)  # Flatten the features for the classifier
        x = self.classifier(x)  # Pass the flattened features through the classifier
        return x  # Return the output
