import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchsummary import summary


# get parameterised model
def get_model(activation='ReLU', dropout_rate=0):
    # decide what activation function to use
    if activation == 'Tanh':
        act = nn.Tanh
    elif activation == 'Sigmoid':
        act = nn.Sigmoid
    elif activation == 'ELU':
        act = nn.ELU
    else:
        act = nn.ReLU
    # sequential model
    model = nn.Sequential(
        # 2D convolutional layer: kernel size 2x2, stride size 2x2, 32 output channels
        nn.Conv2d(1, 32, kernel_size=5, stride=1),
        # activation layer
        act(),
        # max pooling layer: kernel size 2x2, stride size 2x2
        nn.MaxPool2d(kernel_size=2, stride=2),
        # 2D convolutional layer: kernel size 5x5, stride size 1x1, 64 output channels
        nn.Conv2d(32, 64, kernel_size=5, stride=1),
        # activation layer
        act(),
        # max pooling layer: kernel size 2x2, stride size 2x2
        nn.MaxPool2d(kernel_size=2, stride=2),
        # flatten layer
        nn.Flatten(),
        # fully connected layer: input size 1024, output size 1024
        nn.Linear(1024, 1024),
        # activation layer
        act(),
        # fully connected layer: input size 1024, output size 256
        nn.Linear(1024, 256),
        # activation layer
        act(),
        # dropout layer
        nn.Dropout(dropout_rate),
        # fully connected layer: input size 256, output size 10 (i.e., number of label classes)
        nn.Linear(256, 10)
    )

    return model


# initialize weights using Xavier Uniform initialization
def weights_init(layer):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)


# evaluation function to calculate the model's accuracy
def evaluation(model, dataloader):
    n_total, n_correct = 0, 0
    model.eval()
    for data in dataloader:
        inputs, labels = data
        outputs = model(inputs)
        _, pred = torch.max(outputs.data, -1)
        n_total += labels.shape[0]
        n_correct += (pred == labels).sum().item()
    return 100 * n_correct / n_total  # convert to percentage


# train model
def train(model, optimiser, loss_fn, train_loader, test_loader, n_epochs):
    train_accuracy = []
    test_accuracy = []
    loss_epoch_array = []
    for epoch in range(n_epochs):
        loss_epoch = 0
        for data in train_loader:
            model.train()
            inputs, labels = data
            optimiser.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimiser.step()
            loss_epoch += loss.item()

        loss_epoch_array.append(loss_epoch)
        train_accuracy.append(evaluation(model, train_loader))
        test_accuracy.append(evaluation(model, test_loader))

        print(f"Epoch {epoch + 1}: "
              f"loss: {loss_epoch_array[-1]},"
              f" train accuracy: {train_accuracy[-1]},"
              f" test accuracy:{test_accuracy[-1]}")

    return train_accuracy, test_accuracy, loss_epoch_array


# run parameterised experiment
def run_experiment(activation='ReLU', dropout_rate=0, learning_rate=0.1, n_epochs=30):
    # get Fashion MNIST dataset and setup data loader for train set and test set
    train_set = torchvision.datasets.FashionMNIST(root=".", train=True, download=True, transform=transforms.ToTensor())
    test_set = torchvision.datasets.FashionMNIST(root=".", train=False, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

    # fix randomness seed for reproducibility
    torch.manual_seed(0)

    # get classifier
    classifier = get_model(activation, dropout_rate)

    # model summary
    summary(classifier, (1, 28, 28))

    # initialise weights using Xavier Uniform Initialisation
    classifier.apply(weights_init)

    # get optimiser
    optimiser = torch.optim.SGD(list(classifier.parameters()), lr=learning_rate, momentum=0.25)

    # use cross entropy loss function for classification task
    loss_fn = nn.CrossEntropyLoss()

    # train and test the model
    train_accuracy, test_accuracy, loss = train(
        classifier, optimiser, loss_fn, train_loader, test_loader, n_epochs
    )
    # plot and print results
    plt.plot(train_accuracy, label='Train Accuracy')
    plt.plot(test_accuracy, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.title(f'Activation: {activation}, Learning Rate: {learning_rate}, Dropout Rate: {dropout_rate}')
    plt.legend()
    plt.show()

    plt.plot(loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Activation: {activation}, Learning Rate: {learning_rate}, Dropout Rate: {dropout_rate}')
    plt.show()

    print(f'Activation: {activation}, Learning Rate: {learning_rate}, Dropout Rate: {dropout_rate}')
    print(f'Final Train Accuracy: {train_accuracy[-1]}, Final Test Accuracy: {test_accuracy[-1]}')
    print(f'Final Train Loss: {loss[-1]}')


if __name__ == "__main__":
    # different hyperparameters for comparison

    activation = 'ReLU'
    # activation = 'Tanh'
    # activation = 'Sigmoid'
    # activation = 'ELU'

    # dropout_rate = 0
    dropout_rate = 0.3

    # learning_rate = 0.001
    learning_rate = 0.1
    # learning_rate = 0.5
    # learning_rate = 1
    # learning_rate = 10

    n_epochs = 30

    run_experiment(activation, dropout_rate, learning_rate, n_epochs)
