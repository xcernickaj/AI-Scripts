import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


torch.manual_seed(42)


def prepare_data():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    return train_dataset, test_dataset


class MNISTClassifier(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[60, 60], num_classes=10):
        super(MNISTClassifier, self).__init__()


        layers = []
        prev_size = input_size

        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(0.2))
            prev_size = size

        layers.append(nn.Linear(prev_size, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)


def train_model(model, train_loader, test_loader, optimizer_type='sgd', learning_rate=0.01, momentum=0.9, epochs=20):

    criterion = nn.CrossEntropyLoss()


    if optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'sgd_momentum':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    train_losses, test_losses, accuracies = [], [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0


        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()


        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()


                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item() #.item converts the tensor value into a scalar integer value, so we can add it to the integer value


        train_losses.append(train_loss / len(train_loader))
        test_losses.append(test_loss / len(test_loader))
        accuracies.append(100 * correct / total)

        print(
            f'Epoch {epoch + 1}: Train Loss = {train_losses[-1]:.4f}, Test Loss = {test_losses[-1]:.4f}, Accuracy = {accuracies[-1]:.2f}%')

    return train_losses, test_losses, accuracies



def plot_confusion_matrix(model, test_loader):
    model.eval()
    confusion_matrix = np.zeros((10, 10), dtype=int)

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)

            for t, p in zip(target.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, str(confusion_matrix[i, j]),
                ha="center", va="center", color="black" if confusion_matrix[i, j] < confusion_matrix.max() / 2 else "white")

    plt.tight_layout()
    plt.show()



def main():

    batch_size = 64
    learning_rate = 0.01
    epochs = 10


    train_dataset, test_dataset = prepare_data()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #allows multiprocessing, creates batches and shuffles them
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    optimizers = ['sgd', 'sgd_momentum', 'adam']

    results = {}

    for opt in optimizers:
        print(f"\nTraining with {opt.upper()} optimizer")
        model = MNISTClassifier()
        train_losses, test_losses, accuracies = train_model(
            model,
            train_loader,
            test_loader,
            optimizer_type=opt,
            learning_rate=learning_rate,
            epochs=epochs
        )

        results[opt] = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'accuracies': accuracies,
            'model': model
        }


        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.title(f'{opt.upper()} - Training Loss')
        plt.plot(train_losses)
        plt.subplot(132)
        plt.title(f'{opt.upper()} - Testing Loss')
        plt.plot(test_losses)
        plt.subplot(133)
        plt.title(f'{opt.upper()} - Accuracy')
        plt.plot(accuracies)
        plt.tight_layout()
        plt.show()


    best_optimizer = max(results, key=lambda x: max(results[x]['accuracies']))
    best_model = results[best_optimizer]['model']

    print(f"The best optimizer was {best_optimizer}")

    plot_confusion_matrix(best_model, test_loader)


if __name__ == '__main__':
    main()
