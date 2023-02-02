import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        layers = []
        layer_size = 3
        neuron_size = 128

        for i in range(layer_size):
            if i == 0:
                layer = nn.Linear(100, neuron_size)
            elif i == layer_size - 1:
                layer = nn.Linear(neuron_size, 1)
            else:
                layer = nn.Linear(neuron_size, neuron_size)

            layers.append(layer)

        self.module1 = nn.Sequential(*layers)

        # your code (layer architecture)
        ##########################################

        ##########################################

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        act_func = nn.ReLU()

        tmp = data

        for n, layer in enumerate(self.module1):
            if n == len(self.module1) - 1:
                tmp = layer(tmp)
                break
            tmp = act_func(layer(tmp))

        # your code (forward function implementation)
        ##########################################

        ##########################################

        return tmp

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return self.forward(data)


class CustomDataset(Dataset):
    def __init__(self, fname: str):
        df = np.loadtxt(fname, delimiter=",")
        self.x = df[:, :100]
        self.y = df[:, 100:]
        self.length = len(df)

    def __getitem__(self, index):
        x = torch.FloatTensor([self.x[index]])
        y = torch.FloatTensor([self.y[index]])
        return x, y

    def __len__(self):
        return self.length


def train(epochs: int, fname: str):
    dataset = CustomDataset(fname)
    num_train = int(0.8 * len(dataset))
    num_test = len(dataset) - num_train

    train_dataset, test_dataset = random_split(dataset, [num_train, num_test])

    train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=num_test, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Current device: {device}")
    model = MLP().to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        cost = 0.0

        for x, y in train_dataloader:
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cost += loss

        cost = cost / len(train_dataloader)

        with torch.no_grad():
            model.eval()
            for x, y in test_dataloader:
                x = x.to(device)
                y = y.to(device)

                output_test = model(x)
                cost_test = criterion(output_test, y)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1} | Training loss: {cost.item():.4f} | Validation loss: {cost_test.item():.4f}"
            )


if __name__ == "__main__":
    train(10000, "sample10000.csv")
