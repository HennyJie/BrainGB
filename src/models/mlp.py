import torch


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, activation, n_classes=0):
        super(MLP, self).__init__()
        self.net = []
        self.net.append(torch.nn.Linear(input_dim, hidden_dim))
        self.net.append(activation())
        for _ in range(num_layers - 1):
            self.net.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.net.append(activation())
        self.net = torch.nn.Sequential(*self.net)
        self.shortcut = torch.nn.Linear(input_dim, hidden_dim)

        if n_classes != 0:
            self.classifier = torch.nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        out = self.net(x) + self.shortcut(x)
        if hasattr(self, 'classifier'):
            return out, self.classifier(out)
        return out