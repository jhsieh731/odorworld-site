import numpy as np
import torch.nn as nn
import torch
import csv

class AddGaussianNoise(object):
    """Add Gaussian noise to a tensor."""
    def __init__(self, mean=0.1, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Adds Gaussian noise to a tensor.
        """
        noise = np.random.randn(1781) * self.std + self.mean
        return tensor + noise

class SimpleNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, num_classes)  # Output logits, not probabilities

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_model(num_classes, encoding_type):
    n = 100 if encoding_type == 'N' else 500

    # load model
    model = SimpleNet(input_size=1781, num_classes=n)
    model_path = f'models/{encoding_type}_{n}_M{num_classes}.pt'
    model.load_state_dict(torch.load(model_path))
    print(f"Model loaded from {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model


def get_input(odors, dictionary):
    input = np.zeros(1781)
    for odor in odors:
        input += AddGaussianNoise()(torch.tensor(dictionary[odor])).numpy()
    input = torch.tensor(input).float().unsqueeze(0)
    return input


def get_pred(num_classes, encoding_type, odors):
    dictionary = {}
    data = []
    with open('static/dict.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
            dictionary[row['Name'].lower()] = np.array(eval(row['Chemicals']))

    if encoding_type == 'I':
        with open('static/I_dict.csv') as file:
            reader = csv.DictReader(file)
            for row in reader:
                key = list(dictionary.keys())[int(row['idx'])]
                dictionary[key] = np.array(eval(row['chems']))

    # get prediction odors
    model = get_model(num_classes, encoding_type)
    input = get_input(odors, dictionary)
    outputs = model(input).squeeze(1)

    # Convert outputs to predicted labels
    predicted_probs = torch.sigmoid(outputs)
    top_15_preds = predicted_probs.topk(15, dim=1).indices[0].tolist()
    probs = predicted_probs.topk(15, dim=1).values[0].tolist()
    preds = top_15_preds[:num_classes]
    predicted_odors = [list(dictionary.keys())[i] for i in preds]
    top_15 = [list(dictionary.keys())[i] for i in top_15_preds]
    categories = [list(data)[i]["Category"] for i in top_15_preds]
    print(categories)
    return (input, predicted_odors, top_15, probs, categories)
