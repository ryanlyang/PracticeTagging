"""

######################### ATLAS Top Tagging Open Data ##########################

evaluate_torch.py - Script for evaluating PyTorch taggers on test split data.
Model predictions are saved within the checkpoint directory.

For details of the data set and performance baselines, see:
                       https://cds.cern.ch/record/2825328

Modified for PyTorch models

################################################################################

"""

from pathlib import Path
import argparse

# Plotting imports
import numpy as np

# ML imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

################################## PARSE ####################################

# Create an argument parser
parser = argparse.ArgumentParser(description='Test a PyTorch tagger on the ATLAS Top Tagging Open Data set')
parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint .pt file')
parser.add_argument('--test_data', type=str, default='test_split/test_data.npz',
                    help='Path to the test data .npz file')
parser.add_argument('--type', type=str, help='Type of tagger to test',
                    choices=['dnn', 'efn', 'pfn'], default='efn')
parser.add_argument('--max_constits', type=int, default=40,
                    help='Max constituents used in training')

# Parse the command line arguments
args = parser.parse_args()

################################# SETTINGS #####################################


batch_size = 256

########################### Data Preparation ###################################
print("Loading test data...")


test_split = np.load(args.test_data)
test_data = test_split['data']
test_labels = test_split['labels']
test_weights = test_split.get('weights', np.ones(len(test_labels)))  # Default to ones if not present
test_jet_pt = test_split['pt']

print(f"Loaded {len(test_data)} test jets")


num_data_features = test_data.shape[-1]
max_constits = args.max_constits



def mlp(sizes, last_act=None, dropout=0.0):
    layers = []
    for i in range(len(sizes)-1):
        layers += [nn.Linear(sizes[i], sizes[i+1])]
        if i < (len(sizes)-2):
            layers += [nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(p=dropout)]
    if last_act is not None:
        layers += [last_act]
    return nn.Sequential(*layers)

class EFN(nn.Module):
    def __init__(
        self,
        input_dim=2,
        phi_layers=(350, 350, 350, 350, 350),
        f_layers=(300, 300, 300, 300, 300),
        latent_dropout=0.084,
        f_dropouts=0.036,
        output_dim=1,
    ):
        super().__init__()
        K = phi_layers[-1]
        self.phi = mlp([input_dim] + list(phi_layers), dropout=0.0)
        self.latent_dropout = nn.Dropout(p=latent_dropout) if latent_dropout > 0 else nn.Identity()
        self.F = mlp([K] + list(f_layers) + [output_dim], dropout=f_dropouts)

    def forward(self, x):
        angular_adjusted = self.phi(x[1])
        pT_reshaped = x[0].unsqueeze(-1)
        weighted = pT_reshaped * angular_adjusted
        summed = torch.sum(weighted, dim=1)
        dropout_output = self.latent_dropout(summed)
        f_output = self.F(dropout_output)
        output = torch.sigmoid(f_output)
        return output

class DNN(nn.Module):
    def __init__(self, input_dim, hidden_size=400, output_dim=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layers(x)
        x = self.sigmoid(x)
        return x

class PFN(nn.Module):
    def __init__(
        self,
        input_dim=7,
        phi_layers=(250, 250, 250, 250, 250),
        f_layers=(500, 500, 500, 500, 500),
        latent_dropout=0.072,
        f_dropouts=0.022,
        output_dim=1,
    ):
        super().__init__()
        K = phi_layers[-1]
        self.phi = mlp([input_dim] + list(phi_layers), dropout=0.0)
        self.latent_dropout = nn.Dropout(p=latent_dropout) if latent_dropout > 0 else nn.Identity()
        self.F = mlp([K] + list(f_layers) + [output_dim], dropout=f_dropouts)

    def forward(self, x):
        phi_x = self.phi(x)
        summed = torch.sum(phi_x, dim=1)
        dropout_output = self.latent_dropout(summed)
        f_output = self.F(dropout_output)
        output = torch.sigmoid(f_output)
        return output



class EFN_Dataset(Dataset):
    def __init__(self, pt, angular, labels):
        self.pt = torch.as_tensor(pt, dtype=torch.float32)
        self.angular = torch.as_tensor(angular, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return (self.pt[idx], self.angular[idx]), self.labels[idx]

class DNN_Dataset(Dataset):
    def __init__(self, combined, labels):
        self.combined = torch.as_tensor(combined, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.combined[idx], self.labels[idx]

class PFN_Dataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


print(f"Loading {args.type.upper()} model from {args.checkpoint}")


if args.type == 'efn':
    model = EFN()
    test_angular = test_data[:,:,0:2]
    test_pt = test_data[:,:,2]
    dataset = EFN_Dataset(test_pt, test_angular, test_labels)
elif args.type == 'dnn':
    test_data_flat = test_data.reshape(-1, max_constits * num_data_features)
    model = DNN(input_dim=max_constits * num_data_features)
    dataset = DNN_Dataset(test_data_flat, test_labels)
elif args.type == 'pfn':
    model = PFN()
    dataset = PFN_Dataset(test_data, test_labels)


device = torch.device('cpu')
model.load_state_dict(torch.load(args.checkpoint, map_location=device))
model = model.to(device)
model.eval()

print(f"Model loaded on {device}")


test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


print("Running inference...")
predictions_list = []
labels_list = []

with torch.no_grad():
    for inputs, labels in test_loader:
        if args.type == 'efn':
            pt, angular = inputs
            inputs = (pt.to(device), angular.to(device))
        else:
            inputs = inputs.to(device)

        outputs = model(inputs)
        predictions_list.append(outputs.cpu().numpy().squeeze())
        labels_list.append(labels.cpu().numpy())


predictions = np.concatenate(predictions_list)
labels = np.concatenate(labels_list)

print(f"Generated predictions for {len(predictions)} jets")




checkpoint_path = Path(args.checkpoint).parent
save_path = checkpoint_path / "public_test_nominal.npz"


shower_weights = np.ones((len(predictions), 27))  

save_dict = {
    'labels': labels,
    'predictions': predictions,
    'pt': test_jet_pt,
    'shower_weights': shower_weights
}

np.savez(save_path, **save_dict)
print(f"Saved predictions to {save_path}")
