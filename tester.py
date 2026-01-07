from pathlib import Path
import argparse
import numpy as np

# Plotting imports
import matplotlib.pyplot as plt

# ML imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# from energyflow.archs import EFN, PFN
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from smearing import smear_dataset

# Custom imports
import utils

import copy



## Same setup as train.py

################################# SETTINGS #####################################
parser = argparse.ArgumentParser(description='Train top tagging model')
parser.add_argument('--train_path', type=str,
                    default="/home/ryan/ComputerScience/ATLAS/ATLAS-top-tagging-open-data/data",
                    help='Path to directory containing training data files')
args = parser.parse_args()

# Paths to data files. Point this to local directory containing the data files
# in sub-directories
train_path = Path(args.train_path)

# Make glob of the training set files
train_files = sorted(list(train_path.glob("*.h5")))

# Set the amount of data to be used in training. The full training
# set is very large (97 GB) and will not fit in memory all at once. Here, we
# take a subset  of the data. Using the full set will require piping.
n_train_jets = 30000

# Set the fraction of the training data which will be reserved for validation
# After removing 15% for test, we split remaining 85% into ~82% train and ~18% val
# This gives us approximately 70/15/15 overall split
valid_fraction = 0.176  # 15/(100-15) = 0.176 to get 15% of original data

# Max constituents to consider in tagger training (must be <= 200)
max_constits = 40

# Tagger to train, supported options are 'dnn', 'efn', 'pfn', '2dcnn'.
tagger_type = 'efn'
# tagger_type = 'none'

# Training parameters
num_epochs = 30
batch_size = 64

# Path for generated figures
figure_dir = Path().cwd() / "plots"
figure_dir.mkdir(parents=True, exist_ok=True)


########################### Data Preparation ###################################
print("Read data and prepare for tagger training")

# Load data using the functions in preprocessing.py
all_data, all_labels, all_weights, _, all_pt = utils.load_from_files(
    train_files,
    max_jets=n_train_jets,
    max_constits=max_constits,
    use_train_weights=False
)

print("Applying Gaussian smearing to data (10% eta/phi smear)")

smeared_data = copy.deepcopy(all_data)
smeared_data = smear_dataset(smeared_data, eta_smear_factor=0.15, phi_smear_factor=0.10)

# Split: 70% train, 15% validation (handled by train_test_split), 15% test
# First split off 15% for test, save it immediately, then delete from memory
test_fraction = 0.15
test_size = int(len(all_data) * test_fraction)
test_idx = len(all_data) - test_size

print(f"Data split: {test_idx} for train+val (will be split 70/15), {test_size} for test")

# Save test data for later evaluation
test_data_dir = Path().cwd() / "test_split"
test_data_dir.mkdir(exist_ok=True)
np.savez(test_data_dir / "test_data_smeared.npz",
         data=all_data[test_idx:],
         labels=all_labels[test_idx:],
         weights=all_weights[test_idx:],
         pt=all_pt[test_idx:])
print(f"Saved {test_size} test jets to {test_data_dir / 'test_data_smeared.npz'}")


train_data = all_data[:test_idx].copy()
train_labels = all_labels[:test_idx].copy()
train_weights = all_weights[:test_idx].copy()


del all_data, all_labels, all_weights, all_pt

print(f"Remaining {len(train_data)} jets will be split into train/validation by train_test_split")

# Find the number of data features
num_data_features = train_data.shape[-1]



####################### Build Tagger and Datasets  #############################
print("Building tagger and datasets")


# class EFN_Dataset(Dataset):
#     def __init__(self, pt, angular, labels, weights):
#         self.pt = torch.as_tensor(pt, dtype=torch.float32)
#         self.angular = torch.as_tensor(angular, dtype=torch.float32)
#         self.labels = torch.as_tensor(labels, dtype=torch.float32)
#         self.weights = torch.as_tensor(weights, dtype=torch.float32)

#     def __len__(self):
#         return self.labels.shape[0]

#     def __getitem__(self, idx):
#         return (self.pt[idx], self.angular[idx]), self.labels[idx], self.weights[idx]


class PairedEFN_Dataset(Dataset):
    """Dataset that pairs unsmeared and smeared data for student-teacher training"""
    def __init__(self, pt_unsmeared, angular_unsmeared, pt_smeared, angular_smeared, labels, weights):
        self.pt_unsmeared = torch.as_tensor(pt_unsmeared, dtype=torch.float32)
        self.angular_unsmeared = torch.as_tensor(angular_unsmeared, dtype=torch.float32)
        self.pt_smeared = torch.as_tensor(pt_smeared, dtype=torch.float32)
        self.angular_smeared = torch.as_tensor(angular_smeared, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        self.weights = torch.as_tensor(weights, dtype=torch.float32)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        # Return both unsmeared and smeared versions of the same jet
        unsmeared = (self.pt_unsmeared[idx], self.angular_unsmeared[idx])
        smeared = (self.pt_smeared[idx], self.angular_smeared[idx])
        return unsmeared, smeared, self.labels[idx], self.weights[idx]
    
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


        # self.apply(self._init_weights)
        

    def forward(self, x):
        angular_adjusted = self.phi(x[1])
        pT_reshaped = x[0].unsqueeze(-1)
        weighted = pT_reshaped * angular_adjusted
        summed = torch.sum(weighted, dim=1)
        dropout_output = self.latent_dropout(summed)
        f_output = self.F(dropout_output)
        output = torch.sigmoid(f_output)
        return output
    

    

if tagger_type == 'efn':

    # Extract smeared data for training (also need to split it the same way)
    smeared_train_data = smeared_data[:test_idx].copy()
    del smeared_data  # Free up memory

    # Prepare unsmeared data
    train_angular = train_data[:,:,0:2]
    train_pt = train_data[:,:,2]

    # Prepare smeared data
    smeared_train_angular = smeared_train_data[:,:,0:2]
    smeared_train_pt = smeared_train_data[:,:,2]

    # Split both unsmeared and smeared data together to ensure same indices
    (train_angular, valid_angular, train_pt, valid_pt,
     smeared_train_angular, smeared_valid_angular, smeared_train_pt, smeared_valid_pt,
     train_labels, valid_labels, train_weights, valid_weights) = train_test_split(
        train_angular,
        train_pt,
        smeared_train_angular,
        smeared_train_pt,
        train_labels,
        train_weights,
        test_size=valid_fraction
    )

    # Create paired datasets
    dataset = PairedEFN_Dataset(
        train_pt, train_angular,
        smeared_train_pt, smeared_train_angular,
        train_labels, train_weights
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = PairedEFN_Dataset(
        valid_pt, valid_angular,
        smeared_valid_pt, smeared_valid_angular,
        valid_labels, valid_weights
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    efn_unsmeared = EFN()

    efn_smeared = EFN()

    opt_unsmeared = torch.optim.Adam(efn_unsmeared.parameters(), lr=6.3e-5)

    opt_smeared = torch.optim.Adam(efn_smeared.parameters(), lr=6.3e-5)


    criterion_unsmeared = nn.BCELoss(reduction='none')

    criterion_smeared = nn.BCELoss(reduction='none')

    device = torch.device('cpu')
    efn_unsmeared = efn_unsmeared.to(device)
    efn_smeared = efn_smeared.to(device)

    # Setup checkpoint directory
    checkpoint_dir = Path().cwd() / "checkpoints" / "efn_teaching"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')


    for epoch in range(num_epochs):
        efn_unsmeared.train()
        efn_smeared.train()

        total_right_unsmeared = 0
        total_right_smeared = 0

        total = 0

        running_train_loss = 0

        for unsmeared, smeared, labels, weights in train_loader:
            # Unpack unsmeared and smeared inputs
            pt_unsmeared, angular_unsmeared = unsmeared
            pt_smeared, angular_smeared = smeared

            # Move to device
            inputs_unsmeared = (pt_unsmeared.to(device), angular_unsmeared.to(device))
            inputs_smeared = (pt_smeared.to(device), angular_smeared.to(device))
            labels = labels.to(device)
            weights = weights.to(device)

            # For now, just use unsmeared data (you can modify this for student-teacher training)
            forward_pass = efn_model.forward(inputs_unsmeared)

            #Loss calculation

            new_forward_pass = forward_pass.squeeze(-1)

            loss_raw = criterion(new_forward_pass, labels)


            loss_weighted = loss_raw * weights

            loss = (loss_weighted).mean()

            running_train_loss += loss.item()

                
            # Accuracy Calculation
            predictions = (new_forward_pass >= 0.5).float()

            num_right = (predictions == labels).sum().item()
            total_right += num_right
            total += len(predictions)


            opt.zero_grad()
            loss.backward()
            opt.step()

        # running_train_loss /= len(train_loader)
        train_accuracy = total_right / total
        # print(f"Train Epoch {epoch} Loss: {running_loss} Accuracy: {accuracy}")

        efn_model.eval()

        total_right = 0
        total = 0
        running_val_loss = 0

        for unsmeared, smeared, labels, weights in valid_loader:
            with torch.no_grad():
                # Unpack unsmeared and smeared inputs
                pt_unsmeared, angular_unsmeared = unsmeared
                pt_smeared, angular_smeared = smeared

                # Move to device
                inputs_unsmeared = (pt_unsmeared.to(device), angular_unsmeared.to(device))
                inputs_smeared = (pt_smeared.to(device), angular_smeared.to(device))
                labels = labels.to(device)
                weights = weights.to(device)

                # For now, just use unsmeared data (you can modify this for student-teacher training)
                forward_pass = efn_model.forward(inputs_unsmeared)

                new_forward_pass = forward_pass.squeeze(-1)

                loss_raw = criterion(new_forward_pass, labels)

                loss_weighted = loss_raw * weights

                loss = (loss_weighted).mean()

                running_val_loss += loss.item()



                predictions = (new_forward_pass >= 0.5).float()

                num_right = (predictions == labels).sum().item()
                total_right += num_right
                total += len(predictions)

        # running_val_loss /= len(train_loader)

        val_accuracy = total_right / total
        val_loss_avg = running_val_loss/len(valid_loader)
        # print(f"Val Loss: {running_loss} Accuracy: {accuracy}")
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {running_train_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.4f} | Val Loss: {val_loss_avg:.4f}, Val Acc: {val_accuracy:.4f}")

        # Save checkpoint if validation loss improved
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            checkpoint_path = checkpoint_dir / f"best_model.pt"
            torch.save(efn_model.state_dict(), checkpoint_path)
            print(f"  → Saved checkpoint: {checkpoint_path} (val_loss: {val_loss_avg:.4f})")
