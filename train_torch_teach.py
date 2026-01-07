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
n_train_jets = 75000

# Set the fraction of the training data which will be reserved for validation
# After removing 15% for test, we split remaining 85% into ~82% train and ~18% val
# This gives us approximately 70/15/15 overall split
valid_fraction = 0.176  # 15/(100-15) = 0.176 to get 15% of original data

# Max constituents to consider in tagger training (must be <= 200)
max_constits = 40

# Tagger to train, supported options are 'dnn', 'efn', 'pfn', '2dcnn'.
tagger_type = 'efn'

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


class EFN_Dataset(Dataset):
    def __init__(self, pt, angular, labels, weights):
        self.pt = torch.as_tensor(pt, dtype=torch.float32)
        self.angular = torch.as_tensor(angular, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        self.weights = torch.as_tensor(weights, dtype=torch.float32)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return (self.pt[idx], self.angular[idx]), self.labels[idx], self.weights[idx]
    
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

    train_angular = train_data[:,:,0:2]
    train_pt = train_data[:,:,2]

    (train_angular, valid_angular, train_pt,
     valid_pt, train_labels, valid_labels,
     train_weights, valid_weights) = train_test_split(
        train_angular,
        train_pt,
        train_labels,
        train_weights,
        test_size=valid_fraction
    )
    dataset = EFN_Dataset(train_pt, train_angular, train_labels, train_weights)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = EFN_Dataset(valid_pt, valid_angular, valid_labels, valid_weights)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    efn_model = EFN()

    opt = torch.optim.Adam(efn_model.parameters(), lr=6.3e-5)

    criterion = nn.BCELoss(reduction='none')

    device = torch.device('cpu')
    efn_model = efn_model.to(device)

    # Setup checkpoint directory
    checkpoint_dir = Path().cwd() / "checkpoints" / "efn_smeared"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')


    for epoch in range(num_epochs):
        efn_model.train()
        total_right = 0
        total = 0

        running_train_loss = 0

        for inputs, labels, weights in train_loader:
            # inputs = inputs.to(device)
            pt, angular = inputs
            inputs = (pt.to(device), angular.to(device))
            labels = labels.to(device)
            weights = weights.to(device)

            forward_pass = efn_model.forward(inputs)

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

        for inputs, labels, weights in valid_loader:
            with torch.no_grad():
                # inputs = inputs.to(device)
                pt, angular = inputs
                inputs = (pt.to(device), angular.to(device))
                labels = labels.to(device)
                weights = weights.to(device)

                forward_pass = efn_model.forward(inputs)

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






######################
#DNN
######################


class DNN_Dataset(Dataset):
    def __init__(self, combined, labels, weights):
        self.combined = torch.as_tensor(combined, dtype=torch.float32)
        # self.angular = torch.as_tensor(angular, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        self.weights = torch.as_tensor(weights, dtype=torch.float32)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):

        return self.combined[idx], self.labels[idx], self.weights[idx]
    
    
# def mlp(sizes, last_act=None, dropout=0.0):
#     layers = []

#     for i in range(len(sizes)-1):

#         layers += [nn.Linear(sizes[i], sizes[i+1])]
#         if i < (len(sizes)-2):

#             layers += [nn.ReLU()]
#             if dropout > 0:

#                 layers += [nn.Dropout(p=dropout)]
#     if last_act is not None:
#         layers += [last_act]

#     return nn.Sequential(*layers)


class DNN(nn.Module):
    def __init__(
            

        self,
        input_dim,
        # dense_layers=400,
        # kernel_initializer='glorot_uniform',
        # kernel_regularizer=nn.regularizers.l1(l1=2e-4),
        hidden_size = 400,
        output_dim=1,
    ):
        
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
            # nn.BatchNorm1d(hidden_size)
            # nn.ReLU()
        )
        # self.fc1 = nn.Linear(input_dim, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc3 = nn.Linear(hidden_size, hidden_size)
        # self.fc4 = nn.Linear(hidden_size, hidden_size)
        # self.fc5 = nn.Linear(hidden_size, output_dim)

        # self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()

        # self.bn1 = nn.BatchNorm1d(hidden_size)
        

       


        # self.apply(self._init_weights)
        

    def forward(self, x):
        x = self.layers(x)
        x = self.sigmoid(x)

        return x
    

    

if tagger_type == 'dnn':

    # train_angular = train_data[:,:,0:2]
    # train_pt = train_data[:,:,2]
    train_data_flat = train_data.reshape(-1, max_constits * num_data_features)
    # train_combined = torch.outer(train_angular, train_pt).flatten()



    (train_data_fixed, valid_data_fixed, train_labels, valid_labels,
     train_weights, valid_weights) = train_test_split(
        train_data_flat,
        train_labels,
        train_weights,
        test_size=valid_fraction
    )

    dataset = DNN_Dataset(train_data_fixed, train_labels, train_weights)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = DNN_Dataset(valid_data_fixed, valid_labels, valid_weights)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    dnn_model = DNN(input_dim=max_constits * num_data_features)

    opt = torch.optim.Adam(dnn_model.parameters(), lr=1.2e-5)

    criterion = nn.BCELoss(reduction='none')

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    dnn_model = dnn_model.to(device)

    # Setup checkpoint directory
    checkpoint_dir = Path().cwd() / "checkpoints" / "dnn_smeared"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')


    for epoch in range(num_epochs):
        dnn_model.train()
        total_right = 0
        total = 0

        running_train_loss = 0

        for inputs, labels, weights in train_loader:
            # pt, angular = inputs
            inputs = inputs.to(device)
            labels = labels.to(device)
            weights = weights.to(device)

            forward_pass = dnn_model.forward(inputs)

            #Loss calculation

            new_forward_pass = forward_pass.squeeze(-1)

            loss_raw = criterion(new_forward_pass, labels)
            #L1 regularization here

            loss_weighted = loss_raw * weights

            loss = (loss_weighted).mean()

            l1_penalty = 0.0
            lambda_l1 = 2e-4

            for param in dnn_model.named_parameters():
                if 'weight' in param[0]:
                    abs_values = torch.abs(param[1])
                    l1_norm = torch.sum(abs_values)
                    l1_penalty += l1_norm
                    #penalize
                # else:
                #     #skip?
            loss += lambda_l1 * l1_penalty



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

        dnn_model.eval()

        total_right = 0
        total = 0
        running_val_loss = 0

        for inputs, labels, weights in valid_loader:
            with torch.no_grad():
                # pt, angular = inputs
                inputs = inputs.to(device)
                labels = labels.to(device)
                weights = weights.to(device)

                forward_pass = dnn_model.forward(inputs)

                new_forward_pass = forward_pass.squeeze(-1)

                loss_raw = criterion(new_forward_pass, labels)

                loss_weighted = loss_raw * weights

                loss = (loss_weighted).mean()

                l1_penalty = 0.0
                
                lambda_l1 = 2e-4

                for param in dnn_model.named_parameters():
                    if 'weight' in param[0]:
                        abs_values = torch.abs(param[1])
                        l1_norm = torch.sum(abs_values)
                        l1_penalty += l1_norm
                        #penalize
                    # else:
                    #     #skip?
                loss += lambda_l1 * l1_penalty

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
            torch.save(dnn_model.state_dict(), checkpoint_path)
            print(f"  → Saved checkpoint: {checkpoint_path} (val_loss: {val_loss_avg:.4f})")






##################
# PFN
##################



class PFN_Dataset(Dataset):
    def __init__(self, features, labels, weights):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        self.weights = torch.as_tensor(weights, dtype=torch.float32)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.weights[idx]
    


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


        # self.apply(self._init_weights)
        

    def forward(self, x):
        # angular_adjusted = self.phi(x[1])
        # pT_reshaped = x[0].unsqueeze(-1)
        # weighted = pT_reshaped * angular_adjusted
        phi_x = self.phi(x)
        summed = torch.sum(phi_x, dim=1)
        dropout_output = self.latent_dropout(summed)
        f_output = self.F(dropout_output)
        output = torch.sigmoid(f_output)
        return output
    

    

if tagger_type == 'pfn':

    # train_angular = train_data[:,:,0:2]
    # train_pt = train_data[:,:,2]
    

    (train_features, valid_features, train_labels, valid_labels,
     train_weights, valid_weights) = train_test_split(
        train_data,
        train_labels,
        train_weights,
        test_size=valid_fraction
    )
    dataset = PFN_Dataset(train_features, train_labels, train_weights)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = PFN_Dataset(valid_features, valid_labels, valid_weights)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    pfn_model = PFN()

    opt = torch.optim.Adam(pfn_model.parameters(), lr=7.9e-5)

    criterion = nn.BCELoss(reduction='none')

    device = torch.device('cpu')
    pfn_model = pfn_model.to(device)

    # Setup checkpoint directory
    checkpoint_dir = Path().cwd() / "checkpoints" / "pfn_smeared"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')


    for epoch in range(num_epochs):
        pfn_model.train()
        total_right = 0
        total = 0

        running_train_loss = 0

        for inputs, labels, weights in train_loader:
            # pt, angular = inputs
            inputs = inputs.to(device)
            labels = labels.to(device)
            weights = weights.to(device)

            forward_pass = pfn_model.forward(inputs)

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

        pfn_model.eval()

        total_right = 0
        total = 0
        running_val_loss = 0

        for inputs, labels, weights in valid_loader:
            with torch.no_grad():
                # pt, angular = inputs
                inputs = inputs.to(device)
                labels = labels.to(device)
                weights = weights.to(device)

                forward_pass = pfn_model.forward(inputs)

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
            torch.save(pfn_model.state_dict(), checkpoint_path)
            print(f"  → Saved checkpoint: {checkpoint_path} (val_loss: {val_loss_avg:.4f})")
















##################
# Jet Image CNN
##################

import numpy as np


def constituents_to_jet_image(constituents, img_size=32, eta_range=(-0.8, 0.8), phi_range=(-0.8, 0.8)):
    """
    Convert constituent list (eta, phi, pt, ...) into a 2D jet image
    """

    image = np.zeros((img_size, img_size), dtype=np.float32)

    eta_bins = np.linspace(eta_range[0], eta_range[1], img_size + 1)
    phi_bins = np.linspace(phi_range[0], phi_range[1], img_size + 1)


    for i in range(constituents.shape[0]):
        eta = constituents[i, 0]
        phi = constituents[i, 1]
        pt = constituents[i, 2]


        if pt == 0:
            continue



        eta_idx = -1
        for j in range(len(eta_bins) - 1):
            if eta >= eta_bins[j] and eta < eta_bins[j+1]:
                eta_idx = j
                break

        phi_idx = -1
        for k in range(len(phi_bins) - 1):
            if phi >= phi_bins[k] and phi < phi_bins[k+1]:
                phi_idx = k
                break

        if eta_idx >= 0 and eta_idx < img_size and phi_idx >= 0 and phi_idx < img_size:

            image[eta_idx, phi_idx] += pt


    # max_val = image.max()
    # if max_val > 0:
    #     image = image / max_val

    return image


class JetImage_Dataset(Dataset):
    def __init__(self, constituents, labels, weights, img_size=32):
        self.constituents = constituents
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        self.weights = torch.as_tensor(weights, dtype=torch.float32)
        self.img_size = img_size

  
        print("Converting constituents to jet images...")
        self.images = []
        for i in range(len(constituents)):
            img = constituents_to_jet_image(constituents[i], img_size=img_size)
            self.images.append(img)
            # if i % 10000 == 0:
            #     print(f"Processed {i}/{len(constituents)} jets")
        self.images = np.array(self.images)
        self.images = torch.as_tensor(self.images, dtype=torch.float32)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
     
        img = self.images[idx].unsqueeze(0) 
        return img, self.labels[idx], self.weights[idx]


class JetImageCNN(nn.Module):
    def __init__(self, img_size=32, num_filters=64, output_dim=1):
        super().__init__()

        self.conv1 = nn.Conv2d(1, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.pool1 = nn.MaxPool2d(2, 2)  

        self.conv2 = nn.Conv2d(num_filters, num_filters*2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters*2)
        self.pool2 = nn.MaxPool2d(2, 2)  

        # self.conv3 = nn.Conv2d(num_filters*2, num_filters*4, kernel_size=3, padding=1)
        # self.bn3 = nn.BatchNorm2d(num_filters*4)
        # self.pool3 = nn.MaxPool2d(2, 2)  

        self.relu = nn.ReLU()

        flattened_size = (img_size // 4) * (img_size // 4) * (num_filters * 2)


        self.fc1 = nn.Linear(flattened_size, 256)
        # self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        # self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, output_dim)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)


        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

 
        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.relu(x)
        # x = self.pool3(x)

    
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        # x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu(x)
        # x = self.dropout2(x)

        x = self.fc3(x)
        x = self.sigmoid(x)

        return x


if tagger_type == '2dcnn':

    img_size = 32
    # img_size = 64  

    (train_constits, valid_constits, train_labels, valid_labels,
     train_weights, valid_weights) = train_test_split(
        train_data,
        train_labels,
        train_weights,
        test_size=valid_fraction
    )

    dataset = JetImage_Dataset(train_constits, train_labels, train_weights, img_size=img_size)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = JetImage_Dataset(valid_constits, valid_labels, valid_weights, img_size=img_size)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    jetimage_model = JetImageCNN(img_size=img_size)

    opt = torch.optim.Adam(jetimage_model.parameters(), lr=1e-4)
    # opt = torch.optim.Adam(jetimage_model.parameters(), lr=5e-5)

    criterion = nn.BCELoss(reduction='none')

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    jetimage_model = jetimage_model.to(device)


    for epoch in range(num_epochs):
        jetimage_model.train()
        total_right = 0
        total = 0

        running_train_loss = 0

        for inputs, labels, weights in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            weights = weights.to(device)

            forward_pass = jetimage_model.forward(inputs)

         
            new_forward_pass = forward_pass.squeeze(-1)

            loss_raw = criterion(new_forward_pass, labels)

            loss_weighted = loss_raw * weights

            loss = (loss_weighted).mean()

            running_train_loss += loss.item()

           
            predictions = (new_forward_pass >= 0.5).float()

            num_right = (predictions == labels).sum().item()
            total_right += num_right
            total += len(predictions)

            opt.zero_grad()
            loss.backward()
            opt.step()

        # running_train_loss /= len(train_loader)
        train_accuracy = total_right / total

        jetimage_model.eval()

        total_right = 0
        total = 0
        running_val_loss = 0

        for inputs, labels, weights in valid_loader:
            with torch.no_grad():
                inputs = inputs.to(device)
                labels = labels.to(device)
                weights = weights.to(device)

                forward_pass = jetimage_model.forward(inputs)

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
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {running_train_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.4f} | Val Loss: {running_val_loss/len(valid_loader):.4f}, Val Acc: {val_accuracy:.4f}")