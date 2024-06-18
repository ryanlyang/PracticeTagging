"""

######################### ATLAS Top Tagging Open Data ##########################

preprocessing.py - This script defines functions for dealing with the loading
of top tagger training and testing data from .h5 files, and the preprocessing
of that data into the format expected by the taggers.

For a description of the pre-processing and resulting distributions, see
TODO: Add link to the paper

Author: Kevin Greif
Last updated 4/21/2022
Written in python 3

################################################################################

"""

import numpy as np
import h5py


def constituent(data_dict, max_constits=200):
    """ constituent - This function applies a standard preprocessing to the
    jet data contained in train_dict. It will operate on the raw constituent
    level quantities and return 7 constituent level quantities which can be
    used for tagger training.

    Arguments:
    data_dict (dict of np arrays) - The python dictionary containing all of
    the constituent level quantities. Standard naming conventions will be
    assumed.
    max_constits (int) - The maximum number of constituents to consider in
    preprocessing. Cut jet constituents at this number. The maximum is 200,
    which is the maximum number of constituents stored in the .h5 files

    Returns:
    (np array) - The seven constituent level quantities, stacked along the last
    axis.
    """

    ############################## Load Data ###################################

    # Pull data from data dict
    pt = data_dict['fjet_clus_pt'][:,:max_constits]
    eta = data_dict['fjet_clus_eta'][:,:max_constits]
    phi = data_dict['fjet_clus_phi'][:,:max_constits]
    energy = data_dict['fjet_clus_E'][:,:max_constits]

    # Find location of zero pt entries in each jet. This will be used as a
    # mask to re-zero out entries after all preprocessing steps
    mask = np.asarray(pt == 0).nonzero()

    ########################## Angular Coordinates #############################

    print("Preprocessing angular coordinates...")
    print("Applying shift")

    # 1. Center hardest constituent in eta/phi plane. First find eta and
    # phi shifts to be applied
    eta_shift = eta[:,0]
    phi_shift = phi[:,0]

    # Apply them using np.newaxis
    eta -= eta_shift[:,np.newaxis]
    phi -= phi_shift[:,np.newaxis]

    # Fix discontinuity in phi at +/- pi using np.where
    phi = np.where(phi > np.pi, phi - 2*np.pi, phi)
    phi = np.where(phi < -np.pi, phi + 2*np.pi, phi)

    # 2. Rotate such that 2nd hardest constituent sits on negative phi axis
    print("Applying rotation")
    second_eta = eta[:,1]
    second_phi = phi[:,1]
    print("Calculating angle")
    alpha = np.arctan2(second_phi, second_eta) + np.pi/2
    print("Calculating eta")
    eta = (eta * np.cos(alpha[:,np.newaxis]) +
               phi * np.sin(alpha[:,np.newaxis]))
    print("Calculating phi")
    phi = (-eta * np.sin(alpha[:,np.newaxis]) +
               phi * np.cos(alpha[:,np.newaxis]))

    # 3. If needed, reflect so 3rd hardest constituent is in positive eta
    print("Applying flip")
    third_eta = eta[:,2]
    parity = np.where(third_eta < 0, -1, 1)
    eta = (eta * parity[:,np.newaxis]).astype(np.float32)
    # Cast to float32 needed to keep numpy from turning eta to double precision

    # 4. Calculate R with pre-processed eta/phi
    radius = np.sqrt(eta ** 2 + phi ** 2)

    ############################# pT and Energy ################################

    print("Preprocessing pT and energy...")

    # Take the logarithm, ignoring -infs which will be set to zero later
    log_pt = np.log(pt)
    log_energy = np.log(energy)

    # Sum pt and energy in each jet
    sum_pt = np.sum(pt, axis=1)
    sum_energy = np.sum(energy, axis=1)

    # Normalize pt and energy and again take logarithm
    lognorm_pt = np.log(pt / sum_pt[:,np.newaxis])
    lognorm_energy = np.log(energy / sum_energy[:,np.newaxis])

    ########################### Finalize and Return ############################

    print("Stacking preprocessed data...")

    # Reset all of the original zero entries to zero
    eta[mask] = 0
    phi[mask] = 0
    log_pt[mask] = 0
    log_energy[mask] = 0
    lognorm_pt[mask] = 0
    lognorm_energy[mask] = 0
    radius[mask] = 0

    # Stack along last axis
    features = [eta, phi, log_pt, log_energy,
                lognorm_pt, lognorm_energy, radius]
    stacked_data = np.stack(features, axis=-1)

    return stacked_data


def high_level(data_dict):
    """ high_level - This function "standardizes" each of the high level
    quantities contained in data_dict (subtract off mean and divide by
    standard deviation).

    Arguments:
    data_dict (dict of np arrays) - The python dictionary containing all of
    the high level quantities. No naming conventions assumed.

    Returns:
    (array) - The high level quantities, stacked along the last dimension.
    """

    # Empty list to accept pre-processed high level quantities
    features = []

    # Loop through quantities in data dict
    for quant in data_dict.values():

        # Some high level quantities have large orders of magnitude. Can divide
        # off these large exponents before evaluating mean and standard
        # deviation
        if 1e5 < quant.max() <= 1e11:
            # Quantity on scale TeV (sqrt{d12}, sqrt{d23}, ECF1, Qw)
            quant /= 1e6
        elif 1e11 < quant.max() <= 1e17:
            # Quantity on scale TeV^2 (ECF2)
            quant /= 1e12
        elif quant.max() > 1e17:
            # Quantity on scale TeV^3 (ECF3)
            quant /= 1e18

        # Calculated mean and standard deviation
        mean = quant.mean()
        stddev = quant.std()

        # Standardize and append to list
        standard_quant = (quant - mean) / stddev
        features.append(standard_quant)

    # Stack quantities and return
    stacked_data = np.stack(features, axis=-1)

    return stacked_data


def load_from_files(files, max_jets, get_hl=False, **kwargs):
    """ load_from_files - This function loops through a list of strings that 
    give the path to a set of .h5 files containing jet data. It will read the 
    jets from the .h5 files, and return either the constituent or high level
    data, along with the corresponding labels and training weights.

    Arguments:
    files (list of str) - The list of file paths to read jet data from.
    max_jets (int) - The maximum number of jets to read
    get_hl (bool) - If true, get the hl data instead of the constituent data

    Returns:
    data (np array) - The preprocessed jet data
    labels (np array) - The labels for each jet
    weights (np array) - The weights for each jet
    pt (np array) - The pT of each jet
    """

    # Define counter for number of jets read, a list fo accepting the
    # data, labels, weights, and pt
    jets_read = 0
    data_list = []
    label_list = []
    weight_list = []
    pt_list = []

    # Loop through list of files
    for fname in files:

        print("Loading data from file: ", fname)

        # Open file
        f = h5py.File(fname, 'r')

        # Find appropriate names of numpy arrays to read from file attributes
        if get_hl:
            data_vector_names = f.attrs.get('hl')
        else:
            data_vector_names = f.attrs.get('constit')

        # Load data into a python dictionary
        print("Loading data vectors: ", data_vector_names)
        data_dict = {}
        for key in data_vector_names:
            print("Loading key: ", key, " from file.")
            data_dict[key] = f[key][:,...]
        # data_dict = {key: f[key][:,...] for key in data_vector_names}

        # Preprocess data
        if get_hl:
            file_data = high_level(data_dict)
        else:
            file_data = constituent(data_dict, **kwargs)

        # Load labels, weights, and pt
        labels = f['labels'][:]
        weights = f['weights'][:]
        pt = f['fjet_pt'][:]

        # Truncate data, labels, and weights if necessary
        if jets_read + file_data.shape[0] > max_jets:
            file_data = file_data[:max_jets - jets_read]
            labels = labels[:max_jets - jets_read]
            weights = weights[:max_jets - jets_read]
            pt = pt[:max_jets - jets_read]
        
        # Append to lists
        data_list.append(file_data)
        label_list.append(labels)
        weight_list.append(weights)
        pt_list.append(pt)

        # Update counter
        jets_read += file_data.shape[0]

        # Close file
        f.close()

        # Break if we have read the maximum number of jets
        if jets_read >= max_jets:
            break

    # Concatenate lists of data, labels, and weights
    data = np.concatenate(data_list, axis=0)
    labels = np.concatenate(label_list, axis=0)
    weights = np.concatenate(weight_list, axis=0)
    pt = np.concatenate(pt_list, axis=0)

    # Return
    return data, labels, weights, pt