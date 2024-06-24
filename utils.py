"""

######################### ATLAS Top Tagging Open Data ##########################

utils.py - This script defines functions for dealing with the loading
of top tagger training and testing data from .h5 files, and the preprocessing
of that data into the format expected by the taggers.

Author: Kevin Greif
Last updated 6/24/2024
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
    second_eta = eta[:,1]
    second_phi = phi[:,1]
    alpha = np.arctan2(second_phi, second_eta) + np.pi/2
    eta_rot = (eta * np.cos(alpha[:,np.newaxis]) +
               phi * np.sin(alpha[:,np.newaxis]))
    phi_rot = (-eta * np.sin(alpha[:,np.newaxis]) +
               phi * np.cos(alpha[:,np.newaxis]))
    eta = eta_rot
    phi = phi_rot

    # 3. If needed, reflect so 3rd hardest constituent is in positive eta
    third_eta = eta[:,2]
    parity = np.where(third_eta < 0, -1, 1)
    eta = (eta * parity[:,np.newaxis]).astype(np.float32)
    # Cast to float32 needed to keep numpy from turning eta to double precision

    # 4. Calculate R with pre-processed eta/phi
    radius = np.sqrt(eta ** 2 + phi ** 2)

    ############################# pT and Energy ################################

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


def load_from_files(files, max_jets=None, get_hl=False, use_train_weights=True, use_shower_weights=False, use_numbers=False, **kwargs):
    """ load_from_files - This function loops through a list of strings that 
    give the path to a set of .h5 files containing jet data. It will read the 
    jets from the .h5 files, and return either the constituent or high level
    data, along with the corresponding labels and training weights.

    Arguments:
    files (list of str) - The list of file paths to read jet data from.
    max_jets (int) - The maximum number of jets to read, if left to None use all
    use_train_weights (bool) - If false, don't load training weights and return vector of 1's
    use_shower_weights (bool) - If false, don't load shower weights and return vector of 1's
    use_numbers (bool) - If true, return the event numbers for each jet
    get_hl (bool) - If true, get the hl data instead of the constituent data

    Returns:
    data (np array) - The preprocessed jet data
    labels (np array) - The labels for each jet
    train_weights (np array) - The training weights for each jet
    shower_weights (np array) - The shower weights for each jet
    numbers (np array) - The event numbers for each jet
    pt (np array) - The pT of each jet
    """

    # Set max jets to inf if not specified
    if max_jets is None:
        max_jets = np.inf

    # Define counter for number of jets read, a list fo accepting the
    # data, labels, weights, and pt
    jets_read = 0
    data_list = []
    label_list = []
    train_weight_list = []
    shower_weight_list = []
    number_list = []
    pt_list = []

    # Loop through list of files
    for fname in files:

        # Open file
        f = h5py.File(fname, 'r')

        # Find appropriate names of numpy arrays to read from file attributes
        if get_hl:
            data_vector_names = f.attrs.get('hl')
        else:
            if 'constit' in f.keys():
                data_vector_names = ['constit']
            else:
                data_vector_names = f.attrs.get('constit')

        # Load data into a python dictionary
        data_dict = {key: f[key][:,...] for key in data_vector_names}

        # Preprocess data
        if get_hl:
            file_data = high_level(data_dict)
        else:
            if 'constit' in f.keys():
                file_data = data_dict['constit']
            else:
                file_data = constituent(data_dict, **kwargs)

        # Load labels and pt
        labels = f['labels'][:max_jets]
        pt = f['fjet_pt'][:max_jets]

        # Load weights
        if use_train_weights:
            train_weights = f['training_weights'][:]
        else:
            train_weights = np.ones(labels.shape)
        if use_shower_weights:
            shower_weights = f['EventInfo_mcEventWeights'][:,:]
        else:
            shower_weights = np.ones(labels.shape)

        # Load numbers
        if use_numbers:
            numbers = f['EventInfo_mcEventNumber'][:max_jets]
        else:
            numbers = np.zeros(labels.shape)

        # Truncate data, labels, and weights if necessary
        if jets_read + file_data.shape[0] > max_jets:
            file_data = file_data[:max_jets - jets_read]
            labels = labels[:max_jets - jets_read]
            train_weights = train_weights[:max_jets - jets_read]
            shower_weights = shower_weights[:max_jets - jets_read]
            numbers = numbers[:max_jets - jets_read]
            pt = pt[:max_jets - jets_read]
        
        # Append to lists
        data_list.append(file_data)
        label_list.append(labels)
        train_weight_list.append(train_weights)
        shower_weight_list.append(shower_weights)
        number_list.append(numbers)
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
    train_weights = np.concatenate(train_weight_list, axis=0)
    shower_weights = np.concatenate(shower_weight_list, axis=0)
    numbers = np.concatenate(number_list, axis=0)
    pt = np.concatenate(pt_list, axis=0)

    # Return
    return data, labels, train_weights, shower_weights, pt, numbers


def isin_tolerance(A, B, tol):
    A = np.asarray(A)
    B = np.asarray(B)

    Bs = np.sort(B)
    idx = np.searchsorted(Bs, A)

    linvalid_mask = idx==len(B)
    idx[linvalid_mask] = len(B)-1
    lval = Bs[idx] - A
    lval[linvalid_mask] *=-1

    rinvalid_mask = idx==0
    idx1 = idx-1
    idx1[rinvalid_mask] = 0
    rval = A - Bs[idx1]
    rval[rinvalid_mask] *=-1
    return np.minimum(lval, rval) <= tol