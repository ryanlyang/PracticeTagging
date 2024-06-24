"""

######################### ATLAS Top Tagging Open Data ##########################

train.py - This is an example script for training constituent based taggers
on the ATLAS Top Tagging Open Data set.

For details of the data set and performance baselines, see:
                       https://cds.cern.ch/record/2825328

Author: Kevin Greif
Last updated 06/19/2024
Written in python 3

################################################################################

"""

from pathlib import Path

# Plotting imports
import matplotlib.pyplot as plt

# ML imports
import tensorflow as tf
from tensorflow.data import Dataset
from energyflow.archs import EFN, PFN
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split


# Custom imports
import utils

################################# SETTINGS #####################################

# Settings used for data preparation and network training. For model hyper-
# parameters, see "Build Tagger and Datasets" section

# Paths to data files. Point this to local directory containing the data files
# in sub-directories
data_path = Path("/DFS-L/DATA/whiteson/kgreif/JetTaggingH5")
train_path = data_path / "public_train_nominal"

# Make glob of the training set files
train_files = sorted(list(train_path.glob("*.h5")))

# Set the amount of data to be used in training. The full training
# set is very large (97 GB) and will not fit in memory all at once. Here, we
# take a subset  of the data. Using the full set will require piping.
n_train_jets = 6000000

# Set the fraction of the training data which will be reserved for validation
valid_fraction = 0.1

# Max constituents to consider in tagger training (must be <= 200)
max_constits = 80

# Tagger to train, supported options are 'hldnn', 'dnn', 'efn', 'pfn'.
tagger_type = 'pfn'

# Training parameters
num_epochs = 70
batch_size = 256

# Path for generated figures
figure_dir = Path().cwd() / "plots"
figure_dir.mkdir(parents=True, exist_ok=True)

########################### Data Preparation ###################################
print("Read data and prepare for tagger training")

# Load data using the functions in preprocessing.py
train_data, train_labels, train_weights, _, _, _ = utils.load_from_files(
    train_files,
    max_jets=n_train_jets,
    get_hl=True if tagger_type == 'hldnn' else False,
    max_constits=max_constits
)

# Find the number of data features
num_data_features = train_data.shape[-1]

####################### Build Tagger and Datasets  #############################
print("Building tagger and datasets")

# Due to EFN's data shape requirements, the EFN data set build is separate
# from the other models.

if tagger_type == 'efn':

    # Build and compile EFN
    model = EFN(
        input_dim=2,
        Phi_sizes=(350, 350, 350, 350, 350),
        F_sizes=(300, 300, 300, 300, 300),
        Phi_k_inits="glorot_normal",
        F_k_inits="glorot_normal",
        latent_dropout=0.084,
        F_dropouts=0.036,
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=6.3e-5),
        output_dim=1,
        output_act='sigmoid',
        summary=False,
        compile_opts={'weighted_metrics': [tf.keras.metrics.BinaryAccuracy(name='accuracy')]}
    )

    # For EFN, take only eta, phi, and log(pT) quantities, and package into
    # a single dataset. We want each element of the data set to have shape:
    #   ((batch_size, max_constits, 1), (batch_size, max_constits, 2))
    # We can do this using tensorflow Dataset's "zip" function.
    # This code assumes quantities are ordered (eta, phi, pT, ...)
    train_angular = train_data[:,:,0:2]
    train_pt = train_data[:,:,2]

    # Make train / valid split using sklearn train_test_split function
    (train_angular, valid_angular, train_pt,
     valid_pt, train_labels, valid_labels,
     train_weights, valid_weights) = train_test_split(
        train_angular,
        train_pt,
        train_labels,
        train_weights,
        test_size=valid_fraction
    )

    # Build tensorflow data sets
    train_list = [train_pt, train_angular, train_labels, train_weights]
    train_sets = tuple([Dataset.from_tensor_slices(i).batch(batch_size)
                        for i in train_list])
    train_data = Dataset.zip(train_sets[:2])
    train_dataset = Dataset.zip((train_data,) + train_sets[2:])

    valid_list = [valid_pt, valid_angular, valid_labels, valid_weights]
    valid_sets = tuple([Dataset.from_tensor_slices(i).batch(batch_size)
                        for i in valid_list])
    valid_data = Dataset.zip(valid_sets[:2])
    valid_dataset = Dataset.zip((valid_data,) + valid_sets[2:])

# For all other models, data sets can be built using the same process, so
# these are handled together

else:

    if tagger_type == 'hldnn':

        # Build hlDNN
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=train_data.shape[1:]))

        # Hidden layers
        for _ in range(5):
            model.add(tf.keras.layers.Dense(
                180,
                activation='relu',
                kernel_initializer='glorot_uniform')
            )

        # Output layer
        model.add(tf.keras.layers.Dense(
            1,
            activation='sigmoid',
            kernel_initializer='glorot_uniform')
        )

        # Compile hlDNN
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=4e-5),
            # from_logits set to False for uniformity with energyflow settings
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            weighted_metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')]
        )

    elif tagger_type == 'dnn':

        # For DNN, we also need to flatten constituent data into shape:
        # (n_jets, max_constits * num_data_features)
        train_data = train_data.reshape(-1, max_constits * num_data_features)

        # Build DNN
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=train_data.shape[1:]))

        # Hidden layers
        for i in range(5):
            model.add(tf.keras.layers.Dense(
                400,
                kernel_initializer='glorot_uniform',
                kernel_regularizer=tf.keras.regularizers.l1(l1=2e-4)
            ))
            model.add(tf.keras.layers.BatchNormalization(axis=1))
            model.add(tf.keras.layers.ReLU())


        # Output layer
        model.add(tf.keras.layers.Dense(
            1,
            activation='sigmoid',
            kernel_initializer='glorot_uniform',
            kernel_regularizer=tf.keras.regularizers.l1(l1=2e-4))
        )

        # Compile DNN
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1.2e-5),
            # from_logits set to False for uniformity with energyflow settings
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            weighted_metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')]
        )

    elif tagger_type == 'pfn':

        # Build and compile PFN
        model = PFN(
            input_dim=7,
            Phi_sizes=(250, 250, 250, 250, 250),
            F_sizes=(500, 500, 500, 500, 500),
            Phi_k_inits="glorot_normal",
            F_k_inits="glorot_normal",
            latent_dropout=0.072,
            F_dropouts=0.022,
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=7.9e-5),
            output_dim=1,
            output_act='sigmoid',
            summary=False,
            compile_opts={'weighted_metrics': [tf.keras.metrics.BinaryAccuracy(name='accuracy')]}
        )

    else:
        raise ValueError("Tagger type setting not recognized")

    # Make train / valid split using sklearn train_test_split function
    (train_data, valid_data, train_labels,
     valid_labels, train_weights, valid_weights) = train_test_split(
        train_data,
        train_labels,
        train_weights,
        test_size=valid_fraction
    )

    # Build tensorflow datasets.
    # In tf.keras' "fit" API, the first argument is the inputs, the second is
    # the labels, and the third is an optional "sample weight". This is where
    # the training weights should be applied.
    # See: https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
    train_dataset = tf.data.Dataset.from_tensor_slices((
        train_data,
        train_labels,
        train_weights)
    ).batch(batch_size)

    valid_dataset = tf.data.Dataset.from_tensor_slices((
        valid_data,
        valid_labels,
        valid_weights)
    ).batch(batch_size)

############################### Train Tagger ###################################
print("Starting tagger training")

# Callback for storing checkpoints
checkpoint_dir = Path().cwd() / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
checkpoint_path = checkpoint_dir / '{epoch:02d}-{val_loss:.2f}.tf'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=str(checkpoint_path),
    save_best_only=True
)

# Train tagger with keras fit function
train_history = model.fit(
    train_dataset,
    callbacks=[checkpoint_callback],
    validation_data=valid_dataset,
    batch_size=batch_size,
    epochs=num_epochs,
    verbose=1
)

# Plot training and validation loss against training epochs
plt.plot(train_history.history['loss'], label='Training')
plt.plot(train_history.history['val_loss'], label='Validation')
plt.ylabel('Cross-entropy Loss')
plt.xlabel('Training Epoch')
plt.legend()
plt.savefig(figure_dir / 'loss.png', dpi=300)
plt.clf()

