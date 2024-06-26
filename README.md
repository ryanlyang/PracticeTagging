# LHC machine learning challenge: the hunt for boosted top quarks

Machine Learning techniques have revolutionized the identification of top quark decay signatures in experiments at the Large Hadron Collider. This repository describes how to use a public data set for the development of machine learning based top tagging methods, and contribute to the project of discovering new fundamental physics.

Link to data set: http://opendata.cern.ch/record/15013

<p align="center">
<img src="https://user-images.githubusercontent.com/27929701/183285418-b041833f-2249-495a-b3d0-37ae38a1d3a7.png" width=800 class="centerImage" alt="Hits in a particle detector can be represented as images. Here we show such images for single background and signal jets, as well as the averaged background and signal jets. Even for a human, telling signal from background is not easy!">
</p>

*This figure shows individual (top row) and averaged (bottom row) jet images built from the background (left column) and signal (right column) classes.*

## Two minute introduction

Boosted top tagging is an essential binary classification task for experiments at the Large Hadron Collider (LHC) to measure the properties of the top quark. The [ATLAS Top Tagging Open Data Set](http://opendata.cern.ch/record/15013) is a publicly available dataset for the development of Machine Learning (ML) based boosted top tagging algorithms. The dataset consists of a nominal piece used for the training and evaluation of algorithms, and a systematic piece used for estimating the size of systematic uncertainties produced by an algorithm. The nominal data is split into two orthogonal sets, named *train* and *test* and stored in the HDF5 file format, containing about 92 million and 10 million jets respectively. The systematic varied data is split into many more pieces that should only be used for evaluation in most cases. Both nominal sets are composed of equal parts signal (jets initiated by a boosted top quark) and background (jets initiated by light quarks or gluons). For each jet, the datasets contain:

- The four vectors of constituent particles
- 15 high level summary quantities evaluated on the jet
- The four vector of the whole jet
- A training weight (nominal only)
- PYTHIA shower weights (nominal only)
- A signal (1) vs background (0) label

There are two rules for using this data set: the contribution to a loss function from any jet should **always** be [weighted by the training weight](https://gitlab.cern.ch/atlas/ATLAS-top-tagging-open-data/-/blob/master/train.py#L272-293), and any performance claim is incomplete without an estimate of the systematic uncertainties via the method illustrated in this repository. The ideal model shows high performance but also small systematic uncertainties. Happy tagging!

## Introduction to boosted top tagging at the LHC

The top quark is the heaviest known fundamental particle. Its large mass and strong interactions with the Higgs Boson make it an essential piece of the search for new fundamental physics. These quarks are produced in about one in every billion proton-proton collisions at the LHC. Given the rate of collisions, this means a top quark (along with its anti-particle the anti-top quark) is produced every few seconds when the LHC runs at peak luminosity. However its extremely short lifetime makes it a difficult particle to study. A top quark decays well before it could interact with any matter in a particle detector, so the only way to study this particle is to infer its properties from its decay products. When top quarks decay, they most often produce three lighter quarks in the process. These lighter quarks further *hadronize* into many final state particles which can be measured in a detector. Taken together these particles form a *jet*. A common signature of a top quark is then three of these jets. If the top has a large momentum in a direction perpindicular to the beam axis (transverse momentum or p<sub>T</sub>), or has a large *Lorentz boost*, the three jets can overlap and merge into a single large radius jet. 

Light quarks and gluons are produced in copious numbers in LHC collisions. When these particles hadronize they can produce jets that look very similar to jets initiated by boosted top quarks. This means it is difficult to separate the interesting boosted top quark events from the much more numerous light quark and gluon events. To study high momentum top quarks, LHC experiments need to isolate pure samples of boosted top quark jets from the background, requiring dedicated *top tagging* algorithms. These algorithms classify jets as signal or background based on the measured properties of each constituent in the jet. Typically both signal and background jets have around 50 constituent particles, with some jets having as many as 200. Given the high dimensionality of this feature space and the availability of large data sets of jets labeled as signal or background, boosted top tagging is an ideal application of ML techniques.

## Dataset purpose

The ATLAS top tagging open dataset is a public dataset for use in the development of ML based top tagging methods. It is the only public top tagging dataset generated with a GEANT4 based detector simulation and state-of-the-art jet reconstruction methods. A study of the performance of existing top tagging methods on this dataset found that some taggers which showed promise in [previous studies performed with simplified detector simulation](https://arxiv.org/abs/1902.09914) failed to perform in this more realistic setting. This dataset allows future top tagger development to occur directly on a highly realistic dataset. 

Any physics analysis which uses a top tagging algorithm will need to account for the systematic uncertainties produced. This is typically done through measuring a scale factor, which corrects the taggers performance in simulated data to its performance in experimental data. Scale factor measurements are time intensive and require access to experimental data. An alternative approach is to study how the tagger's performance behaves when systematic variations are applied to simulated datasets. The differences in performance between the nominal and _systematic varied_ datasets can be used to estimate the size of the systematic uncertainties that would be produced by a tagger if used in a physics analysis. This dataset includes a suite of systematic varied datasets that can be used for this purpose, allowing the size of systematic uncertainties to be considered in the tagger development process in addition to pure performance.

## Basic recipe for training a tagger and assessing uncertainties

1. **Train a tagger**: An example training script is provided in `train.py`. In practice obtaining good performance will require utilizing the full statistics of the training set. See "Training with large datasets" below.
2. **Evaluate the tagger**: This should be done on the nominal testing set and the systematic varied testing sets. The python script `evaluate.py` evaluates any saved tensorflow model over one of these datasets. The bash script `evaluate_all.sh` repeatedly calls the python script to run evaluation over all datasets.
3. **Calculate performance metrics**: The python script `calc.py` does this using the tagger predictions from step 2 stored as .npz files and produced by the `evaluate.py` script.
4. **Plot performance metrics**: The script `plot_everything.py` will produce a set of plots that detail the tagger performance, the size of the systematic uncertainties, and how they compare to the hlDNN and ParticleNet baselines. **Important**: This script implements the recommended procedure for setting systematic uncertainties using the raw performance metrics generated in step 3. Other methods of setting systematic uncertainties with this dataset are not supported.

## Dataset generation

The ATLAS Top Tagging Open Data Set consists of jets taken from simulated collisions of protons at a center of mass energy of 13 TeV. The nominal signal and background jets come from simulated collision events containing two different processes:

- Signal: A heavy Z boson (termed Z') with mass of 2 tera-electron-volts decaying to a top anti-top quark pair.
- Background: Jets initiated by light quarks and gluons. These particles are copious by-products of proton-proton collisions at the LHC.

Additionally, the dataset used to estimate some systematic uncertainties contains jets taken from collisions containing the standard model production of top / anti-top quark pairs. To be included in the data set, all jets are required to satisfy several conditions which produce sharp cut-offs in the distributions of some of the quantities contained in the data set (the jet pseudo-rapidity for example). For details of these requirements see the paper released in tandem with this data set.

Efficient simulation of background events requires introducing unphysical bumps in the distribution of the background jet's p<sub>T</sub>. To get rid of these bumps, the background jet p<sub>T</sub> spectrum could be reweighted to what is actually observed in LHC collisions, but these weights would cover many orders of magnitude and make the training of a top tagger difficult. Luckily there is no reason the background jet p<sub>T</sub> spectrum needs to be physical in a data set only used for training a top tagger. Searches for new physics at the LHC often bin events by quantities like jet p<sub>T</sub>, and if the tagger learns to associate a particular jet p<sub>T</sub> with signal jets, it can assign background jets as signal because they happen to have the correct p<sub>T</sub>. This effect is known as *background scultping*, and can produce false positive results in a search for new physics if not controlled properly. A first order method for eliminating this effect is to reweight the signal and background jet p<sub>T</sub> spectrum to be identical. The solution to both of these problems is to reweight the background jet p<sub>T</sub> spectrum to match the signal spectrum. This is the purpose of the training weights included in the data set.

## Data set contents

The ATLAS Top Tagging Open Dataset consists of two pieces. The first is a **nominal** dataset used for the training and evaluation of top taggers. The directories named `train_nominal` and `test_nominal` contain HDF5 files that make up the training and testing datasets respecitvely. These sets together make the nominal dataset. The second piece is a suite of datasets that can be used to estimate the systematic uncertainties produced by a top tagger. These datasets are produced with a **systematic variation** that slightly modifies the kinematic properties of the jet constituent kinematics within a given systematic uncertainty. The differences between a tagger's performance on the nominal and the systematic varied datasets can be used to estimate the systematic uncertainties produced by the tagger. See the table below for a list of the systematic uncertainties and the datasets that are used to estimate them.

| Systematic uncertainty      | Description                                               | Dataset directory                                    |
|-----------------------------|-----------------------------------------------------------|---------------------------------------------|
| Cluster energy scale | Vary the energy scale of jet constituents reconstructed with the calorimeter | `ces_up`, `ces_down` |
| Cluster energy resolution | Vary the energy resolution of jet constituents reconstructed with the calorimeter | `cer` |
| Cluster position resolution | Vary the position resolution of jet constituents reconstructed with the calorimeter | `cpos` |
| Track fake rate | Vary the rate of fake jet constiuents produced by the tracking detector | `track_fake_loose`, `track_fake_jet` |
| Track efficiency | Vary the efficiency of jet constituents reconstructed by the tracking detector | `track_eff_global`, `track_eff_jet` |
| Signal parton shower and hadronization modeling | Vary the parton shower and hadronization model for signal jets | `ttbar_pythia`, `ttbar_herwig` |
| Background parton shower | Vary the parton shower model for background jets | `dijet_herwig_cluster`, `dijet_herwig_string` | 
| Background hadronization | Vary the hadronization model for background jets | `dijet_sherpa_angular`, `dijet_sherpa_dipole` |
| Renormalization and factorization scales | Vary the scales for the signal / background jets | Nominal datasets |

Each dataset contains the following information for each jet, except the training weights and PYTHIA shower weights which are only contained in the nominal datasets:

### Constituent four-vectors

Each jet can have anywhere between 3 and 200 constituent particles. Each of these particles is described by four quantities, which collectively make up the particle's *four-vector*:

- Transverse momentum (p<sub>T</sub>): The component of the particle's momentum perpindicular to the beam axis
- Pseudo-rapidity (&eta;): One of two spatial coordinates in the widely used collider physics coordinate system
- Azimuthal angle (&phi;): The other spatial coordinate
- Energy

The constituent four-vectors are contained in branches named `['fjet_clus_pt', 'fjet_clus_eta', 'fjet_clus_phi', 'fjet_clus_E']`. Since jets contain a variable number of constituent particles, these branches have many zero padded entries. Handling the variable length quality of this data is an important challenge in building effective constituent based top taggers. For convenience the constituents are listed in order of decreasing p<sub>T</sub>, but this choice is arbitrary. There is no inherent ordering to the constituents in a jet!

Lastly the angular coordinates (&eta; and &phi;) are unitless, while the p<sub>T</sub> and energy are given in units of mega-electron-volts. This choice of units means these quantities can have large magnitudes (some constituents have energies upwards of 300,000 MeV). This large scale should be dealt with in pre-processing to stabilize training (see below).

### Constituent taste

Each constituent particle also has an associated integer number, termed the taste, which take values of 0, 1, or 2 and are stored in the branch `fjet_clus_taste`. Since zero padded elements are also given a value of 0, the user should use the constituent p<sub>T</sub> to identify masked elements in this branch. The constituent taste signifies how particle-flow and track calo-cluster objects were combined to form unified flow objects within the ATLAS reconstruction software. For more information on the constituent taste, see Section 6.1 of the paper released in tandem with this dataset, and references contained therein.

### High Level Quantities

15 high level quantities are included in this data set. These variables were chosen in two separate studies of high level quantity based top taggers carried out by the ATLAS collaboration: https://cds.cern.ch/record/2259646, https://cds.cern.ch/record/2776782. It suffices to say they "summarize" the information contained in the data describing the jet constituents. They are contained in the following branches:

```
['fjet_C2', 'fjet_D2', 'fjet_ECF1', 'fjet_ECF2', 'fjet_ECF3', 'fjet_L2', 'fjet_L3', 'fjet_Qw', 'fjet_Split12', 'fjet_Split23', 'fjet_Tau1_wta', 'fjet_Tau2_wta', 'fjet_Tau3_wta', 'fjet_Tau4_wta', 'fjet_ThrustMaj']
```

### Jet Four-vector

In addition to the four-vectors of the jet constituents, the data set also includes the four vector of the jet as a whole. The four quantities are stored in branches named `['fjet_pt', 'fjet_eta', 'fjet_phi', fjet_m']`. Notice the four vector of the jet contains the jet mass, as opposed to the energy given for the jet constituents.

**IMPORTANT**: The jet four-vector is not re-calculated from the systematic varied jet constituents in the datasets meant for asssessing experimental systematic uncertainties. This is because the jet four-vector is calibrated. This means the jet transverse momentum will not match the transverse momentum of the sum of the four-vectors of the jet constituents.

### Training Weight (nominal only)

The training weights are contained in the branch `'training_weights'` in the nominal training dataset. These should always be used to weight the loss function in tagger training. Both tensorflow and pytorch's loss functions support applying such a weighting through a simple key-word argument.

### PYTHIA Shower Weights

PYTHIA shower weights are stored in the branch `'EventInfo_mcEventWEights'`. These weights can be used to vary the renormalization and factorization scales, and parton distribution functions (PDFs), used in the QCD calculations that generated the datasets. There are 27 floating point numbers in this branch for each jet. The first weight is a `nominal` event weight. The other 26 vary the scales or PDFs. Most of these are not used in the procedure for setting systematic uncertainties on the tagger performance (see above).

### Labels

Labels are stored in the branch `'labels'` and take the value of 1 for a signal jet and 0 for a background jet.

### MC event number

Each simulated event is assigned an event number. The number of the event from which a jet is taken is included in the branch `EventInfo_mcEventNumber`. This number is useful for ensuring orthogonality of the training and testing sets. Jets taken with event numbers that are a multiple of 10 are assigned to the testing and systematic varied set, and all other jets are assigned to the training set.

### Data Set Attributes

For convenience, each data file also contains a set of attributes which can be used to retrieve branch names and other meta data. These attributes are:

* `constit`: The names of the branches storing constituent kinematic quantities
* `hl`: The names of the branches storing high level quantities
* `jet`: The names of the branches storing jet level kinematic quantities
* `num_cons`: The number of constituent level kinematic quantities stored (4)
* `num_hl`: The number of high level quantities stored (15)
* `num_jet_features`: The number of jet level kinematic quantities stored (4)
* `num_jets`: The number of jets in the data set

## Best Practices for Training Top Taggers

### Pre-processing

The data set is stored "as simulated" with no pre-processing steps applied other than sorting the jet constituents by decreasing p<sub>T</sub>. However machine learning algorithm training often benefits from applying wise pre-processing. For example:

- In top tagging the &eta; and &phi; coordinates of the jet have no bearing on whether the jet is signal or background, so tagger performance can often be improved by shifting all of the jets such that they sit at the origin of the &eta;-&phi; plane.
- Since the p<sub>T</sub> and energy values are given in MeV, they have quite large magnitudes with some constituents having energies above 300,000 MeV. ML training proceeds best with approximately normally distributed (zero mean and unit standard deviation) inputs, so it is advisable to apply some pre-processing to reduce the scale of these inputs.

These are both standard pre-processing tricks, but there are many other ways of pre-processing tagger inputs. The data set is presented with no pre-processing to encourage experimentation in pre-processing schemes, as they can have considerable impact on tagger performance. For the user's reference the pre-processing scheme used in the paper accompanying this data release is implemented in `utils.py`.

### Training and Evaluation

An example tagger training script can be found in `train.py`. This script can run training for a high level quantity based tagger, a simple deep neural network constituent based tagger, and [two purpose built constituent top taggers](https://arxiv.org/abs/1810.05165): the energy flow network and the particle flow network. The user can select between these taggers by setting the `model_type` keyword. The hyper-parameters for these models are hardcoded to match those used in the accompanying ATLAS public note. The user should feel free to experiment with model hyper-parameters and all other settings in the training script. The setting `max_constits` determines the maximum number of constituents considered for each jet. This is default set to 80 to reduce training time and memory consumption, but can be set to the maximum of 200 if the user wishes to ensure all information contained in the data set is available to the tagger.

The training script reports four tagger performance metrics, for all of which higher numbers correspond to better performance. The AUC (area under the reciever operating characteristic curve) and ACC (accuracy) are standard machine learning performance metrics. Background rejection is particle physics terminology for the inverse of the background efficiency (the fraction of correctly classified background events). The script evaluates the background rejection at two fixed signal efficiencies, or *working points*, of 0.5 and 0.8. These metrics are important because any data analysis which makes use of a top tagger will ultimately need to apply a cut on the tagger output to determine which jets are signal and which are background. The location of this cut is often chosen to produce 0.5 or 0.8 signal efficiency, so the performance of a tagger is evaluated by how much of the background is eliminated at these working points.

Top tagger performance is also often evaluated in p<sub>T</sub> bins. The training script also produces plots of the background rejection in bins of jet p<sub>T</sub>, which show how the network performance evolves with jet p<sub>T</sub>.

### Training with Large Data Sets

The training and testing sets require 130GB and 7.6GB respectively when stored on disk. This makes loading all of the data into memory impossible for the vast majority of users. The example training script solves this problem by only using a fraction of the jets in the training set. The user should tune how many jets are taken from the training set, selecting as many jets as will fit within memory constraints. In most cases, using more than a fraction of training set will require data piping. If the user wishes to pursue this option the following resources may be useful:

- Tensorflow: [Data API](https://www.tensorflow.org/guide/data), [Sequence Class](https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence), [Pipeline Optimization](https://www.tensorflow.org/guide/data_performance)
- PyTorch: [Datasets and DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)

## Performance and Uncertainty Baselines of Existing Taggers

**Add uncertainty table!**

<p align="center">
<img src="https://user-images.githubusercontent.com/27929701/166083546-f24d34cf-89b2-4bb3-bc2d-f2ba08b07dfa.png" width=700 class="centerImage" alt="Performance baselines of existing top taggers on the ATLAS top tagging open data set. These are the numbers to beat.">
</p>

For more information on these baselines, see the public note released with this data set.

## Other Resources

- [ATLAS website](https://atlas.cern/)
- [Glossary of particle physics terms](http://opendata.atlas.cern/books/current/get-started/_book/GLOSSARY.html)
- [An introduction to the top quark](https://indico.cern.ch/event/683640/contributions/2808437/attachments/1629658/2597088/IMFP2018.pdf)

## How to Cite

If you use this data in a research paper, please cite: [https://cds.cern.ch/record/2825328](https://cds.cern.ch/record/2825328)


