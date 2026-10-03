## Background

Electroencephalography (EEG) is a medical recording device that recorded the electrical signals produced by the brain from the scalp. EEGs have been shown to decode thoughts about motor function. Two related paradigms exist. Motor imagery is the kinaesthetic imagination of a movement that is never initiated. Motor attempt is a genuine effort to produce the movement, which in a rehabilitation setting may be weak or unsuccessful. This competition uses motor attempt, as it matches how a user would drive a rehabilitation device.

Existing high density EEG devices can be quite expensive and complicated to set up, making them ill-suited to an active rehabilitation clinic. Therefore, a large amount of research and development goes into developing affordable, easy to use EEG recording devices for large scale deployments within clinics or at home rehabilitation use. 

Whether due to head shape, skull thickness, connectivity or slight changes in cortical layout, each person's EEG signals can be different, even though they are thinking about the same thing. 

# Deep learning vs Machine learning entries

When submitting your entry, you must develop a method either using deep learning architectures or traditional signal processing machine learning techniques. The rules of this category can be found below :

### Machine learning
A machine learning entry is defined as any method that does not include any neural network architecture, involving forward and back propagation, other than a single hidden layer multilayer perceptron (for XOR problems). Machine learning entries cannot use algorithms found in libraries such as TensorFlow or PyTorch.

A more in-depth list of restricted functions can be found in the rules section.

### Deep Learning Entries 

Deep learning entries must create an end-to-end deep learning pipeline. The input to these models must be the raw EEG signals or one of the stated data representations below: 

- Raw EEG signal 
- Minimally filtered signal: 50Hz notch filtered (UK Mains frequency) Broad spectrum filtering (1-100 Hz, Removes DC drift and very high frequency components) 
- Expanded data transforms that maintain the time domain as an axis. Examples are: Spectrograms, Continuous Wavelet Transformation etc. These transforms must be performed in the notebook from the raw data within the compute time requirement. 
- Similarity functions for vector database comparison as classification.
- Deep learning entries are allowed to use a non-layer normalisation function before sending any data to the neural network, however this function must remain the same for each participant and not be statically customized for participant with hard coded parameters for each participant. Normalisation functions that adapt at runtime based off function of the loaded data are permitted.


## Training and Test Data

If you are participating in the cross-participant challenge 20% of the participants' data will be held back for the test set with anonymous naming conventions. If you are participating in the intra-participant challenge 20% of each user's data will be held back for the test set. Any use of data from the other competition will result in immediate disqualification. 

## Information we collect

In order to contact you regarding prizes and potential publications, we require you to share your email address linked with your Kaggle account. We will not use your email address for any other purpose other than to contact you directly regarding your participation
