This dataset is a wearable electroencephalography based brain computer interface dataset recorded during a motor attempt paradigm for hand exoskeleton control. The dataset was collected from 20 healthy adult participants using an 8 channel **g.tec Unicorn Hybrid EEG** headset while participants performed two classes of trials: Rest and Motor Attempt. The experimental paradigm was designed to evaluate robust BCI decoding under practical, noisy, real world laboratory conditions rather than idealised shielded recording environments. The work is motivated by the need for wearable, affordable, low channel EEG systems for neuromotor rehabilitation and home based telerehabilitation applications.

The dataset is intended for developing and benchmarking machine learning and deep learning algorithms for binary EEG classification, especially for distinguishing motor attempt from rest using low density wearable EEG. A distinctive feature of this dataset is that the third run includes online visual and robotic hand exoskeleton feedback, allowing competitors to evaluate algorithms in a feedback based BCI control scenario.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F8169116%2F373782d56a088df70dc87fd667101553%2FKaggle%20paradigm%20diagram.png?generation=1779106451750140&alt=media)

Figure 1: Experimental paradigm.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F8169116%2F63188b965a5e491b64646a54dbcdc9cb%2FKaggle%20robotic%20hand.png?generation=1779106502705771&alt=media)

Figure 2: Timing diagram of the training trial.

## Experimental Paradigm
Each participant completed three runs. Each run contains 50 trials, equally divided between the two classes:

| Class | Description |
| --- | --- |
| Rest | The participant was instructed to remain calm and relaxed|
| Motor Imagery| The participant attempted a pinching or squeezing movement using the thumb, index finger, and middle finger, without overt movement.|


The trials were presented in random order to prevent participants from predicting the next condition. Each trial began with a 3 second preparation period, during which a fixation cross was shown. This was followed by a 5 second cue period. The full trial duration was therefore approximately 8 seconds, followed by an inter trial interval of 2 to 3 seconds.
During the motor attempt trials, participants attempted a squeezing or pinching movement. During rest trials, they were asked to relax. The participants were seated approximately 1 metre from a computer screen, with their hands resting on a tabletop. The EEG headset recorded from 8 electrodes while the experimental events were synchronously recorded using Lab Streaming Layer.


## Online Feedback Paradigm

For this Kaggle dataset, each subject contains three files:

| Run | Purpose | Feedback |
| --- | --- | --- |
| Run 1 | Training calibration data | No active exoskeleton feedback |
| Run 2 | Training calibration data | No active exoskeleton feedback |
| Run 3 | Online feedback/test run | Visual and robotic hand exoskeleton feedback |


In the online experiment, participants wore the robotic hand exoskeleton on the dominant hand. During the first two runs, the exoskeleton remained static. During the third run, online neurofeedback was provided. From 5 seconds onward, the classifier output was updated every 0.5 seconds, giving five feedback time points at 5 s, 5.5 s, 6 s, 6.5 s, and 7 s. Correct motor attempt predictions gradually closed the visual hand and the robotic exoskeleton by 20 per cent at each feedback step, reaching full closure after five correct feedback updates.

## Data Acquisition
EEG data were recorded using an 8 channel g.tec Unicorn Hybrid EEG system at a sampling rate of 250 Hz. The electrode montage shown in the paper includes:
Fz, C3, Cz, C4, PO7, Pz, PO8, Oz
Although the Unicorn system provides 17 channels in the CSV, the channels 1-9 correspond to EEG. The remaining channels include auxiliary device signals such as inertial and device related information.


## File Format
The dataset is provided in CSV format:

- S0xx_001_eeg.csv
- S0xx_002_eeg.csv

For example, the attached subject S002 files contain:

- S002_001_eeg.csv
- S002_002_eeg.csv

Test files are labelled in a similar format, with the letter T distinguishing them.

- S008_test_001_eeg.csv
- S008_test_003_eeg.csv

The test files contain no data from any of the subjects in the training set.



## Event Markers

The marker stream contains the timing information needed to segment the EEG into trials. The example files include markers such as:


- trial_start = start of session
- rest = cue to rest
- cue_stop = cue to rest/end of epoch
- move =  cue to attempt the movement


The trial_start marker indicates the beginning of a trial and includes the trial index. The cue_start marker indicates the onset of the class cue. The numeric value following cue_start indicates the trial class. The cue_stop marker indicates the end of the cue period.
In the attached example subject, each run contains 50 trials, with 25 trials from each class. 



- **rest = Rest** 

- **move = Motor Attempt**

In the test file, all cues are replaced by:

- **cue_start**

Which you must predict.


## Prediction Task
The Kaggle task is binary classification:
Rest vs Motor Attempt

Participants may use runs from all participants as calibration/training data and predict labels of all test classes for participants S008, S013 and S015 .

A typical processing strategy is:

1. Load the CSV file.
2. Extract the Unicorn EEG stream.
3. Use only the first 8 EEG channels.
4. Use the event marker stream to identify trial starts and cue starts.
5. Epoch the EEG around each trial, for example from 0 to 8 seconds relative to trial_start. The first 3 seconds are the fixation period and can be used as a baseline, with the cue period running from 3 to 8 seconds.
6. Train a classifier on all other participants data.
7. Evaluate on all sessions from the 3 participants in the test set. Submit in order of <[files].sort()>.

We recommend participants sample each epoch from the cue onset rather than the trial offset.





