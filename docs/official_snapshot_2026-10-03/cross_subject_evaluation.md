Submissions are evaluated on Classification accuracy between the predicted mental action and the observed target.

The Labels you will be predicting are:

- rest = Rest
- move = Motor Attempt

How and how much of the test epoch you use is up to you.

You will be required to submit one file with and ID column and a target collumn detailed below.


## Submission File
you must predict a class label for the TARGET variable. The file should contain a header and have the following format and value (label are picked at random and do not correspond to actual values):

    ID,TARGET
    0,move
    1,rest
    2,move
    ....
    359,rest
    #(Its recommended to pass all target and ID values strings to the CSV writer to avoid changes during file creation)



If you are competing in the cross-participant competition, please submit your labels for each epoch in ascending order of participants and sessions. (Below is an example of the ORDER of the solution file, NOT the corresponding ID and value, please submit with the syntax above)

    ID,TARGET
    <S008_test_001_eeg Epoch 1> ,label
    <S008_test_001_eeg Epoch 2> ,label
    <S008_test_001_eeg Epoch 3> ,label
    ....
    <S008_test_002_eeg Epoch 1> ,label
    <S008_test_002_eeg Epoch 2> ,label
    <S008_test_002_eeg Epoch 3> ,label
    ....
    <S013_test_001_eeg Epoch 1> ,label
    <S013_test_001_eeg Epoch 2> ,label
    <S013_test_001_eeg Epoch 3> ,label
    ....
    <S015_test_003_eeg Epoch n-2> ,label
    <S015_test_003_eeg Epoch n-1> ,label
    <S015_test_003_eeg Epoch n> ,label
