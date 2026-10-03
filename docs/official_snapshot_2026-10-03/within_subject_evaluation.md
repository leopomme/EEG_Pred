Submissions are evaluated on Classification accuracy between the predicted mental action and the observed target.

The Labels you will be predicting are:
- rest = Rest
- move = Motor Attempt

How and how much of the test epoch you use is up to you.

You will be required to submit one file with and ID column and target a collumn detailed below.


## Submission File
you must predict a class label for the TARGET variable. The file should contain a header and have the following format and value (these labels are picked at random and do not correspond to actual values):

    ID,TARGET
    0,rest
    1,move
    2,rest
    ....
    679, move

If you are competing in the single subject competition, please submit your labels for each epoch in ascending order of participants followed by their epoch position in the test set. (Below is an example of the ORDER of the solution file, NOT the corresponding ID and value, please submit with the syntax above)

    ID,TARGET
    <S001_test_epoch_1> ,label
    <S001_test_epoch_2>,label
    <S001_test_epoch_3>,label
    ....
    <S002_test_epoch_1> ,label
    <S002_test_epoch_2>,label
    <S002_test_epoch_3>,label
    ....
    <S003_test_epoch_1> ,label
    <S003_test_epoch_2>,label
    <S003_test_epoch_3>,label
    ....
    <S020_test_epoch_<n-2>>,label
    <S020_test_epoch_<n-1>>,label
    <S020_test_epoch_<n>>,label    

