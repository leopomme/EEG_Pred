<h1> Welcome to the Low Cost Motor Imagery Decoding for Rehab competition. Please read the overview and descriptions carefully, as not following the rules can result in a disqualification</h1>

Brain computer interface powered rehabilitation robots are an emerging technology to help improve and automate the rehab process after neural injury or stroke.  In most forms of stroke rehabilitation, the sooner the hand moves after the user thinks about moving it, the greater the rehabilitative effects.  



For this competition, you are tasked with developing a **multi-user** EEG decoding pipeline for decoding voluntary attempts to move the hand to operate a robotic rehabilitation device, recorded on an affordable headset. The data were recorded from healthy adult volunteers, so the goal is a pipeline validated initially on healthy controls with the intention of later utility in stroke rehabilitation. You'll be provided with a data set of 17 participants, completing the mental task while using the robotic hand exoskeleton. In this competition your aim is to train a single model on every participant, which will be evaluated on 3 participants. 

As an additional challenge, you must decide whether to **design a classical, digital signal processing/linear classification pipeline (for example FBCSP - SVM) , or an end-to-end deep learning pipeline**. You can submit one of each as a final submission, however each pipeline must not contain any of the allowed methods from the other. Details of what is allowed in each category can be found below and in the rules section.

The top three teams of each category will be invited to coauthor a paper discussing the results of the competition and each one of their methods, which will be submitted to a relevant conference or journal. 

If you wish to also participate in the single subject competition, please follow the link below:

https://www.kaggle.com/competitions/low-cost-motor-imagery-decoding-for-rehab/

Both competitions are proudly sponsored by g.tec, whose EEG hardware was used to record the data set: https://www.gtec.at