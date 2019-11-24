# Diagnosing Parkinson's
Team Members: Tobias Carryer, Iyad Okal, Will Wang, Kyle Meade

Developed at HackWestern 2019. [Devpost link](https://devpost.com/software/test-mzb73k)

## Inspiration
Over ten million people worldwide experience Parkinson's disease (PD) yet current methods have only 53% accuracy in detecting early PD symptoms. Recent innovations in machine learning have increased the diagnostic capabilities when diagnosing PD just with a patient's voice, so we’ve developed a tool to help doctors leverage that technology.

## What it does
Our program takes in an audio file of someone talking and outputs the likelihood the person has Parkinson's.

## How we built it
During inference, we used various Python libraries to extract 15 of the 22 features in a person's voice known to be predictive of Parkinson’s. The features are then passed through our neural network hosted on Google Cloud Platform through a Flask server. Finally, a probability is returned to the front-end which tells the user their likelihood of having Parkinson’s. A responsive graph displaying several parameters is then rendered for data visualization.

We trained a neural network to predict Parkinson's from 15 vocal parameters found in a person's voice. The model is trained using a data set from the University of Oxford. The features were extrapolated from dozens of research papers published over the past few years.

## Challenges we ran into
Engineering the extraction methods for each individual feature was especially tough, since none of us had prior experience in the physics behind acoustics and tonality. Using Fourier Fast Transform, we determined the metrics related to the fundamental frequencies. Using Parselmouth, a Python library for the Praat software, we calculated metrics of jitter and shimmer. Each of the 15 features we found took extensive research and obscure Python libraries to compute.

Another challenge we faced was deploying the model to Google Cloud. A huge array of problems arised when deploying the back-end API for the neural network to be inferenced. (i.e. Cross-origin resource sharing (CORS) issues in Google Cloud when making requests from the front-end, Bugs in programmatic uploads & downloads in Google Cloud, Client-Server Architecture challenges, and many more...)

Git issues :P

## Accomplishments that we're proud of
Our model achieved 92% accuracy on a withheld test set and 82% accuracy on the cross validation set! We were happy we could achieve great results even with only 15 of the 22 predictive features.

## What we learned
Deploying to Google Cloud can be really difficult when using machine learning models because instances tend to run out of memory. Also, a great team dynamic can make a hackathon a lot more fun!
