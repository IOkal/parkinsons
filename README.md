# Diagnosing Parkinson's
🏥 Hack Western 6 Entry - Diagnosing Parkinson’s from a person’s voice with 92% accuracy.

> 🏆 Parkinsons.AI won **Best use of Standard Library** and **Best Health Hack** At HackWestern!

> ⚠️ This is a Hackathon entry for HackWestern, and was created in under 36 hours. The code in this repo is likely to be hacky and potentially unstable/insecure, so please be careful if forking this repo. [You can view the project's Devpost page here.](https://devpost.com/software/test-mzb73k)

Team Members: Tobias Carryer, Iyad Okal, Will Wang, Kyle Meade

## Inspiration/Background
Over ten million people worldwide are diagnosed with Parkinson’s Disease (PD), with studies suggesting that there are many more people who are not diagnosed due to the low accuracy of modern day testing methods (53% accurate). Recent studies have shown that Voice can play a strong role in detecting early PD symptoms, and so we set out to develop a tool to help doctors leverage this advance in technology to assist them to more accurately diagnose PD patients.

## What it does
Our app works as a tool for doctors to aid them to detect early signs of Parkinson’s Disease in Patients. It does so by taking an audio file of someone speaking and passing it through to our Custom built Machine Learning Model, there it computes the discrete Fourier transform of the audio file, and analyses the patient’s speech based on 22 features. The ML instance returns the probability of the user showing early sings of the Parkinson’s Disease. It also returns the scores of each of the 22 features so that the doctor can have a detailed explanation as to how the probability was calculated. All of this is displayed in the Dashboard that the doctor would be able to access.

## How we built it
At first we went through dozens of scholarly articles and read up about the relation between Voice and Parkinson’s Disease. Through our research we were able to find 22 features and their mathematical algorithms, from which we were able to recreate 15. We then trained a neural network to predict Parkinson’s from those 15 features. We used a data set from the University of Oxford to train our model. We set up our neural network on the Google Cloud Platform through a Flask server. Finally, we pass back the results from the neural network to the Dashboard for a clean Data visualization.

## Challenges we ran into
Engineering the extraction methods for each individual feature was especially tough, since none of us had prior experience in the physics behind acoustics and tonality. Using Fourier Fast Transform, we determined the metrics related to the fundamental frequencies. Using Parselmouth, a Python library for the Praat software, we calculated metrics of jitter and shimmer. Each of the 15 features we found took extensive research and obscure Python libraries to compute.

Another challenge we faced was deploying the model to Google Cloud. A huge array of problems arised when deploying the back-end API for the neural network to be inferenced. (i.e. Cross-origin resource sharing (CORS) issues in Google Cloud when making requests from the front-end, Bugs in programmatic uploads & downloads in Google Cloud, Client-Server Architecture challenges, and many more...)

Git issues :P

## Accomplishments that we're proud of
Our model achieved 92% accuracy on a withheld test set and 82% accuracy on the cross validation set! We were happy we could achieve great results even with only 15 of the 22 predictive features.

## What we learned
Deploying to Google Cloud can be really difficult when using machine learning models because instances tend to run out of memory. Also, a great team dynamic can make a hackathon a lot more fun!
