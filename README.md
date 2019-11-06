# Fake-News-Detection
This project aims to build a model that has practical potential to detect the subtleties inherent in fake news captured by the unique features selected and differentiate between fake news and real news.
A comparative study was prformed using state-of-the-art algorithms to find the optimally performing model for the methodology selected, supervised learning with source credibility, emotion detection and content-based modelling. Implement the optimally performing model using RShiny as a prototype fake news detecting app.
As per my hypothesis, Deep Learning outperformed more traditional algorithms in identifying patterns with distinct and weak features taken from different methodologies.
Data is pre-processed using various NLP processes in R and additional features are extracted using RapidMiner.
The model is prototyped in R and an MLP Deep learning model is implememted in R with an accuracy of about 80%.
Finally an RShiny application is built that performs real time Fake News Detection.
Working of the RSHINy APP:

First, the input data is uploaded, a CSV file having one or more inputs.
![image](https://user-images.githubusercontent.com/46936497/68316179-eab20200-00b0-11ea-8e21-ec35e8bc06c3.png)
While the data is processed, and the model runs for predictions, the app displays computing progress.
Finally the prediction is displayed as results.
