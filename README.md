# GAN_demo

refernce https://thescipub.com/pdf/jcssp.2021.188.196.pdf

## Original Data

"DATA_update.csv" and "News_update.csv" are the files we will use in our project. "Data.csv" and "News.csv" are the existing data before 2020 that I found. "DATA_update.csv" and "News_update.csv" are the data I scarpped from Yahoo Fiannce and the news headlines scarpped from the seekingalpha website, respectively, and then merged with the data before 2020 based on the Finbert score. It is a complete data set from July 1, 2010 to July 1, 2024.

## Load Data

After downloading the two original data files, you can run the step `1. Load_data.ipynb`. In this file, it will get two more features: Technical indicator and Fourier transform.

## Data Preprocessing

In `2. data_preprocessing.ipynb`, we will get the features and target for our project, and deal with NA values, then normalize our data. In this process, there are two parameters we can set here: `n_steps_in` and `n_steps_out`, we can set how many days we would like to input and how many day/days we would like to predict for the future.

## Baseline LSTM
"3. Baseline_LSTM.ipynb" is the baseline model for our project, which we use LSTM to do the prediction.

## Basic GRU
"3. Basic_GRU.ipynb" is the traditional basic model for our project, which we use GRU to do the prediction.

## Basic GAN
For Basic GAN, "4. Basic_GAN.ipynb" is our basic GAN model.

## WGAN-GP
"5. WGAN_GP.ipynb" is the WGAN-GP model.

## Test Prediction
After we train all the models, we can save the best model, and run "6. Test_prediction.ipynb" to get the prediction.

## Stock-GAN
"7. stock_gan.py" is the Stock_GAN model. The data used is "message_file. csv" and "orderbook_file. csv", and the relevant explanations are all in "LOBSTER_SampleFiles_ReadMe. txt". Because I hope readers can see the results of my run very intuitively, I have been using .ipynb files from 1-6. However, the running time of stock-GAN is too long for codespace to handle, so I ran it in colab and provided .py file.
