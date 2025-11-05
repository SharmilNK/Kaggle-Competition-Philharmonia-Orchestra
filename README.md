# Predictive Model to Identify Patron who will purchase tickets for the 2014–15 Season Subscription.

Brief:
This project predicts which previous patrons are most likely to purchase a 2014–15 concert season subscription using customer, ticket, and demographic data. 
I built a full ML pipeline for data cleaning, feature engineering, and modeling. The final LightGBM + Ridge ensemble achieved strong AUROC performance through 5-fold cross-validation.

## Requirements
Tested with Python 3.8+. Install required packages:
pip install pandas numpy scikit-learn lightgbm pandas numpy scikit-learn

## Project Structure
- main.py              #Code file
- requirements.txt     #Dependencies
- READM.md             #Documentation

## Input files
Here is the link to the Kaggle competition which has the csv files : https://www.kaggle.com/t/1ec87ab81129486a91e96f1b037e1b9b
Place these CSV files in the same working directory before running:
-account.csv — account-level information (billing/shipping, donations, etc.)
-subscriptions.csv — subscription purchase history
-tickets_all.csv — ticket purchase history
-concerts.csv — concert metadata (who, concert_name, season, location)
-zipcodes.csv — zipcode demographics and lat/lon
-train.csv — training set with columns account_id and label (binary target)
-test.csv — test set with  id (used to map account ids)

## How to run
1. Clone the repo
git clone https://github.com/SharmilNK/Kaggle-Competition-Philharmonia-Orchestra.git
cd Kaggle-Competition-Philharmonia-Orchestra
2. Install Dependencies pip install -r requirements.txt
3. Run the code file : python main.py

## The script will:
Load and clean the CSV files.
Impute shipping zip/city information using zipcode lookups.
Merge zipcode demographics to accounts.
Prepare subscription/ticket/donor features and a master account-level feature table.
Select top K correlated features with the training label.
Train a LightGBM + Ridge ensemble using 5-fold stratified CV.
Build test features aligned to the selected features and produce submission.csv.

## Outputs
submission.csv — CSV with predictions for the test set
Also printed to console:Per-fold ROC-AUC, CV OOF ROC-AUC,Top features used
