
import os

# Finding the path to the directory where the current file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Importing modules
from preprocessing import preprocessing
from unsupervised_without_outliers import unsupervised_analysis
from model import XGBoost_Classifier_Improved

# Creating a relative path to the CSV file
df_path = os.path.join(BASE_DIR, "..", "data", "raw", "satisfaction-hosp-psychiatry-2017.xls")

df_cleaned = preprocessing(df_path)
df_predict = unsupervised_analysis(df_cleaned)

print("--------------------------------------------------------------------------------------------------------")

df_for_pred_path = os.path.join(BASE_DIR, "..", "data", "processed", "data_ready_to_predict_without_outliers.csv")
XGBoost_Classifier_Improved(df_for_pred_path)

