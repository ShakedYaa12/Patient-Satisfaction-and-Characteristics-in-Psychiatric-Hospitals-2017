import pandas as pd
from sklearn.cluster import KMeans


def unsupervised_analysis(df_cluster) -> pd.DataFrame: 

    survey_columns = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9',
                      'q10', 'q12', 'q13', 'q14','q15', 'q16', 'q17', 'q18', 'q19',
                      'q20', 'q21', 'q22', 'q23', 'q24', 'q25', 'q26', 'q27','q28', 'q29',
                      'q30','q31', 'q33', 'q34', 'q35', 'q36', 'q71', 'q72', 'q73', 'q74',
                      'q75', 'q76']  
                    
    # Choosing the number of clusters based on the previous test
    optimal_k = 2  
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df_cluster['kmeans_cluster'] = kmeans.fit_predict(df_cluster[survey_columns])

    ## One hot encoder
    # Define categorical variables
    categorical_vars = ['hospital', 'machlaka', 'sugmigdar', 'hosp_size', 
                        'hosp_owner', 'sector', 'peripheral', 'age_bins', 
                        'q39','q40','q41','q42','q43','q44']

    # Apply One-Hot Encoding
    df_cluster = pd.get_dummies(df_cluster, columns=categorical_vars, prefix=categorical_vars)

    ## Converting columns from Bool to Binary
    # Change column type to int64 for columns: 'q9', 'q10' and 145 other columns
    df_cluster = df_cluster.astype({'hospital_1': 'int64', 'hospital_2': 'int64', 'hospital_3': 'int64', 'hospital_4': 'int64', 'hospital_5': 'int64',
                                    'hospital_6': 'int64', 'hospital_7': 'int64', 'hospital_8': 'int64', 'hospital_9': 'int64', 'hospital_10': 'int64',
                                    'hospital_11': 'int64', 'hospital_12': 'int64', 'hospital_13': 'int64', 'machlaka_10 – פעילה סגורה מעורבת': 'int64',
                                    'machlaka_15': 'int64', 'machlaka_16': 'int64', 'machlaka_17': 'int64', 'machlaka_5 - מעורבת': 'int64', 'machlaka_5 – פעילה סגורה גברים': 'int64',
                                    "machlaka_5א' – נשים סגורה פעילה": 'int64', 'machlaka_5ב - מעורבת פעילה פתוחה': 'int64', 'machlaka_6 א - מעורבת': 'int64', 'machlaka_6 ב - מעורבת': 'int64', 
                                    'machlaka_6 – סגורה כרונית נשים': 'int64', 'machlaka_8 - מעורבת': 'int64', 'machlaka_8 – פעילה פתוחה מעורבת': 'int64', 'machlaka_9 – פעילה סגורה גברים': 'int64',
                                    'machlaka_א – גברים סגורה משפטית': 'int64', "machlaka_א' – גברים סגורה.": 'int64', 'machlaka_א-1 סגורה פעילה גברים ונשים .': 'int64', 'machlaka_א-2 גברים ונשים': 'int64',
                                    'machlaka_א-3 סגורה פעילה גברים ונשים': 'int64', 'machlaka_אשפוז יום': 'int64', 'machlaka_ב – גברים סגורה משפטית': 'int64',
                                    "machlaka_ב' פתוחה פעילה גברים נשים .וניפגעות תקיפה מינית.": 'int64', "machlaka_ב' – פתוחה": 'int64', 
                                    'machlaka_ג – מעורבת (גברים נשים) \xa0משפטית – יש אגף פתוח וגם אגף סגור': 'int64', 
                                    "machlaka_ג' פסיכוגריאטריה סגורה גברים ונשים .": 'int64', "machlaka_ד'– נשים - יש אגף פתוח וגם אגף סגור": 'int64',
                                    "machlaka_ה' סגורה פעילה גברים ונשים .": 'int64', "machlaka_ה'- פסיכוגריאטרית סגורה מעורבת": 'int64', 
                                    'machlaka_יב – סגורה משפטית מעורבת (גברים- נשים)': 'int64', 'machlaka_מחלקה א סגורה': 'int64', 'machlaka_מחלקה ב פתוחה': 'int64', 
                                    "machlaka_מחלקה ג' סגורה": 'int64', "machlaka_מחלקה ה' סגורה": 'int64', "machlaka_מחלקה ה' פתוחה": 'int64', 'machlaka_מחלקת נשים סגורה [מחלקה חדשה ללא שיוך אות]': 'int64', 
                                    "machlaka_מחלקת פסיכיאטריה א' - נשים": 'int64', "machlaka_מחלקת פסיכיאטריה ב' - מעורבת פעילה": 'int64', "machlaka_מחלקת פסיכיאטריה ג' - גברים": 'int64', 'machlaka_מיון והשהיה': 'int64', 
                                    'machlaka_פסיכוגריאטריה': 'int64', 'machlaka_פסיכוגריאטריה - סגורה ממושכת פעילה': 'int64', 'machlaka_פסיכוגריאטריה מחלקה 20': 'int64', 'machlaka_פסיכוגריאטריה סגורה': 'int64', 
                                    "machlaka_פסיכיאטריה פעילה ג' (גברים) - סגורה": 'int64', "machlaka_פסיכיאטריה פעילה ד' תחלואה כפולה (גברים) - סגורה": 'int64', "machlaka_פסיכיאטריה פעילה ה' (גברים) - סגורה": 'int64', 
                                    "machlaka_פעילה א' – פתוחה גברים ונשים": 'int64', "machlaka_פעילה ב' – סגורה גברים": 'int64', "machlaka_פעילה ג'- סגורה נשים": 'int64', "machlaka_פעילה ד' –סגורה פסיכוגריאטריה – מעורב גברים ונשים": 'int64', 
                                    "machlaka_פעילה ה' – מחלקה פתוחה תחלואה כפולה מעורב גברים ונשים.": 'int64', 'sugmigdar_גברים': 'int64', 'sugmigdar_מעורבת': 'int64', 'sugmigdar_נשים': 'int64', 'hosp_size_1': 'int64', 'hosp_size_3': 'int64', 
                                    'hosp_owner_1': 'int64', 'hosp_owner_2': 'int64', 'sector_1': 'int64', 'sector_3': 'int64', 'sector_4': 'int64', 'peripheral_1': 'int64', 'peripheral_2': 'int64', 'age_bins_1': 'int64', 'age_bins_2': 'int64',
                                    'age_bins_3': 'int64', 'q39_1': 'int64', 'q39_2': 'int64', 'q39_3': 'int64', 'q39_4': 'int64', 'q39_5': 'int64', 'q39_6': 'int64', 'q39_7': 'int64', 'q40_1': 'int64', 'q40_2': 'int64', 'q40_3': 'int64', 'q40_4': 'int64',
                                    'q40_5': 'int64', 'q41_1': 'int64', 'q41_2': 'int64', 'q41_3': 'int64', 'q41_4': 'int64', 'q41_7': 'int64', 'q41_8': 'int64', 'q41_9': 'int64', 'q42_1': 'int64', 'q42_2': 'int64', 'q43_1': 'int64', 'q43_2': 'int64',
                                    'q43_3': 'int64', 'q44_1': 'int64', 'q44_2': 'int64', 'q44_3': 'int64'})
        
    ## Exporting the DataFrame to a CSV file
    df_cluster.to_csv("C:/Users/LENOVO/Desktop/Patient-Satisfaction-and-Characteristics-in-Psychiatric-Hospitals-2017-1/data/processed/data_ready_to_predict_without_outliers.csv", index=False)

    return df_cluster