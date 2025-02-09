import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def find_outliers(df):

    numeric_columns = df.select_dtypes(include=['number']).columns  # Select only numeric columns
    outlier_rows = pd.DataFrame()  # Create an empty DataFrame to store outlier rows

    for column in numeric_columns:
        # Calculate quartiles and IQR range
        Q1 = df[column].quantile(0.25)  # Lower quartile
        Q3 = df[column].quantile(0.75)  # Upper quartile
        IQR = Q3 - Q1  # Interquartile range

        # Non-outlier range
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter outliers for this column
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        outlier_rows = pd.concat([outlier_rows, outliers])  # Combine outlier rows

    # Remove duplicate rows (if there are outliers in multiple columns)
    outlier_rows = outlier_rows.drop_duplicates()

    return outlier_rows


def fill_missing_by_gender(df, columns_to_fill, gender_column='migdar') -> pd.DataFrame:

       for column in columns_to_fill:
  
        medians = df.groupby(gender_column)[column].median()
        
        df[column] = df.apply(
            lambda row: medians[row[gender_column]] if np.isnan(row[column]) else row[column],
            axis=1)
        return df


def reverse_scale(df, columns, idk_values):
    for col in columns:
        idk_value = idk_values.get(col, None)  
        min_val = df[col][df[col] != idk_value].min()  
        max_val = df[col][df[col] != idk_value].max()  

        df[col] = df[col].apply(
            lambda x: min_val - x + max_val if x != idk_value else x
        )
    return df


def minmax_normalization(df, columns, idk_values):

    for col in columns:
        idk_value = idk_values.get(col, None)  # Finding the min and max values, ignoring the 'Don't Know' value.
        min_val = df[col][df[col] != idk_value].min()
        max_val = df[col][df[col] != idk_value].max()        
        # Min Max Normalization
        df[col] = df[col].apply(
            lambda x: (x - min_val) / (max_val - min_val) if x != idk_value else x)
        
        # Convert 'Don't Know' to 0.5
        df[col] = df[col].replace(idk_value, 0.5)
    
    return df


def preprocessing(path): # xls format
    
    df = pd.read_excel(path)
    
    ## Columns to delete
    columns_to_drop = ["q17", "q32", "q35", "q36", "SurveyCode", "bet", "kampus", "tarich", "q45", "madad1", "q46", "date", 
                       "Q36y", "q17y", "q44_1", "w_bet1",
                       "madad2", "madad3", "madad4", "madad5", "madad6", "madad7", "madad8", "madad9", "district", 
                       "case_manager", "total", "median_age", "q3y", "q4y", "q6y", "q7y", "q8y", "q10y", "q15y", "q19y", "q20y", 
                       "q22y", "q5y", "q9y","q16y", "q24y", "q25y", "q26y", "q27y", "q28y", "q29y", "q18y", "q30y", "q33y", "q34y", 
                       "q35y", "q36t", "Q1y", "q1t", "q2_1", "q21y", "q23y", "q12y", "q13y", "q14y", "q2_1y", "q77", "q71y", "q72y", 
                       "q73y", "q74y", "q75y", "q76y", "q77y", "HAATZAMAPER", "MEIDALATPER", "PEILUTPER", "RETZEFPER", "TNAYMPER", 
                       "YACHASPER", "year"]

    df = df.drop(columns= columns_to_drop)
    df = df.iloc[1: , ]
    
    ## Handling the Bet1 column values ​​and deleting the kodkampus column
    df.loc[df['kodkampus'] == 1, 'bet1'] = 3
    df.loc[df['kodkampus'] == 2, 'bet1'] = 12
    df.loc[df['kodkampus'] == 3, 'bet1'] = 2
    df.loc[df['kodkampus'] == 4, 'bet1'] = 13

    # After we have taken care of the campus column, we can remove it and be left with only the hospital column (bet1)
    df = df.drop(columns= "kodkampus")

    ## Fill in 0 where there is a missing value but the previous column has a value of 1 (indicating that there was no personal caregiver)
    df.loc[df['q11'] == 1, 'q11_1'] = 0
    df.loc[df['q11_1'].isnull(), ['q11','q11_1']]

    ## Calculate the distribution of the data to fill in the missing values ​​in the column
    # Trim the data to exclude 0 and the column name
    df_dist = df.loc[df["q11_1"] != 0]

    column_name = "q11_1"

    # Calculate the distribution of the values ​​present in the column (excluding missing values)
    distribution = df_dist[column_name].value_counts(normalize=True)  # התפלגות באחוזים

    # Calculate the number of missing values ​​in a column
    num_missing = df_dist[column_name].isna().sum()

    # Calculate the number of rows to fill for each value based on the distribution
    values_to_fill = (distribution * num_missing).round().astype(int)  # Round to whole number

    # Ensure that the total number of rows is filled
    difference = num_missing - values_to_fill.sum()
    if difference > 0:
        # Add the difference to the most common value
        most_common_value = values_to_fill.idxmax()
        values_to_fill[most_common_value] += difference

    ## Fill missing values based on the distribution we calculated earlier
    # Distribution of values to be filled (for example: created from the distribution calculation)
    values_to_fill = {
        "4": 2,  # Fill 2 values with "4"
        "3": 1,  # Fill 1 value with "3"
        "1": 1,   
        "2": 1
    }

    # Find indices of missing rows
    missing_indices = df[df[column_name].isna()].index

    # Create a list of values to fill based on the distribution
    fill_values = []
    for value, count in values_to_fill.items():
        fill_values.extend([value] * count)

    # Ensure the number of values matches the number of missing rows
    if len(fill_values) != len(missing_indices):
        raise ValueError("The number of missing rows does not match the distribution of fill values.")

    # Fill missing rows with values
    df.loc[missing_indices, column_name] = fill_values

    ## Fill the "q41" column
    # Column name
    column_name = "q41"

    # Calculate the distribution of existing values in the column (excluding missing values)
    distribution = df[column_name].value_counts(normalize=True)  # Distribution in percentages

    # Calculate the number of missing values in the column
    num_missing = df[column_name].isna().sum()

    # Calculate the number of rows to fill for each value based on the distribution
    values_to_fill = (distribution * num_missing).round().astype(int)  # Round to whole number

    # Ensure that the total number of rows is filled
    difference = num_missing - values_to_fill.sum()
    if difference > 0:
        # Add the difference to the most common value
        most_common_value = values_to_fill.idxmax()
        values_to_fill[most_common_value] += difference

    # Distribution of values to be filled (for example: created from the distribution calculation)
    column_name = "q41"

    values_to_fill = {
        "1": 7,  # Fill 7 values with "1"
        "3": 1,  # Fill 1 value with "3"
        "2": 1,   
        "4": 1,
        "8": 1
    }

    # Find indices of missing rows
    missing_indices = df[df[column_name].isna()].index

    # Create a list of values to fill based on the distribution
    fill_values = []
    for value, count in values_to_fill.items():
        fill_values.extend([value] * count)

    # Ensure the number of values matches the number of missing rows
    if len(fill_values) != len(missing_indices):
        raise ValueError("The number of missing rows does not match the distribution of fill values.")

    # Fill missing rows with values
    df.loc[missing_indices, column_name] = fill_values

    ## Fill the "q42" column
    column_name = "q42"

    # Calculate the distribution of existing values in the column (excluding missing values)
    distribution = df[column_name].value_counts(normalize=True)  # Distribution in percentages

    # Calculate the number of missing values in the column
    num_missing = df[column_name].isna().sum()

    # Calculate the number of rows to fill for each value based on the distribution
    values_to_fill = (distribution * num_missing).round().astype(int)  # Round to whole number

    # Ensure that the total number of rows is filled
    difference = num_missing - values_to_fill.sum()
    if difference > 0:
        # Add the difference to the most common value
        most_common_value = values_to_fill.idxmax()
        values_to_fill[most_common_value] += difference

    # Distribution of values to be filled (for example: created from the distribution calculation)
    values_to_fill = {
        "1": 8,  # Fill 8 values with "1"
        "2": 3,  
    }

    # Find indices of missing rows
    missing_indices = df[df[column_name].isna()].index

    # Create a list of values to fill based on the distribution
    fill_values = []
    for value, count in values_to_fill.items():
        fill_values.extend([value] * count)

    # Ensure the number of values matches the number of missing rows
    if len(fill_values) != len(missing_indices):
        raise ValueError("The number of missing rows does not match the distribution of fill values.")

    # Fill missing rows with values
    df.loc[missing_indices, column_name] = fill_values

    ## Fill the "sector" column
    column_name = "sector"

    # Calculate the distribution of existing values in the column (excluding missing values)
    distribution = df[column_name].value_counts(normalize=True)  # Distribution in percentages

    # Calculate the number of missing values in the column
    num_missing = df[column_name].isna().sum()

    # Calculate the number of rows that need to be filled for each value according to the distribution
    values_to_fill = (distribution * num_missing).round().astype(int)  # עיגול למספר שלם

    # Make sure the total of all rows is filled
    difference = num_missing - values_to_fill.sum()
    if difference > 0:
        # הוסף את ההפרש לערך הנפוץ ביותר
        most_common_value = values_to_fill.idxmax()
        values_to_fill[most_common_value] += difference

    # Distribution of values ​​that need to be filled
    column_name = "sector"
    values_to_fill = {
        "1": 4, 
        "4": 1, 
    }

    # Finding the indexes of the missing rows
    missing_indices = df[df[column_name].isna()].index

    # Create a list of values ​​to fill based on the distribution
    fill_values = []
    for value, count in values_to_fill.items():
        fill_values.extend([value] * count)
    print(fill_values)

    # Verify that the number of rows to fill matches the number of values
    if len(fill_values) != len(missing_indices):
        raise ValueError("The number of missing rows does not match the distribution of fill values.")

    # Fill in the missing rows with values
    df.loc[missing_indices, column_name] = fill_values

    ## Handling hosp_time column
    df = df.dropna(subset=['hosp_time'])
    
    ## Handling mixed column
    df.loc[df['sugmigdar'] == "גברים", 'mixed'] = 1
    df.loc[df['sugmigdar'] == "נשים", 'mixed'] = 2
    df.loc[df['sugmigdar'] == "מעורבת", 'mixed'] = 3

    ## Handling q38, migdar column
    df = fill_missing_by_gender(df, columns_to_fill=['q38'], gender_column='migdar')

    ## Handling Zeman column 
    df['zeman'] = df.groupby('migdar')['zeman'] \
                        .transform(lambda x: x.fillna(x.median(skipna=True)))

    ## Handling 8 (missing) values ​​in the age column
    df['age'] = df.apply(
        lambda row: row['age'] if row['age'] != 8 
        else (1 if 18 <= row['q38'] <= 34 
            else (2 if 35 <= row['q38'] <= 64 
                    else (3 if row['q38'] >= 65 else np.nan))),
        axis=1
    )

    ## Handling 99 (missing) values ​​in the hosp_num column
    df['hosp_num'] = df['hosp_num'].replace(99, np.nan)

    # Truncate the data to exclude 0 and the column name
    df_dist = df.loc[df["hosp_num"] != 0]

    column_name = "hosp_num"

    # Calculate the distribution of the values ​​present in the column (excluding missing values)
    distribution = df_dist[column_name].value_counts(normalize=True)  # התפלגות באחוזים

    # Calculate the number of missing values ​​in a column
    num_missing = df_dist[column_name].isna().sum()

    # Calculate the number of rows that need to be filled for each value according to the distribution
    values_to_fill = (distribution * num_missing).round().astype(int)  # עיגול למספר שלם

    # Make sure the total of all rows is filled
    difference = num_missing - values_to_fill.sum()
    if difference > 0:
        # הוסף את ההפרש לערך הנפוץ ביותר
        most_common_value = values_to_fill.idxmax()
        values_to_fill[most_common_value] += difference

    # Distribution of values ​​that need to be filled
    values_to_fill = {
        3: 9, 
        2: 4,   
        1: 4
    }

    # Finding the indexes of the missing rows
    missing_indices = df[df[column_name].isna()].index

    # Create a list of values ​​to fill based on the distribution
    fill_values = []
    for value, count in values_to_fill.items():
        fill_values.extend([value] * count)

    # Verify that the number of rows to fill matches the number of values
    if len(fill_values) != len(missing_indices):
        raise ValueError("The number of missing rows does not match the distribution of fill values.")

    # Fill in the missing rows with values
    df.loc[missing_indices, column_name] = fill_values

    ## Defining relevant and logical column names
    new_names = {"bet1": "hospital",
            "mixed": "migdarmachlaka", 
             "sug": "sugmachlaka",
             "q117": "q17",
             "q135": "q35",
             "q136": "q36",
             "age": "age_bins",
             "q38": "age",
             "yamim": "num_of_days_until_now",
             "zeman": "num_of_days_to_release"
             }

    df = df.rename(columns= new_names)
    
    ## Exception handling
    outliers_df = find_outliers(df)
    row_index = outliers_df.index
    df = df.drop(row_index)

    ## Ordering the scale of the survey questions
    idk_mapping = {
    5: ['q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11', 'q12', 'q13', 'q15', 'q16', 
        'q19', 'q20', 'q21', 'q22', 'q23', 'q24', 'q25', 'q26', 'q27', 'q28', 'q29', 'q71', 
        'q72', 'q73', 'q74', 'q75', 'q76', 'q17'],
    6: ['q2', 'q14'],
    4: ['q18', 'q30', 'q33', 'q34', 'q35'],
}

    columns = [col for cols in idk_mapping.values() for col in cols]  
    idk_values = {col: idk_value for idk_value, cols in idk_mapping.items() for col in cols}  

    df = reverse_scale(df, columns, idk_values)

    ## Normalize min max
    idk_mapping = {
    5: ['q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11', 'q12', 'q13', 'q15', 'q16', 'q19', 'q20', 'q21','q22', 'q23', 'q24',
        'q25', 'q26', 'q27', 'q28', 'q29', 'q71', 'q72', 'q73', 'q74', 'q75', 'q76', 'q17'],
    6: ['q2', 'q14'],
    4: ['q18', 'q30', 'q33', 'q34', 'q35'],
    11: ['q1', 'q36']}

    columns = [col for cols in idk_mapping.values() for col in cols]
    idk_values = {col: idk_value for idk_value, cols in idk_mapping.items() for col in cols}

    df = minmax_normalization(df, columns, idk_values)

    ## Normalize column q31
    scaler = MinMaxScaler()
    df['q31'] = scaler.fit_transform(df[['q31']])

    ## Feature Engeniring
    # Create a column that displays the total number of days of hospitalization scheduled for the patient
    df['total_hosp_time'] = df['num_of_days_to_release'] + df['num_of_days_until_now']
    
    # Patients admitted voluntarily but later received involuntary treatment during their hospitalization.
    df['forced_during_agreed'] = ((df['q42'] == 1) & (df['q43'] == 1)).astype(int)

    ## Arranging the columns in a logical order    
    df = df[['id', 'hospital', 'migdar', 'machlaka', 'sugmigdar', 'migdarmachlaka',
             'sugmachlaka','hosp_size', 'hosp_owner', 'sector', 'peripheral', 'mitot', 
             'hosp_num', 'num_of_days_until_now', 'num_of_days_to_release', 'total_hosp_time', 'age', 'age_bins',
             'hosp_time', 'hosp_time1', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9',
             'q10', 'q11', 'q11_1', 'q12', 'q13', 'q14', 'q15', 'q16', 'q17', 'q18', 'q19',
             'q20', 'q21', 'q22', 'q23', 'q24', 'q25', 'q26', 'q27', 'q28', 'q29',
             'q30', 'q31', 'q33', 'q34', 'q35', 'q36', 'q37', 'q39', 'q40', 'q41', 'q42',
             'q43', 'q44', 'q71', 'q72', 'q73', 'q74',
             'q75', 'q76', 'forced_during_agreed']]
    
    df_clean = df
    print("The Preprocessing Sucssesful!")
    df.to_csv('data/processed/data_after_preprocessing.csv', index=False)
    
    return df_clean
