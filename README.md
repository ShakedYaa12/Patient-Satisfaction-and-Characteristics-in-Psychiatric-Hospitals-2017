# Satisfaction in Psychiatry Hospitals 2017

## Overview

This project is focused on analyzing patient satisfaction data from psychiatric hospitals in 2017. The data, sourced from [info.data.gov.il](https://info.data.gov.il/home/), will be used to build an end-to-end classification pipeline and conduct unsupervised analyses such as clustering or anomaly detection.

## Dataset

- **File Name:** `satisfaction-hosp-psychiatry-2017.xls`
- **Source:** [https://data.gov.il/dataset/satisfaction-hosp-psychiatry-2017](https://data.gov.il/dataset/satisfaction-hosp-psychiatry-2017)
- **Description:** The dataset contains information on patient satisfaction in psychiatric hospitals in 2015. It includes demographic details such as age and gender, types of treatments received, and reported satisfaction levels. With over 1000 records and more than 100 features, the dataset provides a comprehensive view of patient experiences and interactions with the healthcare system.

## Project Goals

1. Build an end-to-end classification pipeline:

   - Data preprocessing
   - Feature engineering
   - Model selection and training
   - Evaluation and interpretation

2. Perform unsupervised analysis:

   - Clustering
   - Anomaly detection
   - Recommendation system (optional)

## Motivation

We aim to explore the various relationships between patient's characteristics and their satisfaction levels with different aspects of their hospitalization experience. Understanding these connections will provide valuable insights into areas requiring improvement, enabling hospitals to deliver better and more personalized care. Healthcare providers can enhance the overall patient experience and optimize treatment outcomes by identifying factors that significantly impact satisfaction.

## Method

We applied machine learning techniques to perform clustering, classification, and anomaly detection:

- **Clustering:**
  - Techniques: K-Means, GMN, and DBSCAN
  - Final Selection: K-Means
  - Visualizations: Dendrograms to reveal relationships between groups
- **Classification:**
  - Models: XGBoost, LightGBM, Random Forest, Decision Tree, Logistic Regression, and SVM
  - Final Model: XGBoost, due to its superior recall and F1 scores
  - Objective: Predict satisfaction levels
- **Anomaly Detection:**
  - Visualizations: Boxplots to interpret anomalies

## Intended Experiments

1. **Clustering:**

   - Run K-Means for a range of cluster numbers and identify the optimal number using the Elbow Method and cohesion/separation measures.
   - Apply GMN and DBSCAN as alternatives to K-Means and evaluate results.

2. **Classification:**

   - Split the data into training and test sets.
   - Train XGBoost, LightGBM, Random Forest, Decision Tree, Logistic Regression, and SVM models.
   - Validate models using Cross-Validation and evaluate performance with Recall and F1-Score.

3. **Anomaly Detection:**

   - Visualize anomalies with Boxplots.

4. **Feature Importance Analysis:**

   - Analyze feature importance using the trained XGBoost model to determine which patient characteristics significantly impact satisfaction.

## Project Structure

```
project-directory/
│
├── data/
│   ├── raw/                       # Original dataset
│   ├── processed/                 # Processed data ready for analysis
│
├── notebooks/                     # Jupyter notebooks for data exploration and prototyping
│   ├── data-exploration.ipynb
│   ├── modeling.ipynb
│
├── src/                           # Source code for the project
│   ├── unsupervised_analysis.py   # Clustering and anomaly detection
│   ├── classification_pipeline.py # End-to-end classification pipeline                        
│   ├── preprocessing.py           # Scripts for data cleaning and preprocessing
│   ├── model.py                   # model of prediction
│
├── outputs/                       # Results and figures
│   ├── figures/                   # Visualizations
│   ├── reports/                   # Summary reports
│
├── README.md                      # Project documentation
├── requirements.txt               # Dependencies and environment setup
└── LICENSE                        # License information
```

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/ShakedYaa12/Patient-Satisfaction-and-Characteristics-in-Psychiatric-Hospitals-2017.git
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Place the dataset in the `data/raw/` directory.

## Dependencies

The following libraries are required for the project:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `jupyter`
- `scipy`

Install them using:

```bash
pip install -r requirements.txt
```

## Results

Two satisfaction groups were classified:

- **Satisfied** group: average satisfaction of 0.8
- **Less satisfied** group: average satisfaction of 0.58

## Next Steps

1. Perform initial data exploration.
2. Update this README with:
   - Results from data exploration.
   - Details about preprocessing and modeling.
3. Develop and document preprocessing and modeling pipelines.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

Data provided by [https://data.gov.il/dataset/satisfaction-hosp-psychiatry-2017](https://data.gov.il/dataset/satisfaction-hosp-psychiatry-2017).