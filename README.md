# Satisfaction in Psychiatry Hospitals 2017

## Overview
This project is focused on analyzing patient satisfaction data from psychiatric hospitals in 2017. The data, sourced from [info.data.gov.il](https://info.data.gov.il/home/), will be used to build an end-to-end classification pipeline and conduct unsupervised analyses such as clustering or anomaly detection.

## Dataset
- **File Name:** `satisfaction-hosp-psychiatry-2017.xls`
- **Source:** https://data.gov.il/dataset/satisfaction-hosp-psychiatry-2017
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
We will apply machine learning techniques to perform clustering, classification, and anomaly detection:
- **Clustering:**
  - Techniques: K-Means and Hierarchical Clustering
  - Visualizations: Dendrograms to reveal relationships between groups
- **Classification:**
  - Models: Random Forest, Gradient Boosting, and Logistic Regression
  - Objective: Predict satisfaction levels
- **Anomaly Detection:**
  - Technique: Isolation Forest
  - Visualizations: Boxplots to interpret anomalies

## Intended Experiments
1. **Clustering:**
   - Run K-Means for a range of cluster numbers and identify the optimal number using the Elbow Method and cohesion/separation measures.
   - Apply Hierarchical Clustering and visualize results using dendrograms.

2. **Classification:**
   - Split the data into training and test sets.
   - Train Random Forest, Gradient Boosting, and Logistic Regression models.
   - Validate models using Cross-Validation and evaluate performance with Accuracy and F1-Score.

3. **Anomaly Detection:**
   - Use Isolation Forest to detect irregularities or outliers in the dataset.
   - Visualize anomalies with Boxplots.

4. **Feature Importance Analysis:**
   - Analyze feature importance using the trained Random Forest model to determine which patient characteristics significantly impact satisfaction.

## Project Structure
```
project-directory/
│
├── data/
│   ├── raw/                     # Original dataset
│   ├── processed/               # Processed data ready for analysis
│
├── notebooks/                   # Jupyter notebooks for data exploration and prototyping
│   ├── data-exploration.ipynb
│   ├── modeling.ipynb
│
├── src/                         # Source code for the project
│   ├── preprocessing.py         # Scripts for data cleaning and preprocessing
│   ├── classification_pipeline.py # End-to-end classification pipeline
│   ├── unsupervised_analysis.py  # Clustering and anomaly detection
│
├── outputs/                     # Results and figures
│   ├── figures/                 # Visualizations
│   ├── reports/                 # Summary reports
│
├── README.md                    # Project documentation
├── requirements.txt             # Dependencies and environment setup
└── LICENSE                      # License information
```

## Getting Started
1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the dataset in the `data/raw/` directory.

## License
This project is licensed under the **MIT License**:

```
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Next Steps
1. Perform initial data exploration.
2. Update this README with:
   - Results from data exploration.
   - Details about preprocessing and modeling.
3. Develop and document preprocessing and modeling pipelines.

## Acknowledgments
Data provided by https://data.gov.il/dataset/satisfaction-hosp-psychiatry-2017.
