**AGE & GENDER IMPACT ON CARDIOVASCULAR DISEASE PREDICTION IN INDIAN POPULATION USING MACHINE LEARNING**

**Project Overview**

This project focuses on predicting cardiovascular disease (CVD) in the Indian population by analyzing the impact of age and gender along with other clinical factors, using various machine learning (ML) techniques. Early prediction of CVD can help reduce mortality and improve patient care by enabling timely interventions.

The dataset contains clinical data from a multispecialty hospital in India and includes 12 key features such as blood pressure, cholesterol, ECG results, and chest pain type. The goal is to build accurate and interpretable ML models for early heart disease detection.

**Features and Objectives**

Utilize age, gender, and other clinical parameters for CVD prediction.

Compare multiple ML models including Logistic Regression, Decision Trees, Random Forest, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Gradient Boosting (XGBoost), and Naive Bayes.

Evaluate the impact of age and gender by training models with and without these features.

Deploy the best-performing model as a web application for real-time prediction.

**Dataset Details**

Source: Multispecialty hospital dataset from India (1000 subjects).

**Features include:**

Age (years)

Gender (binary: 0 = female, 1 = male)

Chest pain type (0-3)

Resting blood pressure (mm Hg)

Serum cholesterol (mg/dl)

Fasting blood sugar (<=120 mg/dl = 0, >120 mg/dl = 1)

Resting electrocardiogram results (0-2)

Maximum heart rate achieved (bpm)

Exercise induced angina (0/1)

Oldpeak (ST depression induced by exercise)

Slope of peak exercise ST segment (0-3)

Number of major vessels (0-3) colored by fluoroscopy

Target (0 = no heart disease, 1 = heart disease present)

**Data Preprocessing Steps**

Checked and handled missing values and duplicates (none found).

Converted categorical variables using Label Encoding and One-Hot Encoding.

Scaled numeric features using Min-Max scaler.

Feature selection was performed based on correlations, keeping all relevant features including age and gender for impact analysis.

**Machine Learning Models Used**

Logistic Regression

Decision Tree Classifier

Random Forest Classifier

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

Gradient Boosting (XGBoost)

Gaussian Naive Bayes

Each model was trained, evaluated, and fine-tuned using hyperparameter optimization techniques.

**Model Evaluation Metrics**

Accuracy

Precision, Recall, and F1-score

Confusion Matrix

ROC-AUC Score

**Key Results**

The XGBoost model achieved the highest accuracy (~99%) and ROC-AUC (~0.9997).

Random Forest and KNN also demonstrated strong performance.

Logistic Regression and SVM delivered competitive but slightly lower accuracy.

Age and gender inclusion improved model performance by about 1.5% in accuracy.

Feature importance analysis confirmed age and gender contribute meaningful predictive power despite low correlation coefficients.

**Deployment**

The final model was deployed as a web application using Render.

Users can input clinical parameters, such as age and gender, to receive instantaneous CVD risk predictions.

Website is live on URL: https://cvd-prediction-s8jr.onrender.com/


**Future Scope**

Incorporate additional demographic and lifestyle factors for enhanced prediction.

Develop region-specific models to mirror variations within Indian populations.

Explore deep learning methods for improved pattern recognition.

Integrate real-time data from wearables for dynamic risk prediction.

Extend datasets with genomic and advanced medical imaging data.

**References**

A comprehensive list of academic papers, datasets, and online resources used in this project are provided in the project report.

**This project was completed as part of an MBA internship at Qollabb EduTech Private Limited, Bangalore (MBA Candidate, Dr. D. Y. Patil Vidyapeeth, Pune).**
