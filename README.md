# Diabetes-Classification-with-RandomForestClassification
# Dataset : https://www.kaggle.com/datasets/saurabh00007/diabetescsv

Diabetes Classification using RandomForest
This project is a machine learning model that can predict whether a person has diabetes or not based on some medical features. The model uses the RandomForest algorithm, which is an ensemble of decision trees that can handle both numerical and categorical data. The model is trained and tested on the Pima Indians Diabetes Database, which contains 768 records of female patients from a Native American population.

Dataset
The dataset can be downloaded from here. It has 8 features and 1 target variable. The features are:

Pregnancies: Number of times pregnant
Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
BloodPressure: Diastolic blood pressure (mm Hg)
SkinThickness: Triceps skin fold thickness (mm)
Insulin: 2-Hour serum insulin (mu U/ml)
BMI: Body mass index (weight in kg/(height in m)^2)
DiabetesPedigreeFunction: Diabetes pedigree function
Age: Age (years)
The target variable is:

Outcome: Class variable (0 or 1) where 1 means the patient has diabetes and 0 means the patient does not have diabetes
Model
The model is built using the scikit-learn library in Python. The RandomForestClassifier class is used to create the model with the following parameters:

n_estimators: The number of trees in the forest. Default is 100.
criterion: The function to measure the quality of a split. Default is “gini”.
max_depth: The maximum depth of the tree. Default is None, which means the nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
random_state: The seed used by the random number generator. Default is None.
The model is trained on 80% of the data and tested on 20% of the data. The performance metrics used are accuracy, precision, recall and f1-score.

Results
The model achieved an accuracy of 0.7922, a precision of 0.75, a recall of 0.6296 and an f1-score of 0.6842 on the test set. The confusion matrix and the feature importances are also shown below.

Conclusion
The model shows a good performance in predicting diabetes based on the given features. However, there is still room for improvement by tuning the hyperparameters, using other algorithms or adding more features.
