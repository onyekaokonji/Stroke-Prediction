# Stroke-Prediction

The purpose of this was to create a ML model which could predict the risk of stroke in a patient given a few parameters such as BMI, presence of hypertension or any heart disease, smoking history etc.

The dataset can be found here: https://www.kaggle.com/fedesoriano/stroke-prediction-dataset

## Models used
5 different models were trained on the dataset in order to compare performance for which the f1-score was used as the evaluation metric. The models are:
<ol> Logistic Regression </ol>
<ol> Linear Support Vector Classifier </ol>
<ol> SVC Kernel </ol>
<ol> Decision Tree Classifier </ol>
<ol> Extra Trees Classifier </ol>
<ol> Random Forest Classifier </ol>

Of all 6 classifiers trained on the dataset, the best performing model was the Random Forest Classifier with a f1-score of 0.87

## Dataset imbalance
The dataset was greatly imbalanced with respect to the target variable. To display the challenge with data imbalance, the models above were trained on the dataset without any balancing and compared to when the dataset was balanced and a StratifiedKFold split was used. Expectedly, the later showed greater performance.
