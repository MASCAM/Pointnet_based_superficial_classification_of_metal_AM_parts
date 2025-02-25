import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


THREE_CLASSES = True

# Load the dataset
dataframe = pd.read_csv("Features_data/Simulated/Basic_Hollow_Complete_main_features_extracted.csv")
dataframe = pd.DataFrame(dataframe, columns=['Longest Diagonal',	'Thickness',	'Fiedler Number',	'Maximum Height',	'Centroid Distance',	'Std Height',	'Std Centroid Distance',	'Std Thickness',   	'Label'])
#dataframe = pd.DataFrame(dataframe, columns=['Longest Diagonal', 'Thickness', 'Fiedler Number', 'Maximum Height', 'Centroid Distance', 'Label'])
#dataframe = pd.DataFrame(dataframe, columns=['Fiedler Number', 'Std Height', 'Std Centroid Distance', 'Std Thickness', 'Label'])

if THREE_CLASSES:

    dataframe = dataframe[dataframe["Label"].isin([0, 2, 4])]
    dataframe["Label"] = dataframe["Label"].map({0: 0, 2: 1, 4: 2})
    dataframe = dataframe.reset_index(drop=True)

X = dataframe.drop(["Label"], axis=1).fillna(0)
Y = dataframe[["Label"]]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


model = xgb.XGBClassifier(
    max_depth=1,                
    n_estimators=40,           
    learning_rate=0.3,         
    alpha=1,                 
    gamma=0.5,                 
    min_child_weight=6,        
    subsample=0.5,             
    colsample_bytree=0.5,      
    reg_lambda=5.0,            
    eval_metric='mlogloss'     
)

eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(
    X_train, 
    y_train,
    eval_set=eval_set,
    verbose=True
)

y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print evaluation metrics
print("\nModel Performance Metrics:")
print("-" * 30)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Print feature importance
importance = model.feature_importances_
features = X.columns
print("\nFeature Importance:")
print("-" * 30)
for feat, imp in zip(features, importance):
    print(f"{feat}: {imp:.4f}")

# 6. Save a model and load it
import pickle

pickle.dump(model, open("xgboost_classifier_pickle_3_classes.pkl", "wb"))


