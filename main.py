from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

diabetes_130_us_hospitals_for_years_1999_2008 = fetch_ucirepo(id=296)
X = diabetes_130_us_hospitals_for_years_1999_2008.data.features
y = diabetes_130_us_hospitals_for_years_1999_2008.data.targets

columns_to_drop = ['weight', 'payer_code', 'medical_specialty', 'max_glu_serum', 'A1Cresult']

# drop features: weight, payer_code, medical_specialty, max_glu_serum, A1Cresult
X = X.drop(columns=columns_to_drop)
X.drop(X[X['gender'] == 'Unknown/Invalid'].index, inplace=True)
# results in 105 column features after transform

# labels = ['Circulatory', 'Respiratory', 'Digestive', 'Diabetes', 'Injury',
#           'Musculoskeletal', 'Genitourinary', 'Neoplasms', 'Other']
def map_icd_to_category(icd_code):
    try:
        icd_code = float(icd_code)
    except ValueError:
        return "Other"
    if 390 <= icd_code <= 459 or icd_code == 785:
        return 'Circulatory'
    elif 460 <= icd_code <= 519 or icd_code == 786:
        return 'Respiratory'
    elif 520 <= icd_code <= 579 or icd_code == 787:
        return 'Digestive'
    elif str(icd_code).startswith('250'):
        return 'Diabetes'
    elif 800 <= icd_code <= 999:
        return 'Injury'
    # Musculoskeletal
    elif 710 <= icd_code <= 739:
        return 'Musculoskeletal'
    # Genitourinary
    elif 580 <= icd_code <= 629 or icd_code == 788:
        return 'Genitourinary'
    # Neoplasms
    elif 140 <= icd_code <= 239:
        return 'Neoplasms'
    else:
        return 'Other'

X['diag_1'] = X['diag_1'].apply(map_icd_to_category)
X['diag_2'] = X['diag_2'].apply(map_icd_to_category)
X['diag_3'] = X['diag_3'].apply(map_icd_to_category)

categorical_columns = X.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(sparse=False)

# Fit and transform the categorical columns
X_encoded = encoder.fit_transform(X[categorical_columns])
X_encoded = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_columns))
X_encoded = pd.concat([X_encoded, X.drop(columns=categorical_columns)], axis=1)

column_variances = X_encoded.var()
nzv = set()
# set of near zero variance features
for column, variance in column_variances.items():
    if variance < .001:
        nzv.add(column.split('_')[0])

columns_to_drop = []
for column in X_encoded.columns:
    parts = column.split('_')
    if parts[0] in nzv:
        columns_to_drop.append(column)

X_encoded = X_encoded.drop(columns=columns_to_drop)

missing_values = X_encoded.isnull().any(axis=1)
X_encoded = X_encoded[~missing_values]
y = y[~missing_values]

y['readmitted'] = y['readmitted'].map({'<30': 1, '>30': 1, 'NO': 0})

# pd.set_option('display.max_columns', None)
# print(X_encoded.head(1000))


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# kNN classifier
# figure, axes = plt.subplots(1, figsize=(6, 6))
# train_accuracies, test_accuracies = [], []
# k_values = [1, 2, 5, 10, 50, 100, 110]
# for k in k_values:
#     classifier = KNeighborsClassifier(n_neighbors=k)
#     classifier.fit(X_train, y_train)
#     y_pred = classifier.predict(X_train)
#     train_acc = accuracy_score(y_train, y_pred)
#     train_accuracies.append(train_acc)
#     y_pred = classifier.predict(X_val)
#     test_acc = accuracy_score(y_val, y_pred)
#     test_accuracies.append(test_acc)
# axes.semilogx(k_values, train_accuracies, marker='o', color='red')
# axes.semilogx(k_values, test_accuracies, marker='o', color='green')
# plt.show()

# logistic_classifier
train_accuracies = []
test_accuracies = []
cs = [0, 0.1, 1, 10, 50]
for c in cs:
    if c == 0:
        classifier = LogisticRegression(penalty=None, fit_intercept=True)
    else:
        classifier = LogisticRegression(penalty='l1', C=c, solver='liblinear', fit_intercept=True)
    classifier.fit(X_train, y_train)
    train_accuracies.append(classifier.score(X_train, y_train))
    test_accuracies.append(classifier.score(X_val, y_val))
fig, axes = plt.subplots()
axes.semilogx(cs, train_accuracies, color='red', label='training accuracy')
axes.semilogx(cs, test_accuracies, color='blue', label='testing accuracy')
axes.set_xlabel('regulaization strength', fontsize=14)
axes.set_ylabel('accuracy', fontsize=14)
axes.legend()

# neural network
# train_accuracies = []
# val_accuracies = []
# learning_rates = [.001, .0025, .005, .0075, .01]
# for learning_rate in learning_rates:
#     mlp_model = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', solver='sgd', learning_rate_init=learning_rate, batch_size=256, random_state=42)
#     mlp_model.fit(X_train, y_train)
#     mlp_train_pred = mlp_model.predict(X_train)
#     mlp_val_pred = mlp_model.predict(X_val)
#     mlp_train_acc = accuracy_score(y_train, mlp_train_pred)
#     mlp_val_acc = accuracy_score(y_val, mlp_val_pred)
#     train_accuracies.append(mlp_train_acc)
#     val_accuracies.append(mlp_val_acc)
# plt.plot(learning_rates, train_accuracies, marker='o', label='Training Accuracy')
# plt.plot(learning_rates, val_accuracies, marker='o', label='Validation Accuracy')
# plt.xlabel('Learning Rate')
# plt.ylabel('Accuracy')
# plt.title('Accuracy vs. Learning Rate')
# plt.legend()
# plt.grid(True)
# plt.show()

# random forest
# n_estimators_values = [10, 50, 100, 150, 200]
# accuracies = []
# for n_estimators in n_estimators_values:
#     rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
#     rf_classifier.fit(X_train, y_train)
#     y_pred = rf_classifier.predict(X_val)
#     accuracy = accuracy_score(y_val, y_pred)
#     accuracies.append(accuracy)
#
# plt.plot(n_estimators_values, accuracies, marker='o')
# plt.xlabel('Number of Estimators')
# plt.ylabel('Accuracy')
# plt.title('Accuracy vs. Number of Estimators')
# plt.grid(True)
# plt.show()
