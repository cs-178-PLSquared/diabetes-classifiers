from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from imblearn.over_sampling import SMOTE
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

df = pd.read_csv("diabetic_data.csv")
df.drop_duplicates(subset="patient_nbr", keep="first", inplace=True)
df = df.replace("?",np.nan)
df = df.replace({"NO":0,
                    "<30":1,
                    ">30":1})

# def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
#     nunique = df.nunique()
#     df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
#     nRow, nCol = df.shape
#     columnNames = list(df)
#     nGraphRow = math.ceil((nCol + nGraphPerRow - 1) / nGraphPerRow)
#     plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
#     for i in range(min(nCol, nGraphShown)):
#         plt.subplot(nGraphRow, nGraphPerRow, i + 1)
#         columnDf = df.iloc[:, i]
#         if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
#             valueCounts = columnDf.value_counts()
#             valueCounts.plot.bar()
#         else:
#             columnDf.hist()
#         plt.ylabel('counts')
#         plt.xticks(rotation = 90)
#         plt.title(f'{columnNames[i]} (column {i + 1})')
#     plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
#     plt.show()

# plotPerColumnDistribution(X, 15, 5)
columns_to_drop = ['weight', 'payer_code', 'medical_specialty']

# drop features: weight, payer_code, medical_specialty, max_glu_serum, A1Cresult
df.drop(columns=columns_to_drop, inplace=True)
invalid_gender_indices = df[df['gender'] == 'Unknown/Invalid'].index
df.drop(invalid_gender_indices, inplace=True)

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
    elif 710 <= icd_code <= 739:
        return 'Musculoskeletal'
    elif 580 <= icd_code <= 629 or icd_code == 788:
        return 'Genitourinary'
    elif 140 <= icd_code <= 239:
        return 'Neoplasms'
    else:
        return 'Other'
#
df['diag_1'] = df['diag_1'].apply(map_icd_to_category)
df['diag_2'] = df['diag_2'].apply(map_icd_to_category)
df['diag_3'] = df['diag_3'].apply(map_icd_to_category)

# categorical_columns = X.select_dtypes(include=['object']).columns
# encoder = OneHotEncoder(sparse=False)
#
# # Fit and transform the categorical columns
# X_encoded = encoder.fit_transform(X[categorical_columns])
# X_encoded = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_columns))
# X_encoded = pd.concat([X_encoded, X.drop(columns=categorical_columns)], axis=1)



# pd.set_option('display.max_columns', None)
# print(X_encoded.head(1000))

# Alternate approach
df.age = df.age.replace({"[70-80)":75,
                         "[60-70)":65,
                         "[50-60)":55,
                         "[80-90)":85,
                         "[40-50)":45,
                         "[30-40)":35,
                         "[90-100)":95,
                         "[20-30)":25,
                         "[10-20)":15,
                         "[0-10)":5})
mapped = {1.0:"Emergency",
          2.0:"Emergency",
          3.0:"Elective",
          4.0:"New Born",
          5.0:np.nan,
          6.0:np.nan,
          7.0:"Trauma Center",
          8.0:np.nan}
df.admission_type_id = df.admission_type_id.replace(mapped)
mapped_discharge = {1:"Discharged to Home",
                    6:"Discharged to Home",
                    8:"Discharged to Home",
                    13:"Discharged to Home",
                    19:"Discharged to Home",
                    18:np.nan,25:np.nan,26:np.nan,
                    2:"Other",3:"Other",4:"Other",
                    5:"Other",7:"Other",9:"Other",
                    10:"Other",11:"Other",12:"Other",
                    14:"Other",15:"Other",16:"Other",
                    17:"Other",20:"Other",21:"Other",
                    22:"Other",23:"Other",24:"Other",
                    27:"Other",28:"Other",29:"Other",30:"Other"}
df["discharge_disposition_id"] = df["discharge_disposition_id"].replace(mapped_discharge)
mapped_adm = {1:"Referral",2:"Referral",3:"Referral",
              4:"Other",5:"Other",6:"Other",10:"Other",22:"Other",25:"Other",
              9:"Other",8:"Other",14:"Other",13:"Other",11:"Other",
              15:np.nan,17:np.nan,20:np.nan,21:np.nan,
              7:"Emergency"}
df.admission_source_id = df.admission_source_id.replace(mapped_adm)
df['race'] = df['race'].fillna(df['race'].mode()[0])

df['admission_type_id'] = df['admission_type_id'].fillna(df['admission_type_id'].mode()[0])

df['discharge_disposition_id'] = df['discharge_disposition_id'].fillna(df['discharge_disposition_id'].mode()[0])

df['admission_source_id'] = df['admission_source_id'].fillna(df['admission_source_id'].mode()[0])
# df = pd.get_dummies(df)
# column_variances = df.var()
# nzv = set()
# set of near zero variance features
# for column, variance in column_variances.items():
#     if variance < .001:
#         nzv.add(column.split('_')[0])
#
# columns_to_drop = []
# for column in df.columns:
#     parts = column.split('_')
#     if parts[0] in nzv:
#         columns_to_drop.append(column)
#
# df.drop(columns=columns_to_drop)
cat_data = df.select_dtypes('O')
num_data = df.select_dtypes(np.number)
LE = LabelEncoder()
for i in cat_data:
  cat_data[i] = LE.fit_transform(cat_data[i])
data = pd.concat([num_data,cat_data],axis=1)
data.drop(['encounter_id','patient_nbr'],axis=1,inplace=True)
X = data.drop('readmitted',axis=1)
y = data['readmitted']
print("After preprocessing...")
print("Number of features: ", X.shape[1])
print("Number of patients: ", len(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.25, random_state=42)
# SC = StandardScaler()
# X_train = pd.DataFrame(SC.fit_transform(X_train),columns=X_train.columns)
# X_test = pd.DataFrame(SC.transform(X_test),columns=X_test.columns)

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

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

# k = 10 was the best performing k-value from the previous code
# classifier = KNeighborsClassifier(n_neighbors=10)
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_train)
# y_pred_prob = classifier.predict_proba(X_train)[:, 1]
# test_acc = accuracy_score(y_train, y_pred)
# test_pre = precision_score(y_train, y_pred)
# test_rec = recall_score(y_train, y_pred)
# test_auc = roc_auc_score(y_train, y_pred_prob)
# print(f"Test Accuracy: {test_acc}\nTest Precision: {test_pre}\nTest Recall: {test_rec}\nTest AUC: {test_auc}")

# logistic_classifier
# train_accuracies = []
# test_accuracies = []
# cs = [0, 0.1, 1, 10, 50]
# for c in cs:
#     if c == 0:
#         classifier = LogisticRegression(penalty=None, fit_intercept=True)
#     else:
#         classifier = LogisticRegression(penalty='l1', C=c, solver='liblinear', fit_intercept=True)
#     classifier.fit(X_train, y_train)
#     train_accuracies.append(classifier.score(X_train, y_train))
#     test_accuracies.append(classifier.score(X_val, y_val))
# fig, axes = plt.subplots()
# axes.semilogx(cs, train_accuracies, color='red', label='training accuracy')
# axes.semilogx(cs, test_accuracies, color='blue', label='testing accuracy')
# axes.set_xlabel('regulaization strength', fontsize=14)
# axes.set_ylabel('accuracy', fontsize=14)
# axes.legend()

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
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=4, min_samples_leaf=1, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_train)
y_pred_prob = rf_classifier.predict_proba(X_train)[:, 1]
test_acc = accuracy_score(y_train, y_pred)
test_pre = precision_score(y_train, y_pred)
test_rec = recall_score(y_train, y_pred)
test_auc = roc_auc_score(y_train, y_pred_prob)
print(f"Test Accuracy: {test_acc}\nTest Precision: {test_pre}\nTest Recall: {test_rec}\nTest AUC: {test_auc}")

# param_grid = {
#     'n_estimators': [50, 100, 200],  # Number of trees in the forest
#     'max_depth': [None],     # Maximum depth of the tree
#     'min_samples_split': [2, 4, 6, 8], # Minimum number of samples required to split an internal node
#     'min_samples_leaf': [1, 2, 4, 8]   # Minimum number of samples required to be at a leaf node
# }
#
# rf_classifier = RandomForestClassifier(random_state=42)
# grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_train, y_train)
# best_params = grid_search.best_params_
# best_score = grid_search.best_score_
# print("Best Parameters:", best_params)
# print("Best Score:", best_score)

# k-fold cross validation for rf
# rf_classifier = RandomForestClassifier(n_estimators=100,
#                                         max_depth=None,        # Adjust max_depth
#                                         min_samples_split=2, # Adjust min_samples_split
#                                         min_samples_leaf=1,  # Adjust min_samples_leaf
#                                         random_state=42)
# rf_classifier.fit(X_train, y_train)
# cv_scores = cross_val_score(rf_classifier, X_train, y_train, cv=5)
# print("Cross-Validation Scores:", cv_scores)
# mean_cv_score = np.mean(cv_scores)
# print("Mean Cross-Validation Score:", mean_cv_score)

plt.figure(figsize=(5, 4))
y_pred = rf_classifier.predict(X_train)
print(accuracy_score(y_train, y_pred))
cm = confusion_matrix(y_train, y_pred)

conf_matrix = pd.DataFrame(data=cm, columns=['Predicted:0', 'Predicted:1'], index=['Actual:0', 'Actual:1'])

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=['Green'], cbar=False,
            linewidths=0.1, annot_kws={'size': 16})

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()
