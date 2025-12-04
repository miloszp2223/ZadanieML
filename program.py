import seaborn as sns
import shap
from matplotlib import pyplot as plt
from ucimlrepo import fetch_ucirepo
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report

#change display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', '{:.6f}'.format)


bank_marketing = fetch_ucirepo(id=222)

X = bank_marketing.data.features
y = bank_marketing.data.targets
df = pd.concat([X, y], axis=1)

df = df.drop(columns=['duration']) #future variable
df = df.rename(columns={'day_of_week': 'day'}) #day of the month not week

#NaN fields filled with unknown
missing_cols = ['job', 'education', 'contact', 'poutcome']
for c in missing_cols:
    df[c] = df[c].fillna('unknown')

df['previous'] = df['previous'].clip(upper=60)

#set proper type for bools
bool_cols =['default', 'housing', 'loan', 'y']
for c in bool_cols:
    df[c] = df[c].map({'yes': True, 'no': False})
    df[c] = df[c].astype(bool)
num_cols = ['age', 'balance', 'campaign', 'day', 'pdays', 'previous']

X = df.drop('y', axis=1)
y = df['y']

#training/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=12, stratify=y
)


#encoding
categorical_cols = ['job', 'marital', 'education', 'contact',  'month', 'poutcome']
ohe = OneHotEncoder(drop='first', sparse_output=False)
X_train_encoded = pd.DataFrame(
    ohe.fit_transform(X_train[categorical_cols]),
    index=X_train.index,
    columns=ohe.get_feature_names_out(categorical_cols)  # setting names instead of numbers
)
X_test_encoded = pd.DataFrame(
    ohe.transform(X_test[categorical_cols]),
    index=X_test.index,
    columns=ohe.get_feature_names_out(categorical_cols)
)

#change objects to encoded columns
X_train = X_train.drop(categorical_cols, axis=1).join(X_train_encoded)
X_test = X_test.drop(categorical_cols, axis=1).join(X_test_encoded)

#standardization
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

#convert all to float
X_train = X_train.astype(float)
X_test = X_test.astype(float)


"""
#grid search cross validation
param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l2', 'l1'],
    'solver': ['liblinear'],
    'class_weight': ['balanced']
}

LR = GridSearchCV(
    LogisticRegression(max_iter=1000),
    param_grid,
    cv=5,
    scoring='f1'
)

LR.fit(X_train, y_train)
print(LR.best_params_)

param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [None, 10, 20, 15],
    'max_features': ['sqrt', 'log2', 0.3],
    'class_weight': ['balanced']
}

Forest = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='f1'
)

Forest.fit(X_train, y_train)
print(Forest.best_params_)

param_grid = {
    'C': [10, 5, 15],
    'gamma': ['auto', 'scale', 0.01, 0.001],
    'class_weight': ['balanced']
}


svc = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

svc.fit(X_train, y_train)
print(svc.best_params_)
"""
"""
best params:
{'C': 0.1, 'class_weight': 'balanced', 'penalty': 'l1', 'solver': 'liblinear'}
{'class_weight': 'balanced', 'max_depth': 15, 'max_features': 'log2', 'n_estimators': 400}
{'C': 15, 'class_weight': 'balanced', 'gamma': 'auto'}

"""
"""
#worse models
LR_best = LogisticRegression(
    C=0.1,
    penalty='l1',
    solver='liblinear',
    class_weight='balanced',
    max_iter=1000
)
LR_best.fit(X_train, y_train)
SVC_best = SVC(
    C=15,
    gamma='auto',
    class_weight='balanced'
)
SVC_best.fit(X_train, y_train)
"""
Forest_best = RandomForestClassifier(
    n_estimators=400,
    max_depth=15,
    max_features='log2',
    class_weight='balanced'
)
Forest_best.fit(X_train, y_train)
Forest_y_pred = Forest_best.predict(X_test)
print(classification_report(y_test, Forest_y_pred))


"""
#model comparison
LR_y_pred = LR_best.predict(X_test)
print(classification_report(y_test, LR_y_pred))
Forest_y_pred = Forest_best.predict(X_test)
print(classification_report(y_test, Forest_y_pred))
SVC_y_pred = SVC_best.predict(X_test)
print(classification_report(y_test, SVC_y_pred))
summary = {
    "Model": ["Logistic Regression", "Random Forest", "SVC"],
    "Accuracy": [
        accuracy_score(y_test, LR_y_pred),
        accuracy_score(y_test, Forest_y_pred),
        accuracy_score(y_test, SVC_y_pred)
    ],
    "F1_True": [
        f1_score(y_test, LR_y_pred),
        f1_score(y_test, Forest_y_pred),
        f1_score(y_test, SVC_y_pred)
    ]
}

df_summary = pd.DataFrame(summary)
print(df_summary)
"""


#SHAP
X_sample = X_test.sample(500, random_state=12)

explainer = shap.Explainer(Forest_best, X_sample)
shap_values = explainer(X_sample)

shap.summary_plot(shap_values[:,:,1], X_sample, plot_type="bar", show=False)
plt.savefig("shap_bar.png")

shap.summary_plot(shap_values[:,:,1], X_sample, show=False)
plt.savefig("shap_dot.png")
