# Random Forest Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy import interp

# Importing the dataset
df = pd.read_excel('RAPT.xlsx')

# Selecting variables of interest for model
variables = [#'Age',
             #'Sex', 'MaritalStatus', 'FirstRace', 'SmokingStatus',
             #'BMI', 'AsaPhysicalStatus',
             'Question.1:.What.is.your.age.group?',
             'Question.2:.Gender?',
             'Question.3:.How.far.on.average.can.you.walk?',
             'Question.4:.Which.gaid.aid.do.you.use?',
             'Question.5:.Do.you.use.Community.Supports?',
             'Question.6:.Will.you.live.with.someone.who.can.care.for.you.after.your.operation?'
             #'Pt.Pref',
             #'SurgeonName'
            ]

# Selecting outcome of interest
outcome = ['Actual.Disposition']

# Removing excluded subjects
df.dropna(subset=(outcome + variables), inplace=True)
df = df[((df['Actual.Disposition'] !='Hospice'))]

# Converting different discharge types to "Facility" or "Home" discharges
df['Actual.Disposition'].replace({'IRF': 'Facility',
                                  'SNF': 'Facility',
                                  'Home Health Agency': 'Home',
                                  'Self care': 'Home'}, inplace=True)

# Dataframe for input variables
X = df[variables]

# Numoy array of binary output
y = np.ravel(df[outcome])

# LabelEncoder for output variable (y)
le = LabelEncoder()
y = le.fit_transform(y)

# Saving mapping dictionaries for tables
mapping = dict(zip(le.classes_, range(len(le.classes_))))
reverse_mapping = dict(zip(range(len(le.classes_)), le.classes_))
# all([mapping[x] for x in le.inverse_transform(y)] == y)

# OneHotEncode X variables
X = pd.get_dummies(X, columns=X.columns.difference(['Age', 'Rapt.Score']), drop_first=True)

# Splitting the dataset into the Training set and Test set
# Run classifier with cross-validation and plot ROC curves
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_curve
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

cv = StratifiedKFold(n_splits=5)
# classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
# classifier = LogisticRegression(solver='newton-cg')
classifier = XGBClassifier(max_depth=12, learning_rate=0.1,
                           n_estimators=1000, n_jobs=8, booster='gbtree')

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, y):
    probas_ = classifier.fit(X.iloc[train], y[train]).predict_proba(X.iloc[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate (100-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Stratified Cross Validation Receiver Operator Curve:\nHome versus Facility')
plt.legend(loc="lower right")
plt.show()

"""Confusion Matrix"""
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False,
                          title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[classes == unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.4f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

np.set_printoptions(precision=4)

# Model Confusion Matrix
pred_final = classifier.predict(X.iloc[test])
pred_final = [reverse_mapping.get(item, item) for item in pred_final]
y_test = [reverse_mapping.get(item, item) for item in y[test]]

plot_confusion_matrix(y_test, pred_final, classes=np.array(list(mapping)),
                      normalize=False, title='Model 2 - Confusion Matrix')
plt.show()

# # Print the feature ranking
# importances = classifier.feature_importances_
# std = np.std([tree.feature_importances_ for tree in classifier.estimators_],
#              axis=0)
# indices = np.argsort(importances)[::-1]
#
# print("Feature ranking:")
#
# for f in range(X.shape[1]):
#     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
#
# # Plot the feature importances of the forest
# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(X.shape[1]), importances[indices],
#        color="r", yerr=std[indices], align="center")
# plt.xticks(range(X.shape[1]), indices)
# plt.xlim([-1, X.shape[1]])
# plt.show()

"""Original RAPT Confusion Matrix"""
# Uses all subjects and applies the original RAPT algorithm
df['RaptPredictions'] = df['Risk.Prediction'] == 'high risk  prediction: discharge inpatient rehabilitation'
df['RaptPredictions'].replace({True: 'Facility',
                               False: 'Home'}, inplace=True)

plot_confusion_matrix(df['Actual.Disposition'], df['RaptPredictions'], classes=np.array(list(mapping)),
                      normalize=False, title='Original RAPT (Full Dataset) - Confusion Matrix')
plt.show()

"""Original RAPT test subset Confusion Matrix"""
# Limits data to the subset of test data (same test data used for model)
df.reset_index(inplace=True)
df['RaptPredictions'] = df['Risk.Prediction'] == 'high risk  prediction: discharge inpatient rehabilitation'
df['RaptPredictions'].replace({True: 'Facility',
                               False: 'Home'}, inplace=True)

plot_confusion_matrix(df.loc[test, 'Actual.Disposition'], df.loc[test, 'RaptPredictions'], classes=np.array(list(mapping)),
                      normalize=False, title='Original RAPT (Test Subset) - Confusion Matrix')
plt.show()