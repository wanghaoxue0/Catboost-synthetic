import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
import catboost as cb
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.family']='Microsoft YaHei' 
plt.style.use ('ggplot') 
import seaborn as sns
rcParams['figure.figsize'] = 12, 4


filename = "converted_for_catboost.csv"
csv_data = pd.read_csv(filename, low_memory = False)
credit = pd.DataFrame(csv_data)
#Split data into train and test sets + label target value
from sklearn.model_selection import train_test_split
y = credit.MIS_Status
X = credit.drop(['MIS_Status'], axis=1)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=0)
train_X = train_X
train_y = train_y
cat_features = train_X[["State","Bank", "BankState", "RevLineCr",  "LowDoc"]]

from sklearn.metrics import classification_report
from sklearn import metrics   #Additional     scklearn functions
from sklearn.model_selection import GridSearchCV   #Perforing grid search
from sklearn.model_selection import StratifiedKFold # cross validation

def para_tuning(model,param_grid,X_train,Y_train):
    kflod = StratifiedKFold(n_splits=10, shuffle = True,random_state=7)
    grid_search = GridSearchCV(model,param_grid,scoring = 'roc_auc',n_jobs = -1,cv = kflod)
    grid_result = grid_search.fit(X_train, Y_train) 
    print("Best: %f using %s" % (grid_result.best_score_,grid_search.best_params_))

def model_fit(alg, X_train, Y_train, test_X, test_y):
    #Fit the algorithm on the data
    alg.fit(X_train, Y_train)
    
    #Predict training set:
    dtrain_predictions = alg.predict(test_X)
    dtrain_predprob = alg.predict_proba(test_X)[:,1]
    
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(test_y.values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(test_y, dtrain_predprob))
    
    fpr,tpr,threshold = metrics.roc_curve(test_y, dtrain_predictions)
    roc_auc = metrics.auc(fpr,tpr)
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) 
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.title("")
    plt.savefig("pic/roc curve of catboost.jpg")
    plt.show()

# applying optimized parameters
cb = cb.CatBoostClassifier(
    cat_features= cat_features,
    loss_function = "Logloss",
    depth=10,
    l2_leaf_reg = 2,
    n_estimators=500,
    eval_metric = 'Accuracy',
    leaf_estimation_iterations = 10,
    learning_rate=0.1,
    )

print("start tuning...")
model_fit(cb, train_X, train_y, test_X, test_y)
