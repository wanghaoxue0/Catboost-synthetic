import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


filename = "converted_for_other.csv"
csv_data = pd.read_csv(filename, low_memory = False)
credit = pd.DataFrame(csv_data)
credit_defau = credit[credit['MIS_Status'] > 0]
print(len(credit_defau))
term_defau = np.average(credit_defau["Term"])
credit_ok = credit[credit['MIS_Status'] == 0]
print(len(credit_ok))
term_ok = np.average(credit_ok["Term"])
print(term_defau, term_ok)
#Split data into train and test sets + label target value
from sklearn.model_selection import train_test_split
y = credit.MIS_Status
X = credit.drop(['MIS_Status'], axis=1)
fea_name = X.columns.tolist()
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=0)
#We will apply Simple Imputer and Standart Scaler from sklearn package
from sklearn.impute import SimpleImputer 
my_imputer = SimpleImputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)

#Scaling features with Standart Scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_sc =scaler.fit_transform(train_X)
test_sc = scaler.transform(test_X)
train_sc = train_sc
train_y = train_y
#We will train xgboost without any tunning and check results.
from sklearn.metrics import classification_report
from sklearn import metrics   #Additional     scklearn functions
from sklearn.model_selection import GridSearchCV   #Perforing grid search
from sklearn.model_selection import StratifiedKFold 


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
    
    # plt.figure()
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
    plt.savefig("pic/roc curve of lightGBM.jpg")
    plt.show()

# xgboost 
import xgboost as xgb
# applying optimized parameters
model = xgb.XGBClassifier(learning_rate =0.1, n_estimators=62, max_depth=9,
min_child_weight=2, gamma=0, subsample=0.9,colsample_bytree=0.7, nthread=8,scale_pos_weight=1, \
    seed=27,reg_alpha=3,reg_lambda = 3, use_label_encoder=False)
model_fit(model, train_sc, train_y, test_sc, test_y)

from sklearn.linear_model import LogisticRegression
# applying optimized parameters
model = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=1.0, fit_intercept=True, \
    intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, \
        multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
model_fit(model, train_sc, train_y, test_sc, test_y)

from sklearn.svm import SVC    
# applying optimized parameters
model = SVC(kernel='linear', C = 7, gamma=0.1, probability=True)    
param_test = {"gamma":[0, 0.1, 0.2]}
model_fit(model, train_sc, train_y, test_sc, test_y)

from sklearn.ensemble import RandomForestClassifier
# applying optimized parameters
model = RandomForestClassifier(n_estimators=170,max_depth=16, random_state=90, max_features=8)
param_test = {'max_features':np.arange(5,31)}
model_fit(model, train_sc, train_y, test_sc, test_y)

from sklearn.neural_network import MLPClassifier
# applying optimized parameters
model = MLPClassifier(activation='logistic', alpha=1e-05, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08, learning_rate='constant',
       learning_rate_init=0.01, momentum=0.9,max_iter=2000,hidden_layer_sizes=(10,10),
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
model_fit(model, train_sc, train_y, test_sc, test_y)

import lightgbm as lgb
# applying optimized parameters
model = lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',metrics='auc',learning_rate=0.1, \
    n_estimators=127, max_depth=6, num_leaves= 10,max_bin= 95, min_data_in_leaf= 81, bagging_fraction= 0.6,\
         bagging_freq= 0, feature_fraction= 1.0, lambda_l1= 0.7, lambda_l2= 0.3, min_split_gain=0)
model_fit(model, train_sc, train_y, test_sc, test_y)