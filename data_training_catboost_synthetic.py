# Catboost with synthetic features generation according to the importance
# by Haoxue Wang (whx924@gmail.com)

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
import catboost as cb
filename = "converted_for_catboost.csv"
print("reading...")
csv_data = pd.read_csv(filename, low_memory = False)#防止弹出警告
credit = pd.DataFrame(csv_data)

credit.head()
# # method 1 to do the synthetic process
# credit['plus1']=credit['DisbursementGross']+credit['Term']
# credit['plus2']=credit['DisbursementGross']+credit['GrAppv']
# credit['plus3']=credit['DisbursementGross']+credit['SBA_Appv']
# credit['mul1']=credit['DisbursementGross']*credit['Term']
# credit['mul2']=credit['DisbursementGross']*credit['GrAppv']
# credit['mul3']=credit['DisbursementGross']*credit['SBA_Appv']
# credit['plus_mul1']=(credit['DisbursementGross']+credit['Term'])*credit['Term']
# credit['plus_mul2']=(credit['DisbursementGross']+credit['GrAppv'])*credit['DisbursementGross']
# credit['plus_mul3']=(credit['DisbursementGross']+credit['SBA_Appv'])*credit['DisbursementGross']
# credit['plus_divide1']=(credit['DisbursementGross']+credit['Term'])/credit['Term']
# credit['plus_divide2']=(credit['DisbursementGross']+credit['GrAppv'])/credit['DisbursementGross']
# credit['plus_divide3']=(credit['DisbursementGross']+credit['SBA_Appv'])/credit['DisbursementGross']

# method 2 to do the synthetic process
# credit['plus2']=credit['DisbursementGross']+credit['Term']
# #credit['sub2']=credit['DisbursementGross']-credit['Term']
# credit['mul2']=credit['DisbursementGross']*credit['Term']
# #credit['divide2']=credit['DisbursementGross']/credit['Term']
# credit['plus_mul2']=(credit['DisbursementGross']+credit['Term'])*credit['DisbursementGross']
# #credit['sub_mul2']=(credit['DisbursementGross']-credit['Term'])*credit['DisbursementGross']
# credit['plus_divide2']=(credit['DisbursementGross']+credit['Term'])/credit['DisbursementGross']
# #credit['sub_divide2']=(credit['DisbursementGross']-credit['Term'])/credit['DisbursementGross']

# the synthetic process obtaining the best result
credit['plus2']=credit['DisbursementGross']+credit['GrAppv']
credit['plus3']=credit['DisbursementGross']+credit['SBA_Appv']
credit['mul2']=credit['DisbursementGross']*credit['GrAppv']
credit['mul3']=credit['DisbursementGross']*credit['SBA_Appv']
credit['plus_mul2']=(credit['DisbursementGross']+credit['GrAppv'])*credit['DisbursementGross']
credit['plus_mul3']=(credit['DisbursementGross']+credit['SBA_Appv'])*credit['DisbursementGross']
credit['plus_divide2']=(credit['DisbursementGross']+credit['GrAppv'])/credit['DisbursementGross']
credit['plus_divide3']=(credit['DisbursementGross']+credit['SBA_Appv'])/credit['DisbursementGross']

# CatBoost process
from sklearn.model_selection import train_test_split
y = credit.MIS_Status
X = credit.drop(['MIS_Status'], axis=1)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=0)
print("finished split")
train_X = train_X
train_y = train_y
#We will apply Simple Imputer and Standart Scaler from sklearn package
cat_features = train_X[["State","Bank", "BankState", "RevLineCr",  "LowDoc"]]
# from sklearn.impute import SimpleImputer 
# my_imputer = SimpleImputer()
# train_X = my_imputer.fit_transform(train_X)
# test_X = my_imputer.transform(test_X)

# #Scaling features with Standart Scaler
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# train_sc =scaler.fit_transform(train_X)
# test_sc = scaler.transform(test_X)
# train_sc = train_sc
# train_y = train_y
#We will train xgboost without any tunning and check results.
from sklearn.metrics import classification_report
from sklearn import metrics   #Additional     scklearn functions
from sklearn.model_selection import GridSearchCV   #Perforing grid search
from sklearn.model_selection import StratifiedKFold #交叉验证

def para_tuning(model,param_grid,X_train,Y_train):
    kflod = StratifiedKFold(n_splits=10, shuffle = True,random_state=7)#将训练/测试数据集划分5个互斥子集，
    grid_search = GridSearchCV(model,param_grid,scoring = 'roc_auc',n_jobs = -1,cv = kflod)
    #scoring指定损失函数类型，n_jobs指定全部cpu跑，cv指定交叉验证
    grid_result = grid_search.fit(X_train, Y_train) #运行网格搜索
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
     
    # from xgboost import plot_importance
    # plot_importance(alg)
    # plt.title("")
    # plt.savefig("feature importance of catboost.jpg")
    # plt.show()

params = {
            # 'loss_function': ['Logloss', 'CrossEntropy'],
        #   'learning_rate' : [0.05, 0.1, 0.15],
        #  'l2_leaf_reg': range(1, 10, 3)  
         }

cb = cb.CatBoostClassifier(
    cat_features= cat_features,
    loss_function = "Logloss",
    depth=10,
    l2_leaf_reg = 2,
    n_estimators=500,
    eval_metric = 'Accuracy',
    leaf_estimation_iterations = 10,
    learning_rate=0.1
    )

print("start tuning...")
model_fit(cb, train_X, train_y, test_X, test_y)

# for a deep exploration

# # below I provide another idea to generate synthetic features according to the correlation
# train_x=train_X
# def find_corr_pairs_sub(train_x, train_y, eps=0.01):
#     feature_size = len(train_x[0])
#     feature_corr_list = []
#     for i in range(feature_size):
#         if i % 50 == 0:
#             print (i)
#         for j in range(feature_size):
#             if i < j:
#                 corr = stats.pearsonr(train_x[:,i] - train_x[:,j], train_y)
#                 if abs(corr[0]) < eps:
#                     continue
#                 feature_corr = (i, j, abs(corr[0]))
#                 feature_corr_list.append(feature_corr)
    
#     return feature_corr_list

# def find_corr_pairs_plus(train_x, train_y, eps=0.01):
#     feature_size = len(train_x[0])
#     feature_corr_list = []
#     for i in range(feature_size):
#         if i % 50 == 0:
#             print (i)
#         for j in range(feature_size):
#             if i < j:
#                 corr = stats.pearsonr(train_x[:,i] + train_x[:,j], train_y)
#                 if abs(corr[0]) < eps:
#                     continue
#                 feature_corr = (i, j, corr[0])
#                 feature_corr_list.append(feature_corr)
#     return feature_corr_list

# def find_corr_pairs_mul(train_x, train_y,eps=0.01):
#     feature_size = len(train_x[0])
#     feature_corr_list = []
#     for i in range(feature_size):
#         if i % 50 == 0:
#             print (i)
#         for j in range(feature_size):
#             if i < j:
#                 corr = stats.pearsonr(train_x[:,i] * train_x[:,j], train_y)
#                 if abs(corr[0]) < eps:
#                     continue
#                 feature_corr = (i, j, abs(corr[0]))
#                 feature_corr_list.append(feature_corr)
    
#     return feature_corr_list

# def find_corr_pairs_divide(train_x, train_y, eps=0.01):
#     feature_size = len(train_x[0])
#     feature_corr_list = []
#     for i in range(feature_size):
#         if i % 50 == 0:
#             print (i)
#         for j in range(feature_size):
#             if i != j:
#                 try:
#                     res = train_x[:,i] / train_x[:,j]
#                     corr = stats.pearsonr(res, train_y)
#                     if abs(corr[0]) < eps:
#                         continue
#                     feature_corr = (i, j, abs(corr[0]))
#                     feature_corr_list.append(feature_corr)
#                 except ValueError:
#                     print ('divide 0')
    
#     return feature_corr_list

# def find_corr_pairs_sub_mul(train_x, train_y, sorted_corr_sub, eps=0.01):
#     feature_size = len(train_x[0])
#     feature_corr_list = []
#     for i in range(len(sorted_corr_sub)):
#         ind_i = sorted_corr_sub[i][0]
#         ind_j = sorted_corr_sub[i][1]
#         if i % 100 == 0:
#             print (i)
#         for j in range(feature_size):
#             if j != ind_i and j != ind_j :
#                 res = (train_x[:,ind_i] - train_x[:, ind_j]) * train_x[:,j]
#                 corr = stats.pearsonr(res, train_y)
#                 if abs(corr[0]) < eps:
#                     continue
#                 feature_corr = (ind_i, ind_j, j, corr[0])
#                 feature_corr_list.append(feature_corr)
#     return feature_corr_list

# def get_distinct_feature_pairs(sorted_corr_list):
#     distinct_list = []
#     dis_ind = {}
#     for i in range(len(sorted_corr_list)):
#         if sorted_corr_list[i][0] not in dis_ind and sorted_corr_list[i][1] not in dis_ind:
#             dis_ind[sorted_corr_list[i][0]] = 1
#             dis_ind[sorted_corr_list[i][1]] = 1
#             distinct_list.append(sorted_corr_list[i])
#     return distinct_list

# def get_distinct_feature_pairs2(sorted_corr_list):
#     distinct_list = []
#     dis_ind = {}
#     for sorted_corr in sorted_corr_list:
#         cnt = 0
#         for i in range(3):
#             if sorted_corr[i] in dis_ind:
#                 cnt = cnt + 1
#         if cnt > 1:
#             continue
#         for i in range(3):
#             dis_ind[sorted_corr[i]] = 1
#         distinct_list.append(sorted_corr)
#     return distinct_list

# def get_feature_pair_sub_list(train_x, train_y, eps=0.01):
#     sub_list = find_corr_pairs_sub(train_x, train_y, eps)
#     sub_list2 = [corr for corr in sub_list if abs(corr[2])>eps]
#     sorted_sub_list = sorted(sub_list2, key=lambda corr:abs(corr[2]), reverse=True)
#     dist_sub_list = get_distinct_feature_pairs(sorted_sub_list)
#     dist_sub_list2 = [[corr[0], corr[1]] for corr in dist_sub_list]
#     feature_pair_sub_list = [[520, 521], [271, 521], [271, 520]]
#     feature_pair_sub_list.extend(dist_sub_list2[1:])
#     return feature_pair_sub_list

# def get_feature_pair_plus_list(train_x, train_y, eps=0.01):
#     plus_list = find_corr_pairs_plus(train_x, train_y, eps)
#     plus_list2 = [corr for corr in plus_list if abs(corr[2])>eps]
#     sorted_plus_list = sorted(plus_list2, key=lambda corr:abs(corr[2]), reverse=True)
#     feature_pair_plus_list = get_distinct_feature_pairs(sorted_plus_list)
#     feature_pair_plus_list = [[corr[0],corr[1]] for corr in feature_pair_plus_list]
#     return feature_pair_plus_list

# def get_feature_pair_mul_list(train_x, train_y, eps=0.01):
#     mul_list = find_corr_pairs_mul(train_x, train_y, eps)
#     mul_list2 = [corr for corr in mul_list if abs(corr[2])>eps]
#     sorted_mul_list = sorted(mul_list2, key=lambda corr:abs(corr[2]), reverse=True)
#     feature_pair_mul_list = get_distinct_feature_pairs(sorted_mul_list)
#     feature_pair_mul_list = [[corr[0],corr[1]] for corr in feature_pair_mul_list]
#     return feature_pair_mul_list

# def get_feature_pair_divide_list(train_x, train_y, eps=0.01):
#     divide_list = find_corr_pairs_divide(train_x, train_y, eps)
#     divide_list2 = [corr for corr in divide_list if abs(corr[2])>eps]
#     sorted_divide_list = sorted(divide_list2, key=lambda corr:abs(corr[2]), reverse=True)
#     feature_pair_divide_list = get_distinct_feature_pairs(sorted_divide_list)
#     feature_pair_divide_list = [[corr[0],corr[1]] for corr in feature_pair_divide_list]
#     return feature_pair_divide_list

# def get_feature_pair_sub_mul_list(train_x, train_y, eps=0.01):
#     feature_pair_sub_list = get_feature_pair_sub_list(train_x, train_y, eps=0.01)
#     sub_mul_list = find_corr_pairs_sub_mul(train_x, train_y, feature_pair_sub_list, eps=0.01)
#     sub_mul_list2 = [corr for corr in sub_mul_list if abs(corr[3]) > eps]
#     sorted_sub_mul_list = sorted(sub_mul_list2, key=lambda corr:abs(corr[2]), reverse=True)
#     feature_pair_sub_mul_list = get_distinct_feature_pairs2(sorted_sub_mul_list)
#     feature_pair_sub_mul_list = [[corr[0], corr[1], corr[2]] for corr in feature_pair_sub_mul_list]
#     return feature_pair_sub_mul_list

# def get_feature_pair_sub_list_sf(train_x, train_y, eps=0.01):
#     #Owing to the features are selected by random sampling, the returned result may be different from what I provide
#     sub_list = find_corr_pairs_sub(train_x, train_y, eps)
#     sf = random.sample(len(sub_list),500)
#     sub_list_sf = [sub_list[i] for i in sf]
#     sub_list2 = [[corr[0], corr[1]] for corr in sub_list_sf]
#     feature_pair_sub_list_sf = [[520, 521], [271, 521], [271, 520]]
#     feature_pair_sub_list_sf.extend(sub_list2[1:])
#     return feature_pair_sub_list_sf

# def toLabels(train_y):
#     labels = np.zeros(len(train_y))
#     labels[train_y>0] = 1
#     return labels

# # get the top feature indexes by invoking f_regression 
# def getTopFeatures(train_x, train_y, n_features=16):
#     f_val, p_val = f_regression(train_x,train_y)
#     f_val_dict = {}
#     p_val_dict = {}
#     for i in range(len(f_val)):
#         if math.isnan(f_val[i]):
#             f_val[i] = 0.0
#         f_val_dict[i] = f_val[i]
#         if math.isnan(p_val[i]):
#             p_val[i] = 0.0
#         p_val_dict[i] = p_val[i]
    
#     sorted_f = sorted(f_val_dict.items(), key=operator.itemgetter(1),reverse=True)
#     sorted_p = sorted(p_val_dict.items(), key=operator.itemgetter(1),reverse=True)
    
#     feature_indexs = []
#     for i in range(0,n_features):
#         feature_indexs.append(sorted_f[i][0])
    
#     return feature_indexs

# # generate the new data, based on which features are generated, and used
# def get_data(train_x, feature_indexs, feature_minus_pair_list=[], feature_plus_pair_list=[],
#             feature_mul_pair_list=[], feature_divide_pair_list = [], feature_pair_sub_mul_list=[],
#             feature_pair_plus_mul_list = [],feature_pair_sub_divide_list = [], feature_minus2_pair_list = [],feature_mul2_pair_list=[], 
#             feature_sub_square_pair_list=[], feature_square_sub_pair_list=[],feature_square_plus_pair_list=[]):
#     sub_train_x = train_x[:,feature_indexs]
#     for i in range(len(feature_minus_pair_list)):
#         ind_i = feature_minus_pair_list[i][0]
#         ind_j = feature_minus_pair_list[i][1]
#         sub_train_x = np.column_stack((sub_train_x, train_x[:,ind_i]-train_x[:,ind_j]))
    
#     for i in range(len(feature_plus_pair_list)):
#         ind_i = feature_plus_pair_list[i][0]
#         ind_j = feature_plus_pair_list[i][1]
#         sub_train_x = np.column_stack((sub_train_x, train_x[:,ind_i] + train_x[:,ind_j]))
    
#     for i in range(len(feature_mul_pair_list)):
#         ind_i = feature_mul_pair_list[i][0]
#         ind_j = feature_mul_pair_list[i][1]
#         sub_train_x = np.column_stack((sub_train_x, train_x[:,ind_i] * train_x[:,ind_j]))
    
#     for i in range(len(feature_divide_pair_list)):
#         ind_i = feature_divide_pair_list[i][0]
#         ind_j = feature_divide_pair_list[i][1]
#         sub_train_x = np.column_stack((sub_train_x, train_x[:,ind_i] / train_x[:,ind_j]))
    
#     for i in range(len(feature_pair_sub_mul_list)):
#         ind_i = feature_pair_sub_mul_list[i][0]
#         ind_j = feature_pair_sub_mul_list[i][1]
#         ind_k = feature_pair_sub_mul_list[i][2]
#         sub_train_x = np.column_stack((sub_train_x, (train_x[:,ind_i]-train_x[:,ind_j]) * train_x[:,ind_k]))
        
#     return sub_train_x