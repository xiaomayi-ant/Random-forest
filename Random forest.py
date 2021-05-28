import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

#读入数据
train=pd.read_excel(r'C:\Users\ant.zheng\Desktop\dataset\Regression\regre.xlsx',sheet_name='Sheet1')
target='type'  #Disbursed 的值就是二元分类的输出
print(train['type'].value_counts())
print(train.head())
x_columns=[x for x in train.columns if x not in [target]]
X=train[x_columns]
Y=train['type']
rf0=RandomForestClassifier(criterion='gini',oob_score=True,random_state=10)
rf0.fit(X,Y)
print(rf0.oob_score_)
Y_predprob=rf0.predict_proba(X)[:,1]
print("AUC Score (Train): %f"%metrics.roc_auc_score(Y,Y_predprob))

param_test1={'n_estimators':range(20,100,10)}
gsearch1=GridSearchCV(estimator=RandomForestClassifier(random_state=10),
                      param_grid=param_test1,scoring='roc_auc',cv=5)
gsearch1.fit(X,Y)
# print(gsearch1.cv_results_,gsearch1.best_params_,gsearch1.best_score_)
print("分类器数:", gsearch1.best_params_,"模型得分:", gsearch1.best_score_)
rf1=RandomForestClassifier(n_estimators=90,oob_score=True,random_state=10)
rf1.fit(X,Y)
print("最优分类器袋外分数：",rf1.oob_score_)

param_test2={'max_depth':range(2,14,2),'min_samples_split':range(2,201,20)}
gsearch2=GridSearchCV(estimator=RandomForestClassifier(n_estimators=90,random_state=10),
                      param_grid=param_test2,scoring='roc_auc',cv=5)   #,iid=False
gsearch2.fit(X,Y)
# print(gsearch2.cv_results_,gsearch2.best_params_,gsearch2.best_score_)
print("最大深度:",gsearch2.best_params_,"得分:",gsearch2.best_score_)
rf2=RandomForestClassifier(n_estimators=90,max_depth=8,oob_score=True,random_state=10)
rf2.fit(X,Y)
print("最优深度袋外分数：",rf2.oob_score_)

param_test3={'min_samples_split':range(2,150,10),'min_samples_leaf':range(2,60,5)}
gsearch3=GridSearchCV(estimator=RandomForestClassifier(n_estimators=90,max_depth=8,oob_score=True,random_state=10),
                      param_grid=param_test3,scoring='roc_auc',cv=5)#,iid=False

gsearch3.fit(X,Y)
# print(gsearch3.cv_results_,gsearch3.best_params_,gsearch3.best_score_)
print("内部节点最大样本和叶子节点最大样本:", gsearch3.best_params_,"得分:",gsearch3.best_score_)
rf3=RandomForestClassifier(n_estimators=90,max_depth=8,min_samples_split=2,min_samples_leaf=2,oob_score=True,random_state=10)
rf3.fit(X,Y)
print("最优节点袋外分数：",rf3.oob_score_)
param_test4={'max_features':range(3,11,2)}
gsearch4=GridSearchCV(estimator=RandomForestClassifier(n_estimators=90,max_depth=8,min_samples_split=2,
                                                       min_samples_leaf=2,oob_score=True,random_state=10),
                      param_grid=param_test4,scoring='roc_auc',cv=5)   #,iid=False
gsearch4.fit(X,Y)
# print(gsearch4.cv_results_,gsearch4.best_params_,"/n",gsearch4.best_score_)
print("最大特征数:",gsearch4.best_params_,"得分:",gsearch4.best_score_)
rf4=RandomForestClassifier(n_estimators=90,max_depth=8,min_samples_split=2,
                           min_samples_leaf=2,max_features=7,oob_score=True,random_state=10)
rf4.fit(X,Y)
print("调优模型袋外分数:",rf4.oob_score_)

#制作roc曲线
Y_prob=rf4.predict_proba(X)[:,1]
print("AUC Score (Train): %f"%metrics.roc_auc_score(Y,Y_prob))
from sklearn.metrics import  roc_curve,auc
fpr,tpr,thersholds=roc_curve(Y,Y_prob,pos_label=1)

# for  i,value in enumerate(thersholds):
#     print("%f%f%f" %(fpr[i],tpr[i],value))
roc_auc=auc(fpr,tpr)
plt.plot(fpr,tpr,'k--',label='ROC(area={0:.2f})'.format(roc_auc),lw=2)
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.show()







