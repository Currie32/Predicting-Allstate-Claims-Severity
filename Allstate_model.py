
# coding: utf-8

# This is the code for my best submission in Kaggle's "Allstate Claims Severity" competition. I found this competition rather interesting because I have never worked with such anonymized data before. The goal for this competition was to minimize the mean average error when predicting the values of the insurance claims (this feature is referred to as loss). If you would like to learn more about the competition, visit https://www.kaggle.com/c/allstate-claims-severity
# 
# The sections of this analysis are:
# -Inspecting the data 
# -Building the model
# -Summary

# In[ ]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy import stats
import xgboost as xgb
import operator


# In[ ]:

#display max rows and columns
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows',1000)


# First things first, let's load the data.

# In[ ]:

train = pd.read_csv("/Users/Dave/Desktop/Programming/Personal Projects/Allstate-Kaggle/train.csv")
test = pd.read_csv("/Users/Dave/Desktop/Programming/Personal Projects/Allstate-Kaggle/test.csv")


# # Inspecting the Data

# In[ ]:

train.head()


# In[ ]:

train.describe()


# In[ ]:

train.describe(include = ['object'])


# In[ ]:

print train.shape
print test.shape


# In[ ]:

train.isnull().sum()


# In[ ]:

test.isnull().sum()


# Good to see that there are no missing values.

# Let's take a closer look at our target feature, loss.

# In[ ]:

plt.hist(train.loss, bins = 50)
plt.show()


# Let's get a better picture of things by transforming loss by log.

# In[ ]:

plt.hist(train.loss, bins = 50)
plt.yscale('log')
plt.show()

train.loss.describe()


# There is quite the long-tail distribution here, let's see if a boxplot makes things a little easier to understand.

# In[ ]:

plt.boxplot(train.loss, 1)
plt.yscale("log")
plt.show()


# It's surprising to see a loss claim for as low as 67 cents, but the majority of the data is between 1000 and 4000 dollars. 

# Join the train and test datasets to inspect and transform the data quicker.

# In[ ]:

df = pd.concat([train,test], axis = 0, ignore_index = True)


# Group all of the continuous and categorical features together to make exploring and transforming the data easier.

# In[ ]:

cont_features = []
cat_features = []

for i in train.columns:
    if train[i].dtype == 'float':
        cont_features.append(i)
    elif train[i].dtype == 'object':
        cat_features.append(i)
        
for c in range(len(cat_features)):
    train[cat_features[c]] = train[cat_features[c]].astype('category').cat.codes


# In[ ]:

cont_features = []
cat_features = []

for i in test.columns:
    if test[i].dtype == 'float':
        cont_features.append(i)
    elif test[i].dtype == 'object':
        cat_features.append(i)
        
for c in range(len(cat_features)):
    test[cat_features[c]] = test[cat_features[c]].astype('category').cat.codes


# In[ ]:

for cat in train[cat_features]:
    print cat
    print
    print np.corrcoef(x = train.loss, y = train[cat])
    plt.hist(train[cat])
    plt.show()


# Most of the categorical features have uneven distributions. I suspect these feature are asking about what type of claim the customer has, i.e. auto, theft, fire.

# In[ ]:

plt.figure(figsize=(15,25))
gs = gridspec.GridSpec(8, 2)
for i, cn in enumerate(train[cont_features]):
    ax = plt.subplot(gs[i])
    sns.distplot(train[cn], bins=50)
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(cn))
plt.show()


# In[ ]:

cont_corr = train[cont_features].corr()
plt.subplots(figsize=(15, 12))
sns.set(style="white")
mask = np.zeros_like(cont_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(cont_corr, vmax=1, annot=True, mask = mask,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()


# In[ ]:

print train[cont_features].corrwith(train.loss).sort_values(ascending = False)


# Some of the continuous features have high correlations with each other, but nothing is very correlated with loss. I'm tempted to drop cont12 now because it is so highly correlated with cont11, but first I'm am going to look at how tranforming the features impact their correlation with loss.

# In[ ]:

for feature in train[cont_features]:
    print feature
    print "no transformation:", train[feature].corr(train.loss)
    print "sqrt transformation:", np.sqrt(train[feature]).corr(train.loss)
    print "log10 transformation:", np.log10(train[feature]).corr(train.loss)
    print


# In[ ]:

train.cont1 = np.log10(train.cont1)
train.cont2 = np.log10(train.cont2)
train.cont4 = np.log10(train.cont4)
train.cont5 = np.log10(train.cont5)
train.cont8 = np.log10(train.cont8)
train.cont10 = np.sqrt(train.cont10)
train.cont13 = np.log10(train.cont13)


# In[ ]:

test.cont1 = np.log10(test.cont1)
test.cont2 = np.log10(test.cont2)
test.cont4 = np.log10(test.cont4)
test.cont5 = np.log10(test.cont5)
test.cont8 = np.log10(test.cont8)
test.cont10 = np.sqrt(test.cont10)
test.cont13 = np.log10(test.cont13)


# Since I did not transform either cont11 or cont12, I will drop cont12.

# In[ ]:

train = train.drop('cont12', 1)
test = test.drop('cont12', 1)


# In[ ]:

labels = train.loss


# In[ ]:

trainFinal = train.drop('loss', 1)


# Let's check if train and test still have the same number of features.

# In[ ]:

print len(trainFinal.columns)
print len(test.columns)


# All's good there!

# # Building the Model

# In[ ]:

#Because we are transforming y (loss) by log, we want to square its value and 
#subtract the shift that we added to it initially.
def log_xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y)-shift,
                                      np.exp(yhat)-shift)


# In[ ]:

random_state = 2
params = {
        'eta': 0.03,
        'gamma': 0.5,
        'max_depth': 10,
        'min_child_weight': 6,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'alpha': 1,
        'lambda': 1.5,
        'verbose_eval': True,
        'seed': random_state,
    }
''' 
BEST PARAMETERS
params = {
        'eta': 0.03,
        'gamma': 0.5,
        'max_depth': 10,
        'min_child_weight': 6,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'alpha': 1,
        'lambda': 1.5,
        'verbose_eval': True,
        'seed': random_state,
    }
'''


# In[ ]:

from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

n_folds = 10
cv_sum = 0 #The sum of the mean_absolute_error for each fold.
early_stopping_rounds = 100
iterations = 10000
shift = 250
printN = 100
fpred = [] #stores the sums of predicted values for each fold.

trainScaled = trainFinal.apply(lambda x: MinMaxScaler().fit_transform(x))
testScaled = test.apply(lambda x: MinMaxScaler().fit_transform(x))

#Based on f-score and pearson-r, these feature are not useful in our prediction. 
#The code is written at the bottom of this analysis for these measures.
trainScaled = trainScaled.drop(['id','cat15','cat22','cat64','cat70','cat86','cat93','cat97','cat107','cat108'], 1)
testScaled = testScaled.drop(['id','cat15','cat22','cat64','cat70','cat86','cat93','cat97','cat107','cat108'], 1)

testFinal = xgb.DMatrix(testScaled)
ytrain = np.log(labels + shift) 

kf = KFold(trainScaled.shape[0], n_folds=n_folds)
for i, (train_index, test_index) in enumerate(kf):
    print('\n Fold %d' % (i+1))
    X_train, X_val = trainScaled.iloc[train_index], trainScaled.iloc[test_index]
    y_train, y_val = ytrain.iloc[train_index], ytrain.iloc[test_index]
    
    xgtrain = xgb.DMatrix(X_train, label=y_train)
    xgtest = xgb.DMatrix(X_val, label = y_val)
    watchlist = [(xgtrain, 'train'), (xgtest, 'eval')] 
    
    xgbModel = xgb.train(params, 
                         xgtrain, 
                         iterations, 
                         watchlist,
                         verbose_eval = printN,
                         early_stopping_rounds=early_stopping_rounds,
                         feval = log_xg_eval_mae
                        )
    
    scores_val = xgbModel.predict(xgtest, ntree_limit=xgbModel.best_ntree_limit)
    cv_score = mean_absolute_error(np.exp(y_val), np.exp(scores_val))
    print('log_eval-MAE: %.6f' % cv_score)
    y_pred = np.exp(xgbModel.predict(testFinal, ntree_limit=xgbModel.best_ntree_limit)) - shift
    print(xgbModel.best_ntree_limit)

    if i > 0:
        fpred = pred + y_pred
    else:
        fpred = y_pred
    pred = fpred
    cv_sum = cv_sum + cv_score


# In[ ]:

mpred = pred / n_folds
score = cv_sum / n_folds
print('Average eval-MAE: %.6f' % score)


# In[ ]:

print("Writing results")
result = pd.DataFrame(mpred)
result[id] = test.id
result.columns = ['loss','id']
result.to_csv('/Users/Dave/Desktop/Programming/Personal Projects/Allstate-Kaggle/result.csv', index=0)


# In[ ]:

result.head(10)


# # Summary

# Working with anonymized presents different challenges than working with standard data. Without the ability to confidently engineer new features, I was limited in the ways I could optimize my transformations of the data and build my model. Nonetheless, I still believe that I did a good job in creating a useful model. My score for the Kaggle competition was 1130.01427 (mean average error), the winning score was 1109.70772, and the 'Random Forest Benchmark' score was 1227.74974. Clearly my score is far closer to that of the winner, but there was still room for improvement. If I created an ensemble of models, transformed my features differently, or even used different features in my model, my score could have improved.
# 
# The five most important features in my model were: cont14, cont7, cont6, cat100, and cat112. Since I do not know what these features represent, we cannot draw much of a conclusion, but it's still a good practice to identify which features have the greatest impact on a model.

# In[ ]:

#Find the importance of each feature
importance = xgbModel.get_fscore()
importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
print importance
print len(importance)


# In[ ]:

for feature in trainFinal:
    print feature
    print stats.pearsonr(trainFinal[feature], labels)
    print 


# In[ ]:



