import os
import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score , accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , MinMaxScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import pickle



def process(fname ='diabetes_012_health_indicators_BRFSS2015.csv'):
    data = pd.read_csv(fname)
    data.rename(columns = {'Diabetes_012':'target'}, inplace = True)
    y = data["target"]
    X = data.drop(['target'], axis = 1)
    print(y.value_counts())
    standard , minmax , enc =  StandardScaler() , MinMaxScaler(), OneHotEncoder()
    
    continuous = ["BMI", "PhysHlth",  "MentHlth"] # continuous
    ordinal = ['Income', "Education", "Age"] # ordinal with range
    one_hot = ['GenHlth'] # ordinal, categorical 
    binary = [] #"everything else"

    for col in list(X.columns):
        if col in ordinal:
            X[col] = minmax.fit_transform(np.array(X[col]).reshape(len(X),1))
        if col in continuous:
            X[col] = standard.fit_transform(np.array(X[col]).reshape(len(X),1))
        if col in one_hot:
            one_hot = pd.get_dummies(X[col])
            names = [col+"_" +str(list(one_hot)[i]) for i in range(len(list(one_hot)))]
            one_hot.columns = names
            X = pd.concat([X, one_hot], axis = 1)
        else:
            binary.append(col)
    return X,y

def class_partition(X,y):

    X0 = X.iloc[np.where(y==0)]
    X1 = X.iloc[np.where(y==1)]
    X2 = X.iloc[np.where(y==2)]
    
    return  [X0 , X1, X2],  [y.iloc[np.where(y==0)], y.iloc[np.where(y==1)] , y.iloc[np.where(y==2)]]

def balanced_binary_grid_search(X_train, y_train, grid, scoring = 'f1_weighted', c1 = 0, c2 = 2, num_searches = 5):
    counts = list(y_train.value_counts())
    num_models = int(max(counts) / min(counts))
    sample_size = min(counts)
    Xs, ys = class_partition(X_train,y_train)
    X0, X1 = Xs[c1], Xs[c2]
    y0, y1 = ys[c1], ys[c2]
    print(sample_size)
    print("Starting Grid Search on {}-{} classifier".format( c1, c2))
    input()
    best, data, scores = [], [], []            
    for ii in range(num_searches):
        if len(X0) > sample_size:
            part0 = X0.sample(sample_size)
            X0 = X0.drop(part0.index)
        else:
            part0 = Xs[c1].sample(sample_size)
            X0 = Xs[c1]
        if len(X1) > sample_size:
            part1 = X1.sample(sample_size)
            X1 = X1.drop(part1.index)
        else:
            part1 = Xs[c2].sample(sample_size)
            X1 = Xs[c2]
        print("Searching Step : " , ii)
        X_ii , y_ii = pd.concat( [part0, part1]) , pd.concat( [ y0.iloc[0:sample_size] , y1.iloc[0:sample_size] ])
        clf = RandomForestClassifier()
        search = GridSearchCV(clf, param_grid = grid, scoring = scoring, cv=KFold(n_splits=3) )
        search.fit(X_ii, y_ii)
        data.append(pd.DataFrame( search.cv_results_).sort_values("rank_test_score").iloc[0:10])
        best.append(search.best_estimator_)
        scores.append(search.best_score_)
    print("done")
    return best , pd.concat(data), scores

def heatmap(ax,data):
    im = ax.imshow(data, cmap = 'YlGnBu')
    ax.set_xticklabels = [0,1,2]
    ax.set_yticklabels = [0,1,2]


def confusion(y_test, y_pred):

#     print("Accuracy: ", accuracy_score(y_test, y_pred))
#     # print("AUC: " , sklearn.metrics.roc_auc_score(y_test, y_proba, multi_class = 'ovr') )
#     print("F1 Score: ", f1_score(y_test, y_pred, average = "weighted"))

    fig, ax = plt.subplots(figsize = (8,6))
    # sns.heatmap(metrics.confusion_matrix(y_val, y_pred), annot = True, xticklabels = y_test.unique(), yticklabels = y_test.unique(), cmap = 'summer')
    grid = sklearn.metrics.confusion_matrix(y_test, y_pred)
    heatmap(grid)
    size = len(list(set(y_pred)))
    for i in range(size):
        for j in range(size):
            text = ax.text(j, i, grid[i, j],
                           ha="center", va="center", color="red",
                          font = {'weight' : 'bold', 'size'   : 12})

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig("Binary_Voting_Confusion_1.png")
    plt.show()

def decision(model01, model02, model12, X_test):
    
    pred01 = model01.predict(X_test)
    belief01= model01.predict_proba(X_test)
    pred02 = model02.predict(X_test)
    belief02 = model01.predict_proba(X_test)
    pred12 = model12.predict(X_test)
    belief12 = model01.predict_proba(X_test)
    
    y_pred = np.zeros(len(X_test))
    for i in range(len(X_test)):
        if pred01[i] == pred12[i]:
#             print("ones ", pred01[i] , pred12[i])
            y_pred[i] = pred01[i]
        elif pred12[i] == pred02[i]:
#             print("twos " , pred02[i] , pred12[i])
            y_pred[i] = pred12[i]
        elif pred01[i] == pred02[i]:
#             print("zeros" , pred01[i] , pred02[i])
            y_pred[i] = pred02[i]
        else:
#             print("tie")
            use = np.argmax( np.array([max(belief01[i]) , max(belief02[i]) , max(belief12[i])]) )
            y_pred[i] = list([pred01[i], pred02[i], pred12[i]])[int(use)]

    
    return y_pred

def grid_analysis( cv_results , n = 20 ):
    top = cv_results.sort_values("rank_test_score").iloc[0:n]
    fig, ax = plt.subplots(1,3)
    params = ['param_n_estimators', 'param_max_depth', 'param_min_samples_split' ]  #,  'param_min_samples_split' , 'param_min_samples_leaf']
    for n,p in enumerate(params):
        row = 0
        col = n % 3
        group1 = top.groupby(by=p).agg({p :'size',"mean_test_score":'mean',"mean_fit_time": "mean" }
                                                              ).sort_values("mean_test_score", ascending = False)
        group2 = top.groupby(by=p)["mean_test_score"].mean()
        print(group1)
        ax[col].plot(group2, linewidth = 2)
        ax[col].set_title(p, size = 14)  
        
    fig.tight_layout()
    
    return fig, ax

def stack_prediction(stack, X_test, y_test, printout = False):
    num_cat = len(np.unique(y_test))
    votes =  np.zeros([len(y_test) , num_cat])
    probas = np.zeros( [len(y_test) , num_cat] )
    print(num_cat)
    for n,m in enumerate(stack):
        y_pred = m.predict(X_test)
        y_proba = m.predict_proba(X_test)
        probas += y_proba
        # tallying loop
        print(num_cat)
        for n,pred in enumerate(list(y_pred)):
            if num_cat > 2:
                votes[n,int(pred)] += 1
            if num_cat == 2:
                index = int(pred == 0)
                votes[n,index] += 1
            if(printout):
                print("Individual Model Performance " , n) 
                print("Accuracy: ", accuracy_score(y_test, y_pred))
                print("AUC: " , sklearn.metrics.roc_auc_score(y_test, y_proba, multi_class = 'ovr') )
                print("F1 Score: ", f1_score(y_test, y_pred, average = "macro"))
                print("")
    vote_pred = votes.argmax(axis = 1)
#     proba_pred = probas.argmax(axis = 1)
    
    return vote_pred, probas / len(stack)