### Importing modules

import pandas as pd  ## for data frame analysis
import numpy as np   ## for analysis

import matplotlib.pyplot as plt  ## for visulization
import seaborn as sns ## plotting

#import statsmodels.api as sm ## plotting

import scipy.stats as stats

import os  ##  for path and os

from feature_engine.encoding import OneHotEncoder as fe_OneHotEncoder  ## encoding of categorical variables
from sklearn.preprocessing import StandardScaler  ### to scale the variables between 0 and 1 for OLS regression

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import ElasticNet

# for feature engineering
from feature_engine import imputation as mdi
from feature_engine import discretisation as dsc
from feature_engine import encoding as ce

from sklearn.metrics import mean_squared_error, r2_score , mean_absolute_error

import catboost as cb

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.feature_selection import RFECV

from importlib import reload
import utility_func




def diagnostic_plots(df,figsize):

    '''
    # function to plot a histogram and a Q-Q plot
    # side by side, for a certain variable

    '''
    numeric_var = df.select_dtypes(include=[np.number]).columns.tolist()
    # plt.figure(figsize=figsize)
    for var in numeric_var :

        #plt.figure(figsize=figsize, dpi= 100, facecolor='w', edgecolor='k')

        plt.rcParams['figure.figsize'] = figsize
        fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

        ax1.hist(df[var],bins=30)

        ax1.set_title("Histrogram for the variable "+str(var))


        stats.probplot(df[var], dist="norm", plot=ax2)

        #ax2.set_title("QQ PRobabbility plot for variable "+str(var))


        plt.show()


def scatter_plot(df,target,figsize):
    '''
    # function to plot a scatter plot between two numerical variables

    '''
    ## get the list of numerical variables from the df

    numeric_var = df.select_dtypes(include=[np.number]).columns.tolist()

    ## This list also include the target variables so lets exlcue it from list

    numeric_var.remove(target)

    #fig,(ax1,ax2,ax3)=plt.figure(1,3)

    for var in numeric_var :
        x = df[var]
        y = df[target]

        #plt.figure(figsize=figsize, dpi= 100, facecolor='w', edgecolor='k')
        plt.rcParams['figure.figsize'] = figsize

        fig,ax1 = plt.subplots(nrows=1,ncols=1)
        #fig,(ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3)

        ax1.scatter(x, y, marker='o',c='green',edgecolor='black',linewidth=1,alpha=.75)

        ax1.set_xlabel(var)
        ax1.set_ylabel(target)
        ax1.set_title(" Scatter plot for "+str(target)+" and "+str(var))


        plt.show()


def correlation(df,figsize)  :

    numeric_var = df.select_dtypes(include=[np.number]).columns.tolist()

    corr_df = df[numeric_var].corr()

    f = plt.figure(figsize=figsize)
    plt.rcParams['figure.figsize'] = figsize
    plt.matshow(corr_df, fignum=f.number)
    plt.xticks(range(df[numeric_var].select_dtypes(['number']).shape[1]),
                                                        df[numeric_var].select_dtypes(['number']).columns, fontsize=14, rotation=45)
    plt.yticks(range(df[numeric_var].select_dtypes(['number']).shape[1]),
                                        df[numeric_var].select_dtypes(['number']).columns, fontsize=14)
    cbr = plt.colorbar()
    cbr.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)

    plt.show()

    return corr_df


def vif(df,var_list,VIF_threshold=10):

    '''
    This Functionn Takes Three Parameters :
    df : Pandas DataFrame which will be used to calculate VIF
    var_list : Initial variables list which will be used to calculate VIF
    VIF_threshold : Highest threshold for VIF in the data. If a variable has more VIF then this, will be removed

    This Function return two lists :

    var_list : final variables list after removing highly corelated variables

    vif_var_list : Variables dropped because of VIF


    '''

    X = df[var_list]

    vif_var_list = []  ## variables removed from the list
    max_vif =100
    while max_vif > VIF_threshold:

        X = df[var_list]
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                              for i in range(len(X.columns))]
        var_to_remove = vif_data[vif_data["VIF"] == vif_data["VIF"].max()]['feature'].values[0]

        #Numeric_var_2.remove(var_to_remove)
        #X = scaled_train_df[Numeric_var]
        #del X_train_scaled[var_to_remove]

        print("current var is ",var_to_remove)
        max_vif = vif_data["VIF"].max()

        if max_vif >= 10 :

            vif_var_list.append(var_to_remove)

            var_list.remove(var_to_remove)

            print(var_to_remove, "is removed")


        else :
            break

        print("max vif is ",max_vif)

        return var_list,vif_var_list



def model_performance_statistics(model_object,df,feature_list,target,col_name,figsize,classification=False,ols=True) :

    '''
    This function print and return  MSE, RMSE,r square and mean absolute error
    '''

    if classification == False :  ### If its classification model then we need to add code for AUC, accuracy etc

        df[col_name] = model_object.predict(df[feature_list])

        mse = mean_squared_error(df[target], df[col_name])

        rmse = np.sqrt(mean_squared_error(df[target], df[col_name]))

        r_square = r2_score(df[target], df[col_name])

        m_abs_error = mean_absolute_error(df[target], df[col_name])

        print('test mse: {}'.format(mse))
        print('test rmse: {}'.format(rmse))
        print('test r2: {}'.format(r_square))
        print('test mean absolute error: {}'.format(m_abs_error))


        if ols ==True :  #

            df_imp  = pd.DataFrame({'Variable_Name':feature_list,'Variable_Importance':model_object.coef_})

            df_imp['abs_coeff'] = np.abs(df_imp['Variable_Importance'])

            df_imp.sort_values(by=['abs_coeff'],ascending=False,inplace=True)
        else :

            df_imp = pd.DataFrame({'Variable_Name':feature_list,'Variable_Importance':model_object.feature_importances_})
            df_imp.sort_values(by=['Variable_Importance'],ascending=False,inplace=True)

        df_imp = df_imp.set_index('Variable_Name')

        plt.rcParams['figure.figsize'] = figsize

        df_imp['Variable_Importance'].plot.bar( )

        plt.title("Variable importance plot")

        plt.show()

        plt.scatter(df[target],df[col_name])
        plt.xlabel('True '+str(target))
        plt.ylabel('Predicted '+str(target))

        plt.title("True Vs Predicted")

        return df,df_imp,mse,rmse,r_square,m_abs_error




def grid_search_result_summary(grid_object,figsize,graph=False) :

    Grid_result = pd.DataFrame(grid_object.cv_results_['params'])

    Grid_result['means'] = grid_object.cv_results_['mean_test_score']
    Grid_result['abs_means'] = np.abs(Grid_result['means'])
    Grid_result['stds'] = grid_object.cv_results_['std_test_score']
    Grid_result = Grid_result.reset_index()
    Grid_result.sort_values(['abs_means','stds'],inplace=True)


    #plt.figure(figsize=figsize, dpi= 100, facecolor='w', edgecolor='k')

    if graph==True :
        plt.rcParams['figure.figsize'] = figsize

        ax = Grid_result.plot(kind = 'line', x = 'index',
                          y = 'abs_means', color = 'Blue',
                          linewidth = 3)
        #plt.figure(figsize=(200,160))
        ax2 = Grid_result.plot(kind = 'bar', x = 'index',
                           y = 'stds', secondary_y = True,
                           color = 'Red',  linewidth = 3,
                           ax = ax)

        #title of the plot

        plt.title("Grid Search Result Summary")

        #labeling x and y-axis
        ax.set_xlabel('index', color = 'g')
        ax.set_ylabel('abs_means', color = "b")
        ax2.set_ylabel('stds', color = 'r')

        #defining display layout
        plt.tight_layout()

        plt.show()

    return Grid_result






def rfe_cv_backward(df,target,int_var_list,metric,parm_grid,final_feature_num,estimator,random_state=0,validation_data=None,cv=5,

                                       n_jobs=-1 , cross_validation=True) :

    ''' rfe_cv_backward function is a feature selection method. For Each oteration, it tries to drop all
    features one by one and then finally drop the features which causes least drop in overall model performance

    df : Modeling data
    target : Name of the Target Variable
    int_var_list : List of the initial feature selection
    metric : Metric to be use for model performancce evaluation
    parm_grid : parameter grid for grid search
    final_feature_num : Count of final features we want to keep in model
    estimator : Skitlearn Classifier or regressor

    random_state : For reproducibility
    validation_data : If we are not using cross validation then validation data for perofrmance evaluation

    cv : number of fold for cross validation

    n_jobs : cores to be used for parallel processing

    cross_validation : if not True we have to provide validation data set

    '''

    var_current = int_var_list  ## List to keep the features after dropping variable in eact iteration

    dropped_var_list = []  ### List to keep dropped variables after each iteration

    score_aft_var_drop = []  ## list to keep the final score after drop of variable in each iteration

    len_var_current =[] ## List to keep the number of variables left after each iteration

    ## For first iteration it should be equal to the initial variable list


    ## overall_performance dataframe will keep the record of perofrmance after dropping variabl in each iteration
    
    overall_performance = pd.DataFrame(columns=['Iteration_id','Model_id','Dropped_variable','validation_score'])

    dropped_perf_summmary = pd.DataFrame(columns=['Iteration_id','Dropped_variable','Num_feature','validation_score'])

    Iteration_id = 0 ## Initialization of iteration_id


    while len(var_current) > final_feature_num :

        temp_performance = pd.DataFrame(columns=['Model_id','Dropped_variable','validation_score'])

        temp_dropped_perf_summmary = pd.DataFrame(columns=['Iteration_id','Dropped_variable','Num_feature','validation_score'])

        Model_id = 0 ## Initialization of model_id for each iteration if we have n number of features then we build n models
        ## by dropping them one by one and will finally drop the feature which has least performance impact

        for var in var_current : ## Iterate through all variables and drop them one by one

            var_test = list(set(var_current)-set([var]))

            if cross_validation == True  :

                rfecv = GridSearchCV(estimator, parm_grid, scoring=metric, cv=cv, n_jobs=n_jobs,verbose=False)
                # perform the search
                rfecv.fit(df[var_test], df[target])

                temp_performance.loc[Model_id] = [Model_id,var,rfecv.best_score_]

#                 temp_performance['Dropped_variable'] = var

#                 temp_performance['validation_score'] =

                Model_id = Model_id+1

                del var_test

        ### For each iteration finally we have to drop the variable with least importance
        ### The variable with least importance will have the highest validation_score
        ### We can sort the temp_performance_sort by validation_score and sort it in decreaing order
        ### Pick the First var. This variable will be least important

        temp_performance_sort = temp_performance.sort_values('validation_score',ascending=False)

        del temp_performance

        print('temp_performance_sort is ',temp_performance_sort)

        Dropped_variable = temp_performance_sort.iloc[0].values[1]

        score_aft_var_drop = temp_performance_sort.iloc[0].values[2]

        ### Update the List of Dropped Variables

        dropped_var_list.append(Dropped_variable)

        ### Update the list of var_current by dropping dropped variable

        var_current = list(list(set(var_current)-set([Dropped_variable])))

        len_var_current = len(var_current)


        temp_performance_sort['Iteration_id'] = Iteration_id

        overall_performance = overall_performance.append(temp_performance_sort)

        temp_dropped_perf_summmary.loc[Iteration_id] = [Iteration_id,Dropped_variable,len(var_current),score_aft_var_drop]
#         temp_dropped_perf_summmary['Dropped_variable'] = Dropped_variable
#         temp_dropped_perf_summmary['Num_feature'] = len(var_current)
#         temp_dropped_perf_summmary['validation_score'] = score_aft_var_drop

        dropped_perf_summmary = dropped_perf_summmary.append(temp_dropped_perf_summmary)

        print("current iteration is ",Iteration_id)

        print("dropped variabless is ",Dropped_variable)


        Iteration_id = Iteration_id+1

        del temp_dropped_perf_summmary,temp_performance_sort



    return overall_performance,dropped_perf_summmary

    #IndexError: single positional indexer is out-of-bounds
