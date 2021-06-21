''' 
HOCHIMINH CITY APARTMENT PRICE PREDICTION
Data taken in June 2021
https://github.com/QuangTranUTE/Housing-Price-Prediction 
quangtn@hcmute.edu.vn

**Disclaimer:** This code serves research and educational purpose only. No guarantee of accuracy for other purposes.

INSTRUCTIONS:
    TO DO ###############
    + Run entire code: if you want to train your model from scratch. You can easily customize the model and stuff by changing hyperparameters put at the beginning of code parts (marked with comments NOTE: HYPERPARAM)
    + Run only Part 1 & Part 4: if you already trained (and saved) a model and want to do prediction (review analysis).
For other instructions, such as how to prepare your data, please see the github repository given above.

Reference: 
Some parts of this code are based on Chapter 2 in: Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (2nd ed.) by Aurélien Géron. 
'''


# In[1]: PART 1. IMPORT AND FUNCTIONS
#region
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  
from sklearn.preprocessing import OneHotEncoder      
from statistics import mean
from sklearn.model_selection import KFold   
import joblib
#endregion



# In[2]: PART 2. GET THE DATA 
raw_data = pd.read_csv('datasets\GiaChungCu_HCM_June2021_laydulieu_com.csv')



# In[3]: PART 3. DISCOVER THE DATA 
#region
# 3.1 Quick view of the data
print('\n____________ Dataset info ____________')
print(raw_data.info())              
print('\n____________ Statistics of numeric features ____________')
print(raw_data.describe())    
 
# 3.2 Scatter plot pair of features
from pandas.plotting import scatter_matrix   
features_to_plot = ["GIÁ - TRIỆU ĐỒNG", "SỐ PHÒNG", "SỐ TOILETS", "DIỆN TÍCH - M2"]
scatter_matrix(raw_data[features_to_plot], figsize=(12, 8)) 
#plt.savefig('figures/scatter_mat_all_feat.png', format='png', dpi=300)
plt.show()

# 3.3 Plot histogram of numeric features
raw_data.hist(figsize=(10,5))  
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.tight_layout()
# plt.savefig('figures/hist_raw_data.png', format='png', dpi=300) # must save before show()
plt.show()

# 3.4 Compute correlations b/w features and label
corr_matrix = raw_data.corr()
print(corr_matrix["GIÁ - TRIỆU ĐỒNG"].sort_values(ascending=False)) 
#endregion



# In[4]: PART 4. PREPARE THE DATA 
#region
# 4.1 Remove unused features
raw_data.drop(columns = ["GIỐNG - LOẠI", "GIỐNG - NHU CẦU", "GIỐNG - TỈNH THÀNH", "SỐ TẦNG"], inplace=True) 
 
# 4.2 Split training-test sets
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(raw_data, test_size=0.2, random_state=42) 

# 4.3 Separate labels from data, since we do not process label values
train_set_labels = train_set["GIÁ - TRIỆU ĐỒNG"].copy()
train_set = train_set.drop(columns = "GIÁ - TRIỆU ĐỒNG") 
test_set_labels = test_set["GIÁ - TRIỆU ĐỒNG"].copy()
test_set = test_set.drop(columns = "GIÁ - TRIỆU ĐỒNG") 

# 4.4 Define pipelines for processing data. 
# Define ColumnSelector: a transformer for choosing columns:
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, dataframe, labels=None):
        return self
    def transform(self, dataframe):
        return dataframe[self.feature_names].values    
num_feat_names = ['DIỆN TÍCH - M2', 'SỐ PHÒNG', 'SỐ TOILETS'] 
cat_feat_names = ['QUẬN HUYỆN', 'HƯỚNG', 'GIẤY TỜ PHÁP LÝ'] 

# Pipeline for categorical features:
cat_pipeline = Pipeline([
    ('selector', ColumnSelector(cat_feat_names)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="constant", fill_value = "NO INFO", copy=True)),
    ('cat_encoder', OneHotEncoder()) ])    

# Define MyFeatureAdder: a transformer for adding features "TỔNG SỐ PHÒNG",...  
class MyFeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_TONG_SO_PHONG = True): 
        self.add_TONG_SO_PHONG = add_TONG_SO_PHONG
    def fit(self, feature_values, labels = None):
        return self   
    def transform(self, feature_values, labels = None):
        SO_PHONG_id, SO_TOILETS_id = 1, 2 
        TONG_SO_PHONG = feature_values[:, SO_PHONG_id] + feature_values[:, SO_TOILETS_id]
        if self.add_TONG_SO_PHONG:
            feature_values = np.c_[feature_values, TONG_SO_PHONG] 
        return feature_values

# Pipeline for numerical features:
num_pipeline = Pipeline([
    ('selector', ColumnSelector(num_feat_names)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="median", copy=True)),  
    ('attribs_adder', MyFeatureAdder(add_TONG_SO_PHONG = True)),
    ('std_scaler', StandardScaler(with_mean=True, with_std=True, copy=True)) ])  
  
# Combine features transformed by two above pipelines:
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline) ])  

# 4.5 Run the pipeline to process training data           
processed_train_set_val = full_pipeline.fit_transform(train_set)
print('\n____________ Processed feature values ____________')
print(processed_train_set_val[[0, 1, 2],:].toarray())
print(processed_train_set_val.shape)
print('We have %d numeric feature + 1 added features + 35 cols of onehotvector for categorical features.' %(len(num_feat_names)))
joblib.dump(full_pipeline, r'models/full_pipeline.pkl')

# (optional) Add header to create dataframe. Just to see. We don't need header to run algorithms 
if True: 
    onehot_cols = []
    for val_list in full_pipeline.transformer_list[1][1].named_steps['cat_encoder'].categories_: 
        onehot_cols = onehot_cols + val_list.tolist()
    columns_header = train_set.columns.tolist() + ["TỔNG SỐ PHÒNG"] + onehot_cols
    for name in cat_feat_names:
        columns_header.remove(name)
    processed_train_set = pd.DataFrame(processed_train_set_val.toarray(), columns = columns_header)
    print('\n____________ Processed dataframe ____________')
    print(processed_train_set.info())
    print(processed_train_set.head())
#endregion



# In[5]: TRAIN AND EVALUATE MODELS 
#region
# 5.1 Try LinearRegression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(processed_train_set_val, train_set_labels)
print('\n____________ LinearRegression ____________')
print('Learned parameters: ', model.coef_)

# Compute R2 score and root mean squared error
def r2score_and_rmse(model, train_data, labels): 
    r2score = model.score(train_data, labels)
    from sklearn.metrics import mean_squared_error
    prediction = model.predict(train_data)
    mse = mean_squared_error(labels, prediction)
    rmse = np.sqrt(mse)
    return r2score, rmse      
r2score, rmse = r2score_and_rmse(model, processed_train_set_val, train_set_labels)
print('\nR2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
        
# Predict labels for some training instances:
print("\nInput data: \n", train_set.iloc[0:9])
print("\nPredictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))
print("Labels:      ", list(train_set_labels[0:9]))

# Store models to files, to compare latter:
def store_model(model, model_name = ""):
    if model_name == "": 
        model_name = type(model).__name__
    joblib.dump(model,'models/' + model_name + '_model.pkl')
def load_model(model_name):
    model = joblib.load('models/' + model_name + '_model.pkl')
    return model
store_model(model)


#%% 5.2 Try DecisionTreeRegressor model
# Training:
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(processed_train_set_val, train_set_labels)
# Compute R2 score and root mean squared error:
print('\n____________ DecisionTreeRegressor ____________')
r2score, rmse = r2score_and_rmse(model, processed_train_set_val, train_set_labels)
print('\nR2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
store_model(model)
# Predict labels for some training instances:
print("\nPredictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))
print("Labels:      ", list(train_set_labels[0:9]))


#%% 5.3 Try RandomForestRegressor model
# Training:
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators = 5) 
model.fit(processed_train_set_val, train_set_labels)
# Compute R2 score and root mean squared error:
print('\n____________ RandomForestRegressor ____________')
r2score, rmse = r2score_and_rmse(model, processed_train_set_val, train_set_labels)
print('\nR2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
store_model(model)      
# Predict labels for some training instances:
print("\nPredictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))
print("Labels:      ", list(train_set_labels[0:9]))


#%% 5.4 Try polinomial regression model
# Training:
from sklearn.preprocessing import PolynomialFeatures
poly_feat_adder = PolynomialFeatures(degree = 2) 
train_set_poly_added = poly_feat_adder.fit_transform(processed_train_set_val)
new_training = 10
if new_training:
    model = LinearRegression()
    model.fit(train_set_poly_added, train_set_labels)
    store_model(model, model_name = "PolinomialRegression")      
else:
    model = load_model("PolinomialRegression")
# Compute R2 score and root mean squared error:
print('\n____________ Polinomial regression ____________')
r2score, rmse = r2score_and_rmse(model, train_set_poly_added, train_set_labels)
print('\nR2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
# Predict labels for some training instances:
print("\nPredictions: ", model.predict(train_set_poly_added[0:9]).round(decimals=1))
print("Labels:      ", list(train_set_labels[0:9]))


#%% 5.5 EVALUATE MODELS
from sklearn.model_selection import cross_val_score
print('\n____________ K-fold cross validation ____________')
run_new_evaluation = False
if run_new_evaluation:
    cv = KFold(n_splits=5,shuffle=True,random_state=37) 

    # Evaluate LinearRegression:
    model_name = "LinearRegression" 
    model = LinearRegression()             
    nmse_scores = cross_val_score(model, processed_train_set_val, train_set_labels, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    joblib.dump(rmse_scores,'saved_objects/' + model_name + '_rmse.pkl')
    print("LinearRegression rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')

    # Evaluate DecisionTreeRegressor:
    model_name = "DecisionTreeRegressor" 
    model = DecisionTreeRegressor()
    nmse_scores = cross_val_score(model, processed_train_set_val, train_set_labels, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    joblib.dump(rmse_scores,'saved_objects/' + model_name + '_rmse.pkl')
    print("DecisionTreeRegressor rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')

    # Evaluate RandomForestRegressor:
    model_name = "RandomForestRegressor" 
    model = RandomForestRegressor(n_estimators = 5)
    nmse_scores = cross_val_score(model, processed_train_set_val, train_set_labels, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    joblib.dump(rmse_scores,'saved_objects/' + model_name + '_rmse.pkl')
    print("RandomForestRegressor rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')

    # Evaluate Polinomial regression:
    model_name = "PolinomialRegression" 
    model = LinearRegression()
    nmse_scores = cross_val_score(model, train_set_poly_added, train_set_labels, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    joblib.dump(rmse_scores,'saved_objects/' + model_name + '_rmse.pkl')
    print("Polinomial regression rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')
else:
    # Load rmse
    model_name = "LinearRegression" 
    rmse_scores = joblib.load('saved_objects/' + model_name + '_rmse.pkl')
    print("\nLinearRegression rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')

    model_name = "DecisionTreeRegressor" 
    rmse_scores = joblib.load('saved_objects/' + model_name + '_rmse.pkl')
    print("DecisionTreeRegressor rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')

    model_name = "RandomForestRegressor" 
    rmse_scores = joblib.load('saved_objects/' + model_name + '_rmse.pkl')
    print("RandomForestRegressor rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')

    model_name = "PolinomialRegression" 
    rmse_scores = joblib.load('saved_objects/' + model_name + '_rmse.pkl')
    print("Polinomial regression rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')
#endregion



# In[6]: FINE-TUNE MODELS 
# NOTE: this takes time.
#region
print('\n____________ Fine-tune models ____________')
def print_search_result(grid_search, model_name = ""): 
    print("\n====== Fine-tune " + model_name +" ======")
    print('Best hyperparameter combination: ',grid_search.best_params_)
    print('Best rmse: ', np.sqrt(-grid_search.best_score_))  
    print('Best estimator: ', grid_search.best_estimator_) # NOTE: require refit=True in  SearchCV
    print('Performance of hyperparameter combinations:')
    cv_results = grid_search.cv_results_
    for (mean_score, params) in zip(cv_results["mean_test_score"], cv_results["params"]):
        print('rmse =', np.sqrt(-mean_score).round(decimals=1), params) 

method = 1
# 6.1 Method 1: Grid search 
if method == 1:
    from sklearn.model_selection import GridSearchCV
    cv = KFold(n_splits=5,shuffle=True,random_state=37) 
        
    run_new_search = False      
    if run_new_search:        
        # Fine-tune RandomForestRegressor:
        model = RandomForestRegressor()
        param_grid = [
            {'bootstrap': [True], 'n_estimators': [3, 15, 30], 'max_features': [2, 12, 20, 39]},
            {'bootstrap': [False], 'n_estimators': [3, 5, 10, 20], 'max_features': [2, 6, 10]} ]
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_squared_error', return_train_score=True, 
        refit=True) 
        grid_search.fit(processed_train_set_val, train_set_labels)
        joblib.dump(grid_search,'saved_objects/RandomForestRegressor_gridsearch.pkl')
        print_search_result(grid_search, model_name = "RandomForestRegressor")      

        # Fine-tune Polinomial regression:          
        model = Pipeline([ ('poly_feat_adder', PolynomialFeatures()),  
                           ('lin_reg', LinearRegression()) ]) 
        param_grid = [
            {'poly_feat_adder__degree': [1, 2, 3]} ]
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_squared_error', return_train_score=True)
        grid_search.fit(processed_train_set_val, train_set_labels)
        joblib.dump(grid_search,'saved_objects/PolinomialRegression_gridsearch.pkl') 
        print_search_result(grid_search, model_name = "PolinomialRegression") 
    else:
        # Load grid_search
        grid_search = joblib.load('saved_objects/RandomForestRegressor_gridsearch.pkl')
        print_search_result(grid_search, model_name = "RandomForestRegressor")         
        grid_search = joblib.load('saved_objects/PolinomialRegression_gridsearch.pkl')
        print_search_result(grid_search, model_name = "PolinomialRegression") 

# 6.2 Method 2: Random search 
elif method == 2:
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint  
    cv = KFold(n_splits=5,shuffle=True,random_state=37)
        
    run_new_search = 0     
    if run_new_search:
        # Fine-tune RandomForestRegressor:
        model = RandomForestRegressor(random_state=48)
        param_distribs = {
            'n_estimators': randint(low=1, high=150),
            'max_features': randint(low=1, high=39),
            'bootstrap':  randint(low=0, high=2) } 
        rnd_search = RandomizedSearchCV(model, param_distributions=param_distribs, n_iter=30, cv=cv, scoring='neg_mean_squared_error', random_state=42)
        rnd_search.fit(processed_train_set_val, train_set_labels)
        joblib.dump(rnd_search,'saved_objects/RandomForestRegressor_randsearch.pkl') 
        print_search_result(rnd_search, model_name = "RandomForestRegressor") 

        # Fine-tune Polynomial regression:
        # CAUTION: High degree polynomial consumes much memory, you may run out of RAM. 
        model = Pipeline([ ('poly_feat_adder', PolynomialFeatures()),
                           ('lin_reg', LinearRegression()) ]) 
        param_distribs = {
            'poly_feat_adder__degree': randint(low=1, high=4) }     
        rnd_search = RandomizedSearchCV(model, param_distributions=param_distribs, n_iter=2, cv=cv, scoring='neg_mean_squared_error', random_state=42)
        rnd_search.fit(processed_train_set_val, train_set_labels)
        joblib.dump(rnd_search,'saved_objects/PolinomialRegression_randsearch.pkl') 
        print_search_result(rnd_search, model_name = "PolinomialRegression") 
    else:
        # Load grid_search
        rnd_search = joblib.load('saved_objects/RandomForestRegressor_randsearch.pkl')
        print_search_result(rnd_search, model_name = "RandomForestRegressor")         
        rnd_search = joblib.load('saved_objects/PolinomialRegression_randsearch.pkl')
        print_search_result(rnd_search, model_name = "PolinomialRegression") 
#endregion



# In[7]: ANALYZE AND TEST THE BEST MODEL
#region:
# 7.1 Pick the best model (random forest):
search = joblib.load('saved_objects/RandomForestRegressor_gridsearch.pkl')
best_model = search.best_estimator_

# 7.2 Analyse the solution to get more insights about the data:
# NOTE: ONLY for rand forest
print('\n____________ ANALYZE AND TEST YOUR SOLUTION ____________')
print('SOLUTION: ' , best_model)
store_model(best_model, model_name="SOLUTION")   

if type(best_model).__name__ == "RandomForestRegressor":
    # Print features and importance score (ONLY on rand-forest)
    feature_importances = best_model.feature_importances_
    onehot_cols = []
    for val_list in full_pipeline.transformer_list[1][1].named_steps['cat_encoder'].categories_: 
        onehot_cols = onehot_cols + val_list.tolist()
    feature_names = train_set.columns.tolist() + ["TỔNG SỐ PHÒNG"] + onehot_cols
    for name in cat_feat_names:
        feature_names.remove(name)
    print('\nFeatures and importance score: ')
    print(*sorted(zip( feature_names, feature_importances.round(decimals=4)), key = lambda row: row[1], reverse=True),sep='\n')

#%% 7.3 Run on test data:
full_pipeline = joblib.load(r'models/full_pipeline.pkl')
best_model = joblib.load(r'models/SOLUTION_model.pkl')
processed_test_set = full_pipeline.transform(test_set)  
# Compute R2 score and root mean squared error:
r2score, rmse = r2score_and_rmse(best_model, processed_test_set, test_set_labels)
print('\nPerformance on test data:')
print('R2 score (on test data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
# Predict labels for some test instances:
print("\nTest data: \n", test_set.iloc[0:9])
print("\nPredictions: ", best_model.predict(processed_test_set[0:9]).round(decimals=1))
print("Labels:      ", list(test_set_labels[0:9]),'\n')

#endregion



# %%
