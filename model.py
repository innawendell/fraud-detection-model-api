import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import joblib
import os

file_path = 'data/onlinefraud.csv'
# specify numeric and categorical features in the data
numeric_features = ['step', 
                    'amount', 
                    'oldbalanceOrg', 
                    'newbalanceOrig',
                    'oldbalanceDest',
                    'newbalanceDest']

categorical_features = ['type', 
                        'isFlaggedFraud']

fraud_threshold = 0.97

def get_data(file_path, test_frac):
    fraud_df = pd.read_csv(file_path)
    # split data into features and target variable
    y = fraud_df['isFraud']
    X = fraud_df.drop(columns=['isFraud'])
    # split into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac)
    
    return X_train, X_test, y_train, y_test

def create_preprocessor(numeric_features, categorical_features):
    numeric_transformer = StandardScaler() 
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(transformers=[
                                 ('num', numeric_transformer, numeric_features),
                                 ('cat', categorical_transformer, categorical_features)])

    return preprocessor

def save_model(pipeline, file_name):
    exists= os.path.exists(folder)
    if not exists:
    	os.makedirs('models')
    	print('New directory was created.')

    joblib.dump(pipe_lr_param, filename)
    print('Model dumped')

X_train, X_test, y_train, y_test = get_data(file_path, 0.3)
preprocessor = create_preprocessor(numeric_features, categorical_features)

# create a Pipeline with Logistic Regression
# to tackle class imbalance, we will set class_weight to 'balanced', which will adjust weights 
# so that minority class is given more weight proportional to its frequency
# create a Pipeline with Logistic Regression
# to tackle class imbalance, we will set class_weight to 'balanced', which will adjust weights 
# so that minority class is given more weight proportional to its frequency
lr_pipeline = Pipeline([
                       ('preprocessor', preprocessor),
                       ('lr', LogisticRegression(class_weight='balanced',
                       	solver='saga'))
                       ])

param_grid = [
            {'lr__C': [1e-3, 10, 1e2, 1e3],
            'lr__penalty': ['l1', 'l2']
            }
            ]
# find the best parameters
grid_search = GridSearchCV(lr_pipeline, 
                           param_grid, 
                           cv=2,
                           scoring='roc_auc', 
                           verbose=1)
# fit the model
grid_search.fit(X_train, y_train)

print("Best Score: ", grid_search.best_score_)
print("Best Params: ", grid_search.best_params_)

lr_pipeline = Pipeline([
                       ('preprocessor', preprocessor),
                       ('lr', LogisticRegression(class_weight='balanced', 
                                                 penalty=grid_search.best_params_['lr__penalty'], 
                                                 C=grid_search.best_params_['lr__C'], 
                                                 warm_start=True))
                       ])
pipeline_model = lr_pipeline.fit(X_train, y_train)
y_pred = pipeline_model.predict_proba(X_test)
print('roc-auc score:', format(roc_auc_score(y_test, 
                                             y_pred[:, 1]), 
                                             '0.3f'))
precision = round(precision_score(y_test, [1 if pred >= fraud_threshold else 0 for pred in y_pred[:, 1]]), 3)
recall = round(recall_score(y_test, [1 if pred >= fraud_threshold else 0 for pred in y_pred[:, 1]]), 3)


# save the model pipeline
pipe_lr_param = {}
pipe_lr_param['pipeline'] = pipeline_model
pipe_lr_param['params'] = grid_search.best_params_
pipe_lr_param['numeric_features'] = numeric_features
pipe_lr_param['categorical_features'] = categorical_features
pipe_lr_param['precision'] = precision
pipe_lr_param['recall'] = recall
pipe_lr_param['threshold'] = fraud_threshold

# save the pipeline in the models folder
save_model(pipe_lr_param, 'models', 'models/lr_model_pipeline.joblib')

