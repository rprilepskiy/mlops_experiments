from sklearn.metrics import accuracy_score, roc_auc_score
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow

mlflow.set_tag('model_type', 'catboost')
iterations = 10
mlflow.log_param('iterations', iterations)
 
data = pd.read_csv('data/feature_generated.csv')
X, y = data.drop('Risk', axis=1), data['Risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
params = {'n_estimators': 1, 'learning_rate': 0.1, 'depth': 12}
mlflow.log_params(params)
model = CatBoostClassifier(**params)
 
for iter in range(iterations):
    if model.is_fitted():
        model.fit(X_train, y_train, init_model=model)
    else:
        model.fit(X_train, y_train)
    predicted_train = model.predict(X_train)
    predicted_test = model.predict(X_test)
    mlflow.log_metric('Accuracy Train', accuracy_score(y_train, predicted_train), step=iter*params['n_estimators'])
    mlflow.log_metric('Accuracy Test', accuracy_score(y_test, predicted_test), step=iter*params['n_estimators'])
    mlflow.log_metric('ROC AUC Train', roc_auc_score(y_train, predicted_train), step=iter * params['n_estimators'])
    mlflow.log_metric('ROC AUT Test', roc_auc_score(y_test, predicted_test), step=iter * params['n_estimators'])
 
    model.save_model('catboostmodel.cbm')
    mlflow.log_artifact('catboostmodel.cbm')