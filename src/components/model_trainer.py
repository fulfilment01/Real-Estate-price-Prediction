import os 
import sys 
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array, test_array):
        try:
            logging.info("Spliting training and testing data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(max_iter=10000),
                "Ridge": Ridge(max_iter=10000),
                # "Gradient Boosting": GradientBoostingRegressor(),
                # "K-Neighbors Regressor": KNeighborsRegressor(n_jobs=-1),
                # "Decision Tree": DecisionTreeRegressor(),
                # "Random Forest": RandomForestRegressor(n_jobs=-1),
                # "XGB Regressor": XGBRegressor(n_jobs=-1),
                # "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                # "AdaBoost Regressor": AdaBoostRegressor(),
                # "SVM": SVR()   
            }
            params = {
                # "Decision Tree": {
                #     'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                #     'splitter': ['best', 'random'],
                #     'max_features': ['auto', 'sqrt', 'log2', None],
                #     'max_depth': [None, 10, 20, 30, 40, 50],
                #     'min_samples_split': [2, 5, 10],
                #     'min_samples_leaf': [1, 2, 4]
                # },
                # "Random Forest": {
                #     'n_estimators': [50, 100, 200, 300, 400, 500],
                #     'criterion': ['squared_error', 'absolute_error', 'poisson'],
                #     'max_features': ['auto', 'sqrt', 'log2'],
                #     'max_depth': [None, 10, 20, 30, 40, 50],
                #     'min_samples_split': [2, 5, 10],
                #     'min_samples_leaf': [1, 2, 4],
                #     'bootstrap': [True, False]
                # },
                # "Gradient Boosting": {
                #     'loss': ['squared_error', 'huber', 'absolute_error', 'quantile'],
                #     'learning_rate': [0.01, 0.05, 0.1, 0.2],
                #     'n_estimators': [100, 200, 300, 400],
                #     'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                #     'criterion': ['friedman_mse', 'squared_error'],
                #     'max_depth': [3, 4, 5, 6],
                #     'min_samples_split': [2, 5, 10],
                #     'min_samples_leaf': [1, 2, 4],
                #     'max_features': ['auto', 'sqrt', 'log2']
                # },
                "Linear Regression": {
                    'fit_intercept': [True, False],
                    'copy_X': [True, False],
                    'n_jobs': [None, -1],
                    'positive': [True, False]
                },
                "Ridge": {
                    'alpha': [0.1, 1.0, 10.0, 100.0],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga']
                },
                "Lasso": {
                    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
                    'max_iter': [1000, 5000, 10000],
                    'selection': ['cyclic', 'random']
                },
                # "XGB Regressor": {
                #     'learning_rate': [0.01, 0.05, 0.1, 0.2],
                #     'n_estimators': [100, 200, 300, 400],
                #     'max_depth': [3, 4, 5, 6],
                #     'min_child_weight': [1, 3, 5],
                #     'subsample': [0.6, 0.7, 0.8, 0.9],
                #     'colsample_bytree': [0.6, 0.7, 0.8, 0.9]
                # },
                # "CatBoosting Regressor": {
                #     'depth': [6, 8, 10],
                #     'learning_rate': [0.01, 0.05, 0.1],
                #     'iterations': [100, 200, 300]
                # },
                # "AdaBoost Regressor": {
                #     'n_estimators': [50, 100, 200, 300],
                #     'learning_rate': [0.01, 0.05, 0.1, 0.2],
                #     'loss': ['linear', 'square', 'exponential']
                # },
                # "K-Neighbors Regressor": {
                #     'n_neighbors': [3, 5, 7, 9, 11],
                #     'weights': ['uniform', 'distance'],
                #     'metric': ['euclidean', 'manhattan', 'minkowski']
                # },
                # "SVM": {
                #     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                #     'C': [0.1, 1, 10, 100],
                #     'epsilon': [0.1, 0.2, 0.3],
                #     'degree': [3, 4, 5],
                #     'gamma': ['scale', 'auto']
                # }
            }


            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)

            # to get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # to get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score <0.6:
                raise CustomException("No best model found",sys)
            
            logging.info(f"Best model found on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test,predicted)
            return r2_square
        
        except Exception as e:
            raise CustomException(e,sys)