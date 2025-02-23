import sys
import os
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, eval_model

@dataclass
class ModelTrainerConfig:
    train_model_file_path= os.path.join('artifacts','train_model.pkl')

class ModelTrainer:
    def __init__(self,):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Split train and test data")
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            model_report:dict=eval_model(
                X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models
            )
        
            #  To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("Model score is less than 0.6",sys)
            logging.info(f"Best Model: {best_model_name}")
            
            save_object(self.model_trainer_config.train_model_file_path,best_model)

            predict = best_model.predict(X_test)
            r2 = r2_score(y_test, predict)
            logging.info(f"R2 Score: {r2}")
            return r2
        except Exception as e:
            raise CustomException(e,sys)