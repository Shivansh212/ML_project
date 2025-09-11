from sklearn.ensemble import(RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import os

from src.Exceptions import customException
from src.logger import logging
from src.Utils import save_object,evaluateModel

from dataclasses import dataclass
import sys


@dataclass
class Modeltrainerconfig():
    trained_model_file_path=os.path.join('artifact','model.pkl')

class Modeltrainer:
    def __init__(self):
        self.model_trainer_config=Modeltrainerconfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "XGB Regressor":XGBRegressor(),
                "CatBoost Regressor":CatBoostRegressor(verbose=0),
                "AdaBoost Regressor":AdaBoostRegressor()
            }
            model_report:dict=evaluateModel(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise customException("No best model found",sys)
            logging.info(f"Best model found on both training and testing dataset")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2_square=r2_score(y_test,predicted)
            return r2_square
        
        

        

        except Exception as e:
            raise customException(e, sys)
            pass




