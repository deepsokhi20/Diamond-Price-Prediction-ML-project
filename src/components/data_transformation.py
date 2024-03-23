from sklearn.impute import SimpleImputer ##handling missing values
from sklearn.preprocessing import StandardScaler ##handling feature scaling
from sklearn.preprocessing import OrdinalEncoder ##ordinal Encoding
##pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import sys,os
from dataclasses import dataclass
import pandas as pd
import numpy as np
from src.components.data_ingestion import DataIngestion

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

##data transformation config
@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

##data Ingestion class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig()
    
    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            ##define which columns should be ordinal-encoded and which should be scaled
            categorical_cols=['cut','color','clarity']
            numerical_cols=['carat','depth','table','x','y','z']

            ##define the custom ranking for each ordinal variable
            cut_categories=['Fair','Good','Very Good','Premium','Ideal']
            color_categories=['D','E','F','G','H','I','J']
            clarity_categories=['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info('Pipeline Initiated')

            ##numerical pipeline
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            ## categorical pipeline
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ('scaler',StandardScaler())
                ]
            )

            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipeline',cat_pipeline,categorical_cols)
            ]
            )

            return preprocessor
        
            logging.info('Pipeline completed')
        except Exception as e:
            logging.info('Eror in Data Transformation')
            raise CustomException(e,sys)



    def initiate_data_transformation(self,train_path,test_path):
        try:
            ## Reading train and test data
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head:\n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head :\n{test_df.head().to_string()}')
            logging.info('Obtaining preprocessing object')
            preprocessing_obj=self.get_data_transformation_object()

            target_column_name='price'
            drop_columns=[target_column_name,'id']

            ##featues into dependent and independent features

            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=train_df[target_column_name]

            ##apply the transformation
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_train_df)
            logging.info('Applying preprocessing object on trainig and testing datasets.')


            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]


            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info('preprocessor pickle is created and saved')
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    obj=DataTransformation()
    obj.initiate_data_transformation( train_data_path,test_data_path)
