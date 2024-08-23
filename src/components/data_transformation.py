import sys
import os
from dataclasses import dataclass
import numpy as np 
import pandas as pd 
import scipy.sparse as sp 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import category_encoders as ce
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        this function is responsible for data transformation
        """
        try:
            numerical_columns = ['Assessed Value','Sales Ratio',]
            low_cardinality_columns = ['List Year','Date Recorded','Property Type','Residential Type']

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="mean")),
                    ("scaler",StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown='ignore'))
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"categorical columns: {low_cardinality_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,low_cardinality_columns)    
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read Train and test data completed")

            logging.info("obtaining preprocessing objest")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="Sale Amount"
            useless_column_name="Profit"
            high_cardinality_columns = ['Town', 'Address']

            # Prepare input features and target variable
            input_feature_train_df=train_df.drop(columns=[target_column_name,useless_column_name], axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name,useless_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            # Subsample the training data (10% of the original dataset)
            subsample_fraction = 0.1
            X_sample, _, y_sample, _ = train_test_split(input_feature_train_df, target_feature_train_df, train_size=subsample_fraction, random_state=42)

            logging.info("Apply TargetEncoder to high cardinality features")
            target_encoder = ce.TargetEncoder()
            X_high_card_encoded_train = target_encoder.fit_transform(X_sample[high_cardinality_columns], y_sample)
            X_high_card_encoded_test = target_encoder.transform(input_feature_test_df[high_cardinality_columns])

            logging.info("Apply the remaining transformations to the low cardinality and numeric features")
            X_other_transformed_train = preprocessing_obj.fit_transform(X_sample.drop(columns=high_cardinality_columns))
            X_other_transformed_test = preprocessing_obj.transform(input_feature_test_df.drop(columns=high_cardinality_columns))

            # Convert sparse matrices to dense numpy arrays if needed
            if sp.issparse(X_other_transformed_train):
                X_other_transformed_train = X_other_transformed_train.toarray()
            if sp.issparse(X_other_transformed_test):
                X_other_transformed_test = X_other_transformed_test.toarray()

            # Convert DataFrames to numpy arrays
            if isinstance(X_high_card_encoded_train, pd.DataFrame):
                X_high_card_encoded_train = X_high_card_encoded_train.values
            if isinstance(X_high_card_encoded_test, pd.DataFrame):
                X_high_card_encoded_test = X_high_card_encoded_test.values

            logging.info(f"Shape of X_high_card_encoded_train: {X_high_card_encoded_train.shape}")
            logging.info(f"Shape of X_other_transformed_train: {X_other_transformed_train.shape}")
            logging.info(f"Shape of X_high_card_encoded_test: {X_high_card_encoded_test.shape}")
            logging.info(f"Shape of X_other_transformed_test: {X_other_transformed_test.shape}")

            # Combine features
            try:
                logging.info("Combining high cardinality encoded features with other transformed features")
                X_train_preprocessed = np.hstack((X_high_card_encoded_train, X_other_transformed_train))
                X_test_preprocessed = np.hstack((X_high_card_encoded_test, X_other_transformed_test))
            except Exception as e:
                logging.error(f"Error while stacking arrays: {e}")
                raise CustomException(e, sys)
            
            # logging.info("Combine high cardinality encoded features with other transformed features")
            # X_train_preprocessed = np.concatenate((X_high_card_encoded_train, X_other_transformed_train))
            # X_test_preprocessed = np.concatenate((X_high_card_encoded_test, X_other_transformed_test))

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            train_arr = np.c_[X_train_preprocessed, y_sample]
            test_arr = np.c_[X_test_preprocessed, target_feature_test_df]

            # input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df, target_feature_train_df)
            # input_feature_test_arr = preprocessing_obj.fit_transform(input_feature_test_df, target_feature_test_df)

            # train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            # test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            logging.info("saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)