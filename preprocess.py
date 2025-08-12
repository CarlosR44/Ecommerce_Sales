import numpy as np
import pandas as pd
import joblib
import category_encoders as ce
from sklearn.preprocessing import StandardScaler

# df = load_dataset("nprak26/remote-worker-productivity")
df = pd.read_csv('ecommerce_product_sales.csv')

cols_to_encoded = ['Product Name','Category',
                   'Customer Gender',
                   'Payment Method']

class Preprocessor:
    
    def __init__(self):
        self.cols_to_drop = ['Region', 'Sale Date', 'Sale ID']
        self.freq_encoded = ce.CountEncoder(cols=cols_to_encoded, normalize=True)
        self.scaler = StandardScaler()
    
    def fit(self, df):
        df_features = df.drop(columns=self.cols_to_drop + ['Total Sales'], errors='ignore')
        df_encoded = self.freq_encoded.fit_transform(df_features)
        num_cols = df_encoded.select_dtypes(include=['int64', 'float64']).columns # type: ignore
        df_encoded[num_cols] = self.scaler.fit_transform(df_encoded[num_cols])
        
        return df_encoded
    
    def transform(self, df):
        df_features = df.drop(columns=self.cols_to_drop, errors='ignore')
        df_encoded = self.freq_encoded.transform(df_features)
        num_cols = df_encoded.select_dtypes(include=['int64', 'float64']).columns
        df_encoded[num_cols] = self.scaler.transform(df_encoded[num_cols])
        
        return df_encoded
    
preprocessor = Preprocessor()
df_train_preprocessed = preprocessor.fit(df) 


joblib.dump(preprocessor, 'preprocessor.pkl')