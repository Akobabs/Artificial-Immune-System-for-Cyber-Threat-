import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA

def load_kdd_data(file_path):
    # KDD Cup 1999 column names
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
        'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
        'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
        'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
        'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
        'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
    ]
    data = pd.read_csv(file_path, header=None, names=columns)
    
    # Encode categorical features
    categorical_cols = ['protocol_type', 'service', 'flag']
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
    
    # Normalize numerical features
    numerical_cols = [col for col in data.columns if col not in categorical_cols + ['label']]
    scaler = MinMaxScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    
    # Split self (normal) and non-self (attack)
    self_data = data[data['label'] == 'normal.'].drop('label', axis=1)
    non_self_data = data[data['label'] != 'normal.'].drop('label', axis=1)
    
    return self_data, non_self_data

def load_iotnid_data(file_path):
    # Load IoTNID.csv
    data = pd.read_csv(file_path)
    
    # Handle missing values
    data.fillna(0, inplace=True)
    
    # Drop non-feature columns
    drop_cols = ['Flow_ID', 'Timestamp', 'Cat', 'Sub_Cat']
    data = data.drop(columns=[col for col in drop_cols if col in data.columns])
    
    # Encode categorical features
    categorical_cols = ['Src_IP', 'Dst_IP', 'Protocol']
    for col in categorical_cols:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
    
    # Handle infinities and large values in numerical columns
    numerical_cols = [col for col in data.columns if col not in categorical_cols + ['Label']]
    for col in numerical_cols:
        # Replace inf/-inf with NaN
        data[col] = data[col].replace([np.inf, -np.inf], np.nan)
        # Cap large values (e.g., clip to 99th percentile if non-NaN)
        if not data[col].isna().all():
            cap_value = data[col].quantile(0.99) if data[col].quantile(0.99) != 0 else 1e6
            data[col] = data[col].clip(upper=cap_value)
        # Fill NaNs with median or 0
        data[col] = data[col].fillna(data[col].median() if not data[col].isna().all() else 0)
    
    # Normalize numerical features
    scaler = MinMaxScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    
    # Split self (normal) and non-self (attack)
    self_data = data[data['Label'] == 'Normal'].drop('Label', axis=1)
    non_self_data = data[data['Label'] != 'Normal'].drop('Label', axis=1)
    
    return self_data, non_self_data

def apply_pca(data, n_components=10):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return reduced_data, pca

if __name__ == "__main__":
    # Test KDD preprocessing
    try:
        kdd_self, kdd_non_self = load_kdd_data('data/kddcup.data_10_percent')
        print("KDD Self Shape:", kdd_self.shape)
        print("KDD Non-Self Shape:", kdd_non_self.shape)
    except FileNotFoundError:
        print("KDD dataset not found. Please place 'kddcup.data_10_percent' in data/")
    
    # Test IoTNID preprocessing
    try:
        iot_self, iot_non_self = load_iotnid_data('data/IoTNID.csv')
        print("IoTNID Self Shape:", iot_self.shape)
        print("IoTNID Non-Self Shape:", iot_non_self.shape)
    except FileNotFoundError:
        print("IoTNID dataset not found. Please place 'IoTNID.csv' in data/")



import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_kdd_data(file_path):
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
        'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
        'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
        'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
        'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
        'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
    ]
    data = pd.read_csv(file_path, header=None, names=columns)
    
    # Define categorical and numerical columns
    categorical_cols = ['protocol_type', 'service', 'flag']
    numerical_cols = [col for col in data.columns if col not in categorical_cols + ['label']]
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
        ])
    
    # Apply preprocessing
    X = preprocessor.fit_transform(data.drop('label', axis=1))
    processed_data = pd.DataFrame(X, columns=numerical_cols + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)))
    
    # Split self (normal) and non-self (attack)
    self_data = processed_data[data['label'] == 'normal.']
    non_self_data = processed_data[data['label'] != 'normal.']
    
    return self_data, non_self_data

def load_iotnid_data(file_path):
    data = pd.read_csv(file_path)
    
    # Handle missing values
    data.fillna(0, inplace=True)
    
    # Drop non-feature columns
    drop_cols = ['Flow_ID', 'Timestamp', 'Cat', 'Sub_Cat']
    data = data.drop(columns=[col for col in drop_cols if col in data.columns])
    
    # Define categorical and numerical columns
    categorical_cols = ['Src_IP', 'Dst_IP', 'Protocol']
    numerical_cols = [col for col in data.columns if col not in categorical_cols + ['Label']]
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
        ])
    
    # Apply preprocessing
    X = preprocessor.fit_transform(data.drop('Label', axis=1))
    processed_data = pd.DataFrame(X, columns=numerical_cols + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)))
    
    # Split self (normal) and non-self (attack)
    self_data = processed_data[data['Label'] == 'Normal']
    non_self_data = processed_data[data['Label'] != 'Normal']
    
    return self_data, non_self_data

def apply_pca(data, n_components=10):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return reduced_data, pca

if __name__ == "__main__":
    # Test KDD preprocessing
    try:
        kdd_self, kdd_non_self = load_kdd_data('data/kddcup.data_10_percent')
        print("KDD Self Shape:", kdd_self.shape)
        print("KDD Non-Self Shape:", kdd_non_self.shape)
    except FileNotFoundError:
        print("KDD dataset not found. Please place 'kddcup.data_10_percent' in data/")
    
    # Test IoTNID preprocessing
    try:
        iot_self, iot_non_self = load_iotnid_data('data/IoTNID.csv')
        print("IoTNID Self Shape:", iot_self.shape)
        print("IoTNID Non-Self Shape:", iot_non_self.shape)
    except FileNotFoundError:
        print("IoTNID dataset not found. Please place 'IoTNID.csv' in data/")