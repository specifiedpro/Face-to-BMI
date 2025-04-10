# src/data_prep.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_process_data(csv_path: str, image_base_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Create a full file path for each image:
    df['file_path'] = df['name'].apply(lambda x: os.path.join(image_base_path, x))
    # Keep only rows where image exists:
    df = df[df['file_path'].apply(os.path.exists)]
    # Map gender to a numerical value.
    df['sex'] = df['gender'].map({'Male': 0, 'Female': 1})
    return df

def split_dataset(df: pd.DataFrame, valid_ratio: float = 0.1):
    # Assume your CSV has an 'is_training' flag.
    df_train_full = df[df['is_training'] == 1]
    df_test = df[df['is_training'] == 0]
    df_train, df_valid = train_test_split(df_train_full, test_size=valid_ratio, random_state=42)
    return df_train, df_valid, df_test

def build_img_list(df, image_base_path):
    """
    Returns a list of tuples: (full_image_path, bmi, sex)
    Assumes df contains columns: 'name', 'bmi', and 'sex'
    """
    return [
        (os.path.join(image_base_path, row['name']), row['bmi'], row['sex'])
        for _, row in df.iterrows()
    ]
