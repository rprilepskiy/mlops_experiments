import pandas as pd

 
def feature_generation(data_path):
    df = pd.read_csv(data_path)
    df = df.drop('LOCATION_ID', axis=1)
    district_loss_onehot = pd.get_dummies(df['District_Loss'], prefix='District_Loss')
    history_onehot = pd.get_dummies(df['History'], prefix='History')
    df.drop(['District_Loss', 'History', 'Audit_Risk', 'Inherent_Risk', 'CONTROL_RISK', 'Score', 'TOTAL'],
            axis=1, inplace=True)
    df[district_loss_onehot.columns] = district_loss_onehot
    df[history_onehot.columns] = history_onehot
    return df
 
if __name__ == '__main__':
    df = feature_generation('data/audit_data.csv')
    df.to_csv('data/feature_generated.csv', index=False)