# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(filepath='data/erp_financial_data.csv'):
    df = pd.read_csv(filepath)
    
    # 构造特征
    df['diff_ratio'] = df['reconciliation_diff'] / (df['order_amount'] + 1e-5)
    df['credit_utilization'] = df['order_amount'] / (df['credit_limit'] + 1e-5)
    
    # 孤立森林特征（对账异常检测）
    anomaly_features = ['order_amount', 'billing_amount', 'reconciliation_diff', 'diff_ratio']
    
    # XGBoost 特征（违约风险预测）
    risk_features = [
        'order_amount', 'billing_amount', 'reconciliation_diff', 'diff_ratio',
        'overdue_days', 'payment_delay', 'credit_limit', 'historical_default_rate', 'credit_utilization'
    ]
    
    scaler = StandardScaler()
    X_anomaly = scaler.fit_transform(df[anomaly_features])
    X_risk = df[risk_features]
    y_risk = df['is_default']
    
    print('Preprocessing completed successfully.')
    print(f'Anomaly feature matrix shape: {X_anomaly.shape}, Risk feature matrix shape: {X_risk.shape}')
    return df, X_anomaly, X_risk, y_risk

if __name__ == '__main__':
    load_and_preprocess()
