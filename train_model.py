# -*- coding: utf-8 -*-
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
import shap

from preprocess import load_and_preprocess

def main():
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    df, X_anomaly, X_risk, y_risk = load_and_preprocess()
    
    # 1. 孤立森林财务对账异常检测
    print('\n--- Training Isolation Forest for Financial Reconciliation Anomaly Detection ---')
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    df['anomaly_pred'] = iso_forest.fit_predict(X_anomaly)
    # 孤立森林中 -1 表示异常，1 表示正常
    anomaly_count = (df['anomaly_pred'] == -1).sum()
    print(f'Detected reconciliation anomalies: {anomaly_count} out of {len(df)} records.')
    
    # 2. XGBoost 应收账款违约风险预测
    print('\n--- Training XGBoost for Accounts Receivable Default Prediction ---')
    X_train, X_test, y_train, y_test = train_test_split(X_risk, y_risk, test_size=0.2, random_state=42, stratify=y_risk)
    
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    y_pred = xgb_model.predict(X_test)
    y_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))
    auc = roc_auc_score(y_test, y_proba)
    print(f'XGBoost ROC-AUC Score: {auc:.4f}')
    
    # 3. SHAP 可解释性分析
    print('\n--- Generating SHAP Explanations ---')
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_test)
    
    # 保存 SHAP 摘要图
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title('SHAP Summary Plot for AR Default Risk', fontsize=12)
    plt.tight_layout()
    shap_path = 'outputs/shap_summary.png'
    plt.savefig(shap_path, dpi=300)
    plt.close()
    print(f'Saved SHAP summary plot to {shap_path}')
    
    # 保存特征重要性条形图
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, show=False)
    plt.title('SHAP Feature Importance Bar Plot', fontsize=12)
    plt.tight_layout()
    bar_path = 'outputs/shap_importance_bar.png'
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f'Saved SHAP feature importance bar plot to {bar_path}')
    
    # 保存带有异常标签的数据集
    df.to_csv('data/erp_financial_results.csv', index=False)
    print('\nAll modeling and explanation tasks completed successfully!')

if __name__ == '__main__':
    main()
