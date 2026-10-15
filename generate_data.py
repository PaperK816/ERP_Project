# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd

def generate_erp_dataset(n_samples=3000, output_path='data/erp_financial_data.csv'):
    np.random.seed(42)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    customer_ids = [f'CUST_{i:04d}' for i in range(1, 501)]
    cust_mapping = {cid: {
        'credit_limit': np.random.choice([50000, 100000, 200000, 500000]),
        'historical_default_rate': np.random.beta(1, 8)
    } for cid in customer_ids}
    
    data = []
    for i in range(n_samples):
        cid = np.random.choice(customer_ids)
        c_info = cust_mapping[cid]
        
        order_amount = np.random.exponential(scale=20000) + 2000
        noise = np.random.normal(0, 30)
        if np.random.rand() < 0.05:
            noise += np.random.choice([-1, 1]) * np.random.uniform(500, 3000)
        billing_amount = order_amount + noise
        reconciliation_diff = billing_amount - order_amount
        
        overdue_days = np.random.exponential(scale=12) if np.random.rand() > 0.3 else 0
        payment_delay = overdue_days + np.random.normal(0, 2)
        
        risk_score = (overdue_days * 0.04) + (c_info['historical_default_rate'] * 6) + (1.5 if abs(reconciliation_diff) > 400 else 0)
        is_default = 1 if risk_score > 2.2 and np.random.rand() < 0.75 else 0
        
        data.append({
            'transaction_id': f'TXN_{i+1:05d}',
            'customer_id': cid,
            'order_amount': round(order_amount, 2),
            'billing_amount': round(billing_amount, 2),
            'reconciliation_diff': round(reconciliation_diff, 2),
            'overdue_days': int(max(0, overdue_days)),
            'payment_delay': round(max(0, payment_delay), 1),
            'credit_limit': c_info['credit_limit'],
            'historical_default_rate': round(c_info['historical_default_rate'], 4),
            'is_default': is_default
        })
        
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'Successfully generated dataset at {output_path} with shape {df.shape}')
    return df

if __name__ == '__main__':
    generate_erp_dataset()
