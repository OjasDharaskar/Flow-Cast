import os
import pandas as pd
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
import optuna
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

def split_data(df, target_col):
    """Time-based split: 70% train, 15% val, 15% test"""
    df = df.sort_values('timestamp')
    
    drop_cols = ['road_id', 'road_name', 'weather_station_id', 'holiday_name', 'event_name', 'timestamp', target_col]
    features = [c for c in df.columns if c not in drop_cols]
    
    X = df[features]
    y = df[target_col]
    
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test, features

def evaluate_regression(model, X, y):
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    return {'mae': mae, 'rmse': rmse, 'r2': r2}

def evaluate_classification(model, X, y):
    preds = model.predict(X)
    
    # Simple binary or multiclass check for roc_auc handling
    # Here we report accuracy, macro f1, macro precision, macro recall
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average='macro')
    return {'accuracy': acc, 'f1': f1}

def train_baselines(X_train, y_train, X_val, y_val, task_type='regression'):
    print(f"Training baseline models for {task_type}...")
    if task_type == 'regression':
        models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(random_state=42),
            'Lasso': Lasso(random_state=42),
            'ElasticNet': ElasticNet(random_state=42),
            'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=50, random_state=42)
        }
        eval_func = evaluate_regression
    else:
        models = {
            'LogisticRegression': LogisticRegression(random_state=42, n_jobs=-1),
            'RandomForest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=50, random_state=42)
        }
        eval_func = evaluate_classification
    
    results = {}
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        t_time = time.time() - start
        
        metrics = eval_func(model, X_val, y_val)
        metric_str = ", ".join([f"{k.upper()}: {v:.2f}" for k, v in metrics.items()])
        print(f"{name} - {metric_str}, Time: {t_time:.1f}s")
        
        results[name] = {'model': model, 'metrics': metrics, 'time': t_time}
    return results

def optimize_trees(X_train, y_train, X_val, y_val, model_name='xgboost', task_type='regression'):
    print(f"Optimizing {model_name} for {task_type} with Optuna...")
    
    def objective(trial):
        if model_name == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'random_state': 42,
                'n_jobs': -1
            }
            ModelClass = XGBRegressor if task_type == 'regression' else XGBClassifier
        else:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', -1, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            ModelClass = LGBMRegressor if task_type == 'regression' else LGBMClassifier
            
        model = ModelClass(**params)
        model.fit(X_train, y_train)
        
        if task_type == 'regression':
            return mean_absolute_error(y_val, model.predict(X_val))
        else:
            # For classification, Optuna minimizes by default, so return negative F1
            return -f1_score(y_val, model.predict(X_val), average='macro')

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5)
    
    print(f"Best Params: {study.best_params}")
    ModelClass = (XGBRegressor if task_type == 'regression' else XGBClassifier) if model_name == 'xgboost' else (LGBMRegressor if task_type == 'regression' else LGBMClassifier)
    
    best_model = ModelClass(**study.best_params, random_state=42, n_jobs=-1)
    
    start = time.time()
    best_model.fit(X_train, y_train)
    t_time = time.time() - start
    
    metrics = evaluate_regression(best_model, X_val, y_val) if task_type == 'regression' else evaluate_classification(best_model, X_val, y_val)
    return {'model': best_model, 'metrics': metrics, 'time': t_time}

def save_artifacts(results, best_model_name, features, target_col, root_dir):
    models_dir = os.path.join(root_dir, 'data', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    best_model = results[best_model_name]['model']
    
    # Save Model
    with open(os.path.join(models_dir, f'best_model_{target_col}.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
        
    # Save Comparison
    rows = []
    for name, data in results.items():
        row = {'Model': name, 'TrainTime_s': data['time']}
        row.update(data['metrics'])
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(models_dir, f'comparison_{target_col}.csv'), index=False)
    
    # Feature Importance
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]
        plt.figure(figsize=(10, 6))
        plt.title(f"Feature Importances ({best_model_name} - {target_col})")
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), [features[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(models_dir, f'importance_{target_col}.png'))
        plt.close()

def perform_training(df, target_col, task_type='regression'):
    print(f"\n{'='*50}\nStarting {task_type.upper()} for target: {target_col}\n{'='*50}")
    
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    X_train, y_train, X_val, y_val, X_test, y_test, features = split_data(df, target_col)
    
    results = train_baselines(X_train, y_train, X_val, y_val, task_type)
    results['XGBoost'] = optimize_trees(X_train, y_train, X_val, y_val, 'xgboost', task_type)
    results['LightGBM'] = optimize_trees(X_train, y_train, X_val, y_val, 'lightgbm', task_type)
    
    print("\nEvaluating on TEST set...")
    test_results = {}
    best_score = float('inf') if task_type == 'regression' else float('-inf')
    best_name = None
    
    for name, data in results.items():
        metrics = evaluate_regression(data['model'], X_test, y_test) if task_type == 'regression' else evaluate_classification(data['model'], X_test, y_test)
        test_results[name] = {'model': data['model'], 'metrics': metrics, 'time': data['time']}
        
        # Select best model (Minimize MAE for regression, Maximize F1 for classification)
        current_score = metrics['mae'] if task_type == 'regression' else metrics['f1']
        is_better = (current_score < best_score) if task_type == 'regression' else (current_score > best_score)
        
        if is_better:
            best_score = current_score
            best_name = name
            
    print(f"\nBest Model for {target_col}: {best_name}")
    save_artifacts(test_results, best_name, features, target_col, root_dir)
    print(f"Artifacts saved for {target_col}.\n")

def run_regression(df, target_col='traffic_volume'):
    perform_training(df, target_col, task_type='regression')

def run_classification(df, target_col='congestion_level'):
    perform_training(df, target_col, task_type='classification')

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML models for Flow-Cast.")
    parser.add_argument("--target", type=str, help="Target column to predict (e.g., traffic_volume)")
    parser.add_argument("--task", type=str, choices=["regression", "classification"], help="Type of task (regression or classification)")
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root_dir, 'data', 'Processed', 'featured_dataset.csv')
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.dropna()
    
    # If the user specified a target and task from the CLI
    if args.target and args.task:
        if args.task == "classification":
            df[args.target] = df[args.target].astype(int)
            run_classification(df, target_col=args.target)
        else:
            run_regression(df, target_col=args.target)
    else:
        print("No specific target provided via CLI. Running all 4 default PRD targets...")
        
        # 1. Traffic Volume (Regression)
        run_regression(df, target_col='traffic_volume')
        
        # 2. Travel Time (Regression)
        run_regression(df, target_col='travel_time')
        
        # 3. Congestion Level (Classification)
        df['congestion_level'] = df['congestion_level'].astype(int) 
        run_classification(df, target_col='congestion_level')
        
        # 4. Accident Risk (Classification)
        df['accident_count'] = df['accident_count'].astype(int)
        run_classification(df, target_col='accident_count')
