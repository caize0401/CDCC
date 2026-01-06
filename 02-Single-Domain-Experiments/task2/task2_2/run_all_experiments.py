import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

from common_utils import load_datasets, preprocess_data, split_train_test, evaluate_predictions, save_results_to_csv
from automl_surrogate import train_and_eval_automl
from cnn1d_trainer import train_and_eval_cnn1d


warnings.filterwarnings('ignore')


def train_and_eval_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return evaluate_predictions(y_test, y_pred)


def main():
    base_dir = Path(__file__).resolve().parent
    out_root = base_dir / 'results'
    out_root.mkdir(exist_ok=True, parents=True)

    datasets = load_datasets()

    models = {
        'RandomForest': RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1),
        'MLP': MLPClassifier(hidden_layer_sizes=(256, 128), activation='relu', solver='adam', alpha=1e-4, max_iter=500, random_state=42),
        'XGBoost': XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            objective='multi:softprob', eval_metric='mlogloss', tree_method='hist', random_state=42, n_jobs=-1
        ),
        'H2O_AutoML': 'automl_surrogate',
        'CNN1D': 'cnn1d',
    }

    all_rows = []

    for dataset_type in ['data1', 'data2', 'data3']:
        for size_type in ['035', '05']:
            df = datasets[dataset_type][size_type]
            X, y, scaler = preprocess_data(df, dataset_type)
            X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2, seed=42)
            # 统一将标签映射为 0..K-1（部分尺寸可能缺类，如出现 [0,1,2,4]）
            le = LabelEncoder().fit(y_train)
            y_train = le.transform(y_train)
            y_test = le.transform(y_test)

            for model_name, model in models.items():
                print(f"Running {model_name} on {dataset_type}-{size_type} ...")
                if model_name == 'H2O_AutoML':
                    results = train_and_eval_automl(X_train, y_train, X_test, y_test)
                elif model_name == 'CNN1D':
                    if dataset_type != 'data1':
                        # CNN1D 仅适用于原始曲线（data1）
                        continue
                    # 对于 data1，此时 X 是标准化后的 1D 序列，形状 (N, L)
                    results = train_and_eval_cnn1d(X_train, y_train, X_test, y_test, epochs=50, batch_size=128, lr=1e-3)
                else:
                    results = train_and_eval_model(model, X_train, y_train, X_test, y_test)

                # Save per-model CSV
                out_csv = out_root / f"{model_name}_{dataset_type}_{size_type}.csv"
                save_results_to_csv(results, out_csv)

                row = {
                    'model': model_name,
                    'dataset': dataset_type,
                    'size': size_type,
                    'accuracy': results['accuracy'],
                    'precision': results['precision'],
                    'recall': results['recall'],
                    'f1_score': results['f1_score'],
                }
                all_rows.append(row)

    summary = pd.DataFrame(all_rows)
    summary.to_csv(out_root / 'summary.csv', index=False, encoding='utf-8-sig')
    print('All experiments completed. Summary saved to results/summary.csv')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()


