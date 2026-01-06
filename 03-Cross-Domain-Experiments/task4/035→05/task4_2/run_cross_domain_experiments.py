import os
from pathlib import Path
import pandas as pd

from common_utils import load_datasets, preprocess, fit_scaler_on_union, transform_split, encode_labels_union, compute_metrics
from models import make_mlp, make_rf, make_xgb, make_automl_voter, train_eval_cnn1d


def run_all():
    base_dir = Path(__file__).resolve().parent
    out_root = base_dir / 'results'
    out_root.mkdir(parents=True, exist_ok=True)

    datasets = load_datasets(base_dir)
    rows = []

    for dataset_type in ['data1', 'data2', 'data3']:
        # train: 0.35, test: 0.5
        df_train = datasets[dataset_type]['035']
        df_test = datasets[dataset_type]['05']

        X_tr_raw, y_tr_raw = preprocess(df_train, dataset_type)
        X_te_raw, y_te_raw = preprocess(df_test, dataset_type)

        scaler = fit_scaler_on_union(X_tr_raw, X_te_raw)
        X_train = transform_split(scaler, X_tr_raw)
        X_test = transform_split(scaler, X_te_raw)

        # 以 union(train,test) 的 5 类空间编码，保证测试集中新增类可被识别
        y_train, y_test, le = encode_labels_union(y_tr_raw, y_te_raw)
        labels_full = list(range(len(le.classes_)))

        # RF
        rf = make_rf()
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        m = compute_metrics(y_test, y_pred)
        model_dir = out_root / f'RandomForest_{dataset_type}'
        model_dir.mkdir(exist_ok=True)
        pd.DataFrame([m]).to_csv(model_dir / 'metrics.csv', index=False, encoding='utf-8-sig')
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred, labels=labels_full)
        pd.DataFrame(cm, index=[f'True_{i}' for i in range(cm.shape[0])], columns=[f'Pred_{i}' for i in range(cm.shape[1])]).to_csv(model_dir / 'confusion_matrix.csv', encoding='utf-8-sig')
        rows.append({'model': 'RandomForest', 'dataset': dataset_type, **m})

        # MLP
        mlp = make_mlp()
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
        m = compute_metrics(y_test, y_pred)
        model_dir = out_root / f'MLP_{dataset_type}'
        model_dir.mkdir(exist_ok=True)
        pd.DataFrame([m]).to_csv(model_dir / 'metrics.csv', index=False, encoding='utf-8-sig')
        cm = confusion_matrix(y_test, y_pred, labels=labels_full)
        pd.DataFrame(cm, index=[f'True_{i}' for i in range(cm.shape[0])], columns=[f'Pred_{i}' for i in range(cm.shape[1])]).to_csv(model_dir / 'confusion_matrix.csv', encoding='utf-8-sig')
        rows.append({'model': 'MLP', 'dataset': dataset_type, **m})

        # XGB
        xgb = make_xgb()
        # If some classes in labels_full are missing in y_train, pad with tiny-weight synthetic samples
        import numpy as np
        present = set(np.unique(y_train).tolist())
        missing = [c for c in labels_full if c not in present]
        if missing:
            X_pad = X_train.mean(axis=0, keepdims=True)
            X_synth = np.repeat(X_pad, repeats=len(missing), axis=0)
            y_synth = np.array(missing, dtype=y_train.dtype)
            X_train_xgb = np.concatenate([X_train, X_synth], axis=0)
            y_train_xgb = np.concatenate([y_train, y_synth], axis=0)
            sw = np.ones(X_train_xgb.shape[0], dtype=np.float32)
            sw[-len(missing):] = 1e-6
            xgb.fit(X_train_xgb, y_train_xgb, sample_weight=sw)
        else:
            xgb.fit(X_train, y_train)
        y_pred = xgb.predict(X_test)
        m = compute_metrics(y_test, y_pred)
        model_dir = out_root / f'XGBoost_{dataset_type}'
        model_dir.mkdir(exist_ok=True)
        pd.DataFrame([m]).to_csv(model_dir / 'metrics.csv', index=False, encoding='utf-8-sig')
        cm = confusion_matrix(y_test, y_pred, labels=labels_full)
        pd.DataFrame(cm, index=[f'True_{i}' for i in range(cm.shape[0])], columns=[f'Pred_{i}' for i in range(cm.shape[1])]).to_csv(model_dir / 'confusion_matrix.csv', encoding='utf-8-sig')
        rows.append({'model': 'XGBoost', 'dataset': dataset_type, **m})

        # AutoML surrogate
        automl = make_automl_voter()
        automl.fit(X_train, y_train)
        y_pred = automl.predict(X_test)
        m = compute_metrics(y_test, y_pred)
        model_dir = out_root / f'H2O_AutoML_{dataset_type}'
        model_dir.mkdir(exist_ok=True)
        pd.DataFrame([m]).to_csv(model_dir / 'metrics.csv', index=False, encoding='utf-8-sig')
        cm = confusion_matrix(y_test, y_pred, labels=labels_full)
        pd.DataFrame(cm, index=[f'True_{i}' for i in range(cm.shape[0])], columns=[f'Pred_{i}' for i in range(cm.shape[1])]).to_csv(model_dir / 'confusion_matrix.csv', encoding='utf-8-sig')
        rows.append({'model': 'H2O_AutoML', 'dataset': dataset_type, **m})

        # CNN1D（将 data2/3 作为一维向量输入）
        m = train_eval_cnn1d(X_train, y_train, X_test, y_test, epochs=50, batch_size=128, lr=1e-3)
        model_dir = out_root / f'CNN1D_{dataset_type}'
        model_dir.mkdir(exist_ok=True)
        pd.DataFrame([{k: v for k, v in m.items() if k in ['accuracy','precision','recall','f1_score']}]).to_csv(model_dir / 'metrics.csv', index=False, encoding='utf-8-sig')
        cm = confusion_matrix(m['y_true'], m['y_pred'], labels=labels_full)
        pd.DataFrame(cm, index=[f'True_{i}' for i in range(cm.shape[0])], columns=[f'Pred_{i}' for i in range(cm.shape[1])]).to_csv(model_dir / 'confusion_matrix.csv', encoding='utf-8-sig')
        rows.append({'model': 'CNN1D', 'dataset': dataset_type, 'accuracy': m['accuracy'], 'precision': m['precision'], 'recall': m['recall'], 'f1_score': m['f1_score']})

    pd.DataFrame(rows).to_csv(out_root / 'summary.csv', index=False, encoding='utf-8-sig')
    print('All cross-domain experiments (train 0.35, test 0.5) completed.')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_all()


