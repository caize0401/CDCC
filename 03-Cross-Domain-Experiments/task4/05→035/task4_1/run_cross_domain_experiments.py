import os
from pathlib import Path
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from common_utils import load_datasets, preprocess, fit_scaler_on_union, transform_split, encode_labels, compute_metrics
from models import make_mlp, make_rf, make_xgb, make_automl_voter, train_eval_cnn1d


def run_all():
    base_dir = Path(__file__).resolve().parent
    out_root = base_dir / 'results'
    out_root.mkdir(parents=True, exist_ok=True)

    datasets = load_datasets(base_dir)

    # 5 模型 × 3 数据集 = 15 组实验
    # 训练集：0.5 尺寸；测试集：0.35 尺寸
    rows = []
    for dataset_type in ['data1', 'data2', 'data3']:
        # 取出 train/test 的原始数据
        df_train = datasets[dataset_type]['05']
        df_test = datasets[dataset_type]['035']

        # X,y 原始
        X_tr_raw, y_tr_raw = preprocess(df_train, dataset_type)
        X_te_raw, y_te_raw = preprocess(df_test, dataset_type)

        # 标准化：训练+测试联合拟合
        scaler = fit_scaler_on_union(X_tr_raw, X_te_raw)
        X_train = transform_split(scaler, X_tr_raw)
        X_test = transform_split(scaler, X_te_raw)

        # 标签编码：以训练集 fit，映射测试集
        y_train, y_test, le = encode_labels(y_tr_raw, y_te_raw)
        labels_full = list(range(len(le.classes_)))

        # 保存到每个模型的子目录
        # 1) Random Forest
        rf = make_rf()
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        m = compute_metrics(y_test, y_pred)
        model_dir = out_root / f'RandomForest_{dataset_type}'
        model_dir.mkdir(exist_ok=True)
        pd.DataFrame([m]).to_csv(model_dir / 'metrics.csv', index=False, encoding='utf-8-sig')
        # confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred, labels=labels_full)
        pd.DataFrame(cm, index=[f'True_{i}' for i in range(cm.shape[0])], columns=[f'Pred_{i}' for i in range(cm.shape[1])]).to_csv(model_dir / 'confusion_matrix.csv', encoding='utf-8-sig')
        rows.append({'model': 'RandomForest', 'dataset': dataset_type, **m})

        # 2) MLP
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

        # 3) XGBoost
        xgb = make_xgb()
        xgb.fit(X_train, y_train)
        y_pred = xgb.predict(X_test)
        m = compute_metrics(y_test, y_pred)
        model_dir = out_root / f'XGBoost_{dataset_type}'
        model_dir.mkdir(exist_ok=True)
        pd.DataFrame([m]).to_csv(model_dir / 'metrics.csv', index=False, encoding='utf-8-sig')
        cm = confusion_matrix(y_test, y_pred, labels=labels_full)
        pd.DataFrame(cm, index=[f'True_{i}' for i in range(cm.shape[0])], columns=[f'Pred_{i}' for i in range(cm.shape[1])]).to_csv(model_dir / 'confusion_matrix.csv', encoding='utf-8-sig')
        rows.append({'model': 'XGBoost', 'dataset': dataset_type, **m})

        # 4) H2O AutoML（替代：软投票集成）
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

        # 5) 1D-CNN（对 data1/data2/data3 均运行：data2/3 作为一维信号向量处理）
        m = train_eval_cnn1d(X_train, y_train, X_test, y_test, epochs=50, batch_size=128, lr=1e-3)
        model_dir = out_root / f'CNN1D_{dataset_type}'
        model_dir.mkdir(exist_ok=True)
        # write metrics (drop arrays)
        pd.DataFrame([{k: v for k, v in m.items() if k in ['accuracy','precision','recall','f1_score']}]).to_csv(model_dir / 'metrics.csv', index=False, encoding='utf-8-sig')
        # confusion matrix，按训练集类别数固定为5×5（或K×K）
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(m['y_true'], m['y_pred'], labels=labels_full)
        pd.DataFrame(cm, index=[f'True_{i}' for i in range(cm.shape[0])], columns=[f'Pred_{i}' for i in range(cm.shape[1])]).to_csv(model_dir / 'confusion_matrix.csv', encoding='utf-8-sig')
        rows.append({'model': 'CNN1D', 'dataset': dataset_type, 'accuracy': m['accuracy'], 'precision': m['precision'], 'recall': m['recall'], 'f1_score': m['f1_score']})

    # 汇总
    summary = pd.DataFrame(rows)
    summary.to_csv(out_root / 'summary.csv', index=False, encoding='utf-8-sig')
    print('All cross-domain experiments completed. See results folder.')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_all()


