# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import os

COL_EVAL_LABEL = 2                
COL_EVAL_NODULE_TYPE = 44         
COL_EVAL_DIAMETER = 45            
COL_EVAL_AGE = 4                  
COL_EVAL_CTR = 47                 
COL_EVAL_SPIKE = 48               
COL_EVAL_PLEURAL = 49             
COL_EVAL_VACUOLE = 50             
COL_EVAL_LOBULE = 51              
COL_EVAL_CYSTIC = 52              
COL_EVAL_IL6 = 53                 
COL_EVAL_PAR = 55                 
COL_EVAL_LMR = 56                 


LABEL_MAPPING = {'浸润性腺癌': 1, '非浸润性腺癌': 0}
NODULE_NA_FILL = "solid"


BASE_FEATURE_NAMES = [
    "CTR",
    "Imaging_Diameter",
    "Age",
    "Spiculation",
    "Pleural_Traction",
    "Vacuole_Sign",
    "Lobulation",
    "Cystic_Lung_Cancer"
]
OPT_FEATURE_MAP = {
    "IL6_Value": COL_EVAL_IL6,
    "PAR": COL_EVAL_PAR,
    "LMR": COL_EVAL_LMR
}

LOG_COLS = ["IL6_Value", "LMR", "PAR"]

NOLOG_NUM_COLS = ["Age", "CTR", "Imaging_Diameter"]

BIN_COLS = ["Spiculation", "Pleural_Traction", "Vacuole_Sign", "Lobulation", "Cystic_Lung_Cancer"]


def load_eval_data(
    eval_excel_path="eval_data.xlsx",
    use_il6: bool = True,
    use_par: bool = True,
    use_lmr: bool = True,
    train_il6_threshold: float = None,  
    train_nodule_encoder = None        
):

    if not os.path.exists(eval_excel_path):
        raise FileNotFoundError(f"❌ 验证集文件不存在：{eval_excel_path}")
    print(f"\n📥 读取验证集文件：{eval_excel_path}")
    df = pd.read_excel(eval_excel_path, header=0)
    print(f"✅ 验证集原始样本数={df.shape[0]}, 总列数={df.shape[1]}")


    max_col = max(COL_EVAL_LMR, COL_EVAL_NODULE_TYPE, COL_EVAL_LABEL)
    if df.shape[1] <= max_col:
        raise ValueError(f"❌ 验证集列数不足！至少需要 {max_col+1} 列")


    label_col = df.columns[COL_EVAL_LABEL]
    deleted_label = df[df[label_col].isna()].copy()
    if not deleted_label.empty:
        print(f"⚠️ 验证集发现 {len(deleted_label)} 条标签空值，已删除")
    df = df.dropna(subset=[label_col]).copy().reset_index(drop=True)
    print(f"✅ 删除标签空值后，验证集剩余样本：{df.shape[0]}")


    y_raw = df.iloc[:, COL_EVAL_LABEL]
    y_eval = y_raw.map(LABEL_MAPPING).values
    if np.isnan(y_eval).any():
        raise Exception("❌ 验证集标签存在无法映射的文本，请核对Excel")


    if train_il6_threshold is not None:
        il6_series = df.iloc[:, COL_EVAL_IL6].copy()
        keep_mask = il6_series.isna() | (il6_series <= train_il6_threshold)
        deleted_il6 = df[~keep_mask].copy()
        if not deleted_il6.empty:
            print(f"⚠️ 验证集 {len(deleted_il6)} 条IL6异常值(>{train_il6_threshold:.2f})，已删除")
        df = df[keep_mask].reset_index(drop=True)
        y_eval = y_eval[keep_mask]
        print(f"✅ 验证集IL6过滤完成，剩余样本：{df.shape[0]}")


    base_cols = [
        COL_EVAL_CTR, COL_EVAL_DIAMETER, COL_EVAL_AGE,
        COL_EVAL_SPIKE, COL_EVAL_PLEURAL, COL_EVAL_VACUOLE,
        COL_EVAL_LOBULE, COL_EVAL_CYSTIC
    ]
    feature_names = BASE_FEATURE_NAMES.copy()


    opt_list = []
    if use_il6:
        base_cols.append(COL_EVAL_IL6)
        feature_names.append("IL6_Value")
        opt_list.append("IL6_Value")
    if use_par:
        base_cols.append(COL_EVAL_PAR)
        feature_names.append("PAR")
        opt_list.append("PAR")
    if use_lmr:
        base_cols.append(COL_EVAL_LMR)
        feature_names.append("LMR")
        opt_list.append("LMR")


    numeric_features = df.iloc[:, base_cols].copy()


    nodule_series = df.iloc[:, COL_EVAL_NODULE_TYPE].copy().fillna(NODULE_NA_FILL)

    nodule_encoded = train_nodule_encoder.transform(nodule_series.values.reshape(-1, 1))
    nodule_names = [f"nodule_type_{cat}" for cat in train_nodule_encoder.categories_[0]]
    feature_names.extend(nodule_names)

    X_eval = pd.concat([numeric_features, pd.DataFrame(nodule_encoded)], axis=1)
    X_eval.columns = feature_names
    X_eval = X_eval.apply(pd.to_numeric, errors="coerce")


    if "Age" in X_eval.columns:
        X_eval = X_eval.drop(columns=["Age"])
        feature_names.remove("Age")


    if len(X_eval) != len(y_eval):
        raise ValueError("❌ 验证集特征与标签行数不匹配")
    print(f"✅ 验证集特征总数：{X_eval.shape[1]}")
    return X_eval, y_eval, feature_names, df


def preprocess_eval_data(X_eval, mice_imputer, cat_imputer):

    X = X_eval.copy()
    use_log = [c for c in LOG_COLS if c in X.columns]
    use_nolog = [c for c in NOLOG_NUM_COLS if c in X.columns]
    use_bin = [c for c in BIN_COLS if c in X.columns]
    cat_onehot = [col for col in X.columns if col.startswith("nodule_type_")]
    all_cat = use_bin + cat_onehot
    all_numeric = use_log + use_nolog


    if all_numeric:
        X[all_numeric] = mice_imputer.transform(X[all_numeric])


    if use_log:
        X[use_log] = np.log1p(X[use_log])

    if all_cat:
        X[all_cat] = cat_imputer.transform(X[all_cat])

    X_eval_final = X.values
    return X_eval_final