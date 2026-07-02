# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import os
from fancyimpute import IterativeImputer

# data column position
COL_FEATURE_NODULE_TYPE = 44      
COL_FEATURE_CTR = 47              
COL_FEATURE_DIAMETER = 45         
COL_FEATURE_AGE = 4               
COL_FEATURE_SPIKE = 48            
COL_FEATURE_PLEURAL = 49          
COL_FEATURE_VACUOLE = 50          
COL_FEATURE_LOBULE = 51           
COL_FEATURE_CYSTIC = 52           
COL_FEATURE_IL6 = 53              

COL_FEATURE_NLR = 54    
COL_FEATURE_PLR = 55    
COL_FEATURE_LMR = 56    
COL_FEATURE_SII = 57    
COL_FEATURE_SIRI = 58  
COL_FEATURE_AISI = 59   
COL_FEATURE_NPR = 60    
COL_FEATURE_PAR = 61   
COL_FEATURE_PNI = 62   

COL_LABEL_OUTCOME = 2  
COL_PATIENT_ID = 1 


NODULE_NA_FILL = "solid"  
LABEL_MAPPING = {'浸润性腺癌':1, '非浸润性腺癌':0}


BASE_FEATURE_NAMES = [
    "CTR",
    "Imaging_Diameter",
    "Age",
    "Spiculation",
    "Pleural_Traction",
    "Vacuole_Sign",
    "Lobulation",
    "Cystic_Lung_Cancer",  
    "IL6_Value"
]


BLOOD_FEATURE_NAMES = [
    "NLR", "PLR", "LMR", "SII", "SIRI",
    "AISI", "NPR", "PAR", "PNI"
]



COL_FEATURE_WBC = 35      
COL_FEATURE_ALB = 36      
COL_FEATURE_PLT = 37      
COL_FEATURE_NEUT = 38     
COL_FEATURE_LYMPH = 39    
COL_FEATURE_HGB = 40      
COL_FEATURE_MONO = 41     

CELL_FEATURE_NAMES = [
    "WBC", "ALB", "PLT", "NEUT", "LYMPH", "HGB", "MONO"
]


def load_real_data(
    excel_path="data_third.xlsx",
    use_il6: bool = True,          
    use_blood_markers: bool = True,
    use_cell_markers: bool = True,
    filter_percentile: int = 99,
    blood_markers: list = None,
    cell_markers: list = None,
):
    if blood_markers is None:
        blood_markers = []
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"❌ Excel文件不存在：{excel_path}")
    df = pd.read_excel(excel_path, header=0)

    max_col = max([
        COL_LABEL_OUTCOME, COL_FEATURE_IL6, COL_FEATURE_NODULE_TYPE,
        COL_FEATURE_CYSTIC, COL_FEATURE_PNI 
    ])

    print("\n🔍 Excel【结局变量】真实取值：")
    print(df.iloc[:, COL_LABEL_OUTCOME].value_counts(dropna=False))
    print("\n🔍 Excel【结节性质】真实取值：")
    print(df.iloc[:, COL_FEATURE_NODULE_TYPE].value_counts(dropna=False))

    label_col = df.columns[COL_LABEL_OUTCOME]
    deleted_df = df[df[label_col].isna()].copy()
    if not deleted_df.empty:
        for idx, row in deleted_df.iterrows():
            patient_id = row[df.columns[COL_PATIENT_ID]]  
            excel_row = idx + 2  
            print(f"   └─ 住院号：{patient_id}  |  Excel行号：{excel_row}")
    else:
        print(f"\n✅ 标签列无空值，无需删除！")

    df = df.dropna(subset=[label_col]).copy().reset_index(drop=True)

    y_raw = df.iloc[:, COL_LABEL_OUTCOME]
    y = y_raw.map(LABEL_MAPPING).values

    il6_series = df.iloc[:, COL_FEATURE_IL6].copy()
    valid_il6 = il6_series.dropna()
    threshold = np.percentile(valid_il6, filter_percentile)
    keep_mask = il6_series.isna() | (il6_series <= threshold)

    deleted_il6 = df[~keep_mask].copy()
    if not deleted_il6.empty:
        print(f"\n⚠️ 发现 {len(deleted_il6)} 条IL6异常值（>{threshold:.2f}），已删除：")
        for idx, row in deleted_il6.iterrows():
            patient_id = row[df.columns[COL_PATIENT_ID]]
            excel_row = idx + 2
            print(f"   └─ 住院号：{patient_id}  |  Excel行号：{excel_row}")

    df = df[keep_mask].reset_index(drop=True)
    y = y[keep_mask]

    base_cols = [
        COL_FEATURE_CTR, COL_FEATURE_DIAMETER, COL_FEATURE_AGE,
        COL_FEATURE_SPIKE, COL_FEATURE_PLEURAL, COL_FEATURE_VACUOLE,
        COL_FEATURE_LOBULE, COL_FEATURE_CYSTIC
    ]

    feature_names = BASE_FEATURE_NAMES[:8]

    if use_il6:
        base_cols.append(COL_FEATURE_IL6)
        feature_names.append(BASE_FEATURE_NAMES[8])

    BLOOD_COL_MAP = {
        "NLR": COL_FEATURE_NLR,
        "PLR": COL_FEATURE_PLR,
        "LMR": COL_FEATURE_LMR,
        "SII": COL_FEATURE_SII,
        "SIRI": COL_FEATURE_SIRI,
        "AISI": COL_FEATURE_AISI,
        "NPR": COL_FEATURE_NPR,
        "PAR": COL_FEATURE_PAR,
        "PNI": COL_FEATURE_PNI
    }
    CELL_COL_MAP = {
        "WBC": COL_FEATURE_WBC,
        "ALB": COL_FEATURE_ALB,
        "PLT": COL_FEATURE_PLT,
        "NEUT": COL_FEATURE_NEUT,
        "LYMPH": COL_FEATURE_LYMPH,
        "HGB": COL_FEATURE_HGB,
        "MONO": COL_FEATURE_MONO
    }

    selected_blood_cols = []
    selected_blood_names = []
    if use_blood_markers and blood_markers:

        valid_markers = [b for b in blood_markers if b in BLOOD_COL_MAP]
        invalid_markers = [b for b in blood_markers if b not in BLOOD_COL_MAP]
        if invalid_markers:
            raise ValueError(f"无效血清指标：{invalid_markers}，支持：{list(BLOOD_COL_MAP.keys())}")

        selected_blood_names = valid_markers
        selected_blood_cols = [BLOOD_COL_MAP[b] for b in valid_markers]

        base_cols.extend(selected_blood_cols)
        feature_names.extend(selected_blood_names)

    selected_cell_cols = []
    selected_cell_names = []
    if use_cell_markers and cell_markers:
        valid_markers = [c for c in cell_markers if c in CELL_COL_MAP]
        invalid_markers = [c for c in cell_markers if c not in CELL_COL_MAP]
        if invalid_markers:
            raise ValueError(f"无效细胞指标：{invalid_markers}，支持：{list(CELL_COL_MAP.keys())}")
        selected_cell_names = valid_markers
        selected_cell_cols = [CELL_COL_MAP[c] for c in valid_markers]
        base_cols.extend(selected_cell_cols)
        feature_names.extend(selected_cell_names)


    numeric_features = df.iloc[:, base_cols].copy()


    nodule_series = df.iloc[:, COL_FEATURE_NODULE_TYPE].copy().fillna(NODULE_NA_FILL)
    encoder = OneHotEncoder(sparse_output=False, drop=None)
    nodule_type_encoded = encoder.fit_transform(nodule_series.values.reshape(-1, 1))
    nodule_names = [f"nodule_type_{cat}" for cat in encoder.categories_[0]]

    feature_names.extend(nodule_names)


    X = pd.concat([numeric_features, pd.DataFrame(nodule_type_encoded)], axis=1)
    

    X.columns = feature_names  
    X = X.apply(pd.to_numeric, errors="coerce")


    if len(X) != len(y):
        raise ValueError(f"❌ 数据对齐失败！X行数：{len(X)}，y行数：{len(y)}")
    if X.shape[1] != len(feature_names):
        raise ValueError(f"❌ 特征名数量不匹配！")

    print(f"\n✅ 特征整理完成：{X.shape[1]} 个特征")

    print(f"✅ 固定特征：8个 | IL6：{use_il6} | 血液指标：{use_blood_markers}，已选中：{selected_blood_names}")
    print(f"✅ 自动生成特征名：{feature_names}")
    print(f"✅ 标签分布：1={np.sum(y==1)}条，0={np.sum(y==0)}条")
    if "IL6_Value" in feature_names:
        feature_names.remove("IL6_Value")
        feature_names.insert(0, "IL6_Value")  
        X = X[feature_names]  

    delete_feature_name = "Age"  

    if delete_feature_name in X.columns:
        X = X.drop(columns=[delete_feature_name])
        feature_names.remove(delete_feature_name) 

    if "IL6_Value" in BASE_FEATURE_NAMES or "IL6_Value" in feature_names:
        il6_series_raw = df.iloc[:, COL_FEATURE_IL6]
        valid_il6 = il6_series_raw.dropna()
        il6_threshold = np.percentile(valid_il6, filter_percentile)
    else:
        il6_threshold = None

    return X, y, feature_names, df, encoder, il6_threshold


def preprocess_data(X, y, df):

    X = X.copy()

    LOG_COLS = ["IL6_Value","NLR","PLR","LMR","SII","SIRI","AISI","NPR","PAR","PNI"]
    use_log = [c for c in LOG_COLS if c in X.columns]

    NOLOG_NUM_COLS = ["Age","CTR","Imaging_Diameter","WBC","ALB","PLT","NEUT","LYMPH","HGB","MONO"]
    use_nolog = [c for c in NOLOG_NUM_COLS if c in X.columns]

    BIN_COLS = ["Spiculation","Pleural_Traction","Vacuole_Sign","Lobulation","Cystic_Lung_Cancer"]
    use_bin = [c for c in BIN_COLS if c in X.columns]
    CAT_ONEHOT = [col for col in X.columns if col.startswith("nodule_type_")]
    all_cat = use_bin + CAT_ONEHOT


    mice_imputer = IterativeImputer(random_state=42)   
    cat_imputer = SimpleImputer(strategy="most_frequent") 


    all_numeric = use_log + use_nolog
    if all_numeric:
        X[all_numeric] = mice_imputer.fit_transform(X[all_numeric])



    if use_log:
        X[use_log] = np.log1p(X[use_log])

    if all_cat:
        X[all_cat] = cat_imputer.fit_transform(X[all_cat])


    X_final_array = X.values
    X_train, X_test, y_train, y_test, train_idx, test_idx= train_test_split(
        X_final_array, y, df.index.values, test_size=0.3, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, None, mice_imputer, cat_imputer
