# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

EXCEL_PATH = "data_fifth.xlsx"
USE_IL6 = True
USE_BLOOD_MARKERS = True 
USE_CELL_MARKERS = False 
USE_BASIC_MARKERS = False 

ONLY_KEEP_IL6_AND_BLOOD = True  

BLOOD_MARKERS = [
    "NLR",  "LMR", "SII", "SIRI",
    "AISI", "NPR",  "PNI", "PLR", "PAR"
]
CELL_MARKERS = ["WBC", "ALB", "PLT", "NEUT", "LYMPH", "MONO"] 
FILTER_PERCENTILE = 99
SAVE_DIR = "output_lasso_analysis"



sns.set_theme(style='whitegrid', font_scale=1.2)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.linewidth': 0.8,
    'axes.edgecolor': 'black',
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'black',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

from data import load_real_data, preprocess_data

def plot_lasso_cv_path(X_train, y_train, feature_names):
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    n_Cs = 50

    Cs = np.logspace(-4, 1, n_Cs)
    alphas = 1 / Cs
    
    cv_scores = []
    coefs_path = []
    
    print("正在计算参数和系数...")
    for c in Cs:
        model = LogisticRegression(
            penalty='l1', solver='saga', C=c,
            random_state=42, class_weight='balanced', max_iter=20000
        )
        model.fit(X_train, y_train)
        score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        cv_scores.append(score)
        coefs_path.append(model.coef_[0])
    
    cv_scores = np.array(cv_scores)
    scores_mean = np.mean(cv_scores, axis=1)
    scores_std = np.std(cv_scores, axis=1)
    coefs_path = np.array(coefs_path).T
    
    max_score_idx = np.argmax(scores_mean)
    best_C = Cs[max_score_idx]
    best_alpha = alphas[max_score_idx]
    k = 1
    score_1se = scores_mean[max_score_idx] - k * scores_std[max_score_idx]

    index_1se = np.argmin(np.abs(scores_mean[:max_score_idx+1] - score_1se))
    C_1se = Cs[index_1se]
    alpha_1se = alphas[index_1se]
    
    print(f"✅ 最优正则参数 C: {best_C:.4f} (α: {best_alpha:.4f})")
    print(f"✅ 1SE正则参数 C: {C_1se:.4f} (α: {alpha_1se:.4f})")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5)) 
    

    ax1.plot(alphas, scores_mean, 
             color='red', marker='o', markersize=4,
             markeredgecolor='gray', markeredgewidth=0.5,
             linewidth=1.5, label='Mean CV Accuracy')
    ax1.fill_between(alphas, scores_mean - scores_std, scores_mean + scores_std,
                     color='#2c7bb6', alpha=0.15, label='±1 std')
    
    ax1.axvline(best_alpha, linestyle='--', color='#d7191c', linewidth=1.5,
                label=f'α* = {best_alpha:.4f}')
    ax1.axvline(alpha_1se, linestyle='--', color='#fdae61', linewidth=1.5,
                label=f'$α_{{1SE}}$ = {alpha_1se:.4f}')
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Regularization strength (α)', fontsize=12, weight='bold')
    ax1.set_ylabel('Mean CV Accuracy', fontsize=12, weight='bold')
    ax1.set_title('(A) LASSO Cross-Validation', fontsize=13, weight='bold', pad=15)
    ax1.legend(loc='lower right', framealpha=0.95, fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    n_features = coefs_path.shape[0]

    line_colors = [
        "#48A597",  
        "#2D6FB5",  
        "#A67C52",  
        "#7A6BA3",  
        "#E08E45",  
        "#3A9399",  
        "#8B8B7A",  
        "#C2554F",  
        "#5B8C5A"   
    ]
    color_idx = 0
    for i in range(n_features):
        ft_name = feature_names[i]
        if ft_name == "IL6_Value":
            ax2.plot(alphas, coefs_path[i], color='#9C1A1C', linewidth=3.5, alpha=1, label="IL-6")
        else:
            color = line_colors[color_idx % len(line_colors)]
            color_idx += 1
            ax2.plot(alphas, coefs_path[i], color=color, linewidth=3.5, alpha=0.85, label=ft_name)
    ax2.axvline(best_alpha, linestyle='--', color='#d7191c', linewidth=1.5)
    ax2.axvline(alpha_1se, linestyle='--', color='#fdae61', linewidth=1.5)

    
    ax2.set_xscale('log')
    ax2.set_xlim(right=50) 
    ax2.set_xlabel('Regularization strength (α)', fontsize=12, weight='bold')
    ax2.set_ylabel('Coefficient Value', fontsize=12, weight='bold')
    ax2.set_title('LASSO Coefficient Path', fontsize=13, weight='bold', pad=15)
    ax2.grid(True, linestyle=':', alpha=0.6)

    ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left", framealpha=0.9, fontsize=8)
    
    plt.tight_layout()
    
    save_png = os.path.join(SAVE_DIR, "LASSO_CV_Path.png")
    save_pdf = os.path.join(SAVE_DIR, "LASSO_CV_Path.pdf")
    plt.savefig(save_png, dpi=300, bbox_inches='tight')
    plt.savefig(save_pdf, bbox_inches='tight')
    plt.close()
    print(f"✅ 顶刊双图已保存：{save_png}")
    return best_C

def export_feature_importance(X_train, y_train, feature_names, best_C):
    lasso = LogisticRegression(
        penalty='l1', solver='saga', C=best_C,
        random_state=42, class_weight='balanced', max_iter=2000
    )
    lasso.fit(X_train, y_train)
    
    best_coef = lasso.coef_[0].ravel()
    importance = np.abs(best_coef)

    df = pd.DataFrame({
        "Rank": range(1, len(feature_names)+1),
        "Feature": feature_names,
        "Coefficient": best_coef,
        "Importance(Abs)": importance
    }).sort_values("Importance(Abs)", ascending=False)

    print("\n" + "="*60)
    print(f"📊 LASSO特征重要性排名")
    print("="*60)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(SAVE_DIR, "LASSO_Feature_Importance.csv"), index=False, encoding="utf-8-sig")
    print(f"\n✅ 特征重要性已保存")


if __name__ == "__main__":
    
    X, y, feature_names, df, temp1, temp2 = load_real_data(
        excel_path=EXCEL_PATH,
        use_il6=USE_IL6,
        use_blood_markers=USE_BLOOD_MARKERS,
        use_cell_markers=USE_CELL_MARKERS,
        filter_percentile=FILTER_PERCENTILE,
        blood_markers=BLOOD_MARKERS,
        cell_markers=CELL_MARKERS
    )
    keep_features = []
    if USE_BASIC_MARKERS:
        for i in feature_names:
            if (i not in BLOOD_MARKERS) and i != "IL6_Value":
                keep_features.append(i)
    if ONLY_KEEP_IL6_AND_BLOOD:
        if USE_IL6:
            keep_features.append("IL6_Value")
        if USE_BLOOD_MARKERS:
            keep_features.extend(BLOOD_MARKERS)
        if USE_CELL_MARKERS:
            keep_features.extend(CELL_MARKERS)
        X = X[keep_features]
        feature_names = keep_features
        print(f"✅ 已开启：仅保留 IL6 + 炎症指标")
        print(f"📌 最终参与分析的特征：{feature_names}")

    X_train, X_test, y_train, y_test, _, _, _ = preprocess_data(X, y, df)
    
    best_C = plot_lasso_cv_path(X_train, y_train, feature_names)
    export_feature_importance(X_train, y_train, feature_names, best_C)
    
    print("\n🎉 分析完成！结果：", SAVE_DIR)