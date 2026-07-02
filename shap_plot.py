import os
import numpy as np
import matplotlib.pyplot as plt
import shap
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap  
import re
import statsmodels.api as sm  



def plot_shap_analysis(
    model, 
    X_train, X_test, y_train, y_test, 
    feature_names,  
    save_dir, 
    sample_size=150
):

    os.makedirs(save_dir, exist_ok=True)
    model_name = type(model).__name__
    print(f"\n🔍 正在处理 {model_name} 的SHAP分析...")

    if model_name in ['LGBMClassifier', 'XGBClassifier', 'ExtraTreesClassifier', 'RandomForestClassifier', 'BalancedRandomForestClassifier', 'HistGradientBoostingClassifier']:
        explainer = shap.TreeExplainer(model)
        raw_shap = explainer.shap_values(X_test)
        X_plot = X_test
    elif model_name in ['StackingClassifier']:
        background = shap.sample(X_train, 100)
        explainer = shap.KernelExplainer(model.predict_proba, background)

        X_plot = shap.sample(X_test, min(100, len(X_test)))
        raw_shap = explainer.shap_values(X_plot)
    elif model_name in ['LogisticRegression', 'RidgeClassifier', 'PassiveAggressiveClassifier']:
        background = shap.sample(X_train, 100)
        explainer = shap.LinearExplainer(model, background)
        raw_shap = explainer.shap_values(X_test)
        X_plot = X_test

    elif model_name in ['SVC', 'KNeighborsClassifier']:
        background = shap.sample(X_train, sample_size)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        X_plot = shap.sample(X_test, min(sample_size, len(X_test)))
        raw_shap = explainer.shap_values(X_plot)

    else:
        raise ValueError(f"不支持的模型类型: {model_name}")

    if isinstance(raw_shap, list):
        shap_values = raw_shap[1]
    elif isinstance(raw_shap, np.ndarray):
        if len(raw_shap.shape) == 2:
            shap_values = raw_shap
        elif len(raw_shap.shape) == 3:
            class_dim = np.argwhere(np.array(raw_shap.shape) == 2)[0][0]
            if class_dim == 0:
                shap_values = raw_shap[1]
            elif class_dim == 1:
                shap_values = raw_shap[:, 1, :]
            else:
                shap_values = raw_shap[:, :, 1]
    else:
        raise ValueError(f"不支持的SHAP值类型")


    assert shap_values.shape == X_plot.shape, "形状不匹配"
    assert shap_values.shape[1] == len(feature_names), "特征数量不匹配"
    print("✅ SHAP值格式校验通过！")

    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    X_all = pd.concat([X_train_df, X_test_df], axis=0).reset_index(drop=True)
    raw_shap_all = explainer.shap_values(X_all)
    if isinstance(raw_shap_all, list):
        shap_all = raw_shap_all[1]
    elif isinstance(raw_shap_all, np.ndarray):
        if len(raw_shap_all.shape) == 2:
            shap_all = raw_shap_all
        elif len(raw_shap_all.shape) == 3:
            class_dim = np.argwhere(np.array(raw_shap_all.shape) == 2)[0][0]
            if class_dim == 0:
                shap_all = raw_shap_all[1]
            elif class_dim == 1:
                shap_all = raw_shap_all[:, 1, :]
            else:
                shap_all = raw_shap_all[:, :, 1]
    else:
        raise ValueError(f"不支持的SHAP值类型")

    assert shap_all.shape == X_all.shape, "全量数据：SHAP与特征形状不匹配"
    assert shap_all.shape[1] == len(feature_names), "全量数据：特征数量不匹配"
    print("✅ 全量数据SHAP维度解析完成！")

    plt.figure(figsize=(10, 12))
    shap.summary_plot(shap_values, X_plot, feature_names=feature_names, plot_type="dot", max_display=12, show=False)
    base_path = os.path.join(save_dir, "SHAP_蜂群图")
    plt.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{base_path}.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 12))
    shap.summary_plot(shap_values, X_plot, feature_names=feature_names, plot_type="bar", max_display=12, show=False)
    base_path = os.path.join(save_dir, "SHAP_特征重要性")
    plt.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{base_path}.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax1 = plt.subplots(figsize=(10, 8))  

    cmap = LinearSegmentedColormap.from_list("original_ink", ["#4AA698", "#F6FAFA", "#9E1F21"])
    bar_color = "#CFDBF4"

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sort_inds = np.argsort(mean_abs_shap)  
    sorted_features = [feature_names[i] for i in sort_inds]
    sorted_mean_shap = mean_abs_shap[sort_inds]
    total_shap_sum = np.sum(mean_abs_shap)  

    shap_vals_sorted = shap_values[:, sort_inds]
    feat_vals_sorted = X_plot.values[:, sort_inds] if hasattr(X_plot, 'values') else X_plot[:, sort_inds]
    y_pos = np.arange(len(sorted_features))


    ax2 = ax1.twiny()

    ax2.barh(y_pos, sorted_mean_shap, color=bar_color, align='center', alpha=0.8, height=0.6, zorder=2)
    for i in range(len(sorted_features)):
        row_shap = shap_vals_sorted[:, i]
        row_feat = feat_vals_sorted[:, i]
        
        feat_min, feat_max = np.min(row_feat), np.max(row_feat)
        row_feat_norm = (row_feat - feat_min)/(feat_max - feat_min) if feat_max > feat_min else np.zeros_like(row_feat)
        
        jitter = np.random.normal(0, 0.1, size=len(row_shap))
        ax1.scatter(
            row_shap, np.repeat(i, len(row_shap)) + jitter,
            c=row_feat_norm, cmap=cmap, s=15, alpha=0.7, edgecolors='none', zorder=4
        )

    max_mean_val = np.max(sorted_mean_shap)
    ax2.set_xlim(0, max_mean_val * 1.2)

    for i, v in enumerate(sorted_mean_shap):
        pct = (v / total_shap_sum)*100
        offset = max_mean_val * 0.01
        ax1.text(
            v + offset, i, f"{pct:.1f}%", va='center', ha='left', fontsize=10,
            transform=ax2.transData, zorder=10,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1)
        )

    ax1.set_zorder(ax2.get_zorder()+1)
    ax1.patch.set_visible(False)

    ax1.set_yticks(y_pos)
    display_labels = []
    for name in sorted_features:
        if name == "IL6_Value":
            display_labels.append("IL-6")
        elif name == "nodule_type_pGGN":
            display_labels.append("Nodule_Type_pGGN")
        elif name == "nodule_type_mGGN":
            display_labels.append("Nodule_Type_mGGN")
        elif name == "nodule_type_hmGGN":
            display_labels.append("Nodule_Type_hmGGN")
        elif name == "Lobulation":
            display_labels.append("Lobulation_Sign")
        elif name == "Spiculation":
            display_labels.append("Spiculation_Sign")
        elif name == "Pleural_Traction":
            display_labels.append("Pleural_Traction_Sign")
        else:
            display_labels.append(name)
    ax1.set_yticklabels(display_labels, fontsize=11)
    ax1.set_xlabel("SHAP value (impact on model output)", fontsize=12)
    ax2.set_xlabel("Mean Absolute SHAP Value of Metalearning Model", fontsize=12)
    ax1.grid(True, axis='x', linestyle='--', alpha=0.4)

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    sm_map = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm_map.set_array([])
    cbar = fig.colorbar(sm_map, cax=cax)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Low', 'High'])
    cbar.set_label('Feature value', rotation=270, labelpad=15, fontsize=12)

    base_path = os.path.join(save_dir, "Fig2_Global_Contribution")
    plt.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{base_path}.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ {model_name} SHAP分析完成！")



    analyze_clinical_features(X_train, X_test, y_train, y_test, shap_values, X_plot, feature_names, save_dir)


def analyze_clinical_features(X_train, X_test, y_train, y_test, shap_values, X_plot, feature_names, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    target_features = ["IL6_Value", "PAR", "LMR"]
    
    X_all = np.vstack([X_train, X_test])
    y_all = np.hstack([y_train, y_test])
    total_samples = len(X_train) + len(X_test)

    print(f"\n📊 Total samples check: {total_samples} cases")
    print(f"Train set: {len(X_train)} | Test set: {len(X_test)}")
    print(f"✅ Total adenocarcinoma cases: {y_all.sum()}")

    for feat_name in target_features:

        feat_idx = feature_names.index(feat_name) if feat_name in feature_names else -1
        
        if feat_idx == -1:
            print(f"\n⚠️ 未找到特征 {feat_name}，跳过该特征分析")
            continue
        
        feat_all = X_all[:, feat_idx]


        plt.figure(figsize=(8, 5))
        shap.dependence_plot(
            feat_idx, shap_values, X_plot,
            feature_names=feature_names,
            interaction_index=None, show=False, dot_size=50
        )
        plt.axhline(y=0, color='red', linestyle='--', lw=2, label='SHAP = 0 (cut-off boundary)')

        plt.xlabel(f"{feat_name.split('_')[0]} Level")
        plt.ylabel("SHAP Value (Positive = Promotes Adenocarcinoma)")
        plt.title(f"Prognostic Value of {feat_name.split('_')[0]} for Invasive Adenocarcinoma Prediction")
        plt.legend()
        plt.tight_layout()
        
        base_path = os.path.join(save_dir, f"{feat_name.split('_')[0]}_cutoff_analysis")
        plt.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{base_path}.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ {feat_name} 图表保存完成：{base_path}.png/.pdf")

    print(f"\n🎉 所有特征分析完成！")
