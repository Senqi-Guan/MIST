import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, roc_auc_score
import os
import pandas as pd
from scipy import stats
import os

from scipy import stats
import numpy as np
from sklearn.metrics import roc_auc_score
from matplotlib.colors import LinearSegmentedColormap

def compute_delong_p(y_true, y_pred1, y_pred2):

    y_true = np.asarray(y_true).flatten()
    y_pred1 = np.asarray(y_pred1).flatten()
    y_pred2 = np.asarray(y_pred2).flatten()
    

    pos_mask = y_true == 1
    neg_mask = y_true == 0
    n_pos, n_neg = pos_mask.sum(), neg_mask.sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    

    pred1_pos, pred1_neg = y_pred1[pos_mask], y_pred1[neg_mask]
    pred2_pos, pred2_neg = y_pred2[pos_mask], y_pred2[neg_mask]
    

    V10_1 = _calc_v10(pred1_pos, pred1_neg)  
    V10_2 = _calc_v10(pred2_pos, pred2_neg)  
    V01_1 = _calc_v01(pred1_pos, pred1_neg)  
    V01_2 = _calc_v01(pred2_pos, pred2_neg)  
    

    var1 = np.var(V10_1, ddof=1) / n_pos + np.var(V01_1, ddof=1) / n_neg
    var2 = np.var(V10_2, ddof=1) / n_pos + np.var(V01_2, ddof=1) / n_neg
    

    cov_v10 = np.cov(V10_1, V10_2, ddof=1)[0, 1]
    cov_v01 = np.cov(V01_1, V01_2, ddof=1)[0, 1]
    cov_12 = cov_v10 / n_pos + cov_v01 / n_neg
    

    se = np.sqrt(var1 + var2 - 2 * cov_12)
    auc1 = roc_auc_score(y_true, y_pred1)
    auc2 = roc_auc_score(y_true, y_pred2)
    if se < 1e-10:
        return 1.0 if abs(auc1 - auc2) < 1e-10 else 0.0
    z = (auc1 - auc2) / se
    
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return p

def _calc_v10(pos_pred, neg_pred):
    """V10：每个阳性样本的预测值 大于 所有阴性样本预测值的比例"""
    return np.mean(pos_pred[:, np.newaxis] > neg_pred[np.newaxis, :], axis=1)

def _calc_v01(pos_pred, neg_pred):
    """V01：每个阴性样本的预测值 小于 所有阳性样本预测值的比例"""
    return np.mean(neg_pred[np.newaxis, :] < pos_pred[:, np.newaxis], axis=0)


def compute_nri_idi(y_true, prob_base, prob_new, threshold=0.5):
    base_pred = (prob_base >= threshold).astype(int)
    new_pred = (prob_new >= threshold).astype(int)

    p1_new = prob_new[y_true == 1].mean()
    p0_new = prob_new[y_true == 0].mean()
    p1_base = prob_base[y_true == 1].mean()
    p0_base = prob_base[y_true == 0].mean()
    idi = (p1_new - p1_base) - (p0_new - p0_base)

    case_up = np.sum((y_true == 1) & (new_pred > base_pred))
    case_down = np.sum((y_true == 1) & (new_pred < base_pred))
    control_up = np.sum((y_true == 0) & (new_pred > base_pred))
    control_down = np.sum((y_true == 0) & (new_pred < base_pred))
    n_case = np.sum(y_true == 1)
    n_control = np.sum(y_true == 0)
    nri = (case_up - case_down) / n_case - (control_up - control_down) / n_control

    return nri, idi



def plot_all_figures(y_test, model_names, prob_list, save_dir):
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    for i, (name, prob) in enumerate(zip(model_names, prob_list)):
        frac_pos, mean_pred = calibration_curve(y_test, prob, n_bins=5)
        plt.plot(mean_pred, frac_pos, marker='o', linewidth=2, color=colors[i], label=name)
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title('Calibration Curves of the Test Cohort', fontsize=14, pad=15)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    base_path = os.path.join(save_dir, 'figure_calibration_curve')
    plt.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{base_path}.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    youden_thresholds = []  
    plt.figure(figsize=(7, 6))
    for i, (name, prob) in enumerate(zip(model_names, prob_list)):
        fpr, tpr, thresholds = roc_curve(y_test, prob)
        youden_j = tpr - fpr  
        best_idx = np.argmax(youden_j)
        best_thr = thresholds[best_idx]
        youden_thresholds.append(best_thr)
        auc = roc_auc_score(y_test, prob)
        plt.plot(fpr, tpr, linewidth=2, color=colors[i], label=f'{name} (AUC={auc:.3f})')
    plt.plot([0,1],[0,1],'k--', linewidth=2, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves of the Test Cohort', fontsize=14, pad=15)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    base_path = os.path.join(save_dir, 'figure_AUC')
    plt.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{base_path}.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(7, 6))
    thresholds = np.linspace(0.01, 0.99, 100)
    for i, (name, prob) in enumerate(zip(model_names, prob_list)):
        net_benefit = []
        for th in thresholds:
            y_pred = (prob >= th).astype(int)
            tp = np.sum((y_test == 1) & (y_pred == 1))
            fp = np.sum((y_test == 0) & (y_pred == 1))
            nb = (tp - fp * (th/(1-th))) / len(y_test)
            net_benefit.append(nb)
        net_benefit = np.array(net_benefit)
        net_benefit[net_benefit < 0] = 0.0  
        plt.plot(thresholds, net_benefit, linewidth=2, color=colors[i], label=name)
    plt.plot(thresholds, np.zeros_like(thresholds), 'k--', linewidth=2, label='None')
    plt.plot(thresholds, thresholds, 'k-.', linewidth=2, label='All')
    plt.xlabel('Threshold Probability', fontsize=12)
    plt.ylabel('Net Benefit', fontsize=12)
    plt.title('Decision Curve for the Test Cohort', fontsize=14, pad=15)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    base_path = os.path.join(save_dir, 'figure_DCA')
    plt.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{base_path}.pdf", dpi=300, bbox_inches='tight')
    plt.close()



    delong_results = []
    n_models = len(model_names)
    for i in range(n_models):
        for j in range(i+1, n_models):
            m1, m2 = model_names[i], model_names[j]
            p_val = compute_delong_p(y_test, prob_list[i], prob_list[j])
            delong_results.append([m1, m2, round(p_val, 4)])


    df_delong = pd.DataFrame(delong_results, columns=["模型1", "模型2", "Delong_p值"])
    df_delong.to_csv(os.path.join(save_dir, "delong_test_results.csv"), index=False, encoding="utf-8-sig")
    print("✅ Delong检验完成，结果已保存")


    base_idx = 5 
    base_name = model_names[base_idx]
    base_prob = prob_list[base_idx]
    nri_threshold = youden_thresholds[base_idx]


    nri_idi_results = []
    for i in range(n_models):
        if i == base_idx:
            continue
        new_name = model_names[i]
        new_prob = prob_list[i]
        nri, idi = compute_nri_idi(y_test, base_prob, new_prob, nri_threshold)
        nri_idi_results.append([base_name, new_name, round(nri, 4), round(idi, 4)])


    df_nri_idi = pd.DataFrame(nri_idi_results, columns=["基准模型", "对比模型", "NRI", "IDI"])
    df_nri_idi.to_csv(os.path.join(save_dir, "nri_idi_results.csv"), index=False, encoding="utf-8-sig")
    print("✅ NRI/IDI计算完成，结果已保存")




def plot_single_model_cross_cohort(model_name, y_train, train_prob, y_test, test_prob, y_eval, eval_prob, save_dir):

    shap_cmap = LinearSegmentedColormap.from_list(
        "custom_shap",
        ["#48A597", "#FFFFFF", "#9C1A1C"]
    )

    colors = ['#48A597', '#9C1A1C', '#2D6FB5']
    cohort_labels = ['Train Cohort', 'Test Cohort', 'Validation Cohort']
    y_list = [y_train, y_test, y_eval]
    prob_list = [train_prob, test_prob, eval_prob]

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    for i, (y, prob, label) in enumerate(zip(y_list, prob_list, cohort_labels)):
        frac_pos, mean_pred = calibration_curve(y, prob, n_bins=5)
        plt.plot(mean_pred, frac_pos, marker='o', linewidth=2, color=colors[i], label=label)
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title(f'Calibration Curves of {model_name} across Cohorts', fontsize=14, pad=15)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    base_path = os.path.join(save_dir, 'figure_cross_cohort_calibration')
    plt.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{base_path}.pdf", dpi=300, bbox_inches='tight')
    plt.close()


    plt.figure(figsize=(7, 6))
    n_bootstrap = 1000
    mean_fpr = np.linspace(0, 1, 100)  
    rng = np.random.default_rng(42)
    for i, (y, prob, label) in enumerate(zip(y_list, prob_list, cohort_labels)):
        y = np.array(y)
        prob = np.array(prob)
        n_samples = len(y)
        
        fpr, tpr, _ = roc_curve(y, prob)
        auc = roc_auc_score(y, prob)
        
        boot_tprs = []
        for _ in range(n_bootstrap):
            idx = rng.choice(n_samples, size=n_samples, replace=True)
            boot_fpr, boot_tpr, _ = roc_curve(y[idx], prob[idx])
            interp_tpr = np.interp(mean_fpr, boot_fpr, boot_tpr)
            interp_tpr[0] = 0.0
            boot_tprs.append(interp_tpr)
        boot_tprs = np.array(boot_tprs)
        ci_lower = np.percentile(boot_tprs, 2.5, axis=0)
        ci_upper = np.percentile(boot_tprs, 97.5, axis=0)
        
        plt.plot(fpr, tpr, linewidth=2, color=colors[i], label=f'{label} (AUC={auc:.3f})')
        plt.fill_between(mean_fpr, ci_lower, ci_upper, color=colors[i], alpha=0.2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves of {model_name} across Cohorts', fontsize=14, pad=15)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    base_path = os.path.join(save_dir, 'figure_cross_cohort_roc')
    plt.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{base_path}.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(7, 6))
    thresholds = np.linspace(0.01, 0.99, 100)
    for i, (y, prob, label) in enumerate(zip(y_list, prob_list, cohort_labels)):
        net_benefit = []
        for th in thresholds:
            y_pred = (prob >= th).astype(int)
            tp = np.sum((y == 1) & (y_pred == 1))
            fp = np.sum((y == 0) & (y_pred == 1))
            nb = (tp - fp * (th / (1 - th))) / len(y)
            net_benefit.append(nb)
        net_benefit = np.array(net_benefit)
        net_benefit[net_benefit < 0] = 0.0
        plt.plot(thresholds, net_benefit, linewidth=2, color=colors[i], label=label)
    plt.plot(thresholds, np.zeros_like(thresholds), 'k--', linewidth=2, label='None')
    plt.plot(thresholds, thresholds, 'k-.', linewidth=2, label='All')
    plt.xlabel('Threshold Probability', fontsize=12)
    plt.ylabel('Net Benefit', fontsize=12)
    plt.title(f'Decision Curve of {model_name} across Cohorts', fontsize=14, pad=15)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    base_path = os.path.join(save_dir, 'figure_cross_cohort_dca')
    plt.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{base_path}.pdf", dpi=300, bbox_inches='tight')
    plt.close()

