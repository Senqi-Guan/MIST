# -*- coding: utf-8 -*-
import pandas as pd
import os
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from data import preprocess_data, load_real_data
from models import get_model
from eval import evaluate_model
from plot_figures import plot_all_figures, plot_single_model_cross_cohort
from shap_plot import plot_shap_analysis
from eval_data import load_eval_data, preprocess_eval_data
import argparse  
import numpy as np
import joblib 


warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

ROOT_OUTPUT = "output"
os.makedirs(ROOT_OUTPUT, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


def main():
    parser = argparse.ArgumentParser(description="肺结节预测模型")
    parser.add_argument('--our_index', type=int, default=1, choices=[0,1], help='0=无IL6, 1=有IL6')
    parser.add_argument('--use_cell_markers',type=int, default=1, choices=[0,1], help='0=无细胞指标, 1=有细胞指标')
    parser.add_argument('--cell_markers', nargs='*', default=[], help='指定要加入的细胞指标，例：--cell_markers WBC ALB PLT')
    parser.add_argument('--use_blood_markers',type=int, default=1, choices=[0,1], help='0=无血液新指标, 1=有血液新指标')
    parser.add_argument('--blood_markers', nargs='*', default=[], help='指定要加入的血清指标，例：--blood_markers NLR PLR')
    parser.add_argument('--shap_model', type=int, default=5, choices=[0,1,2,3,4,5,6,7], help='选择SHAP分析的模型: 0=lr,1=svm,2=knn,3=extratrees,4=xgboost,5=lightgbm')
    parser.add_argument('--data_per', type=int, default=99, help='保留数据的百分比，消除极值的影响')
    parser.add_argument('--data_root', type=str, default='data_fifth.xlsx', help='输入excel的文件位置')
    parser.add_argument('--use_metalearning', type=int, default=0, choices=[0,1], help='0=不使用元学习模型, 1=使用堆叠元学习模型')

    parser.add_argument('--eval_data', type=int, default=0, choices=[0,1], help='0=不跑验证集, 1=运行外部验证集')
    parser.add_argument('--eval_excel', type=str, default='eval_data.xlsx', help='验证集Excel路径')
    parser.add_argument('--eval_use_il6', type=int, default=1, choices=[0,1], help='验证集是否使用IL6')
    parser.add_argument('--eval_use_par', type=int, default=1, choices=[0,1], help='验证集是否使用PLR')
    parser.add_argument('--eval_use_lmr', type=int, default=1, choices=[0,1], help='验证集是否使用LMR')

    args = parser.parse_args()
    

    use_il6 = (args.our_index == 1)
    use_blood_markers = (args.use_blood_markers == 1)
    use_cell_markers = (args.use_cell_markers == 1)
    use_eval = (args.eval_data == 1)  
    eval_use_il6 = (args.eval_use_il6 == 1)
    eval_use_par = (args.eval_use_par == 1)
    eval_use_lmr = (args.eval_use_lmr == 1)
    print("🚀 开始运行")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(ROOT_OUTPUT, timestamp)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"📂 本次实验文件夹：{experiment_dir}")
    model_weight_dir = os.path.join(experiment_dir, "model_weight")
    os.makedirs(model_weight_dir, exist_ok=True)
    print(f"📦 模型权重文件夹：{model_weight_dir}")

    X, y , feature_names, df, nodule_encoder, il6_threshold= load_real_data(
        excel_path=args.data_root,
        use_il6=use_il6,
        use_blood_markers=use_blood_markers,
        use_cell_markers=use_cell_markers,  
        filter_percentile=args.data_per,
        blood_markers=args.blood_markers,
        cell_markers=args.cell_markers)
    X_train, X_test, y_train, y_test, scaler, mice_imputer, cat_imputer = preprocess_data(X, y, df)

    model_list = ['lr', 'svm', 'randomforest', 'extratrees', 'xgboost', 'LightGBM']
    if args.use_metalearning == 1:
        model_list.append('Metalearning Model')
    results = []
    prob_list = []
    trained_models = []  

    for name in model_list:
        model = get_model(name)
        

        model.fit(X_train, y_train)
        trained_models.append(model)  
        model_save_path = os.path.join(model_weight_dir, f"{name}.pt")

        auc, acc, sen, spe, ppv, npv, f1, brier, prob = evaluate_model(model, X_test, y_test)
        results.append({"model":name,"auc":auc,"acc":acc,"sen":sen,"spe":spe,"ppv":ppv,"npv":npv,"f1":f1,"brier":brier})
        prob_list.append(prob)

    csv_path = os.path.join(experiment_dir, 'metrics.csv')
    pd.DataFrame(results).to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ 指标已保存：metrics.csv")

    plot_all_figures(y_test, model_list, prob_list, experiment_dir)
    
    print(f"✅ 测试集特征数量：{X_test.shape[1]} 列")

    plot_shap_analysis(trained_models[args.shap_model], X_train, X_test, y_train, y_test, feature_names, experiment_dir, sample_size=100)

    if use_eval:
        print("\n" + "="*60)
        print("🔍 开始运行【外部验证集】评估")

        eval_save_dir = os.path.join(experiment_dir, "eval_result")
        os.makedirs(eval_save_dir, exist_ok=True)
        print(f"📂 验证集结果保存目录：{eval_save_dir}")

        X_eval_raw, y_eval, _, _ = load_eval_data(
            eval_excel_path=args.eval_excel,
            use_il6=eval_use_il6,
            use_par=eval_use_par,
            use_lmr=eval_use_lmr,
            train_il6_threshold=il6_threshold,
            train_nodule_encoder=nodule_encoder
        )

        X_eval = preprocess_eval_data(X_eval_raw, mice_imputer, cat_imputer)
        eval_results = []
        eval_prob_list = []
        for idx, name in enumerate(model_list):
            model = trained_models[idx]
            auc, acc, sen, spe, ppv, npv, f1, brier, prob = evaluate_model(model, X_eval, y_eval)
            eval_results.append({
                "model": name, "auc": auc, "acc": acc, "sen": sen, "spe": spe,
                "ppv": ppv, "npv": npv, "f1": f1, "brier": brier
            })
            eval_prob_list.append(prob)

        eval_csv_path = os.path.join(eval_save_dir, 'eval_metrics.csv')
        pd.DataFrame(eval_results).to_csv(eval_csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ 验证集指标已保存：eval_metrics.csv")

        plot_all_figures(y_eval, model_list, eval_prob_list, eval_save_dir)
        print(f"✅ 验证集绘图完成，图片保存在 eval_result 文件夹")
        print("="*60 + "\n")
        
        ablation_dir = os.path.join(experiment_dir, "ablation")
        os.makedirs(ablation_dir, exist_ok=True)
        print(f"📂 单模型对比图保存目录：{ablation_dir}")

        target_idx = args.shap_model 

        target_model_name = model_list[target_idx]
        target_model = trained_models[target_idx]

        train_prob = target_model.predict_proba(X_train)[:, 1]

        test_prob = prob_list[target_idx]
        eval_prob = eval_prob_list[target_idx]


        plot_single_model_cross_cohort(
            model_name=target_model_name,
            y_train=y_train,
            train_prob=train_prob,
            y_test=y_test,
            test_prob=test_prob,
            y_eval=y_eval,
            eval_prob=eval_prob,
            save_dir=ablation_dir
        )
        print(f"✅ 单模型跨队列对比图绘制完成")
    

    print(f"\n🎉 全部实验完成！所有文件保存在：\n{experiment_dir}")

if __name__ == "__main__":
    main()