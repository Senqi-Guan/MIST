'''
1.conda create -n mist python=3.10
2.conda activate mist
3.pip install pandas scikit-learn lightgbm matplotlib
4.pip install openpyxl
5.pip install numpy==1.26.4 shap==0.45.0
6.pip install xgboost==2.0.3 -i https://pypi.tuna.tsinghua.edu.cn/simple

python main.py --our_index 1 --use_cell_markers 0 --cell_markers WBC --use_blood_markers 1 --blood_markers LMR PAR --shap_model 3 --data_per 99 --use_metalearning 1 --eval_data 1 --eval_use_il6 1 --eval_use_par 1 --eval_use_lmr 1
'''
