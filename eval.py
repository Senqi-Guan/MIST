from sklearn.metrics import (
    roc_auc_score, accuracy_score, recall_score, precision_score,
    confusion_matrix, f1_score, brier_score_loss
)

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]


    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    sen = recall_score(y_test, y_pred)  
    ppv = precision_score(y_test, y_pred) 
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    spe = tn / (tn + fp)  
    npv = tn / (tn + fn)  


    f1 = f1_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_pred_proba)



    print(f"\n{'='*50}\n模型：{model.__class__.__name__}\n{'='*50}")
    print(f"AUC: {auc:.4f}\n准确率(Acc): {acc:.4f}\n灵敏度(Sen): {sen:.4f}")
    print(f"特异度(Spe): {spe:.4f}\n阳性预测值(PPV): {ppv:.4f}\n阴性预测值(NPV): {npv:.4f}")
    print(f"F1分数: {f1:.4f}\nBrier分数: {brier:.4f}")


    return (
        round(auc,3), round(acc,3), round(sen,3), round(spe,3),
        round(ppv,3), round(npv,3), round(f1,3), round(brier,3),
        y_pred_proba
    )