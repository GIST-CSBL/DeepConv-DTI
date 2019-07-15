import pandas as pd
import argparse
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, auc, roc_curve

parser = argparse.ArgumentParser()
parser.add_argument("predictions", help="prediction result to evaluate")
parser.add_argument("--test-name", "-n", help="name of test data set", nargs="*")
parser.add_argument("--threshold", "-T", help="threshold for prediction", type=float, default=0.5)
parser.add_argument("--no-threshold", "-N", help="performance evaluation without threshold (AUC,AUPR)", action="store_true")
parser.add_argument("--evaluation-output","-o" , help="output for result evaluation")
args = parser.parse_args()
no_th = args.no_threshold
prediction_dir = args.predictions
test_names = args.test_name
th = args.threshold
output_file = args.evaluation_output

extension = prediction_dir.split(".")[-1]

if extension=='csv':
   result_df = pd.read_csv(prediction_dir,header=[0,1])
elif extension=='tsv':
   result_df = pd.read_table(prediction_dir, header=[0,1])

th = args.threshold
def label_by_th(y_pred, threshold=0.5):
    y_pred_copy = y_pred.copy()
    y_pred_copy[y_pred>= threshold] = 1 
    y_pred_copy[y_pred<threshold] = 0 
    return y_pred_copy
if no_th:
   evaluation_df = pd.DataFrame(index=["Sen", "Spe", "Pre", "Acc", "F1","AUC", "AUPR"])
else:
   evaluation_df = pd.DataFrame(index=["Sen", "Spe", "Pre", "Acc", "F1"])


#print result_df.head()
for dataset in test_names:
    tn, fp, fn, tp = confusion_matrix(result_df[dataset,"label"].dropna(), label_by_th(result_df[dataset,"predicted"].dropna(), th)).ravel()
    print("Evaluation of the %s set " % dataset)
    sen = float(tp)/(fn+tp)
    pre = float(tp)/(tp+fp)
    spe = float(tn)/(tn+fp)
    acc = float(tn+tp)/(tn+fp+fn+tp)
    f1 = (2*sen*pre)/(sen+pre)
    print( "\tSen : ", sen )
    print("\tSpe : ", spe )
    print("\tAcc : ", acc )
    print("\tPrecision : ", pre)
    print("\tF1 : ", f1)
    result_dic = {"Acc": acc, "Sen" : sen, "Pre":pre, "F1":f1, "Spe":spe}
    if no_th:
       fpr, tpr, thresholds_AUC = roc_curve(result_df[dataset,"label"], result_df[dataset,"predicted"])
       AUC = auc(fpr, tpr)
       precision, recall, thresholds = precision_recall_curve(result_df[dataset,"label"],result_df[dataset,"predicted"])
       AUPR = auc(recall,precision)
       print("\tArea Under ROC Curve(AUC): %0.3f" % AUC)
       print("\tArea Under PR Curve(AUPR): %0.3f" % AUPR)
       print("=================================================")
       result_dic.update({"AUC":AUC, "AUPR":AUPR}) 
    evaluation_df[dataset] = pd.Series(result_dic)
evaluation_output =  args.evaluation_output
if evaluation_output:
   print("save to %s"%output_file)
   dir_name, file_name = os.path.split(evaluation_output)
   if not os.path.isdir(dir_name):
      os.system("mkdir -p "+dir_name)
      print("No directory named %s : create directory" % dir_name)
   evaluation_df.to_csv(evaluation_output)

