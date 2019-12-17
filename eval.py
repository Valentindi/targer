import sys
import pandas as pd
import json
from sklearn.metrics import classification_report, multilabel_confusion_matrix, confusion_matrix

def main(truth_fn, predicted):

	df = pd.read_csv(truth_fn, sep=" ")
	df = df.rename(columns={df.columns[2]:"truth"}) 
	f = open(predicted)
	json_pred = json.load(f)
	json_pred = [[float("NaN")] + doc for doc in json_pred]
	df["pred"] = sum(json_pred, [])
	labels = list(set(df.truth.to_list()))
	print(classification_report(df.truth.to_list(), df.pred.to_list()))
	print(confusion_matrix(df.truth.to_list(), df.pred.to_list()))



if __name__ == "__main__":
	print(sys.argv)
	main(sys.argv[1], sys.argv[2])



