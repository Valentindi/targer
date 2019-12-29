import sys
import pandas as pd
import json
from sklearn.metrics import classification_report, accuracy_score, average_precision_score

def generate_question_table(token, truth, pred):
	correct = [a == b for a, b in zip(truth, pred)]

	row_token = "".join([str.format("<td>{}</td>", x) if c else str.format("<td><red>{}</red></td>", x) for c, x in zip(correct, token)])
	row_truth = "".join([str.format("<td>{}</td>", x) if c else str.format("<td><red>{}</red></td>", x) for c, x in zip(correct, truth)])
	row_pred = "".join([str.format("<td>{}</td>", x) if c else str.format("<td><red>{}</red></td>", x) for c, x in zip(correct, pred)])


	return str.format("<table><tr><td>Token:</td>{}</tr><tr><td>Truth:</td>{}</tr><tr><td>Predicted:</td>{}</tr></table>", row_token, row_truth, row_pred)

def main(truth_fn, predicted_fn, output_fn):

	df = pd.read_csv(truth_fn, sep=" ")
	df = df.rename(columns={df.columns[0]:"tokens", df.columns[2]:"truth"}) 
	f = open(predicted_fn)
	json_pred = json.load(f)
	json_pred = [[float("NaN")] + doc for doc in json_pred]
	df["pred"] = sum(json_pred, [])
	
	pairs = []

	truth_question = []
	pred_question = []
	token_question = []
	tokens = [x for x in df.tokens]
	predicted = [x for x in df["pred"]]
	for it, token in enumerate(df.truth):
		if str(token) == "nan":
			if len(truth_question) < 1:
				continue
			accuracy = accuracy_score(truth_question, pred_question)
			pairs += [
				{
				"token": token_question, 
				"truth": truth_question, 
				"pred": pred_question,
				"accuracy": accuracy
				}
			]
			truth_question = []
			pred_question = []
			token_question = []
		else:
			truth_question += [token]
			pred_question += [predicted[it]]
			token_question += [tokens[it]]


	pairs = sorted(pairs, key=lambda x: x["accuracy"], reverse=True)


	style = "<style> red {color:red} </style>"
	output_html = "<html><head>{}</head><body><h2>Detailed Report on {} Samples</h2>".format(style, len(pairs))
	output_html += str.format("INPUT {} and {}", truth_fn, predicted_fn)
	for it, x in enumerate(pairs):

		report = classification_report(x["truth"], x["pred"], output_dict=True)

		output_html += "<br/>"
		output_html += "<br/>"
		output_html += "<br/>"
		output_html += "<br/>"
		output_html += "<h3>" +  " ".join(x["token"]) + "</h3>"
		output_html += generate_question_table(x["token"], x["truth"], x["pred"])
		output_html += pd.DataFrame(report).transpose().to_html()

	output_html += "</body></html>"
	f = open(output_fn, "w")
	f.write(output_html)
	f.close()
	print("ready, written to", output_fn)





if __name__ == "__main__":
	print(sys.argv)
	main(sys.argv[1], sys.argv[2], sys.argv[3])



