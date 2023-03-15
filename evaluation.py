# this script is written by Marieke Weultjes
# it is used to evaluate the model output(s) for my master thesis

import pandas as pd
import csv
import sys
import nltk
from comet import download_model, load_from_checkpoint
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

#nltk.download('punkt')
smooth = SmoothingFunction()
comet_model = download_model("wmt21-comet-da")
load_comet = load_from_checkpoint(comet_model)

def main():

	# store model output in list
	with open(sys.argv[1], 'r') as f:
		output = [word_tokenize(line.strip(), language='dutch') for line in f.readlines()]

	# open dataset csv 
	df = pd.read_csv("/data/s3238903/dutch-style-transfer/data/DuFo/dutch_validate_test.csv", header=0)
	df_test = df[df['Topic'] == 'ff-boulevard-8']

	# store source data in list
	remove_dups = df_test.head(100)
	source = remove_dups['Informal'].values.tolist()
	#source_tok = [word_tokenize(item, language='dutch') for item in source_list] # tokenization not needed for COMET

	# store the reference data in list
	df_a0 = df_test[df_test['Annotator'] == 'Marieke']
	df_a1 = df_test[df_test['Annotator'] == 'Iris']
	df_a2 = df_test[df_test['Annotator'] == 'Elvira']
	a0_sen = df_a0['Formal'].values.tolist()
	a1_sen = df_a1['Formal'].values.tolist()
	a2_sen = df_a2['Formal'].values.tolist()

	reference = [[a0_sen[i], a1_sen[i], a2_sen[i]] for i in range(100)]
	reference_tok = [[word_tokenize(a0_sen[i], language='dutch'), word_tokenize(a1_sen[i], language='dutch'), word_tokenize(a2_sen[i], language='dutch')] for i in range(100)]

	# test tokenization: everything looks great
	# print(output, "\n", source, "\n", reference)

	bleu_score_sum = 0
	comet_score_sum = 0

	# getting BLEU and COMET scores
	for i in range(100):
		model_bleu = round(sentence_bleu(reference_tok[i], output[i], smoothing_function=smooth.method1), 4)
		bleu_score_sum = bleu_score_sum + model_bleu

		inputs = [{"src": source[i], "mt": output[i], "ref": reference[i]}]
		comet_score_raw = load_comet.predict(inputs, gpu=0)
		#comet_score = round(load_comet.predict(inputs, show_progress=False)[-1][0], 4)

		print(comet_score_raw)
		#print(comet_score)


	print("sentence BLEU score {} = {}".format(sys.argv[1], bleu_score_sum / 100))


if __name__ == '__main__':
    main()
