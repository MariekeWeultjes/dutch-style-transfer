# this script is written by Marieke Weultjes
# it is used to evaluate the model output(s) for my master thesis

import pandas as pd
import csv
import sys
import nltk
import torch
import gc
import numpy as np
from comet import download_model, load_from_checkpoint
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

#nltk.download('punkt')
smooth = SmoothingFunction()
model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(model_path)

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

	bleu_scores = []
	comet_scores_all = []
	comet_scores_means = []

	# freaking memory fucked up
	torch.cuda.empty_cache()
	del df_a0, df_a1, df_a2, remove_dups, df_test, df
	gc.collect()

	# getting BLEU and COMET scores
	for i in range(100):
		# calculate BLEU
		bleu_scores.append(round(sentence_bleu(reference_tok[i], output[i], smoothing_function=smooth.method1), 4))
		# calculate COMET (mean of references)
		inputs = [{"src": source[i], "mt": output[i], "ref": a0_sen[i]}]
		comet_score1 = comet_model.predict(inputs, batch_size=8, gpus=1)
		inputs = [{"src": source[i], "mt": output[i], "ref": a1_sen[i]}]
		comet_score2 = comet_model.predict(inputs, batch_size=8, gpus=1)
		inputs = [{"src": source[i], "mt": output[i], "ref": a2_sen[i]}]
		comet_score3 = comet_model.predict(inputs, batch_size=8, gpus=1)
		comet_score_mean = np.mean([comet_score1['system_score'], comet_score2['system_score'], comet_score3['system_score']])
		# calculate COMET (all references)
		inputs = [{"src": source[i], "mt": output[i], "ref": reference[i]}]
		comet_score_all = comet_model.predict(inputs, batch_size=8, gpus=1)
		# print scores
		# print(f"comet score all refs: {comet_score_all['system_score']}\ncomet score mean {comet_score_mean['system_score']}")
		# add scores to lists
		comet_scores_all.append(comet_score_all['system_score'])
		comet_scores_means.append(comet_score_mean)

	print(f"Evaluation scores {sys.argv[1]}:\n sentence BLEU score = {np.mean(bleu_scores)}\n COMET score (means) = {np.mean(comet_scores_means)}\n COMET score (all refs) = {np.mean(comet_scores_all)}")



if __name__ == '__main__':
    main()
