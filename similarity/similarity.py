# This code is written by Marieke Weultjes
# This code calculates the similarity scores between formal and informal sentence, but also between informal sentences from different annotators.

import pandas as pd
import csv
import nltk
import torch
from comet import download_model, load_from_checkpoint
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
#import sacrebleu

#nltk.download('punkt')
smooth = SmoothingFunction()
model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(model_path)

def main():

	# load dataframe for only the topic with multiples rewrites
	df = pd.read_csv("/data/s3238903/dutch-style-transfer/data/DuFo/dutch_validate_test.csv", header=0)
	df_t0 = df[df['Topic'] == 'ff-boulevard-8']
	df_a0 = df_t0[df_t0['Annotator'] == 'Marieke']
	df_a1 = df_t0[df_t0['Annotator'] == 'Iris']
	df_a2 = df_t0[df_t0['Annotator'] == 'Elvira']

	# get lists of sentences
	informal_sen = df_a0['Informal'].values.tolist()
	formal_a0_sen = df_a0['Formal'].values.tolist()
	formal_a1_sen = df_a1['Formal'].values.tolist()
	formal_a2_sen = df_a2['Formal'].values.tolist()

	# FOR THE CORPUS BLEU, MIGHT BE REMOVED
	#informal = []
	#formal_a0 = []
	#formal_a1 = []
	#formal_a2 = []

	# create lists to store scores
	inf_a0_scores, inf_a1_scores, inf_a2_scores, a0_a1_scores, a0_a2_scores, a1_a2, inf_for = [], [], [], [], [], [], []

	# create list with the instances and their scores to get some more detailed insights
	all_scores, all_scores_all_references = [], []

	print("Calculating BLEU and COMET Scores...")
	# loop through all instances
	for i in range(100):
		# tokenize sentences
		t_inf = word_tokenize(informal_sen[i], language='dutch')
		t_a0 = word_tokenize(formal_a0_sen[i], language='dutch')
		t_a1 = word_tokenize(formal_a1_sen[i], language='dutch')
		t_a2 = word_tokenize(formal_a2_sen[i], language='dutch')
		t_all_references = [t_a0, t_a1, t_a2]
		# add tokenized sentence to list for corpus bleu, MIGHT BE REMOVED
		#informal.append(t_inf)
		#formal_a0.append(t_a0)
		#formal_a1.append(t_a1)
		#formal_a2.append(t_a2)
		# calculcate sentence blue scores
		inf_a0 = round(sentence_bleu([t_inf], t_a0, smoothing_function=smooth.method1), 4)
		inf_a1 = round(sentence_bleu([t_inf], t_a1, smoothing_function=smooth.method1), 4)
		inf_a2 = round(sentence_bleu([t_inf], t_a2, smoothing_function=smooth.method1), 4)
		a0_a1 = round(sentence_bleu([t_a0], t_a1, smoothing_function=smooth.method1), 4)
		a0_a2 = round(sentence_bleu([t_a0], t_a2, smoothing_function=smooth.method1), 4)
		a1_a2 = round(sentence_bleu([t_a1], t_a2, smoothing_function=smooth.method1), 4)
		inf_for = round(sentence_bleu(t_all_references, t_inf, smoothing_function=smooth.method1), 4)
		# add to their respective score variables
		inf_a0_scores.append(inf_a0)
		inf_a1_scores.append(inf_a1)
		inf_a2_scores.append(inf_a2)
		a0_a1_scores.append(a0_a1)
		a0_a2_scores.append(a0_a2)
		a1_a2_scores.append(a1_a2)
		inf_for_scores.append(inf_for)
		# put the sentence_bleu scores in the score list as a list with the respective tokenized sentences
		all_scores.append([t_inf, t_a0, inf_a0])
		all_scores.append([t_inf, t_a1, inf_a1])
		all_scores.append([t_inf, t_a2, inf_a2])
		all_scores.append([t_a0, t_a1, a0_a1])
		all_scores.append([t_a0, t_a2, a0_a2])
		all_scores.append([t_a1, t_a2, a1_a2])
		all_scores_all_references.append([t_a0, t_a1, t_a2, t_inf, inf_for])
		# calculate comet scores
        inputs = [{"src": "", "mt": t_a0[i], "ref": informal_sen[i]}]
        comet_score = comet_model.predict(inputs, batch_size=8, gpus=1)
        inputs = [{"src": "", "mt": t_a1[i], "ref": informal_sen[i]}]
        comet_score = comet_model.predict(inputs, batch_size=8, gpus=1)
        inputs = [{"src": "", "mt": t_a2[i], "ref": informal_sen[i]}]
        comet_score = comet_model.predict(inputs, batch_size=8, gpus=1)

	print("Starting Corpus Bleu Scores...")

	# calculate corpus bleu scores, MIGHT BE REMOVED
	#print("corpus BLEU score inf-a0 = ", round(corpus_bleu(informal, formal_a0), 4))
	#print("corpus BLEU score inf-a1 = ", round(corpus_bleu(informal, formal_a1), 4))
	#print("corpus BLEU score inf-a2 = ", round(corpus_bleu(informal, formal_a2), 4))
	#print("corpus BLEU score a0-a1 = ", round(corpus_bleu(formal_a0, formal_a1), 4))
	#print("corpus BLEU score a0-a2 = ", round(corpus_bleu(formal_a0, formal_a2), 4))
	#print("corpus BLEU score a1-a2 = ", round(corpus_bleu(formal_a1, formal_a2), 4))

	# calculate average BLEU of individual sentence scores
	print("sentence BLEU score inf-a0 = ", np.mean(inf_a0_scores))
	print("sentence BLEU score inf-a1 = ", np.mean(inf_a1_scores))
	print("sentence BLEU score inf-a2 = ", np.mean(inf_a2_scores))
	print("sentence BLEU score a0-a1 = ", np.mean(a0_a1_scores))
	print("sentence BLEU score a0-a2 = ", np.mean(a0_a2_scores))
	print("sentence BLEU score a1-a2 = ", np.mean(a1_a2_scores))
	print("sentence BLEU score inf_for = ", np.mean(inf_for_scores))



	# put sentences and their sentence_bleu score in a file for futher investigation
	with open('similarty_scores.csv', 'w', newline='') as f:
		writer = csv.writer(f)
		# create header and write rest of data
		writer.writerow(["sentence 1", "sentence 2", "score"])
		writer.writerows(all_scores)

	with open('bleu_scores_all_references.csv', 'w', newline='') as f:
		writer = csv.writer(f)
		# create header and write rest of data
		writer.writerow(["references", "source", "score"])
		writer.writerows(all_scores_all_references)

if __name__ == '__main__':
    main()
