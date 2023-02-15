# This code is written by Marieke Weultjes
# This code calculates the similarity scores between formal and informal sentence, but also between informal sentences from different annotators.

import pandas as pd
#from nltk.translate import sentence_bleu, corpus_bleu
#from nltk.tokenize import word_tokenize

def main():

	# load dataframe for only the topic with multiples rewrites
	df = pd.read_csv("data/test_overview.tsv", sep="\t", header=0)
	df_t0 = df[df['Topic'] == 'ff-boulevard-8']
	df_a0 = df_t0[df_t0['Annotator'] == 'Marieke']
	df_a1 = df_t0[df_t0['Annotator'] == 'Iris']
	df_a2 = df_t0[df_t0['Annotator'] == 'Elvira']

	# get lists of sentences
	informal_sen = df_a0['Informal'].values.tolist()
	formal_a0_sen = df_a0['Formal'].values.tolist()
	formal_a1_sen = df_a1['Formal'].values.tolist()
	formal_a2_sen = df_a2['Formal'].values.tolist()

	informal = []
	formal_a0 = []
	formal_a1 = []
	formal_a2 = []

	# create variable to calculate average sentence bleu scores
	inf_a0 = 0
	inf_a1 = 0
	inf_a2 = 0
	a0_a1 = 0
	a0_a2 = 0
	a1_a2 = 0

	# tokenize all sentences and append to new lists
	for i in range(100):
		t_inf = word_tokenize(informal_sen[i], language='dutch')
		t_a0 = word_tokenize(formal_a0_sen[i], language='dutch')
		t_a1 = word_tokenize(formal_a1_sen[i], language='dutch')
		t_a2 = word_tokenize(formal_a2_sen[i], language='dutch')
		informal.append(t_inf)
		formal_a0.append(t_a0)
		formal_a1.append(t_a1)
		formal_a2.append(t_a2)
		# calculcate sentence blue scores and add to their respective score variables
		inf_a0 = inf_a0 + sentence_blue(t_inf, t_a0)
		inf_a1 = inf_a1 + sentence_blue(t_inf, t_a1)
		inf_a2 = inf_a2 + sentence_blue(t_inf, t_a2)
		a0_a1 = a0_a1 + sentence_blue(t_a0, t_a1)
		a0_a2 = a0_a2 + sentence_blue(t_a0, t_a2)
		a1_a2 = a1_a2 + sentence_blue(t_a1, t_a2)

	# calculate corpus bleu scores
	print("corpus BLUE score inf-a0 = ", corpus_bleu(informal, formal_a0))
	print("corpus BLUE score inf-a1 = ", corpus_bleu(informal, formal_a1))
	print("corpus BLUE score inf-a2 = ", corpus_bleu(informal, formal_a2))
	print("corpus BLUE score a0-a1 = ", corpus_bleu(formal_a0, formal_a1))
	print("corpus BLUE score a0-a2 = ", corpus_bleu(formal_a0, formal_a2))
	print("corpus BLUE score a1-a2 = ", corpus_bleu(formal_a1, formal_a2))

	# calculate average BLEU of individual sentence scores
	print("sentence BLUE score inf-a0 = ", inf_a0 / 100)
	print("sentence BLUE score inf-a1 = ", inf_a1 / 100)
	print("sentence BLUE score inf-a2 = ", inf_a2 / 100)
	print("sentence BLUE score a0-a1 = ", a0_a1 / 100)
	print("sentence BLUE score a0-a2 = ", a0_a2 / 100)
	print("sentence BLUE score a1-a2 = ", a1_a2 / 100)

if __name__ == '__main__':
    main()


