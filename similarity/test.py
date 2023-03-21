
import torch
from comet import download_model, load_from_checkpoint

model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(model_path)


def main():
	# set sample in variables
	informal_source = "haha een Brabander in hart en nieren… gewoon een manier van praten"
	ref1 = "Een Brabander in hart en nieren, dat is een manier van praten."	
	ref2 = "Mensen uit Brabant hebben een bepaalde manier van praten."
	ref3 = "Het is een Brabander in hart en nieren. Die hebben een eigen manier van praten."
	#output_vanilla = "I'm a Brabander in hart and nieren. It's just a way of talking."
	#output_all_vanilla = "Een Brabander in hart en nieren. Het is gewoon een manier van praten."
	# testing different inputs
	inputs = [{"src": "", "mt": ref1, "ref": informal_source}]
	comet_score = comet_model.predict(inputs, batch_size=8, gpus=1)
	print(f"test 1: empty-ref1-informal: {comet_score['system_score']}")
	# score: 0.8563
	inputs = [{"src": "", "mt": ref2, "ref": informal_source}]
	comet_score = comet_model.predict(inputs, batch_size=8, gpus=1)
	print(f"test 2: empty-ref2-informal: {comet_score['system_score']}")
	# score: 0.6698
	inputs = [{"src": "", "mt": ref3, "ref": informal_source}]
	comet_score = comet_model.predict(inputs, batch_size=8, gpus=1)
	print(f"test 3: empty-ref3-informal: {comet_score['system_score']}")
	# score: 0.8167
	inputs = [{"src": "", "mt": ref2, "ref": ref1}]
	comet_score = comet_model.predict(inputs, batch_size=8, gpus=1)
	print(f"test 4: empty-ref2-ref1: {comet_score['system_score']}")
	# score: 0.6256
	inputs = [{"src": "", "mt": ref3, "ref": ref1}]
	comet_score = comet_model.predict(inputs, batch_size=8, gpus=1)
	print(f"test 5: empty-ref3-ref1: {comet_score['system_score']}")
	# score: 0.8429
	inputs = [{"src": "", "mt": ref3, "ref": ref2}]
	comet_score = comet_model.predict(inputs, batch_size=8, gpus=1)
	print(f"test 6: empty-ref3-ref2: {comet_score['system_score']}")
	# score: 0.6404

if __name__ == '__main__':
    main()
