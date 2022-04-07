from googletrans import Translator, constants
from pprint import pprint

def get_lines_from_file(filename):
    """Read datafiles and return lists with sentences"""

    # create lists to store sentences
    english_sentences = []

    # open files and store lines
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            english_sentences.append(line)

    f.close()

    return english_sentences


def translate(sentences):
    """Translate the data from English to Dutch"""
    # init the Google API translator
    translator = Translator()

    translated_sentences = []

    # translate the given list
    for line in sentences:
        dutch_translation = translator.translate(line, dest="nl")
        translated_sentences.append(dutch_translation.text)
        print("translation done")

    return translated_sentences

def create_new_data_file(translated_sentences, new_filename):
    """create a new file to use for dutch style transfer"""
    with open(new_filename, 'w') as f:
        for line in translated_sentences:
            line = line + "\n"
            f.write(line)
    f.close()

def main():

    # gather all English data; for macbook use path "../data/em/train.0"
    #train_src_en = get_lines_from_file("/data/s3238903/style_transfer/data/em/train.0")
    #train_tgt_en = get_lines_from_file("/data/s3238903/style_transfer/data/em/train.1")
    ##valid_src_en = get_lines_from_file("./data/em/valid.0")
    valid_src_en = get_lines_from_file("/data/s3238903/style_transfer/data/em/valid.0")
    #valid_tgt_en = get_lines_from_file("/data/s3238903/style_transfer/data/em/valid.1")

    # translate all English sentences
    #train_src_nl = translate(train_src_en)
    #train_tgt_nl = translate(train_tgt_en)
    valid_src_nl = translate(valid_src_en)
    #valid_tgt_nl = translate(valid_tgt_en)

    ##create_new_data_file(valid_src_nl,"./data/em-dutch/valid.0")
    create_new_data_file(valid_src_nl,"/data/s3238903/style_transfer/data/em-dutch/valid.0")


if __name__ == '__main__':
    main()