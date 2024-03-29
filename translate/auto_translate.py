from googletrans import Translator

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


def translate(sentences,part):
    """Translate the data from English to Dutch"""
    # init the Google API translator
    translator = Translator()

    translated_sentences = []
    failed_translations = [part]

    # translate the given list
    for line in sentences:
        try:
            dutch_translation = translator.translate(line, dest="nl")
            translated_sentences.append(dutch_translation.text)
        except:
            failed_translations.append(line)

    print("Translation ({}/4) Finished...".format(part))

    return translated_sentences, failed_translations


def create_new_data_file(translated_sentences, new_filename,part):
    """create a new file to use for dutch style transfer"""
    with open(new_filename, 'w') as f:
        for line in translated_sentences:
            line = line + "\n"
            f.write(line)

    print("Writing ({}/5) Finished...".format(part))
    f.close()


def main():

    # gather all English data; for macbook use path "../data/em/train.0"
    print("Gathering Data...")
    train_src_en = get_lines_from_file("/data/s3238903/dutch-style-transfer/data/em/train.0")
    train_tgt_en = get_lines_from_file("/data/s3238903/dutch-style-transfer/data/em/train.1")
    valid_src_en = get_lines_from_file("/data/s3238903/dutch-style-transfer/data/em/valid.0")
    valid_tgt_en = get_lines_from_file("/data/s3238903/dutch-style-transfer/data/em/valid.1")

    # translate all English sentences
    print("Start Translation...")
    train_src_nl,failed_train_src = translate(train_src_en,"1")
    train_tgt_nl,failed_train_tgt = translate(train_tgt_en,"2")
    valid_src_nl,failed_valid_src = translate(valid_src_en,"3")
    valid_tgt_nl,failed_valid_tgt = translate(valid_tgt_en,"4")

    all_failed = failed_train_src + failed_train_tgt + failed_valid_src + failed_valid_tgt

    print("Writing to new file...")
    #create_new_data_file(valid_src_nl,"./data/em-dutch/valid.0")
    create_new_data_file(train_src_nl,"/data/s3238903/dutch-style-transfer/data/em-dutch/train.0","1")
    create_new_data_file(train_tgt_nl,"/data/s3238903/dutch-style-transfer/data/em-dutch/train.1","2")
    create_new_data_file(valid_src_nl,"/data/s3238903/dutch-style-transfer/data/em-dutch/valid.0","3")
    create_new_data_file(valid_tgt_nl,"/data/s3238903/dutch-style-transfer/data/em-dutch/valid.1","4")

    create_new_data_file(all_failed, "/data/s3238903/dutch-style-transfer/data/em-dutch/failed", "5")

    print("Script Completed")

if __name__ == '__main__':
    main()