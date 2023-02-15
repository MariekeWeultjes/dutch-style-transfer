from googletrans import Translator

def main():
    """Translate the data from English to Dutch"""
    # init the Google API translator
    translator = Translator()

    failed_translations = ["D.R.I , DEAD KENNEDY'S , CIRCLE JERKS", 
    					   "'Cause he broke up with Jennifer Garner :P", 
    					   "Yea that was like the 1st commercially successfull rap song.",
    					   "I have no idea, but I like that song.",
    					   "I like Tom and Katie as a couple, but I don't  think they'll last forever.",
    					   "Please advise, otherwise I will keep looking."]

    # translate the whole list and print translations
    for i in failed_translations:
        dutch_translation = translator.translate(i, dest="nl")
        print(dutch_translation.text)

if __name__ == '__main__':
    main()
