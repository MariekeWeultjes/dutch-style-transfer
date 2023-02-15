# Author: Marieke Weultjes
# This is some code to scrape the html_output_test data for my master thesis
# Used simple YouTube tutorial from Tinkernut

# used libraries
from bs4 import BeautifulSoup
import nltk
import requests
import pandas as pd
import re
import emoji
import json
import time

# When first time running the script:
#nltk.download('punkt')

# safe useful information to this data dictionary
data_stats = {}
topic_dict = {}
data_dict = {}
#all_comments = []

def get_comments(topic_link, topic_comments):
    """retrieve all comments from a given webpage"""
    print("Retrieving data from: " + topic_link)
    # try statement?
    topic_page = requests.get(topic_link)
    soup = BeautifulSoup(topic_page.text, 'html.parser')
    comment_elements = soup.find_all("div", attrs={"class": "post"})

    if len(comment_elements) == 0:
        time.sleep(10)
    else:
        for c in comment_elements:
            #all_comments.append(c.text)
            topic_comments.append(c.text)

    return topic_comments

def clean_up(data):
    """remove useless and noisy data from the comments"""

    new_data = []

    for d in data:
        # replace the images, URL's, mentions and emojis
        d = re.sub(r'.* \d+.?\d? KB', 'IMAGE', d, flags=re.IGNORECASE)
        d = re.sub(r'http\S+', 'URL', d)
        d = re.sub('@[A-Za-z0-9]+', 'MENTION', d, flags=re.IGNORECASE)
        d = emoji.demojize(d)
        d = re.sub(':[a-z_]*:', '', d)
        # add to new list of data
        new_data.append(d)

    return new_data

def tokenize_data(comments):
    """Tokenize the comments to get seperated sentences"""
    data = []
    for comment in comments:
        sentences = nltk.sent_tokenize(comment.strip(), "dutch")
        for sen in sentences: # split a second time on newlines
            if sen.__contains__("\n"):
                more_sentences = sen.split("\n")
                data = data + more_sentences
            else:
                data.append(sen)

    return data

def preprocessing(data):
    """This function will strip the data of useless instances"""
    new_data = []
    less = []
    more = []
    mentions = []
    urls = []
    # remove all instances shorter than 5, longer than 25 or contains multiple lines.
    for item in data:
        split_sentence = nltk.word_tokenize(item, "dutch")
        all_words = [word for word in split_sentence if word.isalnum()]
        if len(all_words) < 5:
            less.append(item)
        elif len(all_words) > 25:
            more.append(item)
        elif item.__contains__("MENTION"):
            mentions.append(item)
        elif item.__contains__("URL"):
            urls.append(item)
        else:
            new_data.append(item)


    return new_data, less, more, mentions, urls

def main():
    """This is a description of the function"""
    category_page = requests.get("https://www.forumfeminarum.nl/c/entertainment/8")
    soup = BeautifulSoup(category_page.text, 'html.parser')

    # find all 30 different topic-links
    topic_hrefs = soup.find_all('a', attrs={'class': 'title raw-link raw-topic-link'})

    topic_links = [t['href'] for t in topic_hrefs]
    #topic_links = ["https://www.forumfeminarum.nl/t/mommybloggerstopic-9/31513", "https://www.forumfeminarum.nl/t/where-yo-memes-at-2/22044"]
    # meme_topic = "https://www.forumfeminarum.nl/t/where-yo-memes-at-2/22044" # len = 57

    for tl in topic_links:
        comments = []
        new_dict = {}
        url_info = tl[32:] # save info for data_stats
        ti = topic_links.index(tl)
        topic_dict.update({tl: ti}) # save topic index in dictionary
        for i in range(0, 5000, 26):
            comments = get_comments(tl + "/{}".format(i), comments)
        # remove duplicates by transforming the list to a set.
        # this is necessary, since comments can be retrieved multiple times due to the scraping method.
        unique_comments = list(set(comments))
        clean_comments = clean_up(unique_comments)
        separate_sentences = tokenize_data(clean_comments)
        unique_sentences = list(set(separate_sentences))
        data, less, more, mentions, urls = preprocessing(unique_sentences)

        # store usable data in dict with the topic indicator
        for d in data:
            new_dict.update({d: ti})
        data_dict.update(new_dict)

        # store topic data stats
        data_stats[url_info + "_unique_comments"] = len(unique_comments)
        data_stats[url_info + "_unique_sentences"] = len(unique_sentences)
        data_stats[url_info + "_usable_sentences"] = len(data)
        data_stats[url_info + "_less"] = len(less)
        data_stats[url_info + "_more"] = len(more)
        data_stats[url_info + "_mentions"] = len(mentions)
        data_stats[url_info + "_urls"] = len(urls)

    # HANDLE ALL COMMENTS: remove duplicates by transforming the list to a set
    #all_unique_comments = list(set(all_comments))
    #all_clean_comments = clean_up(all_unique_comments)

    # HAC: tokenize the 'clean' comments
    #all_separate_sentences = tokenize_data(all_clean_comments)
    #all_unique_sentences = list(set(all_separate_sentences))
    #all_data, l, m, n = preprocessing(all_unique_sentences)

    # store some useful data information
    #data_stats["total_unique_comments"] = len(all_unique_comments)
    #data_stats["total_unique_sentences"] = len(all_unique_sentences)
    #data_stats["total_usable_sentences"] = len(all_data)
    #data_stats["removed_short_sentences"] = len(l)
    #data_stats["removed_long_sentences"] = len(m)
    #data_stats["removed_newline_comments"] = len(n)

    text_data = list(data_dict.keys())
    topic_index = list(data_dict.values())
    for i in range(len(set(topic_index))):
        data_stats["final_sen_topic_{}".format(i)] = topic_index.count(i)

    data_stats["final_usable_sentences"] = len(text_data)

    # store in usable file-format
    df_dict = {"Comments": text_data, "Index": topic_index}
    df = pd.DataFrame(df_dict)
    df.to_csv("final_all_data_with_index.csv", index=False)

    pretty_data_stats = json.dumps(data_stats, indent=4)
    print(pretty_data_stats)
    print(topic_dict)

if __name__ == '__main__':
    main()
