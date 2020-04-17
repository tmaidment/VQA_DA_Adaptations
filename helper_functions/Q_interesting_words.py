
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))


def extract_four_words_from_question(question):
    """
    Returns 4 words from question after removing stop-words
    :param question:
    :return: list of 4 words (strings)
    """
    filtered_words = [word for word in question.split(' ') if word not in stopwords]
    temp = ' ' * (4-len(filtered_words))
    filtered_words.extend(temp) # it will not add any elements if value < 0
    return filtered_words[:4]

