'''
You can remove the import and the stop_words definition from the function
and put it in a global "seen" place if possible, it will make the execution
a bit faster (instead of importing and loading the set every time)
'''

def extract_four_words_from_question(question):
    from nltk.corpus import stopwords
    stopwords = set(stopwords.words('english'))
    filtered_words = [word for word in question if word not in stopwords]
    filtered_words.extend('' * 4-len(filtered_words)) # it will not add any elements if value < 0
    return filtered_words[:4]