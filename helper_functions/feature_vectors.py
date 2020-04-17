
from gensim.models import Word2Vec

model = Word2Vec.load("./models/w2v_mscocovqa_2.model")

def question_vector(question_str):
    return model.wv[question_str]

def answer_vector(answer_str):
    return model.wv[answer_str]

# This is a driver test
if __name__ == '__main__':
    question = "what is using umbrellas as a central theme"
    answer = "sculpture"
    print("Question: ", question)
    print(question_vector(question))
    print("Answer: ", answer)
    print(answer_vector(answer))
