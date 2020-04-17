
from gensim.models import Word2Vec

q_path = "W:\\Github_rep\\VQA-master\\PythonHelperTools\\Q_words_test.txt"
a_path = "W:\\Github_rep\\VQA-master\\PythonHelperTools\\A_words_test.txt"

def create_features():
    print("VQA Questions .. building word2vec feature vectors")
    q_file = open(q_path, 'r')
    model = Word2Vec.load("questions_vqa.model")
    lines = q_file.readlines()
    for line in lines:
        words = line.split(':')[2]
        words = words.split(" ")
        word1_vector = model.wv[words[0]]
        word2_vector = model.wv[words[1]]
        # TODO: save them somehwere? Or keep them in a list and pass them
    q_file.close()

    print("VQA Answers .. building word2vec vector features")
    a_file = open(a_path, 'r')
    model = Word2Vec.load("answers_vqa.model")
    lines = a_file.readlines()
    # TODO: each 4 answers for 1 questions are saved in a seperate line
    for line in lines:
        word = line.split(':')[3]
        word1_vector = model.wv[word]
        # TODO: save somehwere? Or keep them in a list and pass them
    a_file.close()

def train_word2vec_models():

    print("VQA Questions .. building word2vec model")
    q_file = open(q_path, 'r')
    sentences = []
    lines = q_file.readlines()
    for line in lines:
        sentences.append([line.split(':')[1]])
    q_file.close()
    print("Training model")
    model = Word2Vec(sentences, min_count=1)
    print("Saving model ")
    model.save("questions_vqa.model")
    print("Model for question is done, it is saved in \'questions_vqa.model\' file")

    print("VQA Answers .. building word2vec model")
    a_file = open(a_path, 'r')
    sentences = []
    lines = a_file.readlines()
    for line in lines:
        sentences.append([line.split(':')[2]])
    a_file.close()
    print("Training model")
    model = Word2Vec(sentences, min_count=1)
    print("Saving model ")
    model.save("answers_vqa.model")
    print("Model for answers is done, it is saved in \'answers_vqa.model\' file")

