
from gensim.models import Word2Vec

print("Creating/Loading word2vec model")
mscoco_q_path = "datasets\\qa\\cocoqa\\train\\questions.txt"
q_file = open(mscoco_q_path, 'r')
sentences = []
lines = q_file.readlines()
for line in lines:
    #print(line.strip())
    sentences.append(line.strip().split(' '))
q_file.close()
#print(sentences)
print("VQA Answers .. building word2vec model")
mscoco_a_path = "datasets\\qa\\cocoqa\\train\\answers.txt"
a_file = open(mscoco_a_path, 'r')
lines = a_file.readlines()
for line in lines:
    sentences.append([line.strip()])
a_file.close()

print("Training model")
model = Word2Vec(sentences, min_count=1)
print("Saving model ")
model.save("w2v_mscocovqa_2.model")
print("Model is done, saved in \'w2v_mscocovqa.model\' file")

