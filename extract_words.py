# coding: utf-8

from PythonHelperTools.vqaTools.vqa import VQA
import spacy


def extract_word(text, type=""):
    doc = nlp(text.replace(type, ""))
    nouns = [chunk.text for chunk in doc.noun_chunks]
    verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
    word2 = ""
    if (len(nouns) > 0):
        word2 = nouns[0].split(' ')[-1]
    elif (len(verbs) > 0):
        word2 = verbs[0].split(' ')[-1]
    elif len(text) > 0:
        word2 = text[0]
    return word2


# dataDir		='../../VQA'
dataDir		='W:\\Spring2020\\CV\\project\\datasets\\VQA 2.0\\'
versionType ='v2_' # this should be '' when using VQA v2.0 dataset
taskType    ='OpenEnded' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
dataType    ='mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0.
dataSubType ='train2014'
annFile     ='%s/Annotations/%s%s_%s_annotations.json'%(dataDir, versionType, dataType, dataSubType)
quesFile    ='%s/Questions/%s%s_%s_%s_questions.json'%(dataDir, versionType, taskType, dataType, dataSubType)
imgDir 		= '%s/Images/%s/%s/' %(dataDir, dataType, dataSubType)

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")

# initialize VQA api for QA annotations
vqa=VQA(annFile, quesFile)

# load and display QA annotations for given question types
"""
quesTypes for abstract and mscoco has been provided in  ../QuestionTypes/ folder.
"""
file_qs = open("Q_words.txt", 'w')
file_ans = open("A_words.txt", 'w')
qtype_file = open("W:\\Github_rep\\VQA-master\\QuestionTypes\\mscoco_question_types.txt")
types = qtype_file.readlines()
for type in types:
    print(type.strip())
    annIds = vqa.getQuesIds(quesTypes=type.strip())
    anns = vqa.loadQA(annIds)
    for ann in anns:
        quesId = ann['question_id']
        question = (vqa.qqa[quesId]['question']).lower()
        word2 = extract_word(question, type)
        interesting_words = type.replace(" ", "") + ' ' + word2
        line = str(quesId) + ":" + question + ":" + interesting_words + '\n'
        #print(line)
        file_qs.write(line)

        for ans in ann['answers'][:4]:
            answer = ans['answer'].lower()
            word = extract_word(answer)
            line = str(quesId) + ":" + str(ans['answer_id']) + ":" + answer + ":" + word + '\n'
            #print(line)
            file_ans.write(line)

file_qs.close()
file_ans.close()
print("Format of Questions => ID:Question:Interesting_words")
print("Format of Answers   => QID:AID:Answer:Interesting_word")


'''
interesting_words = type.replace(" ", "") + ' '.join(nouns[:3])
interesting_words += ' '.join(verbs[:4-len(interesting_words.split(' '))])
#print("nouns: ", nouns)
#print("length of words: ", len(interesting_words.split(' ')))
#print("nouns added: ", len(nouns[:3]))
#print("remainder: ", 4-len(interesting_words.split(' ')))
#print("verbs added: ", ' '.join(verbs[:4-len(interesting_words.split(' '))]))
'''