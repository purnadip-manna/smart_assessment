import numpy as np
from keras.models import load_model
from BertSemanticDataGenerator import *
import csv
import pandas as pd

# Labels in our dataset.
labels = ["contradiction", "entailment", "neutral"]

model=load_model('./models/testmodel1.h5')

'''
op = 1
while(op == 1):
    sentence1 = input("Sentence1: ")
    sentence2 = input("Sentence2: ")
    sentence_pairs = np.array([[str(sentence1), str(sentence2)]])
    test_data = BertSemanticDataGenerator(
        sentence_pairs, labels=None, batch_size=1, shuffle=False, include_targets=False,
    )
    result = model.predict(test_data)
    print(result)
    index = np.argmax(result, axis=None)
    print("\n")
    l = labels[index]
    percentage = str(result[0][index])
    print(l + ": " + percentage + "%")
    # ["contradiction", "entailment", "neutral"]

    with open('test.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([sentence1, sentence2, l, percentage, str(result)])

    op = int(input("Enter 1 to continue: "))
'''
df = pd.read_csv('data.csv')
length = len(df.index)
for i in range(length):
    question = df.loc[i, "questions"]
    sentence1 = df.loc[i, "sentence1"]
    sentence2 = df.loc[i, "sentence2"]
    label = df.loc[i, "label"]

    sentence_pairs = np.array([[str(sentence1), str(sentence2)]])
    test_data = BertSemanticDataGenerator(
        sentence_pairs, labels=None, batch_size=1, shuffle=False, include_targets=False,
    )
    result = model.predict(test_data)
    index = np.argmax(result, axis=None)
    pred_label = labels[index]
    percentage = str(result[0][index])
    sucess = 0
    if(pred_label.strip() == label.strip()):
        sucess = 1
        
    with open('result.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([question, sentence1, sentence2, label, pred_label, percentage, sucess, str(result)])
