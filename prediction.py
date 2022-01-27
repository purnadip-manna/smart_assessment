import numpy as np 
import re 
import transformers  
from keras.models import load_model 
from BertDataGenerator import *

tokenizer=transformers.BertTokenizer.from_pretrained("./static/model/bert-base-uncased/", do_lower_case=True)
max_length=128
bert_model=transformers.TFBertModel.from_pretrained("./static/model/bert-base-uncased/")
bert_model.trainable=True
label_cols = ["contradiction", "entailment", "neutral"]
model=load_model("./static/model/testmodel/testmodel0.h5")

def filter_comment(comment): 
    comment=re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", comment)
    return comment


def predict_sense(standard_answer,students_answer,api_mode=True):
    if len(standard_answer)!=0 and len(students_answer)!=0:
        cl_sn_ans=filter_comment(students_answer)
        cl_st_ans=filter_comment(standard_answer)
        sentence_pairs = np.array([[str(cl_sn_ans), str(cl_st_ans)]])
        test_data = BertSemanticDataGenerator(
            sentence_pairs, labels=None, batch_size=1, shuffle=False, include_targets=False,
        )
        result = model.predict(test_data)
        predictions=list(result[0])
    else: 
        predictions=[0,0,0]
    if api_mode:
        json={}
        for i in range(len(label_cols)):
            json[label_cols[i]]=float(predictions[i])
        return json   


 