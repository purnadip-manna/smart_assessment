import tensorflow as tf
import numpy as np
import transformers
from keras.models import load_model

bert_model = transformers.TFBertModel.from_pretrained("./static/model/bert-base-uncased/") 
max_length = 128  # Maximum length of input sentence to the model.
batch_size = 32

class BertSemanticDataGenerator(tf.keras.utils.Sequence): 
    def __init__(
        self,
        sentence_pairs,
        labels,
        batch_size=batch_size,
        shuffle=True,
        include_targets=True,
    ):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.include_targets = include_targets
        
        # Load our BERT Tokenizer to encode the text.
        # We will use base-base-uncased pretrained model.
        
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "./static/model/bert-base-uncased/", do_lower_case=True
        )
        self.indexes = np.arange(len(self.sentence_pairs))
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch.
        return len(self.sentence_pairs) // self.batch_size

    def __getitem__(self, idx):
        # Retrieves the batch of index.
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        sentence_pairs = self.sentence_pairs[indexes]

        # With BERT tokenizer's batch_encode_plus batch of both the sentences are
        # encoded together and separated by [SEP] token.
        encoded = self.tokenizer.batch_encode_plus(
            sentence_pairs.tolist(),
            add_special_tokens=True,
            max_length=max_length,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_tensors="tf",
        )   

        bert_output = bert_model(**encoded)
        sequence_output = bert_output.last_hidden_state
         
        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype="int32")
            return sequence_output, labels
        else:
            return sequence_output

    def on_epoch_end(self):
        # Shuffle indexes after each epoch if shuffle is set to True.
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indexes)

