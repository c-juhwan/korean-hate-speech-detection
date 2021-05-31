#!/usr/bin/env python
# coding: utf-8

# # Chapter 4.1 - 영어 텍스트 분류, P145 
# 
# ## Initialize

# For M1 macs: https://cpuu.postype.com/post/9091007
#from tensorflow.python.compiler.mlcompute import mlcompute
#mlcompute.set_mlc_device(device_name='gpu')

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tqdm import tqdm
import re
import json
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.python.ops.math_ops import reduce_prod
#from tensorflow.math import reduce_prod

DATA_PATH = "/Users/choijuhwan/Workspace/Git_Repos/korean-hate-speech-detection/study/word2vec-nlp-tutorial/"

## Load Data

df_train = pd.read_csv(DATA_PATH + "labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
df_test = pd.read_csv(DATA_PATH + "testData.tsv", header=0, delimiter="\t", quoting=3)

# ## Preprocessing

def preprocessing(text):

    # Use Regular Expression
    # Remove HTML tag
    # Lower case and split to each words
    # Remove stopwords
    # Make splited words to sentences

    regular_expression1 = "^[a-zA-Z0-9]"

    text = BeautifulSoup(text, 'html5lib').get_text()
    text = re.sub(regular_expression1, " ", text)

    words = text.lower().split()
    stop_words = stopwords.words('english')
    words = [w for w in words if not w in stop_words]

    clean_text = ' '.join(words)

    return clean_text

for i in tqdm(range(0, df_train.shape[0])):
    df_train['review'][i] = preprocessing(df_train['review'][i])

for i in tqdm(range(0, df_test.shape[0])):
    df_test['review'][i] = preprocessing(df_test['review'][i])

# ### Integer index

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_train['review'])
train_sequences = tokenizer.texts_to_sequences(df_train['review'])
test_sequences = tokenizer.texts_to_sequences(df_test['review'])

# ### Make vocabulary

vocab = tokenizer.word_index
vocab["<PAD>"] = 0

datas = {}
datas['vocab'] = vocab
datas['vocab_size'] = len(vocab)

# ### Padding for train data & test data

MAX_SEQUENCE_LENGTH = 180

train_inputs = pad_sequences(train_sequences, maxlen = MAX_SEQUENCE_LENGTH, padding='post')
train_labels = np.array(df_train['sentiment'])

test_inputs = pad_sequences(test_sequences, maxlen = MAX_SEQUENCE_LENGTH, padding='post')
test_id = np.array(df_test['id'])

# ## RNN Model

# ### Seed 

SEED_NUM = 1234
tf.random.set_seed(SEED_NUM)

# ### Define hyperparameter

model_name = 'RNN_classifier_en'
BATCH_SIZE = 128
NUM_EPOCHS = 10
VALID_SPLIT = 0.1
MAX_LEN = train_inputs.shape[1]

kargs = {'model_name': model_name,
        'vocab_size': datas['vocab_size'],
        'embedding_dimension': 150,
        'dropout_rate': 0.2,
        'lstm_dimension': 150,
        'dense_dimension': 150,
        'output_dimension': 1}


# ### Model implementation

class RNNclassifier(tf.keras.Model):
    def __init__(self, **kargs):
        super(RNNclassifier, self).__init__(name=kargs['model_name'])

        self.dropout = tf.keras.layers.Dropout(kargs['dropout_rate'])
        self.embedding = tf.keras.layers.Embedding(input_dim=kargs['vocab_size'], output_dim=kargs['embedding_dimension'])
        self.lstm1 = tf.keras.layers.LSTM(kargs['lstm_dimension'], return_sequences=True)
        self.lstm2 = tf.keras.layers.LSTM(kargs['lstm_dimension'], return_sequences=True)
        self.lstm3 = tf.keras.layers.LSTM(kargs['lstm_dimension'])
        self.fc1 = tf.keras.layers.Dense(units=kargs['dense_dimension'], activation=tf.keras.activations.relu)
        self.fc2 = tf.keras.layers.Dense(units=kargs['output_dimension'], activation=tf.keras.activations.sigmoid)

    def call(self, x):
        x = self.embedding(x)
        x = self.dropout(x)

        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.lstm3(x)
        x = self.dropout(x)

        x = self.fc1(x)
        x = self.dropout(x)

        x = self.fc2(x)

        return x

# ### Generate model instance

model = RNNclassifier(**kargs)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')])

# ### Model learning
# 
# > Memo 
# >
# > numpy 1.20 이상은 뭔가 [에러](https://stackoverflow.com/questions/58479556/notimplementederror-cannot-convert-a-symbolic-tensor-2nd-target0-to-a-numpy)가 있음
# >
# > 난 pip로 Build 안돼서 conda로 1.19.5 설치했음
# >
# > numpy array 말고 Tensor가 들어가야하는듯 -> 꼭 그건 아니고 버전만 맞추면 됨

earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=2)

checkpoint_path = DATA_PATH + model_name + '/weights.h5'
checkpoint_dir = os.path.dirname(checkpoint_path)

if os.path.exists(checkpoint_dir):
    pass
else:
    os.makedirs(checkpoint_dir, exist_ok=True)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True)

history = model.fit(train_inputs, train_labels, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_split=VALID_SPLIT, callbacks=[earlystop_callback, cp_callback])

# ### Show results

plt.plot(history.history['accuracy'], label = "Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation_Accuracy")

plt.legend()
plt.show()


# ### Prediction & Submit to Kaggle
# 
# > Memo
# >
# > AttributeError: 'str' object has no attribute 'decode'
# >
# > h5py package를 2.10.0 버전으로 

SAVE_FILE_NAME = 'weight.h5'
DATA_OUT_PATH = "/Users/choijuhwan/Workspace/Git_Repos/korean-hate-speech-detection/study/word2vec-nlp-tutorial/data_out/"

model.load_weights(os.path.join(checkpoint_path))

predictions = model.predict(test_inputs, batch_size=BATCH_SIZE)
predictions = predictions.squeeze(-1)

def configure(x):
    if x >= 0.5:
        return 1
    else:
        return 0

for i in tqdm(range(0, predictions.shape[0])):
    predictions[i] = configure(predictions[i])

predictions = np.array(predictions).astype('int8')

if not os.path.exists(DATA_OUT_PATH):
    os.makedirs(DATA_OUT_PATH)

output = pd.DataFrame(data={"id": list(test_id), "sentiment":list(predictions)})
output.to_csv(DATA_OUT_PATH+'movie_review_result_rnn.csv', index=False, quoting=3)