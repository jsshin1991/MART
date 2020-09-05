from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from Text.SimpleRNNModel.rnn_utils import train_RNNModel, show_rank
from Text.run_vocab_MART import run_vocab_MART

### Test ###
np.random.seed(1)
# urllib.request.urlretrieve("https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv", filename="spam.csv")
data = pd.read_csv('./Text/spam.csv',encoding='latin1')

del data['Unnamed: 2']
del data['Unnamed: 3']
del data['Unnamed: 4']
data['v1'] = data['v1'].replace(['ham','spam'],[0,1])
# Remove duplicates in column v2
data.drop_duplicates(subset=['v2'], inplace=True)

X_data = data['v2']
y_data = data['v1']

# Perform tokenization and convert to index
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_data)
sequences = tokenizer.texts_to_sequences(X_data)

word_to_index = tokenizer.word_index
index_to_word = dict((v, k) for k, v in word_to_index.items())

threshold = 2
total_cnt = len(word_to_index)
# Count the number of words whose appearance frequency is less than the threshold
rare_cnt = 0
# Total sum of all word frequencies in training data
total_freq = 0
# The sum of the number of occurrences of words whose appearance frequency is less than the threshold
rare_freq = 0

# Receives a pair of words and frequencies as key and value
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

vocab_size = len(word_to_index) + 1
n_of_train = int(len(sequences) * 0.8)
n_of_test = int(len(sequences) - n_of_train)

X_data = sequences
max_len = 189
data = pad_sequences(X_data, maxlen = max_len)
X_test = data[n_of_train:]
y_test = np.array(y_data[n_of_train:])
X_train = data[:n_of_train]
y_train = np.array(y_data[:n_of_train])

# Build a model and save (run at first)
# model = train_RNNModel(X_train, y_train, vocab_size)
# model.save('./Text/SimpleRNNModel/my_model.h5')
# print("\n Test accuracy: %.4f" % (model.evaluate(X_test, y_test)[1]))

# Simple RNN model
model = load_model('./Text/SimpleRNNModel/my_model.h5')
y_data_arr = y_data.to_numpy()

candidate = range(len(y_data_arr))
spam_idx = 1
for idx in candidate:
    if y_data_arr[idx] == 1:
        mart = run_vocab_MART(np.array([data[idx]]), model, vocab_size, list(index_to_word.keys()), steps=30)
        # ascending order according to mart result
        imp_rank = show_rank(data[idx], mart[0], index_to_word, threshold=1e-2)
        print("spam_index: " + str(spam_idx))
        print(imp_rank)
        spam_idx += 1

