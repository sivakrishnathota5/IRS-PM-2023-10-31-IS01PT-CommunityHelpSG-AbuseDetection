import os

import nltk
import pandas as pd
import numpy as np
from nltk.stem.lancaster import LancasterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.svm import SVC

from vectorizers.factory import get_vectoriser

import faiss
nltk.download('punkt')

#######  ######
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import time
import os
import tensorflow as tf
import pandas as pd
import re
import unicodedata
from nltk.tokenize import RegexpTokenizer
import nltk
from nltk.corpus import stopwords
import joblib
######   ####


#############################
import string
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd

import random
import joblib
import pickle
device = torch.device("cpu")


### Encoder RNN Implementation
class EncoderLSTM(nn.Module):
    def __init__(self,size_input,size_embbeding,size_hidden,layers,p):
        super(EncoderLSTM,self).__init__()
        self.size_input=size_input
        self.size_embbeding=size_embbeding
        self.size_hidden=size_hidden
        self.layers=layers
        self.dropout=nn.Dropout(p)
        self.tag=True

        self.embbed_layer=nn.Embedding(self.size_input,self.size_embbeding)
        self.lstm=nn.LSTM(self.size_embbeding,self.size_hidden,self.layers,dropout=p)

    def forward(self, x):
        # print(x.shape)
        embbeding=self.dropout(self.embbed_layer(x))
        # print(embbeding.shape)
        output, (hidden_st,cell_st) = self.lstm(embbeding)
        return hidden_st, cell_st # will return only hidden and cell state from the Encoder
    
### Decoder RNN Implementation
class DecoderLSTM(nn.Module):
    def __init__(self,size_input,size_embbeding,size_hidden,layers,p,size_output):
        super(DecoderLSTM,self).__init__()
        self.size_input=size_input
        self.size_embbeding=size_embbeding
        self.size_hidden=size_hidden
        self.layers=layers
        self.size_output=size_output
        self.dropout=nn.Dropout(p)
        # self.tag=True

        self.embbed_layer=nn.Embedding(self.size_input,self.size_embbeding) # input_size X embedding_size
        self.lstm=nn.LSTM(self.size_embbeding,self.size_hidden,self.layers,dropout=p) #embedding_size * hidden_size
        self.fc=nn.Linear(self.size_hidden,self.size_output) # hidden_size*output_size

    def forward(self,x,hidden_st,cell_st):
        x=x.unsqueeze(0)
        embbeding=self.dropout(self.embbed_layer(x))
        outputs, (hidden_st, cell_st) = self.lstm(embbeding, (hidden_st,cell_st))
        preds=self.fc(outputs)
        preds=preds.squeeze(0)
        return preds,hidden_st,cell_st


class Seq2seq_model(nn.Module):
    def __init__(self,encoder_net,decoder_net):
        super(Seq2seq_model,self).__init__()
        self.encoder_net=encoder_net
        self.decoder_net=decoder_net

    def forward(self,src,target,teacher_forcing=0.5):
        batch_length=src.shape[1]
        target_len=target.shape[0]
        hindi_lang=joblib.load('hindi_lang-1.joblib')
        eng_lang=joblib.load('eng_lang-1.joblib')
        
        target_vocab_len=eng_lang.num_of_words

        output_tensor=torch.zeros(target_len,batch_length,target_vocab_len).to(device)
        hidden_st_enc, cell_st_enc=self.encoder_net(src)
        x=target[0]

        for i in range(1,target_len):
            output,hidden_st_dec,cell_st_dec=self.decoder_net(x,hidden_st_enc,cell_st_enc)
            output_tensor[i]=output
            pred=output.argmax(1)
            x=target[i] if random.random()<teacher_forcing else pred #teacher forcing is used with probability 0.5

        return output_tensor    
SOS_token=0
EOS_token=1
PAD_token=2
MAX_LENGTH=10
class Vocab_class:
    def __init__(self):
        self.word_to_index={"<SOS>":0,"<EOS>":1,"<PAD>":2,"<UKN>":3}  #dict to map each token to an index
        self.word_counts={} # will keep track of each token in the vocabulary
        self.index_to_word={0:"<SOS>", 1:"<EOS>", 2:"<PAD>", 3:"<UKN>"} # will map each index to a token
        self.num_of_words=4

    def sentence_add(self, sentence): # function to add the words of a sentence into the vocabulary
        words=sentence.split(" ")
        for word in words:
            if word not in self.word_to_index:
                self.word_to_index[word]=self.num_of_words
                self.word_counts[word]=1

                self.index_to_word[self.num_of_words]=word
                self.num_of_words+=1
            else:
                self.word_counts[word]+=1
                

#############################





# Encoder
class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))

# Attention Mechanism
class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    hidden_with_time_axis = tf.expand_dims(query, 1)
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))
    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)
    return context_vector, attention_weights

# Decoder
class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    context_vector, attention_weights = self.attention(hidden, enc_output)
    x = self.embedding(x)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
    output, state = self.gru(x)
    output = tf.reshape(output, (-1, output.shape[2]))
    x = self.fc(output)
    return x, state, attention_weights

#############################
class FaqEngine:
    def __init__(self, faqslist, type='tfidf'):
        self.faqslist = faqslist
        self.vector_store = None
        self.stemmer = LancasterStemmer()
        self.le = LE()
        self.classifier = None
        self.build_model(type)

    def cleanup(self, sentence):
        word_tok = nltk.word_tokenize(sentence)
        stemmed_words = [self.stemmer.stem(w) for w in word_tok]
        return ' '.join(stemmed_words)

    def build_model(self, type):

        self.vectorizer = get_vectoriser(type)  # TfidfVectorizer(min_df=1, stop_words='english')
        dataframeslist = [pd.read_csv(csvfile).dropna() for csvfile in self.faqslist]
        self.data = pd.concat(dataframeslist, ignore_index=True)
        self.data['Clean_Question'] = self.data['Question'].apply(lambda x : self.cleanup(x))
        self.data['Question_embeddings'] = list(self.vectorizer.vectorize(self.data['Clean_Question'].tolist()))
        self.questions = self.data['Question'].values
        X = self.data['Question_embeddings'].tolist()
        
        X = np.array(X)
        d = X.shape[1]
        index = faiss.IndexFlatL2(d)
        if index.is_trained :
            index.add(X)
        self.vector_store = index
        # Loop wise version for question embedding generation
        # questions_cleaned = []
        # for question in self.questions:
        #     questions_cleaned.append(self.cleanup(question))

        # X = self.vectorizer.vectorize(questions_cleaned)

        # Under following cases, we dont do classification
        # 'Class' column abscent
        # 'Class' column has same values
        if 'Class' not in list(self.data.columns):
            return

        y = self.data['Class'].values.tolist()
        if len(set(y)) < 2: # 0 or 1
            return

        y = self.le.fit_transform(y)

        trainx, testx, trainy, testy = tts(X, y, test_size=.25, random_state=42)

        self.classifier = SVC(kernel='linear')
        self.classifier.fit(trainx, trainy)
        # print("SVC:", self.model.score(testx, testy))    
        
    def unicode_to_ascii(self,s):
        return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')
    def preprocess_sentence(self,w):
        w = self.unicode_to_ascii(w.lower().strip())
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)
        w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
        w = w.rstrip().strip()
        return w
    def evaluate(self,sentence):
        
        BUFFER_SIZE=31904
        BATCH_SIZE=64
        steps_per_epoch=498
        embedding_dim=128
        units=256
        vocab_inp_size=25249
        vocab_tar_size=27436
        max_length_targ=33
        max_length_inp=23
        encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
        decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
        targ_lang=joblib.load('targ_lang.joblib')
        inp_lang=joblib.load('inp_lang.joblib')
        checkpoint_dir = 'training_checkpoints-100'
        attention_plot = np.zeros((max_length_targ, max_length_inp))
        sentence = self.preprocess_sentence(sentence)
        inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_inp,
                                                           padding='post')
        inputs = tf.convert_to_tensor(inputs)
        result = ''
        hidden = [tf.zeros((1, units))]
        enc_out, enc_hidden = encoder(inputs, hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)
        for t in range(max_length_targ):
           predictions, dec_hidden, attention_weights = decoder(dec_input,dec_hidden,enc_out)
           predicted_id = tf.argmax(predictions[0]).numpy()
           result += targ_lang.index_word[predicted_id] + ' '
           if targ_lang.index_word[predicted_id] == '<end>':
               return result, sentence
           dec_input = tf.expand_dims([predicted_id], 0)
        return result, sentence
    SOS_token=0
    EOS_token=1
    PAD_token=2
    MAX_LENGTH=10
    
    def clean_sentence(self,sentence):
        punctuations=list(string.punctuation)
        cleaned=""
        for letter in sentence:
            if letter=='<' or letter=='>' or letter not in punctuations:
                cleaned+=letter
        return cleaned

    def predict_translation(self,sentence,device,max_length=MAX_LENGTH):
        model=joblib.load('model-1.joblib')
        model=torch.load("phase2_v2-1.pth")
        hindi_lang=joblib.load('hindi_lang-1.joblib')
        eng_lang=joblib.load('eng_lang-1.joblib')
        sentence=self.clean_sentence(sentence)
        tokens=sentence.split(" ")
        indexes=[]
        for token in tokens:
            if token in hindi_lang.word_to_index:
                indexes.append(hindi_lang.word_to_index[token])
            else:
                indexes.append(hindi_lang.word_to_index["<UKN>"])
        tensor_of_sentence=torch.LongTensor(indexes).unsqueeze(1).to(device)
        with torch.no_grad():
            hidden,cell=model.encoder_net(tensor_of_sentence)
        outputs=[SOS_token]
        for _ in range(max_length):
            prev_word=torch.LongTensor([outputs[-1]]).to(device)
            with torch.no_grad():
                output,hidden,cell=model.decoder_net(prev_word, hidden,cell)
                pred=output.argmax(1).item()

            outputs.append(pred)

            if eng_lang.index_to_word[pred] =="<EOS>":
                break

        final=[]

        for i in outputs:
            if i == "<PAD>":
                break
            final.append(i)

        final = [eng_lang.index_to_word[idx] for idx in final]
        translated=" ".join(final)
        return translated

    
    


    def query(self, usr):
        # print("User typed : " + usr)
        try:
            translated=self.predict_translation(usr,device)
            #result, sentence = self.evaluate(usr)
            #result1='Predicted translation: {} {} '.format(result)
            #########cleaned_usr = self.cleanup(usr)
            #########t_usr_array = self.vectorizer.query(cleaned_usr)
            #########if self.classifier:
                #########prediction = self.classifier.predict(t_usr_array)[0]
                #########class_ = self.le.inverse_transform([prediction])[0]
                ######### print("Class " + class_)
                #########questionset = self.data[self.data['Class'] == class_]
            #########else:
                #########questionset = self.data

            # threshold = 0.7
            
            # Vectorized implementation of cosine similarity usage for fast execution
            #cos_sims = cosine_similarity(questionset['Question_embeddings'].tolist(), t_usr_array)

            # Top most similar question
            #########top_k = 1
            
            #calling FAISS search
            #########D, I = self.vector_store.search(t_usr_array, top_k)
            
            #########question_index = int(I[0][0])
            #########return self.data['Answer'][question_index]
            return translated
            # Loop wise implementation of cosine similarity usage
            # cos_sims = []
            # for question in questionset['Question']:
            #     cleaned_question = self.cleanup(question)
            #     question_arr = self.vectorizer.query(cleaned_question)
            #     sims = cosine_similarity(question_arr, t_usr_array)
            #     # if sims > threshold:
            #     cos_sims.append(sims)

            # print("scores " + str(cos_sims))
            # commenting this code as we use FAISS vector store. 
            '''if len(cos_sims) > 0:
                ind = np.argmax(cos_sims)
                return self.data['Answer'][questionset.index[ind]]'''
                # ind = cos_sims.index(max(cos_sims))
                # print(ind)
                # print(questionset.index[ind])
                
        except Exception as e:
            print(e)
            return "Could not follow your question [" + usr + "], Try again"


if __name__ == "__main__":
    base_path = os.path.join(os.path.dirname(os.path.abspath( __file__ )),"data")
    faqslist = [os.path.join(base_path,"Greetings.csv"), os.path.join(base_path,"GST FAQs 2.csv")]
    faqmodel = FaqEngine(faqslist, 'tfidf')
    response = faqmodel.query("Hi")
    print(response)
