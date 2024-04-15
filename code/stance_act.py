import tensorflow as tf
from keras import backend as k
from keras.layers.core import Lambda
from keras import layers
import numpy as np
from numpy import array
import pandas as pd
from keras import layers
from keras.preprocessing.text import Tokenizer
from  keras.utils import pad_sequences
from tensorflow.keras.layers import LSTM,Dropout,Bidirectional,Input, Embedding, Dense,Concatenate,Flatten, Multiply,Average,Subtract,GRU
from tensorflow_addons.layers import MultiHeadAttention
from keras.models import Model
from tensorflow.python.keras.layers.merge import concatenate
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from transformers import TFAutoModel, AutoTokenizer
import ast


li_text_train=[]
li_stance_train=[]
li_tox_train=[]
li_id_train=[]
li_vad_train=[]
li_moral_train=[]
li_act_train=[]

li_text_test=[]
li_stance_test=[]
li_id_test=[]
li_vad_test=[]
li_tox_test=[]
li_moral_test=[]
li_act_test=[]

# Train data
data=pd.read_csv("../emnlp_data/train_clean.csv", delimiter=";", na_filter= False) 
for i in range(len(data)):
    id_=str(data.tweetid.values[i])
    li_id_train.append(data.tweetid.values[i])
    li_stance_train.append(str(data.stance.values[i]))
    li_text_train.append(str(data.text.values[i]))
    li_vad_train.append((data.vad.values[i]))
    li_tox_train.append((data.toxic.values[i]))
    li_moral_train.append((data.moral.values[i]))
    li_act_train.append((data.act.values[i]))

# Test data    
data=pd.read_csv("../emnlp_data/test_clean.csv", delimiter=";", na_filter= False) 
for i in range(len(data)):       
    li_id_test.append(data.tweetid.values[i])
    li_stance_test.append(str(data.stance.values[i]))
    li_text_test.append(str(data.text.values[i]))
    li_vad_test.append((data.vad.values[i]))
    li_tox_test.append((data.toxic.values[i]))
    li_moral_test.append((data.moral.values[i]))
    li_act_test.append((data.act.values[i]))

print("li_stance np unique:::",np.unique(li_stance_train,return_counts=True))

# Converting labels into categorical labels
label_encoder=LabelEncoder()
final_lbls=li_stance_train+li_stance_test
values=array(final_lbls)
total_integer_encoded=label_encoder.fit_transform(values)
train_stance_enc=total_integer_encoded[:len(li_stance_train)]
test_stance_enc=total_integer_encoded[len(li_stance_train):]
train_stance=to_categorical(train_stance_enc)
test_stance=to_categorical(test_stance_enc)

# Converting toxic labels into categorical labels
label_encoder=LabelEncoder()
final_lbls=li_tox_train+li_tox_test
values=array(final_lbls)
total_integer_encoded=label_encoder.fit_transform(values)
train_tox_enc=total_integer_encoded[:len(li_stance_train)]
test_tox_enc=total_integer_encoded[len(li_stance_train):]
train_tox=to_categorical(train_tox_enc)
test_tox=to_categorical(test_tox_enc)

# Converting act labels into categorical labels
label_encoder=LabelEncoder()
final_lbls=li_act_train+li_act_test
values=array(final_lbls)
total_integer_encoded=label_encoder.fit_transform(values)
train_act_enc=total_integer_encoded[:len(li_stance_train)]
test_act_enc=total_integer_encoded[len(li_stance_train):]
train_act=to_categorical(train_act_enc)
test_act=to_categorical(train_act_enc)

# Converting moral labels in numpy array
li_moral_test=[np.array(ast.literal_eval(x), dtype=float) for x in li_moral_test]
li_moral_train=[np.array(ast.literal_eval(x), dtype=float) for x in li_moral_train]
li_moral_test=np.array(li_moral_test)
li_moral_train=np.array(li_moral_train)

# Reshaping VAD input features
li_vad_train=[np.array(ast.literal_eval(x), dtype=float) for x in li_vad_train]
li_vad_test=[np.array(ast.literal_eval(x), dtype=float) for x in li_vad_test]
li_vad_train=np.array(li_vad_train)
li_vad_test=np.array(li_vad_test)


# Model preparation
# Define the BERTweet tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
train_encodings = tokenizer(li_text_train, padding=True, truncation=True, return_tensors="tf")
test_encodings = tokenizer(li_text_test, padding=True, truncation=True, return_tensors="tf")

# Instantiate the BERTweet model
bertweet_model = TFAutoModel.from_pretrained("vinai/bertweet-base")
bertweet_model.trainable = False

# Define the tweet inputs
input_ids = Input(shape=(None,), dtype='int32', name='input_ids')
attention_mask = Input(shape=(None,), dtype='int32', name='attention_mask')
# Pass the inputs through the BERTweet model
outputs = bertweet_model(input_ids=input_ids, attention_mask=attention_mask)
# Get the last hidden state from BERTweet
sequence_output = outputs.pooler_output
sequence_output1= Dense(128, activation="relu")(sequence_output)

# VAD input feature
input_vad = Input(shape=(3,))
# Transform the vad features
reshape_layer = Lambda(lambda x: tf.expand_dims(x, axis=1))
reshaped_tensor = reshape_layer(input_vad)
print("reshaped_tensor:::",reshaped_tensor)
input_vad1= Dense(128,activation="relu")(reshaped_tensor)
input_vad1=Flatten()(input_vad1)

# VAD Embedder
int_diff=layers.subtract([sequence_output1,input_vad1])
int_mul=Multiply()([sequence_output1,input_vad1])
IM_output=Concatenate()([sequence_output1,input_vad1,int_diff,int_mul])
print("IM_output:::",IM_output,IM_output)

# Apply MultiHeadAttention
attention = MultiHeadAttention(num_heads=8, head_size=64)
attention_output = attention([IM_output, IM_output, IM_output])

# Outputs
attention_output_stance= Dense(128,activation="relu")(attention_output)
stance_output=Dense(3, activation="softmax", name="task_stance")(attention_output_stance)

attention_output_act= Dense(128,activation="relu")(attention_output)
act_output=Dense(5, activation="softmax", name="task_act")(attention_output_act) 

# Define the model
model=Model(inputs=[input_ids, attention_mask,input_vad],outputs=[stance_output,act_output])

# Model Compile
model.compile(optimizer=Adam(0.0001),loss={'task_stance':'categorical_crossentropy','task_act':'categorical_crossentropy'},
    loss_weights={'task_stance':1.0,'task_act':0.4}, metrics=['accuracy'])    
print(model.summary())

# Model Fit
model.fit([train_encodings['input_ids'], train_encodings['attention_mask'],li_vad_train],[train_stance,train_act],batch_size=32,epochs=20,verbose=2)
predicted = model.predict([test_encodings['input_ids'], test_encodings['attention_mask'],li_vad_test])
print(predicted)

# Results
result_=predicted[0]
p_1 = np.argmax(result_,axis=1)
print("p_1:::",p_1)
print("test_stance_enc::",test_stance_enc)
p_1 = np.argmax(result_, axis=1)
test_accuracy=accuracy_score(test_stance_enc, p_1)
print("test accuracy::::",test_accuracy)
target_names = ['AGAINST','FAVOR','NONE']
class_rep=classification_report(test_stance_enc, p_1)
print("specific confusion matrix",confusion_matrix(test_stance_enc, p_1))
print(class_rep)
class_rep=classification_report(test_stance_enc, p_1, target_names=target_names,output_dict=True)
macro_avg=class_rep['macro avg']['f1-score']
macro_prec=class_rep['macro avg']['precision']
macro_rec=class_rep['macro avg']['recall']
print("macro f1 score",macro_avg)
print("macro_prec",macro_prec)
print("macro_rec",macro_rec)
