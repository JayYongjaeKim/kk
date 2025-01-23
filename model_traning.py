import time
import random
import pandas as pd
import numpy as np

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import os
import numpy as np
import joblib

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, MaxPooling2D, Conv2D, Dropout, Lambda, Dense, Flatten, Activation, Input, Embedding, BatchNormalization
from tensorflow.keras.initializers import glorot_normal, Zeros, TruncatedNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy

from collections import defaultdict
import math

data_path = '/Users/jake/Downloads/ml-1m'
movielens_rcmm = pd.read_csv(f"{data_path}/movielens_rcmm_v2.csv", dtype=str)

label_encoders = {col: LabelEncoder() for col in movielens_rcmm.columns[:-1]}
for col, le in label_encoders.items():
    movielens_rcmm[col] = le.fit_transform(movielens_rcmm[col])

movielens_rcmm['label'] = movielens_rcmm['label'].astype(np.float32)

train_df, test_df = train_test_split(movielens_rcmm, test_size=0.2, random_state=42)

u_i_feature = ['user_id', 'movie_id']
meta_features = ['movie_decade', 'movie_year', 'rating_year', 'rating_month', 'rating_decade', 'genre1','genre2', 'genre3', 'gender', 'age', 'occupation', 'zip']
label = 'label'
field_dims = np.max(movielens_rcmm[u_i_feature + meta_features].astype(np.int64).values, axis=0) + 1

class MultiHeadSelfAttention(Layer):  
    def __init__(self, att_embedding_size=8, head_num=2, use_res=True, scaling=False, seed=1024, **kwargs):
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.use_res = use_res
        self.seed = seed
        self.scaling = scaling
        super(MultiHeadSelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(input_shape)))
        embedding_size = int(input_shape[-1])
        self.W_Query = self.add_weight(name='query', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                       dtype=tf.float32, initializer=TruncatedNormal(seed=self.seed))
        self.W_key = self.add_weight(name='key', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                     dtype=tf.float32, initializer=TruncatedNormal(seed=self.seed + 1))
        self.W_Value = self.add_weight(name='value', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                       dtype=tf.float32, initializer=TruncatedNormal(seed=self.seed + 2))
        if self.use_res:
            self.W_Res = self.add_weight(name='res', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                         dtype=tf.float32, initializer=TruncatedNormal(seed=self.seed))
        super(MultiHeadSelfAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 3:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))
        
        querys = tf.tensordot(inputs, self.W_Query, axes=(-1, 0))  
        keys = tf.tensordot(inputs, self.W_key, axes=(-1, 0))
        values = tf.tensordot(inputs, self.W_Value, axes=(-1, 0))

        querys = tf.stack(tf.split(querys, self.head_num, axis=2))
        keys = tf.stack(tf.split(keys, self.head_num, axis=2))
        values = tf.stack(tf.split(values, self.head_num, axis=2))
        
        inner_product = tf.matmul(querys, keys, transpose_b=True)
        if self.scaling:
            inner_product /= self.att_embedding_size ** 0.5
        self.normalized_att_scores = tf.nn.softmax(inner_product)
        
        result = tf.matmul(self.normalized_att_scores, values)
        result = tf.concat(tf.split(result, self.head_num, axis=0), axis=-1)
        result = tf.squeeze(result, axis=0) 

        if self.use_res:
            result += tf.tensordot(inputs, self.W_Res, axes=(-1, 0))
        result = tf.nn.relu(result)
        return result

    def compute_output_shape(self, input_shape):
        return (None, input_shape[1], self.att_embedding_size * self.head_num)

class FeaturesEmbedding(Layer):
    def __init__(self, field_dims, embed_dim, **kwargs):
        super(FeaturesEmbedding, self).__init__(**kwargs)
        self.total_dim = sum(field_dims)
        self.embed_dim = embed_dim
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.longlong)
        self.embedding = tf.keras.layers.Embedding(input_dim=self.total_dim, output_dim=self.embed_dim)

    def build(self, input_shape):
        self.embedding.build(input_shape)
        self.embedding.set_weights([tf.keras.initializers.GlorotUniform()(shape=self.embedding.weights[0].shape)])

    def call(self, x):
        x = x + tf.constant(self.offsets)
        return self.embedding(x)

class AutoInt(Layer): 
    def __init__(self, field_dims, embedding_size, att_layer_num=3, att_head_num=2, att_res=True, 
                 l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False, dnn_dropout=0.4, init_std=0.0001):
        super(AutoInt, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embedding_size)
        self.num_fields = len(field_dims)
        self.embedding_size = embedding_size
        self.final_layer = Dense(1, use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=init_std))
        self.int_layers = [MultiHeadSelfAttention(att_embedding_size=embedding_size, head_num=att_head_num, use_res=att_res) for _ in range(att_layer_num)]

    def call(self, inputs):
        att_input = self.embedding(inputs)
        for layer in self.int_layers:
            att_input = layer(att_input)
        att_output = Flatten()(att_input)
        att_output = self.final_layer(att_output)
        y_pred = tf.nn.sigmoid(att_output)
        return y_pred

class AutoIntModel(Model):
    def __init__(self, field_dims, embedding_size, att_layer_num=3, att_head_num=2, att_res=True, 
                 l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False, dnn_dropout=0.4, init_std=0.0001):
        super(AutoIntModel, self).__init__()
        self.autoInt_layer = AutoInt(field_dims, embedding_size, att_layer_num=att_layer_num, att_head_num=att_head_num, 
                                     att_res=att_res, l2_reg_dnn=l2_reg_dnn, dnn_dropout=dnn_dropout, init_std=init_std)

    def call(self, inputs, training=False):
        return self.autoInt_layer(inputs, training=training)

epochs = 5
learning_rate = 0.0001
dropout = 0.4
batch_size = 2048
embed_dim = 16

autoInt_model = AutoIntModel(field_dims, embed_dim, att_layer_num=3, att_head_num=2, att_res=True, 
                             l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False, dnn_dropout=dropout, init_std=0.0001)
optimizer = Adam(learning_rate=learning_rate)
loss_fn = BinaryCrossentropy(from_logits=False)

autoInt_model.compile(optimizer=optimizer, loss=loss_fn, metrics=['binary_crossentropy'])

history = autoInt_model.fit(train_df[u_i_feature + meta_features], train_df[label], 
                            epochs=epochs, batch_size=batch_size, validation_split=0.1)

class AutoIntMLPModel(Model):
    def __init__(self, field_dims, embedding_size, att_layer_num=3, att_head_num=2, att_res=True, 
                 dnn_hidden_units=(32, 32), dnn_activation='relu', l2_reg_dnn=0, 
                 l2_reg_embedding=1e-5, dnn_use_bn=False, dnn_dropout=0.4, init_std=0.0001):
        super(AutoIntMLPModel, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embedding_size)
        self.num_fields = len(field_dims)
        self.embedding_size = embedding_size

        self.final_layer = Dense(1, use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=init_std))
        
        self.dnn = []
        for units in dnn_hidden_units:
            self.dnn.append(Dense(units, activation=dnn_activation, kernel_regularizer=tf.keras.regularizers.l2(l2_reg_dnn)))
            if dnn_use_bn:
                self.dnn.append(BatchNormalization())
            self.dnn.append(Dropout(dnn_dropout))
        
        self.int_layers = [MultiHeadSelfAttention(
            att_embedding_size=embedding_size, head_num=att_head_num, use_res=att_res) for _ in range(att_layer_num)]

        # 추가된 Dense 레이어로 출력 shape 조정
        self.output_layer = Dense(1, activation='sigmoid')

    def call(self, inputs):
        embed_x = self.embedding(inputs)
        dnn_embed = tf.reshape(embed_x, shape=(-1, self.embedding_size * self.num_fields))

        att_input = embed_x
        for layer in self.int_layers:
            att_input = layer(att_input)

        att_output = Flatten()(att_input)
        att_output = self.final_layer(att_output)
        
        dnn_output = dnn_embed
        for dnn_layer in self.dnn:
            dnn_output = dnn_layer(dnn_output)

        y_pred = tf.add(att_output, dnn_output)

        # 최종 출력 shape을 (None, 1)로 조정
        return self.output_layer(y_pred)



autoIntMLP_model = AutoIntMLPModel(
    field_dims=field_dims,
    embedding_size=16,
    att_layer_num=3,
    att_head_num=2,
    dnn_hidden_units=(64, 32),
    dnn_dropout=0.5
)

optimizer = Adam(learning_rate=0.001)
loss_fn = BinaryCrossentropy()
autoIntMLP_model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

history_mlp = autoIntMLP_model.fit(
    train_df[u_i_feature + meta_features],
    train_df[label],
    epochs=10,
    batch_size=2048,
    validation_split=0.2,
    verbose=1
)


os.makedirs(f'{data_path}/autoint_plus_model2', exist_ok=True)
autoIntMLP_model.save_weights(f'{data_path}/autoint_plus_model2/autoInt_plus_model2.weights.h5')

np.save(f'{data_path}/autoint_plus_model2/field_dims2.npy', field_dims)

joblib.dump(label_encoders, f'{data_path}/autoint_plus_model2/label_encoders2.pkl')

print("모델 저장 완료")


def evaluate_model(model, test_df):
    def ndcg_at_k(y_true, y_pred, k=10):
        order = np.argsort(y_pred)[::-1][:k]
        ideal_order = np.argsort(y_true)[::-1][:k]
        dcg = np.sum([y_true[i] / np.log2(i + 2) for i in order])
        idcg = np.sum([y_true[i] / np.log2(i + 2) for i in ideal_order])
        return dcg / idcg if idcg != 0 else 0

    def hitrate_at_k(y_true, y_pred, k=10):
        top_k_items = np.argsort(y_pred)[::-1][:k]
        hits = np.sum(np.isin(top_k_items, np.where(y_true == 1)[0]))
        return hits / k

    ndcg_scores = []
    hitrate_scores = []
    user_ids = test_df['user_id'].unique()

    for i, user_id in enumerate(user_ids):
        print(f"Processing user {i + 1}/{len(user_ids)} with ID: {user_id}")
        user_data = test_df[test_df['user_id'] == user_id]
        if user_data.empty:
            print(f"No data for user {user_id}")
            continue

        y_true = user_data['label'].values
        y_pred = model.predict(user_data[u_i_feature + meta_features], batch_size=128)
        
        if len(y_pred) != len(y_true):
            print(f"Mismatch in predictions for user {user_id}: y_true={len(y_true)}, y_pred={len(y_pred)}")
            continue
        
        ndcg_scores.append(ndcg_at_k(y_true, y_pred))
        hitrate_scores.append(hitrate_at_k(y_true, y_pred))
    
    print(f"AutoInt+ NDCG: {np.mean(ndcg_scores):.4f}")
    print(f"AutoInt+ Hitrate: {np.mean(hitrate_scores):.4f}")

