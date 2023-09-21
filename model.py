import numpy as np
import pandas as pd
import pickle
from gensim.models import word2vec
import MeCab
from gensim import corpora
from gensim import models
import scipy.spatial.distance
from gensim.models import KeyedVectors

tagger = MeCab.Tagger("-Owakati")#タグはMeCab.Tagger（neologd辞書）を使用

tagger.parse('')
def tokenize_ja(text, lower):
    node = tagger.parseToNode(str(text))
    while node:
        if lower and node.feature.split(',')[0] in ["名詞","形容詞"]:#分かち書きで取得する品詞を指定
            yield node.surface.lower()
        node = node.next
def tokenize(content, token_min_len, token_max_len, lower):
    return [
        str(token) for token in tokenize_ja(content, lower)
        if token_min_len <= len(token) <= token_max_len and not token.startswith('_')
    ]



#学習データの読み込み

df = pd.read_csv('C:/Users/shun/Desktop/travel/じゃらん.csv')
df_travel = df.groupby(['場所','スコア'])['口コミ'].apply(list).apply(' '.join).reset_index().sort_values('スコア', ascending=False)

#コーパス作成
wakati_travel_text = []
for i in df_travel['口コミ']:
    txt = tokenize(i, 2, 10000, True)
    wakati_travel_text.append(txt)

np.savetxt("../travel_corpus.txt", wakati_travel_text, fmt = '%s', delimiter = ',')

# モデル作成
word2vec_travel_model = word2vec.Word2Vec(wakati_travel_text, sg = 3, vector_size = 200, window = 5, min_count = 2, epochs = 100, workers = 3)
#sg（0: CBOW, 1: skip-gram）,size（ベクトルの次元数）,window（学習に使う前後の単語数）,min_count（n回未満登場する単語を破棄）,iter（トレーニング反復回数）

# モデルのセーブ
#word2vec_travel_model.save("word2vec_travel_model.model")
word2vec_travel_model.wv.save_word2vec_format("word2vec_travel_model.kv", binary=False)

#word2vec_travel_model =word2vec.Word2Vec.load("word2vec_travel_model.model")
model = KeyedVectors.load_word2vec_format("word2vec_travel_model.kv", binary=False)
    
def predict(input_word):
    results=word2vec_travel_model.wv.most_similar(input_word)
    
    return results

def calculate_language_vector(words):
    features = 200
    feature_vec = np.zeros((features))
    for word in words:
        try:
            feature_vec = np.add(feature_vec, model[word])
        #例外処理.辞書にない文字が出たときは処理をスキップする
        except KeyError:
            pass
    if len(words) > 0:
        feature_vec = np.divide(feature_vec, len(words))
    return feature_vec

def search_most_similar(vector):
    df = pd.read_csv('C:/Users/shun/Desktop/travel/じゃらん.csv')
    df_travel = df.groupby(['場所','スコア'])['口コミ'].apply(list).apply(' '.join).reset_index().sort_values('スコア', ascending=False)
    wakati_travel_text = []
    for i in df_travel['口コミ']:
        txt = tokenize(i, 2, 10000, True)
        wakati_travel_text.append(txt)

    df_travel['wakati_travel_text'] = wakati_travel_text

    place = df_travel.iloc[0,1]
    tmp_max =0
    for i in range(len(df)):
        vect = calculate_language_vector(df_travel.iloc[i,3])
        score = 1-scipy.spatial.distance.cosine(vector, vect)
        if score > tmp_max:
            place = df_travel.iloc[i,1]
            tmp_max = score
    return place,tmp_max,vect,df_travel.iloc[0,4]

def calculate_emotion_vector(key_list):
    feature_vec = np.zeros((200),dtype = "float32")
    for word in key_list:
        if word in model:
            feature_vec = np.add(feature_vec, model[word])
    if len(key_list) > 0:
        feature_vec = np.divide(feature_vec, len(key_list))
    return feature_vec