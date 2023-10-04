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

feature_vec = np.zeros((200),dtype = "float32")
feature_vec = np.add(feature_vec, model['景色'])
feature_vec = np.divide(feature_vec, len(key_list))