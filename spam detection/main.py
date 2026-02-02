import re
import math
import pandas as pd
import numpy as np
from scipy.stats import norm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

class preprocessor:

    def __init__(self, String):
        self.text = String
        self.lowercaser()
        self.punctauation_number_remover()
        self.tokeniser()
        self.stopwords_removal()
        self.lematizer()

    def lowercaser(self):
        self.text = self.text.lower()

    def punctauation_number_remover(self):
        self.text = re.sub(r'[\d\W_]+', ' ', self.text)

    def tokeniser(self):
        self.tokens = word_tokenize(self.text)

    def stopwords_removal(self):
        stop_words = set(stopwords.words('english'))
        self.filtered = [w for w in self.tokens if w.lower() not in stop_words]
        
    def lematizer(self):
        self.pos_tags = pos_tag(self.filtered)
        lematizer = WordNetLemmatizer()
        self.lematized = [
            lematizer.lemmatize(word, self.get_wordnet_pos(tag)) 
            for word, tag in self.pos_tags
        ]

    def get_wordnet_pos(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        if tag.startswith('N'):
            return wordnet.NOUN
        if tag.startswith('R'):
            return wordnet.ADV
        if tag.startswith('V'):
            return wordnet.VERB
        else: return wordnet.NOUN

class TF_IDF_vector_finder:

    def __init__(self, docs):
        self.docs = docs
        self.vocabulary = []
        for d in docs:
            for w in d:
                if w not in self.vocabulary:
                    #print(w)
                    self.vocabulary.append(w)
        self.total_docs = len(self.docs)
        self.idf = {}
        #print(len(self.vocabulary))
        loop = 0
        for w in self.vocabulary:
            count = 0
            for d in self.docs:
                if w in d:
                    count += 1
            self.idf[w] = math.log10(self.total_docs/count)
            #print(loop)
            loop += 1
        self.total_tf_idf_finder()
        
    def total_tf_idf_finder(self):
        self.feature_matrix = []
        for d in self.docs:
            vec = [0 for w in self.vocabulary]
            index = 0
            for w in self.vocabulary:
                count = 0
                for t in d:
                    if t == w:
                        count += 1
                vec[index] = (count/len(d))*self.idf[w]
                index += 1
            #print(vec)
            self.feature_matrix.append(vec)
    
class Testing_tf_idf_finder:

    def __init__(self, V, idf):
        self.vocabulary = V
        self.idf = idf

    def tf_idf_calculater(self, x):
        vec = [0 for w in self.vocabulary]
        index = 0
        for w in self.vocabulary:
            count = 0
            for t in x:
                if t == w:
                    count += 1
            vec[index] = (count/len(x))*self.idf[w]
            index += 1
        return vec

class training:

    def __init__(self, data):
        self.data = data
        self.positive_count = 0
        self.negetive_count = 0
        self.positive_group = []
        self.negetive_group = []
        for row in data.itertuples(index = True):
            if row[-2] == 1:
                self.negetive_count += 1
                self.negetive_group.append(row[-1])
            else: 
                self.positive_count += 1
                self.positive_group.append(row[-1])
        self.positive_prior = self.positive_count/data.shape[0]
        self.negetive_prior = self.negetive_count/data.shape[0]
        #print(self.positive_group)
        self.mean_var_calculater()
        
    def mean_var_calculater(self):
        self.feature_size = len(self.data.iloc[0, 3])
        self.positive_per_feature_mean_var = []
        self.negetive_per_feature_mean_var = []
        np_p_group = np.array(self.positive_group)
        np_n_group = np.array(self.negetive_group)
        for f in range(self.feature_size):
            var_p = np.var(np_p_group[:, f]) + 1e-9
            var_n = np.var(np_n_group[:, f]) + 1e-9
            mean_p = np.mean(np_p_group[:, f])
            mean_n = np.mean(np_n_group[:, f])
            self.positive_per_feature_mean_var.append([mean_p, var_p])
            self.negetive_per_feature_mean_var.append([mean_n, var_n])

class prediction:

    def __init__(
            self, 
            x_test, 
            p_prior, 
            n_prior, 
            p_per_feature_mean_var, 
            n_per_feature_mean_var
        ):
        self.x_test = x_test
        self.p_prior = p_prior
        self.n_prior = n_prior
        self.p_mean_variance = p_per_feature_mean_var
        self.n_mean_variance = n_per_feature_mean_var
        self.main()
        self.verdict()

    def main(self):
        self.score_p = math.log10(self.p_prior)
        self.score_n = math.log10(self.n_prior)
        f_size = len(self.p_mean_variance)
        for f in range(f_size):
            mean_p, var_p = self.p_mean_variance[f]
            mean_n, var_n = self.n_mean_variance[f]
            self.score_p += norm.logpdf(self.x_test[f], loc = mean_p, scale = math.sqrt(var_p))
            self.score_n += norm.logpdf(self.x_test[f], loc = mean_n, scale = math.sqrt(var_n))

    def verdict(self):
        if self.score_p > self.score_n:
            self.verdict = 'ham'
        else: self.verdict = 'spam'

data = pd.DataFrame(pd.read_csv("spam_ham_dataset.csv"))
data = data.drop(data.columns[0], axis = 1)
data_size = len(data)
train_frac = 0.8

train_data = data.iloc[:int(train_frac * data_size)]
test_data = data.iloc[int(train_frac * data_size):]

#print("OK")

preprocessed = []
count = 0
for row in train_data.itertuples():
    P = preprocessor(row[-2])
    preprocessed.append(P.lematized)
    #print("done", count)
    count += 1

Tivf = TF_IDF_vector_finder(preprocessed)
train_data['feature_vec'] = Tivf.feature_matrix

#train_data.to_csv('check.csv', index = False)

#print(train_data)

T = training(train_data)
print(T.negetive_per_feature_mean_var)
print("-------------------------------------------")
print(T.positive_per_feature_mean_var)


prior_prob_dict = {
    'positive_prior': T.positive_prior,
    'negetive_prior': T.negetive_prior
}

rows = []
for i, (fp, fn) in enumerate(zip(
    T.positive_per_feature_mean_var,
    T.negetive_per_feature_mean_var
)):
    rows.append({
        "feature": i,
        "word": Tivf.vocabulary[i],
        "idf": Tivf.idf[Tivf.vocabulary[i]],
        "pos_mean": fp[0],
        "pos_var": fp[1],
        "neg_mean": fn[0],
        "neg_var": fn[1]
    })

df = pd.DataFrame(rows)
df.to_csv("feature_stats.csv", index=False, mode='w')

df_prior = pd.DataFrame([prior_prob_dict])
df_prior.to_csv("prior.csv", index = False, mode='w')


for row in test_data.itertuples():
    P = preprocessor(row[-2])
    tivf = Testing_tf_idf_finder(Tivf.vocabulary, Tivf.idf)
    row_vec = tivf.tf_idf_calculater(P.lematized)
    Pr = prediction(
        row_vec, 
        T.positive_prior, 
        T.negetive_prior, 
        T.positive_per_feature_mean_var, 
        T.negetive_per_feature_mean_var
    )
    print(Pr.verdict)


'''Doc =  ["I like machine learning", "I like deep learning", "I like learning"]
lematized_docs = []
for d in Doc:
    P = preprocessor(d)
    lematized_docs.append(P.lematized)

tivf = TF_IDF_vector_finder(lematized_docs)
print(tivf.feature_matrix)'''

