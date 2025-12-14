#!/usr/bin/env python
# coding: utf-8

# In[82]:


import pandas as pd


# In[85]:


df = pd.read_csv("data/lod_multilingual_words.csv", sep="\t")[["lemma", "pos", "meaning", "word", "sentence"]]


# In[86]:


df.head()


# In[137]:


data = pd.merge(df, df, on="lemma")
data = data[
    (data.meaning_x <= data.meaning_y) &
    (data.sentence_x != data.sentence_y) &
    (data.pos_x == data.pos_y) &
    data.pos_x.isin(["VRB", "SUBST", "NP"])
    ]


# In[138]:


data["label"] = (data.meaning_x == data.meaning_y).astype(int)


# In[139]:


data = data.groupby(["lemma", "pos_x", "label"]).first().reset_index()
data = data[["lemma", "pos_x", "sentence_x", "sentence_y", "word_x", "word_y", "label"]]
data = data.rename(columns={"pos_x":"pos"})


# In[140]:


lemmas = set(data.lemma[data.label == 1].values) & set(data.lemma[data.label == 0].values)


# In[141]:


data = data[data.lemma.isin(lemmas)].sample(frac=1.0)


# In[142]:


data


# In[120]:


train = data[data.lemma.str.lower() > "j"]
test = data.drop(train.index)
dev = test.sample(1000)
test = test.drop(dev.index)


# In[121]:


train.to_json("lu_wic.train.json", indent=2, orient="records")
dev.to_json("lu_wic.dev.json", indent=2, orient="records")
test.to_json("lu_wic.test.json", indent=2, orient="records")


# In[122]:


train.shape


# In[123]:


test.shape


# In[124]:


dev.shape


# In[125]:


from sentence_transformers import SentenceTransformer


# In[126]:


model = SentenceTransformer("pierluigic/xl-lexeme")


# In[133]:


test["sentence_x"] = test.apply(lambda x: x["sentence_x"].replace(x["word_x"], f"<t>{x["word_x"]}</t>"), axis=1)
test["sentence_y"] = test.apply(lambda x: x["sentence_y"].replace(x["word_y"], f"<t>{x["word_y"]}</t>"), axis=1)


# In[135]:


embs1 = model.encode(test["sentence_x"].values)
embs2 = model.encode(test["sentence_y"].values)


# In[145]:


sim = model.similarity(embs1, embs2)


# In[156]:


from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


# In[149]:


spearmanr(sim.diag(), test.label)


# In[168]:


precision_recall_fscore_support(test.label, (sim.diag() > 0.8).long().tolist(), average="binary")


# In[167]:


accuracy_score(test.label, (sim.diag() > 0.8).long().tolist())


# In[163]:


a = sim.diag() > 0.8


# In[165]:


a.d


# In[ ]:




