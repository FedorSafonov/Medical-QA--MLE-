#!/usr/bin/env python
# coding: utf-8

# # Проект: Сервис поиска похожих вопросов для медицинских форумов
# 
# ## Описание
# 
# ### Цель
# 
# Целью данного проекта является разработка сервиса для поиска похожих вопросов на медицинских форумах. Сервис поможет удержать пользователей на платформе и увеличить количество просмотров страниц, что в свою очередь повысит эффективность рекламных кампаний и улучшит общие показатели форумов.
# 
# ### Исходные данные
# 
# Для обучения и тестирования моделей используется датасет "medical_questions_pairs" от Hugging Face.  Датасет содержит пары вопросов на английском языке, связанных с медицинской тематикой.
# 
# ### Методы
# 
# В проекте будут использованы следующие методы и технологии:
# 
# * Обработка естественного языка (NLP) для анализа и предобработки текстовых данных.
# * Машинное обучение с использованием различных моделей NLP, таких как Bag-of-Words, TF-IDF, Word2Vec, SentenceTransformers и BERT. 
# * FastAPI для создания микросервиса, предоставляющего функциональность поиска похожих вопросов.
# * Streamlit для разработки интерактивного веб-приложения для демонстрации работы сервиса.
# 
# ### Метрики оценки
# 
# Качество матчинга вопросов будет оцениваться с использованием следующих метрик: 
# 
# * Accuracy@5 - `ключевая метрика`
# * Hits@K
# * MRR@K (Mean Reciprocal Rank)
# * DCG@K (Discounted Cumulative Gain)

# ## Импорты

# In[1]:


import random
from datetime import datetime
from datasets import load_dataset
import pandas as pd
import numpy as np
import spacy
import nltk
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import dcg_score
from skimpy import skim
import time
import chime
from tqdm import tqdm
chime.theme('mario')
get_ipython().run_line_magic('load_ext', 'chime')
from sentence_transformers import SentenceTransformer, util, losses, InputExample, LoggingHandler
from gensim.models import Word2Vec
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
from pickle import dump
from torch.utils.data import DataLoader

plt.style.use('dark_background')


# ## Загрузка и обзор данных

# In[2]:


# Загрузка датасета medical_questions_pairs
try:
    train_df = pd.read_csv('train.csv')

except:
    
    dataset = load_dataset("medical_questions_pairs")

    # Преобразование датасета в pandas DataFrame
    train_df = pd.DataFrame(dataset['train'])
    train_df.to_csv('train.csv',index=False)

# Изучение структуры датасета
print('TRAIN')
print(train_df.info())
print('-------------------')
print()
print('TRAIN')
skim(train_df)
display(train_df.head(3))
print(f'Количество дубликатов: {train_df.duplicated().sum()}')
print(f'Количество уникальных question_1: {train_df.question_1.unique().shape[0]}')
print(f'Количество уникальных question_2: {train_df.question_2.unique().shape[0]}')
all_unique_questions = set(list(train_df.question_1.unique())+list(train_df.question_2.unique()))
sum_len = len(all_unique_questions)
print(f'Суммарное количество уникальных вопросов: {sum_len}')


# In[3]:


# Анализ длины вопросов
train_df['question_1_length'] = train_df['question_1'].str.len()
train_df['question_2_length'] = train_df['question_2'].str.len()

# Визуализация распределения длины вопросов
plt.figure(figsize=(10, 6))
train_df['question_1_length'].hist(bins=50, alpha=0.5, label='question_1')
train_df['question_2_length'].hist(bins=50, alpha=0.5, label='question_2')
plt.set_cmap('Greens')
plt.xlabel('Длина вопроса')
plt.ylabel('Количество')
plt.legend()
plt.show()


# In[4]:


train_df['question_1_length'].describe()


# In[5]:


train_df['question_2_length'].describe()


# ## Предобработка

# In[6]:


nltk.download('punkt')
nltk.download('stopwords')

# Загрузка модели spaCy
nlp = spacy.load("en_core_web_trf")

def preprocess_text(text):
    """
    Функция для предобработки текста.

    Args:
        text (str): Входной текст.

    Returns:
        list: Список обработанных токенов.
    """
    # Токенизация
    tokens = nltk.word_tokenize(text.lower())  # Приведение к нижнему регистру

    # Удаление пунктуации
    tokens = [token for token in tokens if token.isalnum()]

    # Удаление стоп-слов
    stop_words = nltk.corpus.stopwords.words('english')
    tokens = [token for token in tokens if token not in stop_words]

    # Лемматизация
    tokens = [token.lemma_ for token in nlp(" ".join(tokens))]

    # Обработка неизвестных слов (замена на UNK)
    tokens = [token if token != '-PRON-' else 'UNK' for token in tokens]

    return " ".join(tokens)


# In[7]:


def preprocess_dataframe(df):
    """
    Функция для предобработки DataFrame с вопросами.

    Args:
        df (pd.DataFrame): DataFrame с вопросами.

    Returns:
        pd.DataFrame: DataFrame с предобработанными вопросами.
    """
    df['processed_question_1'] = df['question_1'].apply(preprocess_text)
    df['processed_question_2'] = df['question_2'].apply(preprocess_text)
    return df


# In[8]:


try:
    train_df = pd.read_csv('prep_train.csv')
except:
    train_df = preprocess_dataframe(train_df).fillna('No')
    train_df.to_csv('prep_train.csv', index=False)


# ## Тестирование моделей
# 
# ### Функции

# In[9]:


def create_index(df, model=None, text_cols = ['processed_question_1','processed_question_2'], dump_pkl=False):
    """
    Создает индекс-эмбеддингов и словари для поиска похожих вопросов.

    Args:
        df (pd.DataFrame): DataFrame с текстовыми вопросами.
        model (str): Модель SentenceTransformer для создание эмбеддингов. Если None, то словарь индекс-имбеддингов не создаётся.
                    По умолчанию - None
        text_cols (list): Названия текстовых столбцов - 2 текстовых элемента в списке.
                        По умолчанию - ['processed_question_1','processed_question_2'].
        dump (bool): Если True, то сохраняет словари в файлы 'pkl'. По умолчанию - False.
        
    Returns:
        tuple: (словарь индекс-эмбеддингов, словарь индекс-вопрос, словарь вопрос-индекс, словарь с ground truth)
        (Или без словаря индекс-эмбеддингов, если model = None)
    """
    corpus = list(set(df[text_cols[0]].tolist() + df[text_cols[1]].tolist()))
    sen2idx = {text: i for i, text in enumerate(corpus)}
    idx2sen = {v: k for k, v in sen2idx.items()}
    if model:
        idx2emd = {i: model.encode(str(text)) for i, text in idx2sen.items()}

    if dump_pkl:
        with open("idx2emb.pkl", "wb") as fid:
            dump(idx2emd, fid)
        with open("idx2sen.pkl", "wb") as fid:
            dump(idx2sen, fid)
        with open("sen2idx.pkl", "wb") as fid:
            dump(sen2idx, fid)

    gt = defaultdict(list)
    for _, row in df.iterrows():
        idx1 = sen2idx[row[text_cols[0]]]
        idx2 = sen2idx[row[text_cols[1]]]

        if row.label:
            gt[idx1].append(idx2)
            gt[idx2].append(idx1)

    if model:    
        return idx2emd, idx2sen, sen2idx, gt
    else:
        return idx2sen, sen2idx, gt


# In[10]:


def evaluate_model(idx2emd, gt, n=5, model_type = 'SentenceTransformer', return_top_n_idx = False):
    """
    Оценивает модель SentenceTransformer на предоставленных данных.

    Args:
        idx2emd (dict): Словарь с индекс-эмбеддингами.
        gt (dict): Словарь с ground truth.
        n (int): Число, обозначающее N. По умолчанию - 5.

    Returns:
        tuple: (accuracy@n, mrr@k, dcg@k)
    """
    if model_type == 'SentenceTransformer':
        emb_list = []
        for i in range(len(idx2emd)):
            emb_list.append(idx2emd[i])
        arr = np.array(emb_list)
        dist = pdist(arr, metric="cosine")
        dist_matrix = squareform(dist)
        top_n_idx = np.argsort(dist_matrix)[:, :n+1]

    else:
        dist_matrix = cosine_similarity(idx2emd)
        mask = np.eye(dist_matrix.shape[0], dtype=bool)
        dist_matrix[mask] = 0  
        top_n_idx = np.argsort(dist_matrix)[:, -n:][:, ::-1]

    acc = []
    mrr_at_k = []
    dcg_at_k = []
    for j in range(top_n_idx.shape[0]):
        
        if model_type == 'SentenceTransformer':
            # Исключаем первый схожий вопрос
            rec_idx = top_n_idx[j, 1:]
        else:
            rec_idx = top_n_idx[j, :]

        gt_idx = gt.get(j, [])
        if len(gt_idx) > 0:
            
            # Accuracy@N
            intersection = set(rec_idx).intersection(set(gt_idx))
            acc.append(len(intersection) > 0)

            # MRR@K
            ranking = [idx in gt_idx for idx in rec_idx]
            if any(ranking):
                mrr_at_k.append(1 / (ranking.index(True) + 1))
            else:
                mrr_at_k.append(0)

            # DCG@K 
            relevance = [1 if idx in gt_idx else 0 for idx in rec_idx]
            ideal_ranking = sorted(relevance, reverse=True)
            dcg_at_k.append(dcg_score([relevance], [ideal_ranking], k=n))

    if return_top_n_idx:
        return np.mean(acc), np.mean(mrr_at_k), np.mean(dcg_at_k), top_n_idx
    else:
        return np.mean(acc), np.mean(mrr_at_k), np.mean(dcg_at_k)


def evaluate_SentenceTransformer(model_name, df, text_cols=['processed_question_1', 'processed_question_2']):
    """
    Оценивает модель SentenceTransformer на предоставленных данных.

    Args:
        model_name (str): Название модели SentenceTransformer.
        df (pd.DataFrame): DataFrame с текстовыми вопросами.
        text_cols (list): Названия текстовых столбцов - 2 текстовых элемента в списке.
                        По умолчанию - ['processed_question_1', 'processed_question_2'].

    Returns:
        tuple: (accuracy@5, mrr@5, dcg@5)
    """

    model = SentenceTransformer(model_name)
    idx2emd, _, _, gt = create_index(df, model, text_cols)
    accuracy, mrr, dcg = evaluate_model(idx2emd, gt)
    return accuracy, mrr, dcg


# ### Bag-of-Words (BoW)

# In[18]:


# Получание ground truth
_, _, gt = create_index(train_df, text_cols = ['question_1','question_2'])

# Объединение question_1 и question_2 в один DataFrame
all_questions_df = pd.DataFrame({'question': list(all_unique_questions)})

# Создание объекта CountVectorizer
vectorizer = CountVectorizer()

# Векторизация всех вопросов
all_question_vectors = vectorizer.fit_transform(all_questions_df['question'])

# Оценка качества модели
accuracy, mrr, dcg = evaluate_model(all_question_vectors, gt, model_type = 'BoW')

print('Без предобработки текста.')
print("Accuracy@5 (Bag-of-Words):", accuracy)
print("MRR@5 (Bag-of-Words):", mrr)
print("DCG@5 (Bag-of-Words):", dcg)


# In[19]:


# Получание ground truth из предобработанных вопросов
_, _, prep_gt = create_index(train_df)

# Объединение предобработанных question_1 и question_2 в один DataFrame
all_unique_prep_questions = set(
    list(train_df['processed_question_1'].unique()) +
    list(train_df['processed_question_2'].unique())
)
all_prep_questions_df = pd.DataFrame({'question': list(all_unique_prep_questions)})

# Создание объекта CountVectorizer
vectorizer = CountVectorizer()

# Векторизация всех вопросов
all_question_vectors = vectorizer.fit_transform(all_prep_questions_df['question'])

# Оценка качества модели
accuracy, mrr, dcg = evaluate_model(all_question_vectors, prep_gt, model_type = 'BoW')
print('С предобработкой текста.')
print("Accuracy@5 (Bag-of-Words):", accuracy)
print("MRR@5 (Bag-of-Words):", mrr)
print("DCG@5 (Bag-of-Words):", dcg)


# ***Выводы по результатам Bag-of-Words:***
# 
# * **Предобработка текста значительно улучшает качество модели Bag-of-Words.** Accuracy@5 увеличилась с 0.57 до 0.80, MRR@5 - с 0.47 до 0.68, а DCG@5 - с 0.49 до 0.70. Это говорит о том, что предобработка текста, такая как токенизация, лемматизация и удаление стоп-слов, помогает модели лучше понимать семантическое сходство между вопросами.
# 
# * **Модель Bag-of-Words может быть хорошим выбором, если важна простота и скорость работы.** Она требует меньше ресурсов для обучения и использования, чем более сложные модели такие как SentenceTransformers.
# 
# *Код для создания словаря индекс-эмбеддингов (если нужно):*
# 
# ```python
# idx2emd = {i: all_prep_questions_df[i].toarray()[0] for i in range(all_prep_questions_df.shape[0])}
# ```

# ### TF-IDF

# In[58]:


# Создание объекта CountVectorizer
vectorizer = TfidfVectorizer()

# Векторизация всех вопросов
all_question_vectors = vectorizer.fit_transform(all_prep_questions_df['question'])

# Оценка качества модели
accuracy, mrr, dcg = evaluate_model(all_question_vectors, prep_gt, model_type='TF-IDF')
print('С предобработкой текста.')
print("Accuracy@5 (TF-IDF):", accuracy)
print("MRR@5 (TF-IDF):", mrr)
print("DCG@5 (TF-IDF):", dcg)


# ***Выводы по результатам TF-IDF:***
# 
# * **TF-IDF превосходит Bag-of-Words по всем метрикам.** Accuracy@5 увеличилась с 0.80 до 0.86, MRR@5 - с 0.68 до 0.73, а DCG@5 - с 0.70 до 0.75. Это говорит о том, что веса TF-IDF, которые учитывают важность слов в коллекции документов, позволяют модели лучше различать похожие и непохожие вопросы.
# * **TF-IDF может быть хорошим компромиссом между качеством и сложностью.** Она обеспечивает значительно лучшее качество, чем Bag-of-Words, при этом оставаясь относительно простой и быстрой в использовании.
# 
# *Код для создания словаря индекс-эмбеддингов (если нужно):*
# 
# ```python
# idx2emd = {i: all_prep_questions_df[i].toarray()[0] for i in range(all_prep_questions_df.shape[0])}
# ```

# ### Word2Vec

# In[18]:


def vectorize_sentence(sentence, model_w2v):
    """
    Векторизует предложение с помощью Word2Vec, усредняя векторы слов.

    Args:
        sentence (str): Предложение для векторизации.
        model_w2v (gensim.models.Word2Vec): Модель Word2Vec.

    Returns:
        numpy.ndarray: Векторное представление предложения.
    """
    words = sentence.split()
    word_vectors = [model_w2v.wv[word] for word in words if word in model_w2v.wv]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model_w2v.vector_size)


# In[69]:


# Объединение всех вопросов в один список
all_questions = train_df['processed_question_1'].tolist() + train_df['processed_question_2'].tolist()

# Разделение вопросов на списки токенов
all_tokens = [question.split() for question in all_questions]

# Обучение Word2Vec
model_w2v = Word2Vec(sentences=all_tokens, vector_size=512, window=5, min_count=5, workers=4)

# Векторизация всех вопросов
all_question_embeddings = []
for question in all_prep_questions_df['question']:
    all_question_embeddings.append(vectorize_sentence(question, model_w2v))

# Создание словаря idx2emd
idx2emd = {i: embedding for i, embedding in enumerate(all_question_embeddings)}

# Оценка качества модели
accuracy, mrr, dcg = evaluate_model(all_question_embeddings, prep_gt, model_type='Word2Vec')
print('С предобработкой текста.')
print("Accuracy@5 (Word2Vec):", accuracy)
print("MRR@5 (Word2Vec):", mrr)
print("DCG@5 (Word2Vec):", dcg) 


# ***Выводы по результатам Word2Vec:***
# 
# * **Word2Vec показал значительно худшие результаты, чем Bag-of-Words и TF-IDF по всем метрикам.** 
#    Это говорит о том, что усреднение векторов слов, полученных из Word2Vec, 
#    не является эффективным способом представления семантического сходства между вопросами в нашей задаче.
# * **Возможные причины:** 
#    * **Потеря информации при усреднении:** Усреднение векторов слов может привести к потере информации о порядке слов и синтаксической структуре предложения. 
#    * **Размер датасета:** Наш датасет может быть недостаточно большим для обучения качественной модели Word2Vec. 
#    * **Выбор параметров Word2Vec:** Возможно, выбранные параметры (размерность векторов, размер окна контекста, минимальная частота слов) не оптимальны для нашей задачи. (Были опробованы и другие параметры, но их изменение не привело к сильному улучшению метрики. В данном коде оставлен лучший вариант.)
# 
# * **В целом, результаты показывают, что модель Word2Vec не подходит для нашей задачи поиска похожих вопросов.**  
#    Мы можем попробовать использовать другие модели, которые лучше учитывают семантику и структуру предложений.

# ### USE

# In[1]:


import tensorflow_hub as hub

# URL модели USE
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 

# Загрузка модели
model_use = hub.load(module_url)


# In[24]:


# Векторизация всех вопросов
all_question_embeddings = model_use(all_prep_questions_df['question']).numpy()

# Создание словаря idx2emd
idx2emd = {i: embedding for i, embedding in enumerate(all_question_embeddings)}

# Оценка качества модели
accuracy, mrr, dcg = evaluate_model(all_question_embeddings, prep_gt, model_type='USE')
print('С предобработкой текста.')
print("Accuracy@5 (USE):", accuracy)
print("MRR@5 (USE):", mrr)
print("DCG@5 (USE):", dcg) 


# In[22]:


# Векторизация всех вопросов
all_question_embeddings = model_use(all_questions_df['question']).numpy()

# Создание словаря idx2emd
idx2emd = {i: embedding for i, embedding in enumerate(all_question_embeddings)}

# Оценка качества модели
accuracy, mrr, dcg = evaluate_model(all_question_embeddings, gt, model_type='USE')
print('Без предобработки текста.')
print("Accuracy@5 (USE):", accuracy)
print("MRR@5 (USE):", mrr)
print("DCG@5 (USE):", dcg) 


# ***Выводы по результатам USE:***
# 
# * **USE  показал отличные результаты,  как на исходных,  так и на предобработанных данных.** 
#    Accuracy@5  для обоих вариантов выше,  чем у Bag-of-Words и TF-IDF.
# * **Предобработка текста не привела к значительному улучшению или ухудшению качества.**  
#    Это может говорить о том,  что USE  уже достаточно хорошо справляется с обработкой текста и не так чувствительна к предобработке,  как более простые модели.
# * **USE  может быть хорошим выбором,  если нужно высокое качество и простота использования.**  
#    Модель уже предобучена и готова к использованию,  не требуя дополнительного обучения. 
# 
# * **В целом,  результаты показывают,  что USE  - это эффективная модель для поиска похожих вопросов,  которая обеспечивает высокое качество и простоту использования.**

# ### SentenceTransformers

# In[9]:


def colorize_metrics_df(df):
    return (df.sort_values('DCG@5', ascending=0)
            .reset_index(drop=True)
            .style
            .set_caption("Метрики моделей SentenceTransformer")
            .format("{:.2%}",subset=metrics_cols)
            .format(precision=2,subset=['time'])
            .background_gradient(cmap='summer',subset=metrics_cols)
            .background_gradient(cmap='summer_r',subset=['time']))


# In[16]:


get_ipython().run_cell_magic('time', '', "%%chime\n\nmodel_names = [ 'all-MiniLM-L6-v2',\n                'paraphrase-MiniLM-L3-v2',\n                'all-MiniLM-L12-v2',\n                'multi-qa-MiniLM-L6-cos-v1',\n                'paraphrase-multilingual-mpnet-base-v2',\n                'all-distilroberta-v1',\n                'multi-qa-distilbert-cos-v1',\n                'multi-qa-mpnet-base-dot-v1',\n                'all-mpnet-base-v2'\n                ]\n\nmetrics_cols = ['Accuracy@5','MMR@5','DCG@5']\n\ntry:\n    metrics = metrics.dropna()\n    counter = metrics.shape[0]+1\nexcept:\n    try:\n        metrics = pd.read_csv('metrics.csv').dropna()\n        counter = metrics.shape[0]+1\n    except:\n        metrics = pd.DataFrame(columns=['model','text_type'])\n        counter = 0\n    \n\nfor text_type in ['Изначальный', 'Предобработаный']:\n    for model_name in model_names:\n\n        if (metrics.shape[0] == 0 or\n            model_name+text_type not in (metrics.model + metrics.text_type).to_list()):\n\n            metrics.loc[counter,'model'] = model_name\n            metrics.loc[counter,'text_type'] = text_type\n            \n            if text_type == 'Изначальный':\n                text_cols=['question_1', 'question_2']\n            else:\n                text_cols=['processed_question_1','processed_question_2']\n\n            start = time.time()\n            accuracy, mrr, dcg = evaluate_SentenceTransformer(model_name, train_df, text_cols)\n            end = time.time()\n\n            metrics.loc[counter,'Accuracy@5'] = round(accuracy, 4)\n            metrics.loc[counter,'MMR@5'] = round(mrr, 4)\n            metrics.loc[counter,'DCG@5'] = round(dcg, 4)\n            metrics.loc[counter,'time'] = round(end-start, 4)\n\n            counter +=1\n            \n            metrics_colored = colorize_metrics_df(metrics)\n\n            display(metrics_colored, clear = True)\n\n        else:\n            metrics_colored = colorize_metrics_df(metrics)\n            pass\n\ndisplay(metrics_colored, clear = True)\n")


# In[14]:


metrics.to_csv('metrics.csv', index = False)


# **Наблюдения:**
# * **Все модели SentenceTransformer показали более высокое качество по всем метрикам чем дергие модели, рассмотренные в этом проекте.** При этом скорость работы этих моделей значительно ниже, но так как для данной задачи скорость работы всё ранво остаётся в пределах нормы, то выбор лучшей модели будет осуществляться среди моделей SentenceTransformer.
# * ***Модель all-mpnet-base-v2 показала наилучшую точность (Accuracy@5) как на исходных, так и на предобработанных данных. Точность на исходных данных - 96.82%*** Это говорит о том, что эта модель хорошо подходит для нашей задачи поиска похожих вопросов.
# * ***Предобработка текста не привела к улучшению точности.*** В большинстве случаях, точность даже немного снизилась. Это может быть связано с тем, что предобработка удаляет некоторую информацию, которая может быть полезна для модели.
# * ***Модели среднего размера (all-MiniLM-L6-v2, all-MiniLM-L12-v2) также показали хорошую точность. Они могут быть хорошим выбором, если важна скорость работы.***
# * *Модели, специально обученные для задач Question Answering (multi-qa-MiniLM-L6-cos-v1, multi-qa-distilbert-cos-v1, multi-qa-mpnet-base-dot-v1), показали немного худшую точность, чем all-mpnet-base-v2, all-MiniLM-L6-v2 и all-MiniLM-L12-v2.*
# * Метрики `MMR@5`и`DCG@5` на уровене 90%+ (0.9+) указывают на то что в большинстве случаев модель ранжирует правильный вопрос на первое место и реже на второе и ниже.
# 
# **Выбор модели:**
# - Для дальнейшей разработки я буду использовать модель **`all-MiniLM-L6-v2`**, которая показала почти такой же результат (точность 96.36%) что и `all-mpnet-base-v2`, но при этом работает почти в 6 раз быстрее.

# ## Fine-tuning модели SentenceTransformers

# In[12]:


best = 'all-MiniLM-L6-v2'


# In[18]:


def print_metrics(idx2emd, gt, n = 5):
    """
    Печатет метрики модели.

    Args:
        idx2emd (dict): Словарь с индекс-эмбеддингами.
        gt (dict): Словарь с ground truth.
        n (int): Число, обозначающее N. По умолчанию - 5.

    Returns:
        Ничего не вовзращает, только печетает.
        Для получения метрик нужно использовать функцию evaluate_model().
    """
    

    # Вызов функции и evaluate_model
    accuracy, mrr, dcg = evaluate_model(idx2emd, gt, n)

    # Печать результатов
    print(f"Accuracy@{n}: {accuracy:.4f}")
    print(f"MRR@{n}: {mrr:.4f}")
    print(f"DCG@{n}: {dcg:.4f}")


# In[31]:


# Создание списка InputExample
train_data = []
for index, row in train_df.iterrows():
    train_data.append(InputExample(texts=[row['question_1'], row['question_2']], label=float(row['label'])))

# Создание DataLoader
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16)


# In[32]:


get_ipython().run_cell_magic('time', '', '%%chime\n\n# Создание объекта SentenceTransformer\ntuned_CosSim_model = SentenceTransformer(best)\n\n# Определение loss-функции\ntrain_loss = losses.CosineSimilarityLoss(tuned_CosSim_model)\n\n# Fine-tuning модели\ntuned_CosSim_model.fit(train_objectives=[(train_dataloader, train_loss)],\n                                        epochs=3)\n')


# In[33]:


idx2emd, _, _, gt = create_index(train_df, tuned_CosSim_model, text_cols = ['question_1','question_2'])

print_metrics(idx2emd, gt, 5)


# * **Метрики ухудшились, попробую использовать другую loss-функцию - `TripletLoss`**

# In[26]:


def create_triplets(df, triplet_type = 'InputExample'):
    """
    Создает триплеты данных для TripletLoss.

    Args:
        df (pd.DataFrame): DataFrame с вопросами и метками.

    Returns:
        list: Список объектов InputExample.
    """
    triplets = []

    for anchor in train_df.question_1.unique():
        for positive in train_df.query('label == 1').loc[train_df['question_1']==anchor,'question_2'].to_list():
            for negative in train_df.query('label == 0').loc[train_df['question_1']==anchor,'question_2'].to_list():
            
                if triplet_type == 'InputExample':
                    triplets.append(InputExample(texts=[anchor, positive, negative]))
                    
                elif triplet_type =='tuple':
                    triplets.append((anchor, positive, negative))
                
    return triplets


# In[27]:


def evaluate_triplets(triplets, model):
    """
    Оценивает модель SentenceTransformer на триплетах данных.

    Args:
        triplets (list): Список триплетов (anchor, positive, negative).
        model (SentenceTransformer): Объект SentenceTransformer.

    Returns:
        float: Доля правильно классифицированных триплетов.
    """
    correct = 0
    for anchor, positive, negative in triplets:
        
        # Векторизация предложений
        anchor_embedding = model.encode(anchor)
        positive_embedding = model.encode(positive)
        negative_embedding = model.encode(negative)

        # Вычисление расстояний
        positive_distance = cosine_similarity(anchor_embedding.reshape(1, -1), positive_embedding.reshape(1, -1))[0][0]
        negative_distance = cosine_similarity(anchor_embedding.reshape(1, -1), negative_embedding.reshape(1, -1))[0][0]

        # Проверка условия
        if positive_distance > negative_distance:
            correct += 1

    return correct / len(triplets)


# In[34]:


get_ipython().run_cell_magic('time', '', "%%chime\n\n# Создание триплетов данных\ntrain_triplets = create_triplets(train_df)\n\n# Создание DataLoader\ntrain_dataloader = DataLoader(train_triplets, shuffle=True, batch_size=16)\n\n# Создание объекта SentenceTransformer\ntuned_TripletLoss_model = SentenceTransformer(best)\n\n# Определение loss-функции\ntrain_loss = losses.TripletLoss(model=tuned_TripletLoss_model)\n\n# Fine-tuning модели\ntuned_TripletLoss_model.fit(train_objectives=[(train_dataloader, train_loss)], \n                            epochs=3, \n                            warmup_steps=100\n                            )\n\n# Оценка качества модели\nidx2emd, _, _, gt = create_index(train_df, tuned_TripletLoss_model,\n                                text_cols = ['question_1','question_2'])\n\nprint_metrics(idx2emd, gt, 5)\n")


# In[40]:


tuple_train_triplets = create_triplets(train_df, triplet_type='tuple')

base_model = SentenceTransformer(best)

models = [base_model, tuned_CosSim_model, tuned_TripletLoss_model]
names = ['Базовая модель', 'tuned_CosSim_model', 'tuned_TripletLoss_model']

for model, name in zip(models,names):
    AccuracyTriplet =  evaluate_triplets(tuple_train_triplets, model)

    print(f'{name} AccuracyTriplet: {AccuracyTriplet:.2%}')


# * **Базовая модель без дообучения показывает более высокую метрику `AccuracyTriplet` и значительно лучшие показатели целевых метрик `Accuracy@5, MRR@5, DCG@5`, чем модель дообученная с помощью `TripletLoss`***
# * **Базовая модель уступает по метрике `AccuracyTriplet` модели дообученой с помощью `CosineSimilarityLoss`, но показывает лучшие показатели целевых метрик `Accuracy@5, MRR@5, DCG@5`**, это означает что дообученная модель правильнее расставляет приоритеты этих вопросов относительно друг-друга, но правильный вопрос реже попадает в топ-5 и оказывается на первых местах.
# 
# * В будущем можно попробовать использовать MarginRankingLoss и MultipleNegativesRankingLoss которые подходят для оптимизации MRR@5 и DCG@5, так как они направлены на улучшение ранжирования примеров.
# 
# * ***Для построения приложения и микро-сервиса принято решение использовать базовую предобученую модель SentenceTransformer `all-MiniLM-L6-v2`.***
# 
# *AccuracyTriplet - доля правильно классифицированных триплетов. Правильно классифицированный триплет - триплет в котором косиносовое сходство между между заданным вопросом и истинным схожим больше чем с неправильным.
# 
# .
# 
# Рассмотрим примеры выдачи топ-5 схожих вопросов и заодно сохраним получившиеся эмбеддинги.

# In[114]:


N = 5

idx2emd, idx2sen, sen2idx, gt = create_index(train_df, base_model,
                                             text_cols=['question_1', 'question_2'],
                                             dump_pkl=True)

# Получение предсказаний (матрица top_n_idx)
_, _, _, top_n_idx = evaluate_model(idx2emd, gt, return_top_n_idx=True)  # Нам нужна только матрица top_n_idx

# Анализ предложенных вопросов
for i in range(len(top_n_idx[:N+1])):
    predicted_indices = top_n_idx[i,1:]
    true_indices = gt.get(i, [])
    if set(predicted_indices).intersection(set(true_indices)):
        print("Вопрос:")
        display(idx2sen[i])
        print("Предсказанные похожие вопросы:")
        display([idx2sen[j] for j in predicted_indices])
        print("Истинный похожий вопрос:")
        display([idx2sen[j] for j in true_indices])
        print("------------------------------------------") 


# Видно, что модель и правда ставит истинный похожий вопрос на 1-2 позицию.

# ## Сохранение лучшей модели

# In[14]:


base_model.save('model')

