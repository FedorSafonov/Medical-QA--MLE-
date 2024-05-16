import numpy as np
from sentence_transformers import util

def find_similar_questions(question, model, idx2emd, idx2sen, top_n=5):
    """
    Находит n наиболее похожих вопросов.

    Args:
        question (str): Входной вопрос.
        model (SentenceTransformer): Модель SentenceTransformers.
        idx2emd (dict): Словарь индекс-эмбеддинг.
        idx2sen (dict): Словарь индекс-вопрос.
        top_n (int, optional): Количество возвращаемых похожих вопросов. По умолчанию 5.

    Returns:
        list: Список кортежей (вопрос, оценка сходства).
    """
    question_embedding = model.encode(question)
    similarities = [util.cos_sim(question_embedding, emb) for emb in idx2emd.values()]
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    return [(idx2sen[idx], similarities[idx].item()) for idx in top_indices]