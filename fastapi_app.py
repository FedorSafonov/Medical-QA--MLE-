from fastapi import FastAPI
import pickle
from sentence_transformers import SentenceTransformer, util
import numpy as np
from contextlib import asynccontextmanager


def find_similar_questions(question, model, idx2emd, idx2sen, top_n=5):
    """
    Находит n наиболее похожих вопросов.

    Args:
        question (str): Входной вопрос.
        model (SentenceTransformer): Модель SentenceTransformers.
        idx2emd (dict): Словарь индекс-эмбеддинг.
        idx2sen (dict): Словарь индекс-вопрос.
        top_n (int, optional): Количество возвращаемых похожих вопросов. Defaults to 5.

    Returns:
        list: Список кортежей (вопрос,  оценка сходства).
    """
    question_embedding = model.encode(question)
    similarities = [util.cos_sim(question_embedding, emb) for emb in idx2emd.values()]
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    return [(idx2sen[idx], similarities[idx].item()) for idx in top_indices]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, idx2emd, idx2sen
    # Load the ML model
    with open("idx2emb.pkl", "rb") as f:
        idx2emd = pickle.load(f)

    with open("idx2sen.pkl", "rb") as f:
        idx2sen = pickle.load(f)

    model = SentenceTransformer('model')
    yield
    # Clean up the ML models and release the resources
    model.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/similar_questions")
async def get_similar_questions(question: str):
    """
    Endpoint для поиска похожих вопросов.

    Args:
        question (str): Входной вопрос.

    Returns:
        list: Список похожих вопросов со степенью сходства, где 1 - 100% сходство.
    """
    similar_questions = find_similar_questions(question, model, idx2emd, idx2sen)
    return {"similar_questions": similar_questions}