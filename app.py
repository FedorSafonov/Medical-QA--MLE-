from fastapi import FastAPI
import pickle
from sentence_transformers import SentenceTransformer
from contextlib import asynccontextmanager

from ml_functions import find_similar_questions

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, idx2emd, idx2sen
    # Load the ML model
    with open("/data/idx2emb.pkl", "rb") as f:
        idx2emd = pickle.load(f)

    with open("/data/idx2sen.pkl", "rb") as f:
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
        list: Список похожих вопросов со степенью сходства, где 1 = 100% сходство.
    """
    similar_questions = find_similar_questions(question, model, idx2emd, idx2sen)
    return {"similar_questions": similar_questions}