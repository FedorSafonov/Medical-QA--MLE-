import asyncio
import httpx
from fastapi import FastAPI, Depends, Query
from similar_questions import index as sq_index
import pandas as pd


app = FastAPI()

# Загрузка датасета и создание индекса
df = pd.read_csv("prep_train.csv")  # Путь к вашему датасету
model, index = sq_index.load_model_and_create_index("all-MiniLM-L6-v2", df)


@app.get("/search")
async def search(
    query: str = Query(..., description="Запрос для поиска"), top_n: int = 5
):
    """
    Поиск похожих вопросов.

    Args:
        query (str): Запрос для поиска.
        top_n (int): Количество возвращаемых результатов.

    Returns:
        list: Список из top_n похожих вопросов.
    """
    query_embedding = model.encode(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_n)
    results = [
        {"question": df['question_1'].iloc[idx], "score": score}
        for score, idx in zip(distances[0], indices[0])
    ]
    return results


async def test_search():
    # Тестовый запрос
    test_query = """i recently lost my job and the love of my life. The RN who saw me yesterday said I have high blood pressure. Will I be diagnosed with depression?"""

    # Асинхронный клиент HTTP
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=10.0) as client:
        # Отправка GET-запроса
        response = await client.get("/search", params={"query": test_query, "top_n": 5})

        # Вывод результатов
        print(response.json())


@app.on_event("startup")
async def startup_event():
    await test_search()