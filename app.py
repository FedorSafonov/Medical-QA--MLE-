from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import joblib
import uvicorn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests

with open("idx2emb.pkl", "rb") as f:
    idx2emb = pickle.load(f)

with open("idx2sen.pkl", "rb") as f:
    idx2sen = pickle.load(f)

with open("sen2idx.pkl", "rb") as f:
    sen2idx = pickle.load(f)

model = joblib.load('model.pkl')

def calculate_similarity(df, model):
    results = []
    for _, rec in df.iterrows():
        emb1 = model.encode([rec.question_1])
        emb2 = model.encode([rec.question_2])
        cos_sim = cosine_similarity(emb1, emb2)
        results.append({"id": rec.dr_id, "label": rec.label, "dist": cos_sim[0][0]})
    return results

class Question(BaseModel):
    question: str

def find_similar_questions(question: str, idx2emb, idx2sen, sen2idx):
    emb = model.encode([question])
    cos_sim = cosine_similarity(emb, list(idx2emb.values()))
    top5_idx = np.argsort(cos_sim)[0][-5:]
    similar_questions = [idx2sen[idx] for idx in top5_idx]
    return similar_questions

app = FastAPI()

@app.post("/similar-questions")
async def find_similar_questions(question: Question):
    similar_questions = find_similar_questions(question.question, idx2emb, idx2sen, sen2idx)
    return {"similar_questions": similar_questions}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

question = "How to treat the flu?"
data = {"question": question}

response = requests.post("http://localhost:8000/similar-questions", json=data)

if response.status_code == 200:
    similar_questions = response.json()["similar_questions"]
    print(similar_questions)
else:
    print(f"Error: {response.status_code} - {response.text}")