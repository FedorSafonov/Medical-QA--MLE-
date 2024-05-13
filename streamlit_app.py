import streamlit as st
import pickle
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Загрузка модели и эмбеддингов
@st.cache_resource
def load_data():
    with open("idx2emb.pkl", "rb") as f:
        idx2emd = pickle.load(f)

    with open("idx2sen.pkl", "rb") as f:
        idx2sen = pickle.load(f)

    model = SentenceTransformer('model')
    return model, idx2emd, idx2sen

model, idx2emd, idx2sen = load_data()

# Заголовок приложения
st.title("Поиск похожих вопросов")

# Описание приложения
st.write("Введите ваш вопрос,  и мы найдем наиболее похожие вопросы из базы данных.")

# Поле ввода вопроса
question = st.text_input("Введите ваш вопрос:")

st.markdown(
    """
    <script>
    const input = document.querySelector('input[type="text"]');
    input.addEventListener('keydown', function(event) {
        if (event.key === 'Enter') {
            event.preventDefault();
            document.querySelector('button').click();
        }
    });
    </script>
    """,
    unsafe_allow_html=True,
)

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

# Кнопка "Найти похожие вопросы"
if st.button("Найти похожие вопросы"):
    # Поиск похожих вопросов
    similar_questions = find_similar_questions(question, model, idx2emd, idx2sen)

    # Отображение результатов
    st.subheader("Похожие вопросы:")
    for question, score in similar_questions:
        st.write(f"- {question} (сходство: {score * 100:.2f}%)")