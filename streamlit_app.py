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

# Словарь с переводами
translations = {
    "ru": {
        "title": "Поиск похожих медицинских вопросов",
        "description": "Введите ваш вопрос, и мы найдем наиболее похожие вопросы из базы данных.",
        "input_label": "Введите ваш вопрос:",
        "button_label": "Найти похожие вопросы",
        "results_header": "Похожие вопросы:",
        "similarity_label": "сходство"
    },
    "en": {
        "title": "Similar Medical Questions Search",
        "description": "Enter your question, and we will find the most similar questions from the database.",
        "input_label": "Enter your question:",
        "button_label": "Find Similar Questions",
        "results_header": "Similar Questions:",
        "similarity_label": "similarity"
    }
}

# Инициализация состояния языка
if 'language' not in st.session_state:
    st.session_state.language = 'ru'

# Кнопка выбора языка
selected_language = st.radio("Language / Язык", ['Русский', 'English'], index=0 if st.session_state.language == 'ru' else 1)
if selected_language == 'Русский':
    st.session_state.language = 'ru'
else:
    st.session_state.language = 'en'

# Получение переводов для выбранного языка
lang = st.session_state.language
texts = translations[lang]

# Заголовок приложения
st.title(texts['title'])

# Описание приложения
st.write(texts['description'])

# Поле ввода вопроса
question = st.text_input(texts['input_label'])

# JavaScript код для поиска по Enter
st.markdown(
    """
    <script>
    const input = document.querySelector('input[type="text"]');
    input.addEventListener('keyup', function(event) {
        if (event.keyCode === 13) {
            event.preventDefault();
            document.querySelector('button[kind="primary"]').click();
        }
    });
    </script>
    """,
    unsafe_allow_html=True,
)

# Кнопка "Найти похожие вопросы"
if st.button(texts["button_label"]):
    # Поиск похожих вопросов
    similar_questions = find_similar_questions(question, model, idx2emd, idx2sen)

    # Отображение результатов
    st.subheader(texts["results_header"])
    for question, score in similar_questions:
        st.write(f"- {question} ({texts['similarity_label']}: {score * 100:.2f}%)")