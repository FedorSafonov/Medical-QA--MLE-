from sentence_transformers import SentenceTransformer
import faiss
import os


def load_model_and_create_index(model_name, df):
    """
    Загружает модель SentenceTransformer и создает индекс Faiss.

    Args:
        model_name (str): Название модели SentenceTransformer.
        df (pd.DataFrame): DataFrame с вопросами.

    Returns:
        tuple: (SentenceTransformer model, faiss index)
    """
    model_path = f"{model_name}.model"
    index_path = f"{model_name}.index"

    if os.path.exists(model_path) and os.path.exists(index_path):
        model = SentenceTransformer(model_path)
        index = faiss.read_index(index_path)
    else:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(df['question_1'].tolist())
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        model.save(model_path)
        faiss.write_index(index, index_path)

    return model, index