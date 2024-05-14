# Сервис поиска похожих вопросов для медицинских форумов

- Полный отчёт по проекту можно посмотреть в [report.md](https://github.com/FedorSafonov/Medical-QA--MLE-/blob/reveiw/report.md)
- [Ссылка на веб приложение в streamlit](https://fedorsafonov-medical-qa--mle--streamlit-app-agoi6t.streamlit.app/)

## Описание

### Задача поиска похожих вопросов актуальна во многих областях,  таких как онлайн-форумы,  системы вопросов-ответов и чат-боты.  Она позволяет пользователям быстро находить информацию,  которая уже была обсуждена,  избегая дублирования вопросов  и повышая эффективность коммуникации.

### Цель

Целью данного проекта является разработка сервиса для поиска похожих вопросов на медицинских форумах. Сервис поможет удержать пользователей на платформе и увеличить количество просмотров страниц, что в свою очередь повысит эффективность рекламных кампаний и улучшит общие показатели форумов.

### Исходные данные

Для обучения и тестирования моделей используется датасет "medical_questions_pairs" от Hugging Face.  Датасет содержит пары вопросов на английском языке, связанных с медицинской тематикой.

### Методы

В проекте были использованы следующие методы и технологии:

* Обработка естественного языка (NLP)  для анализа и предобработки текстовых данных,  включая токенизацию,  лемматизацию и удаление стоп-слов.
* Машинное обучение с использованием различных моделей NLP:
    * **SentenceTransformers:**  `all-mpnet-base-v2`,  `all-MiniLM-L6-v2`,  `all-MiniLM-L12-v2`,  `paraphrase-multilingual-mpnet-base-v2`,  `multi-qa-MiniLM-L6-cos-v1`,  `all-distilroberta-v1`,  `multi-qa-distilbert-cos-v1`,  `multi-qa-mpnet-base-dot-v1`.
    * **Bag-of-Words:**  Классическая модель с использованием  `CountVectorizer`  из scikit-learn.
    * **TF-IDF:**  Классическая модель с использованием  `TfidfVectorizer`  из scikit-learn.
    * **Word2Vec:**  Модель для обучения word embeddings с использованием `gensim.models.Word2Vec`.
    * **Universal Sentence Encoder (USE):**  Модель от Google  для создания семантических представлений предложений. 
* FastAPI  для создания микросервиса,  предоставляющего функциональность поиска похожих вопросов.
* Streamlit  для разработки интерактивного веб-приложения для демонстрации работы сервиса.

### Метрики оценки

Качество матчинга вопросов будет оцениваться с использованием следующих метрик: 

* **Accuracy@5:**  Доля вопросов,  для которых хотя бы один из 5  наиболее похожих предсказанных вопросов совпадает с истинным похожим вопросом.  **Ключевая метрика.**
* **MRR@K (Mean Reciprocal Rank):**  Средний обратный ранг первого правильно предсказанного похожего вопроса.
* **DCG@K (Discounted Cumulative Gain):**  Метрика,  которая учитывает не только наличие правильного ответа в топ-5,  но и его позицию в списке.
* **AccuracyTriplet:**  Доля правильно классифицированных триплетов.  Правильно классифицированный триплет - триплет,  в котором косинусное сходство между заданным вопросом и истинным похожим вопросом больше, чем с не похожим. 

## Результаты

* Результаты всех моделей представлены в таблицах ниже:

| Модель                      | Accuracy@5 | MRR@5     | DCG@5     | AccuracyTriplet |
|-----------------------------|------------|-----------|-----------|-------------|
| SentenceTransformers (базовая - all-MiniLM-L6-v2) | 0.9682     | 0.9039    | 0.9167    | 0.9482      |
| SentenceTransformers (fine-tune CosineSimilarityLoss) | 0.9242     | 0.8572    | 0.8657    | 0.9639     |
| SentenceTransformers (fine-tune  TripletLoss)       | 0.3320     | 0.2687    | 0.2805    | 0.8983     |
| Bag-of-Words (без предобработки)          | 0.5686     | 0.4736    | 0.4879    |         |
| Bag-of-Words (с предобработкой)          | 0.8036     | 0.6804    | 0.6968    |         |
| TF-IDF (с предобработкой)               | 0.8619     | 0.7321    | 0.7469    |         |
| Word2Vec (с предобработкой)              | 0.4355     | 0.3589    | 0.3710    |        |
| USE (без предобработки)                   | 0.8888     | 0.7894    | 0.8018    |        |
| USE (с предобработкой)                   | 0.8709     | 0.7615    | 0.7734    |        |

<style type="text/css">
#T_0a5c8_row0_col2, #T_0a5c8_row0_col3, #T_0a5c8_row0_col4, #T_0a5c8_row12_col5 {
  background-color: #ffff66;
  color: #000000;
}
#T_0a5c8_row0_col5, #T_0a5c8_row14_col2, #T_0a5c8_row17_col3, #T_0a5c8_row17_col4 {
  background-color: #008066;
  color: #f1f1f1;
}
#T_0a5c8_row1_col2 {
  background-color: #eff766;
  color: #000000;
}
#T_0a5c8_row1_col3 {
  background-color: #f3f966;
  color: #000000;
}
#T_0a5c8_row1_col4 {
  background-color: #eef666;
  color: #000000;
}
#T_0a5c8_row1_col5, #T_0a5c8_row2_col4, #T_0a5c8_row4_col5 {
  background-color: #eaf466;
  color: #000000;
}
#T_0a5c8_row2_col2 {
  background-color: #ddee66;
  color: #000000;
}
#T_0a5c8_row2_col3 {
  background-color: #edf666;
  color: #000000;
}
#T_0a5c8_row2_col5 {
  background-color: #bede66;
  color: #000000;
}
#T_0a5c8_row3_col2 {
  background-color: #d3e966;
  color: #000000;
}
#T_0a5c8_row3_col3 {
  background-color: #e8f366;
  color: #000000;
}
#T_0a5c8_row3_col4 {
  background-color: #e5f266;
  color: #000000;
}
#T_0a5c8_row3_col5 {
  background-color: #9cce66;
  color: #000000;
}
#T_0a5c8_row4_col2 {
  background-color: #dbed66;
  color: #000000;
}
#T_0a5c8_row4_col3 {
  background-color: #cee666;
  color: #000000;
}
#T_0a5c8_row4_col4 {
  background-color: #cbe566;
  color: #000000;
}
#T_0a5c8_row5_col2 {
  background-color: #c6e266;
  color: #000000;
}
#T_0a5c8_row5_col3 {
  background-color: #c8e366;
  color: #000000;
}
#T_0a5c8_row5_col4 {
  background-color: #c9e466;
  color: #000000;
}
#T_0a5c8_row5_col5 {
  background-color: #038166;
  color: #f1f1f1;
}
#T_0a5c8_row6_col2 {
  background-color: #b5da66;
  color: #000000;
}
#T_0a5c8_row6_col3, #T_0a5c8_row6_col4 {
  background-color: #a9d466;
  color: #000000;
}
#T_0a5c8_row6_col5 {
  background-color: #9acc66;
  color: #000000;
}
#T_0a5c8_row7_col2, #T_0a5c8_row9_col4 {
  background-color: #4fa766;
  color: #f1f1f1;
}
#T_0a5c8_row7_col3, #T_0a5c8_row7_col4 {
  background-color: #74ba66;
  color: #f1f1f1;
}
#T_0a5c8_row7_col5 {
  background-color: #178b66;
  color: #f1f1f1;
}
#T_0a5c8_row8_col2 {
  background-color: #71b866;
  color: #f1f1f1;
}
#T_0a5c8_row8_col3, #T_0a5c8_row8_col4 {
  background-color: #57ab66;
  color: #f1f1f1;
}
#T_0a5c8_row8_col5 {
  background-color: #b6db66;
  color: #000000;
}
#T_0a5c8_row9_col2 {
  background-color: #7ebe66;
  color: #000000;
}
#T_0a5c8_row9_col3 {
  background-color: #50a866;
  color: #f1f1f1;
}
#T_0a5c8_row9_col5 {
  background-color: #e4f266;
  color: #000000;
}
#T_0a5c8_row10_col2 {
  background-color: #64b266;
  color: #f1f1f1;
}
#T_0a5c8_row10_col3 {
  background-color: #48a366;
  color: #f1f1f1;
}
#T_0a5c8_row10_col4 {
  background-color: #45a266;
  color: #f1f1f1;
}
#T_0a5c8_row10_col5 {
  background-color: #aad466;
  color: #000000;
}
#T_0a5c8_row11_col2 {
  background-color: #5faf66;
  color: #f1f1f1;
}
#T_0a5c8_row11_col3 {
  background-color: #44a266;
  color: #f1f1f1;
}
#T_0a5c8_row11_col4 {
  background-color: #41a066;
  color: #f1f1f1;
}
#T_0a5c8_row11_col5 {
  background-color: #cfe766;
  color: #000000;
}
#T_0a5c8_row12_col2, #T_0a5c8_row15_col5 {
  background-color: #56ab66;
  color: #f1f1f1;
}
#T_0a5c8_row12_col3 {
  background-color: #359a66;
  color: #f1f1f1;
}
#T_0a5c8_row12_col4 {
  background-color: #369a66;
  color: #f1f1f1;
}
#T_0a5c8_row13_col2 {
  background-color: #43a166;
  color: #f1f1f1;
}
#T_0a5c8_row13_col3 {
  background-color: #269266;
  color: #f1f1f1;
}
#T_0a5c8_row13_col4 {
  background-color: #259266;
  color: #f1f1f1;
}
#T_0a5c8_row13_col5 {
  background-color: #b9dc66;
  color: #000000;
}
#T_0a5c8_row14_col3, #T_0a5c8_row15_col3, #T_0a5c8_row15_col4 {
  background-color: #138966;
  color: #f1f1f1;
}
#T_0a5c8_row14_col4, #T_0a5c8_row16_col2 {
  background-color: #148a66;
  color: #f1f1f1;
}
#T_0a5c8_row14_col5 {
  background-color: #47a366;
  color: #f1f1f1;
}
#T_0a5c8_row15_col2 {
  background-color: #2e9666;
  color: #f1f1f1;
}
#T_0a5c8_row16_col3 {
  background-color: #0f8766;
  color: #f1f1f1;
}
#T_0a5c8_row16_col4 {
  background-color: #108866;
  color: #f1f1f1;
}
#T_0a5c8_row16_col5 {
  background-color: #3e9e66;
  color: #f1f1f1;
}
#T_0a5c8_row17_col2 {
  background-color: #219066;
  color: #f1f1f1;
}
#T_0a5c8_row17_col5 {
  background-color: #fbfd66;
  color: #000000;
}
</style>
<table id="T_0a5c8">
  <caption>Метрики моделей SentenceTransformers</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_0a5c8_level0_col0" class="col_heading level0 col0" >model</th>
      <th id="T_0a5c8_level0_col1" class="col_heading level0 col1" >text_type</th>
      <th id="T_0a5c8_level0_col2" class="col_heading level0 col2" >Accuracy@5</th>
      <th id="T_0a5c8_level0_col3" class="col_heading level0 col3" >MMR@5</th>
      <th id="T_0a5c8_level0_col4" class="col_heading level0 col4" >DCG@5</th>
      <th id="T_0a5c8_level0_col5" class="col_heading level0 col5" >time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_0a5c8_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_0a5c8_row0_col0" class="data row0 col0" >all-mpnet-base-v2</td>
      <td id="T_0a5c8_row0_col1" class="data row0 col1" >Изначальный</td>
      <td id="T_0a5c8_row0_col2" class="data row0 col2" >96.82%</td>
      <td id="T_0a5c8_row0_col3" class="data row0 col3" >90.94%</td>
      <td id="T_0a5c8_row0_col4" class="data row0 col4" >91.67%</td>
      <td id="T_0a5c8_row0_col5" class="data row0 col5" >655.86</td>
    </tr>
    <tr>
      <th id="T_0a5c8_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_0a5c8_row1_col0" class="data row1 col0" >all-MiniLM-L6-v2</td>
      <td id="T_0a5c8_row1_col1" class="data row1 col1" >Изначальный</td>
      <td id="T_0a5c8_row1_col2" class="data row1 col2" >96.36%</td>
      <td id="T_0a5c8_row1_col3" class="data row1 col3" >90.39%</td>
      <td id="T_0a5c8_row1_col4" class="data row1 col4" >90.94%</td>
      <td id="T_0a5c8_row1_col5" class="data row1 col5" >119.97</td>
    </tr>
    <tr>
      <th id="T_0a5c8_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_0a5c8_row2_col0" class="data row2 col0" >all-MiniLM-L12-v2</td>
      <td id="T_0a5c8_row2_col1" class="data row2 col1" >Изначальный</td>
      <td id="T_0a5c8_row2_col2" class="data row2 col2" >95.87%</td>
      <td id="T_0a5c8_row2_col3" class="data row2 col3" >90.13%</td>
      <td id="T_0a5c8_row2_col4" class="data row2 col4" >90.75%</td>
      <td id="T_0a5c8_row2_col5" class="data row2 col5" >221.17</td>
    </tr>
    <tr>
      <th id="T_0a5c8_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_0a5c8_row3_col0" class="data row3 col0" >multi-qa-distilbert-cos-v1</td>
      <td id="T_0a5c8_row3_col1" class="data row3 col1" >Изначальный</td>
      <td id="T_0a5c8_row3_col2" class="data row3 col2" >95.60%</td>
      <td id="T_0a5c8_row3_col3" class="data row3 col3" >89.95%</td>
      <td id="T_0a5c8_row3_col4" class="data row3 col4" >90.57%</td>
      <td id="T_0a5c8_row3_col5" class="data row3 col5" >298.25</td>
    </tr>
    <tr>
      <th id="T_0a5c8_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_0a5c8_row4_col0" class="data row4 col0" >multi-qa-MiniLM-L6-cos-v1</td>
      <td id="T_0a5c8_row4_col1" class="data row4 col1" >Изначальный</td>
      <td id="T_0a5c8_row4_col2" class="data row4 col2" >95.83%</td>
      <td id="T_0a5c8_row4_col3" class="data row4 col3" >88.83%</td>
      <td id="T_0a5c8_row4_col4" class="data row4 col4" >89.49%</td>
      <td id="T_0a5c8_row4_col5" class="data row4 col5" >119.93</td>
    </tr>
    <tr>
      <th id="T_0a5c8_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_0a5c8_row5_col0" class="data row5 col0" >multi-qa-mpnet-base-dot-v1</td>
      <td id="T_0a5c8_row5_col1" class="data row5 col1" >Изначальный</td>
      <td id="T_0a5c8_row5_col2" class="data row5 col2" >95.24%</td>
      <td id="T_0a5c8_row5_col3" class="data row5 col3" >88.55%</td>
      <td id="T_0a5c8_row5_col4" class="data row5 col4" >89.37%</td>
      <td id="T_0a5c8_row5_col5" class="data row5 col5" >647.01</td>
    </tr>
    <tr>
      <th id="T_0a5c8_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_0a5c8_row6_col0" class="data row6 col0" >all-distilroberta-v1</td>
      <td id="T_0a5c8_row6_col1" class="data row6 col1" >Изначальный</td>
      <td id="T_0a5c8_row6_col2" class="data row6 col2" >94.78%</td>
      <td id="T_0a5c8_row6_col3" class="data row6 col3" >87.23%</td>
      <td id="T_0a5c8_row6_col4" class="data row6 col4" >88.05%</td>
      <td id="T_0a5c8_row6_col5" class="data row6 col5" >301.65</td>
    </tr>
    <tr>
      <th id="T_0a5c8_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_0a5c8_row7_col0" class="data row7 col0" >paraphrase-multilingual-mpnet-base-v2</td>
      <td id="T_0a5c8_row7_col1" class="data row7 col1" >Изначальный</td>
      <td id="T_0a5c8_row7_col2" class="data row7 col2" >91.96%</td>
      <td id="T_0a5c8_row7_col3" class="data row7 col3" >84.95%</td>
      <td id="T_0a5c8_row7_col4" class="data row7 col4" >85.82%</td>
      <td id="T_0a5c8_row7_col5" class="data row7 col5" >601.49</td>
    </tr>
    <tr>
      <th id="T_0a5c8_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_0a5c8_row8_col0" class="data row8 col0" >multi-qa-distilbert-cos-v1</td>
      <td id="T_0a5c8_row8_col1" class="data row8 col1" >Предобработаный</td>
      <td id="T_0a5c8_row8_col2" class="data row8 col2" >92.91%</td>
      <td id="T_0a5c8_row8_col3" class="data row8 col3" >83.69%</td>
      <td id="T_0a5c8_row8_col4" class="data row8 col4" >84.59%</td>
      <td id="T_0a5c8_row8_col5" class="data row8 col5" >239.22</td>
    </tr>
    <tr>
      <th id="T_0a5c8_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_0a5c8_row9_col0" class="data row9 col0" >all-MiniLM-L6-v2</td>
      <td id="T_0a5c8_row9_col1" class="data row9 col1" >Предобработаный</td>
      <td id="T_0a5c8_row9_col2" class="data row9 col2" >93.27%</td>
      <td id="T_0a5c8_row9_col3" class="data row9 col3" >83.41%</td>
      <td id="T_0a5c8_row9_col4" class="data row9 col4" >84.27%</td>
      <td id="T_0a5c8_row9_col5" class="data row9 col5" >133.37</td>
    </tr>
    <tr>
      <th id="T_0a5c8_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_0a5c8_row10_col0" class="data row10 col0" >all-MiniLM-L12-v2</td>
      <td id="T_0a5c8_row10_col1" class="data row10 col1" >Предобработаный</td>
      <td id="T_0a5c8_row10_col2" class="data row10 col2" >92.55%</td>
      <td id="T_0a5c8_row10_col3" class="data row10 col3" >83.05%</td>
      <td id="T_0a5c8_row10_col4" class="data row10 col4" >83.87%</td>
      <td id="T_0a5c8_row10_col5" class="data row10 col5" >265.13</td>
    </tr>
    <tr>
      <th id="T_0a5c8_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_0a5c8_row11_col0" class="data row11 col0" >multi-qa-MiniLM-L6-cos-v1</td>
      <td id="T_0a5c8_row11_col1" class="data row11 col1" >Предобработаный</td>
      <td id="T_0a5c8_row11_col2" class="data row11 col2" >92.41%</td>
      <td id="T_0a5c8_row11_col3" class="data row11 col3" >82.88%</td>
      <td id="T_0a5c8_row11_col4" class="data row11 col4" >83.67%</td>
      <td id="T_0a5c8_row11_col5" class="data row11 col5" >181.84</td>
    </tr>
    <tr>
      <th id="T_0a5c8_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_0a5c8_row12_col0" class="data row12 col0" >paraphrase-MiniLM-L3-v2</td>
      <td id="T_0a5c8_row12_col1" class="data row12 col1" >Изначальный</td>
      <td id="T_0a5c8_row12_col2" class="data row12 col2" >92.16%</td>
      <td id="T_0a5c8_row12_col3" class="data row12 col3" >82.24%</td>
      <td id="T_0a5c8_row12_col4" class="data row12 col4" >83.21%</td>
      <td id="T_0a5c8_row12_col5" class="data row12 col5" >70.26</td>
    </tr>
    <tr>
      <th id="T_0a5c8_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_0a5c8_row13_col0" class="data row13 col0" >all-distilroberta-v1</td>
      <td id="T_0a5c8_row13_col1" class="data row13 col1" >Предобработаный</td>
      <td id="T_0a5c8_row13_col2" class="data row13 col2" >91.63%</td>
      <td id="T_0a5c8_row13_col3" class="data row13 col3" >81.60%</td>
      <td id="T_0a5c8_row13_col4" class="data row13 col4" >82.50%</td>
      <td id="T_0a5c8_row13_col5" class="data row13 col5" >231.91</td>
    </tr>
    <tr>
      <th id="T_0a5c8_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_0a5c8_row14_col0" class="data row14 col0" >paraphrase-multilingual-mpnet-base-v2</td>
      <td id="T_0a5c8_row14_col1" class="data row14 col1" >Предобработаный</td>
      <td id="T_0a5c8_row14_col2" class="data row14 col2" >89.79%</td>
      <td id="T_0a5c8_row14_col3" class="data row14 col3" >80.78%</td>
      <td id="T_0a5c8_row14_col4" class="data row14 col4" >81.79%</td>
      <td id="T_0a5c8_row14_col5" class="data row14 col5" >493.40</td>
    </tr>
    <tr>
      <th id="T_0a5c8_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_0a5c8_row15_col0" class="data row15 col0" >all-mpnet-base-v2</td>
      <td id="T_0a5c8_row15_col1" class="data row15 col1" >Предобработаный</td>
      <td id="T_0a5c8_row15_col2" class="data row15 col2" >91.07%</td>
      <td id="T_0a5c8_row15_col3" class="data row15 col3" >80.77%</td>
      <td id="T_0a5c8_row15_col4" class="data row15 col4" >81.75%</td>
      <td id="T_0a5c8_row15_col5" class="data row15 col5" >457.85</td>
    </tr>
    <tr>
      <th id="T_0a5c8_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_0a5c8_row16_col0" class="data row16 col0" >multi-qa-mpnet-base-dot-v1</td>
      <td id="T_0a5c8_row16_col1" class="data row16 col1" >Предобработаный</td>
      <td id="T_0a5c8_row16_col2" class="data row16 col2" >90.34%</td>
      <td id="T_0a5c8_row16_col3" class="data row16 col3" >80.60%</td>
      <td id="T_0a5c8_row16_col4" class="data row16 col4" >81.64%</td>
      <td id="T_0a5c8_row16_col5" class="data row16 col5" >513.54</td>
    </tr>
    <tr>
      <th id="T_0a5c8_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_0a5c8_row17_col0" class="data row17 col0" >paraphrase-MiniLM-L3-v2</td>
      <td id="T_0a5c8_row17_col1" class="data row17 col1" >Предобработаный</td>
      <td id="T_0a5c8_row17_col2" class="data row17 col2" >90.71%</td>
      <td id="T_0a5c8_row17_col3" class="data row17 col3" >79.94%</td>
      <td id="T_0a5c8_row17_col4" class="data row17 col4" >80.94%</td>
      <td id="T_0a5c8_row17_col5" class="data row17 col5" >80.79</td>
    </tr>
  </tbody>
</table>

## Выбор модели

Для построения приложения Streamlit  и микросервиса FastAPI  была выбрана базовая предобученная модель SentenceTransformers  `all-MiniLM-L6-v2`,  так как она показала наилучший баланс между качеством и скоростью работы.  Эта модель обеспечивает высокую точность  и при этом достаточно быстра для использования в интерактивном приложении  и API.  

## Приложение Streamlit

[Ссылка на веб приложение](https://fedorsafonov-medical-qa--mle--streamlit-app-agoi6t.streamlit.app/)


Приложение Streamlit позволяет пользователю ввести вопрос и получить список из 5  наиболее похожих вопросов из базы данных.  Сходство между вопросами отображается в процентах. 

**Инструкции по использованию:**
- Открыть приложение по [ссылке](https://fedorsafonov-medical-qa--mle--streamlit-app-agoi6t.streamlit.app/)
- Выбрать язык (по умолчанию - Русский)
- Ввести вопрос на английском языке в поле "Введите ваш вопрос"
- Нажать на кнопку "Найти похожие вопросы"

**Результат:**
- 5 похожих вопрос на английском языке со степенью сходства в скобках.

*Скриншот приложения:*
![image 1](/images/image.png)

## Микросервис FastAPI

Микросервис FastAPI  предоставляет API  для поиска похожих вопросов.  API  имеет один endpoint  `/similar_questions`,  который принимает на вход вопрос в виде строки  и возвращает JSON  список похожих вопросов и их оценок сходства. 

**Инструкции по использованию:**
- 1. Скачать файлы 'model', 'idx2emb.pkl', 'idx2sen.pkl', 'fastapi_app.py', 'requirements.txt'
- 2. Установить зависимости:
   ```bash
   pip install -r requirements.txt
   ```
- 3. Запуститm API  FastAPI:
   ```bash
   uvicorn fastapi_app:app --reload
   ```
- 4. Тестирование:
    - Открыть адрес http://127.0.0.1:8000/docs в браузере. Здесь можно увидеть интерактивную документацию Swagger UI, которая позволяет отправлять запросы к API и просматривать ответы.

Внешний вид документации.
![alt text](/images/image-1.png)

Ввод конкретного запроса в браузере.
![alt text](/images/image-2.png)

## Итоги

* **Создано приложение Streamlit.**
* **Создан приложение FastAPI**
* **Основные выводы:**  В ходе проекта были исследованы различные подходы к поиску похожих вопросов,  включая классические модели (Bag-of-Words,  TF-IDF)  и модели на основе глубокого обучения (SentenceTransformers,  Word2Vec,  USE).  Результаты показали,  что модели SentenceTransformers  и USE  обеспечивают наилучшее качество,  а предобработка текста  значительно улучшает качество классических моделей.  Fine-tuning  моделей SentenceTransformers  может как улучшить,  так и ухудшить качество,  в зависимости от выбранной loss-функции  и параметров обучения. 
* **Дальнейшее развитие:**  Проект может быть развит в следующих направлениях:
   * **Использование других моделей:**  Можно попробовать использовать другие модели,  такие как Graph Neural Networks, модели Question Answering или модели Paraphrase Detection.
   * **Увеличение размера датасета:**  Обучение моделей на большем датасете может улучшить их качество  и обобщающую способность.
   * **Улучшение предобработки текста:**  Можно попробовать использовать более сложные методы предобработки текста,  такие как нормализация,  стемминг  или учет синтаксической структуры предложений.
   * **Fine-tuning  моделей:**  Можно провести более глубокое исследование fine-tuning  моделей SentenceTransformers,  экспериментируя с различными loss-функциями,  параметрами обучения  и методами отбора данных. 
   * **Разработка более сложного интерфейса:**  Можно добавить в приложение Streamlit  возможности фильтрации,  сортировки,  пагинации  и другие функции,  которые улучшат пользовательский опыт.
   * **Интеграция с другими сервисами:**  API  FastAPI  может быть интегрирован с другими сервисами,  такими как чат-боты,  системы вопросов-ответов  или поисковые системы. 

## Cтатус: 
Завершён.


## Ссылки

* **Датасет:**  [https://huggingface.co/datasets/medical_questions_pairs](https://huggingface.co/datasets/medical_questions_pairs)
* **SentenceTransformers:**  [https://www.sbert.net/](https://www.sbert.net/)
* **Streamlit:**  [https://streamlit.io/](https://streamlit.io/)
* **FastAPI:**  [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)