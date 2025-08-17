# Movie Recommender System

![Streamlit](https://img.shields.io/badge/Streamlit-Enabled-red?logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/ML-sklearn-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

This project is an interactive movie recommendation system built with **Streamlit**, powered by **collaborative filtering**, **content-based filtering**, and **hybrid algorithms**. It leverages datasets of up to **32 million ratings** to provide tailored suggestions based on user preferences or selected movies.

---

## Features

- User-based and item-based collaborative filtering
- Content-based filtering (genres, year)
- Hybrid approaches combining ratings and content
- Multiple recommendation modes:
  - By movie title
  - By user ID
  - Manual movie selection
  - Popular picks by genre, year, or globally
- Ready-to-use **Streamlit app**

---

## Project Structure

- `app_with_1m_ratings.py` — Streamlit app using the 1M MovieLens dataset
- `app_with_32m_ratings.py` — Streamlit app using the 32M MovieLens dataset
- `movie_recommender_system.ipynb` — Development notebook with core logic
- `requirements.txt` — Python dependencies
- `LICENSE` — MIT License

---

## Try the App

Try the 1M MovieLens app: [Link to deployed Streamlit app](hhttps://movierecommendersystem-qqxh2wfxieni9cftjchdxq.streamlit.app//)

---

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/nasser-chaouchi/movie_recommender_system.git
cd movie_recommender_system
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit app (choose one)

```bash
streamlit run app_with_1m_ratings.py
# or
streamlit run app_with_32m_ratings.py
```

---

## Dataset Sources

The datasets are loaded directly from Hugging Face:

- [**1M Ratings Dataset**](https://huggingface.co/datasets/nasserCha/movielens_rating_1m)

- [**32M Ratings Dataset**](https://huggingface.co/datasets/nasserCha/movielens_ratings_32m)

---

## Algorithms

- **Content-Based Filtering** using movie genres
- **Collaborative Filtering** (User-User & Item-Item)
- **Hybrid Recommendations** combining both approaches
- **Popularity-based Suggestions** for new users or quick discovery

---

## Business Impact

Personalized recommendations are at the heart of modern digital platforms.  
This project showcases how machine learning can enhance:

- **User engagement** by tailoring movie suggestions  
- **Retention** through improved discovery and reduced churn  
- **Scalability** from small datasets (1M ratings) to large-scale systems (32M ratings)  

Such recommender systems form the backbone of platforms like Netflix, Amazon Prime, and Spotify, highlighting the critical role of **AI-driven personalization** in the entertainment industry.

---

## Disclaimer

**Important Notice**  
This project is developed **for educational and research purposes only**.  
It is not a production-grade system and should not be used as-is for commercial deployment.  


## Built With

- Python
- pandas, NumPy, scikit-learn, SciPy
- Streamlit
- Hugging Face Datasets
- MovieLens data

---

## Author

**Nasser Chaouchi**  
Data Scientist & ML Engineer  
[LinkedIn](https://www.linkedin.com/in/nasser-chaouchi)

---

## License

This project is licensed under the [MIT License](LICENSE).