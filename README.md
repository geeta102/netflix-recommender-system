# Netflix Recommendation System

A machine learning project that builds a movie recommendation system using the Netflix Prize dataset.

## Overview

This project predicts user ratings and generates personalized Top-10 movie recommendations using collaborative filtering techniques.

The full Netflix Prize dataset is very large, so this project uses a reproducible subset of 250,000 ratings for local training and evaluation.

## Tech Stack

Python, pandas, NumPy, SciPy, scikit-learn, Matplotlib, Seaborn

## Models

| Model | Purpose |
|---|---|
| Popularity Recommender | Recommends globally popular and highly rated movies |
| Bias Baseline | Uses global average, user bias, and movie bias |
| Matrix Factorization SVD | Learns hidden user-movie preference patterns |

## Evaluation

| Model | RMSE | MAP@10 |
|---|---:|---:|
| Bias Baseline | 1.1127 | 0.9295 |
| Matrix Factorization SVD | 1.1682 | 0.9010 |

A movie is considered relevant for MAP@10 if its actual rating is greater than or equal to 3.5.

## Project Structure

```text

src/
├── 01_inspect_dataset.py
├── 02_prepare_data.py
├── 03_eda.py
├── 04_add_movie_titles.py
├── 05_popularity_recommender.py
├── 06_bias_baseline_model.py
├── 07_svd_model.py
├── 08_compare_models.py
├── 09_map_at_10.py
└── 10_final_results.py
 ```

 ## How to run

Install dependencies:
pip install -r requirements.txt

Place the Netflix dataset zip file at:
data/archive.zip

```

Run the scripts in order:
python src/01_inspect_dataset.py
python src/02_prepare_data.py
python src/03_eda.py
python src/04_add_movie_titles.py
python src/05_popularity_recommender.py
python src/06_bias_baseline_model.py
python src/07_svd_model.py
python src/08_compare_models.py
python src/09_map_at_10.py
python src/10_final_results.py

```

 ## Key Insight
The Bias Baseline model performed best on this subset because the data is highly sparse. Matrix Factorization SVD may improve with more data, better sampling, and hyperparameter tuning.

 ## Future Work
Train on a larger subset
Add item-based collaborative filtering
Tune SVD parameters
Add movie metadata features
Build an interactive dashboard

 ## Presentation and Technical Report
[Presentation](https://docs.google.com/presentation/d/11NM2JJ7BXgd1pV55hdfnkK2SjAgxpJo0/edit?usp=sharing&ouid=103462315502597560969&rtpof=true&sd=true)
[Technical Report](https://drive.google.com/file/d/10NA7sM4AJbmjmTpsoT_3bRKbJOjKeTlu/view?usp=sharing)


