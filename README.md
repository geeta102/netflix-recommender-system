# Netflix Recommendation System

This project builds a recommendation system for personalized content discovery using the Netflix Prize dataset.

## Problem Statement

The goal is to learn user preferences from historical user-movie ratings, predict ratings for unseen movies, and generate personalized Top-10 recommendations.

## Dataset

Dataset: Netflix Prize Dataset

The dataset contains:
- User IDs
- Movie IDs
- Ratings from 1 to 5
- Rating dates
- Movie titles and release years

The original dataset is very large, so this project uses a reproducible subset of 250,000 ratings for practical training and evaluation.

## Tech Stack

- Python
- pandas
- numpy
- scipy
- scikit-learn
- matplotlib
- seaborn

## Project Structure

```text
netflix-recommender-system/
├── data/
│   └── archive.zip
├── notebooks/
├── src/
│   ├── 01_inspect_dataset.py
│   ├── 02_prepare_data.py
│   ├── 03_eda.py
│   ├── 04_add_movie_titles.py
│   ├── 05_popularity_recommender.py
│   ├── 06_bias_baseline_model.py
│   ├── 07_svd_model.py
│   ├── 08_compare_models.py
│   ├── 09_map_at_10.py
│   └── 10_final_results.py
├── outputs/
├── reports/
├── requirements.txt
├── .gitignore
└── README.md

Models Implemented
1. Popularity-Based Recommender
This recommends movies with high average ratings and enough rating count. It is useful as a simple baseline and for cold-start users.
2. Bias Baseline Model
This model predicts ratings using:
predicted rating = global average + user bias + movie bias
It captures whether a user generally rates higher or lower and whether a movie is generally liked more or less.
3. Matrix Factorization using SVD
SVD learns hidden user and movie factors from the sparse user-movie rating matrix. It captures latent preference patterns.
Evaluation Metrics
RMSE
Root Mean Squared Error measures rating prediction accuracy.
Lower RMSE is better.
MAP@10
Mean Average Precision at 10 measures ranking quality.
A movie is considered relevant if:
actual rating >= 3.5
Higher MAP@10 is better.
Results
Model	RMSE	MAP@10
Bias Baseline	1.1127	0.9295
Matrix Factorization SVD	1.1682	0.9010

The Bias Baseline model performed best on this subset.
How To Run
Install dependencies:
pip install -r requirements.txt
Run scripts in order:
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
If using Anaconda Python on macOS:
/opt/anaconda3/bin/python3 src/01_inspect_dataset.py
Use the same prefix for all scripts.
Key Insights
The dataset is highly sparse, which is common in recommendation systems.
Popular movies receive many more ratings than long-tail movies.
Bias Baseline performed better on the sampled dataset because it is more stable with sparse data.
SVD can improve with more data, better sampling, and hyperparameter tuning.
Cold Start Strategy
For new users:
Recommend popular high-rated movies.
Ask onboarding questions about favorite genres or movies.
For new movies:
Use metadata such as genre, actors, director, year, or description.
Recommend initially based on content similarity or popularity.
Future Improvements
Train on a larger subset or full dataset.
Add item-based collaborative filtering.
Tune SVD components.
Add genre/content metadata.
Build an interactive dashboard.