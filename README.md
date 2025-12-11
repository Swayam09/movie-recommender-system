Movie Recommender System (Content-Based Filtering)
This project is a simple content-based movie recommender system built using the TMDB 5000 Movies Dataset.
It recommends similar movies based on overview, genres, keywords, cast, and crew information.
The application is deployed using Streamlit, and the recommendation logic is implemented in Python using CountVectorizer, NLTK stemming, and Cosine Similarity.

Live Demo
Streamlit App: https://movie-recommender-system-eqwzilehefsfugyfdgymsx.streamlit.app/

Features
Extracts movie metadata (genres, keywords, cast, crew)
Cleans and transforms data into meaningful tags
Applies stemming to standardize words
Converts tags into vectors using CountVectorizer
Computes similarity between movies using cosine similarity
Simple UI to select a movie and get top 5 recommendations

How It Works
Load and merge the TMDB movie and credits datasets
Extract important text features
Preprocess and clean the text
Build a tags column combining overview, genres, keywords, cast, director
Convert tags into vectors
Compute cosine similarity
Recommend the most similar movies

Project Structure
.
├── app.py                     # Streamlit application
├── tmdb_5000_movies.csv       # Movie metadata
├── tmdb_5000_credits.csv      # Cast and crew metadata
├── requirements.txt           # Dependencies for Streamlit Cloud
└── README.md                  # Project documentation

Technologies Used
Python 3
Pandas
NumPy
NLTK
Scikit-Learn
Streamlit
Ast

Running Locally
Clone the repository:
git clone https://github.com/Swayam09/movie-recommender-system.git
cd movie-recommender-system
Install dependencies:
pip install -r requirements.txt
Run the Streamlit app:
streamlit run app.py

Deploying on Streamlit Cloud
Push your code + CSV files to GitHub
Go to https://share.streamlit.io
Select your repo
Set app.py as the entry file
Deploy
Streamlit Cloud automatically installs dependencies from requirements.txt.


Credits
Dataset:
TMDB 5000 Movie Dataset (Kaggle)
