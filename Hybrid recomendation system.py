import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load the ratings data
df_ratings = pd.read_csv('data/ml-100k/u.data', sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])

# Load the movies data
movies_df = pd.read_csv('data/ml-100k/u.item', sep='|', header=None, encoding='latin-1', 
                         names=['item_id', 'title', 'release_date', 'video_release_date', 
                                'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 
                                'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 
                                'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 
                                'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])

# Create a features column by combining genre columns
genre_columns = movies_df.columns[6:]
movies_df['features'] = movies_df[genre_columns].apply(lambda x: ' '.join(x.index[x == 1]), axis=1)

# Step 1: Collaborative Filtering
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df_ratings[['user_id', 'item_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

model = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
model.fit(trainset)

# Evaluate the model
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# Get top N recommendations for a specific user
def get_collaborative_recommendations(user_id, n=5):
    user_movies = df_ratings[df_ratings['user_id'] == user_id]
    user_movie_ids = user_movies['item_id'].tolist()
    
    # Get predictions for all movies
    all_movie_ids = df_ratings['item_id'].unique()
    predictions = [model.predict(user_id, movie_id) for movie_id in all_movie_ids if movie_id not in user_movie_ids]
    
    # Sort predictions based on estimated ratings
    top_n = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]
    return [pred.iid for pred in top_n]

# Example user ID
user_id = 196  # Change this to any user ID you want to test
collab_recommendations = get_collaborative_recommendations(user_id, n=10)

# Step 2: Content-Based Filtering (Using user genre preferences)
def get_content_based_recommendations(user_genres, movie_ids):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['features'])
    
    # Create a user preference TF-IDF vector
    user_tfidf = tfidf.transform([' '.join(user_genres)])
    
    # Calculate cosine similarities
    cosine_similarities = linear_kernel(user_tfidf, tfidf_matrix[movies_df['item_id'].isin(movie_ids)])
    
    # Create a DataFrame for similarity scores
    sim_scores = pd.DataFrame(cosine_similarities.T, columns=['similarity'])
    sim_scores['item_id'] = movies_df[movies_df['item_id'].isin(movie_ids)]['item_id'].values
    sim_scores = sim_scores.sort_values(by='similarity', ascending=False)
    
    return sim_scores

# Get user preferences
user_preferences = input("Enter your preferred genres (comma-separated): ")
available_genres = movies_df.columns[6:].tolist()
user_genres = [genre.strip() for genre in user_preferences.split(',') if genre.strip() in available_genres]

# Get content-based recommendations based on collaborative filtering results
content_recommendations = get_content_based_recommendations(user_genres, collab_recommendations)

# Step 3: Display the Top 5 Recommendations
top_content_recommendations = content_recommendations.head(5)
recommendations = top_content_recommendations.merge(movies_df, on='item_id')[['title', 'features']]
print("Top 5 Recommended Movies:\n", recommendations)
