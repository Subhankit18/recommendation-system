import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Sample dataset
data = pd.DataFrame({
    'title': ['Movie A','Movie B','Movie C','Movie D'],
    'genre': ['action adventure','action thriller','romance drama','romance comedy']
})

# Vectorize
cv = CountVectorizer()
matrix = cv.fit_transform(data['genre'])

# Similarity
similarity = cosine_similarity(matrix)

def recommend(movie):
    idx = data[data['title'] == movie].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    for i in scores[1:3]:
        print(data.iloc[i[0]].title)

# Test
recommend('Movie A')
