import grank

print('loading dataset ...')
ratings_dataset = grank.load_dataset('movielens100_ratings.csv')
total_users_count = ratings_dataset['userId'].max()
total_movies_count = ratings_dataset['movieId'].max()

print('generating ratings matrix ...')
ratings_matrix = grank.generate_ratings_matrix(ratings_dataset)

print('generating tripartite preference graph ...')
tpg = grank.generate_tpg(ratings_matrix)

print('computing pagerank for user {0} ...'.format(1))
pagerank = grank.compute_pagerank(tpg, 1)

print('computing grank and top k recommended movies for user {0}'.format(1))
top_recommendations = grank.get_top_k_recommendations(pagerank, total_movies_count, 20)
print(top_recommendations)
