import grank

# region loading dataset and training/test

print('loading dataset ...')
ratings_dataset = grank.load_dataset('movielens100k_ratings.csv')

user_ids = ratings_dataset['userId'].unique().tolist()

print('generating training and test datasets ...')
t = int(input('Please Provide T: '))
training_dataset, test_dataset = grank.generate_training_test_datasets(ratings_dataset, t, user_ids)

# getting user and movies list
user_ids = ratings_dataset['userId'].unique().tolist()  # f
movie_ids = ratings_dataset['movieId'].unique().tolist()  # f

total_users_count = len(user_ids)
print('{0} users'.format(total_users_count))
total_movies_count = len(movie_ids)
print('{0} movies'.format(total_movies_count))

stats = (max(user_ids), max(movie_ids))

# endregion

print('generating ratings matrix ...')
training_matrix = grank.generate_ratings_matrix(ratings_dataset)  # f
ratings_matrix = grank.generate_ratings_matrix(ratings_dataset)

del ratings_dataset

print('generating tripartite preference graph ...')
tpg = grank.generate_tpg_alt(stats, ratings_matrix, user_ids, movie_ids)  # f
print(tpg)

print('computing accuracy ...')
k_list = [5, 7, 9]
for k in k_list:
    print('computing NDCG@{0} ...'.format(k))
    accuracy_k = grank.compute_accuracy(stats, tpg, ratings_matrix, user_ids, movie_ids, k)
    print('accuracy using ndcg@{0} is {1}'.format(k, accuracy_k))
