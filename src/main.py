import grank

# region loading dataset and training/test

print('loading dataset ...')
ratings_dataset = grank.load_dataset('test.csv')

user_ids = ratings_dataset['userId'].unique().tolist()

print('generating training and test datasets ...')
t = int(input('Please Provide T: '))
training_dataset, test_dataset = grank.generate_training_test_datasets(ratings_dataset, t, user_ids)

# getting user and movies list
user_ids = ratings_dataset['userId'].unique().tolist()#############
movie_ids = ratings_dataset['movieId'].unique().tolist()#############

total_users_count = len(user_ids)
print('{0} users'.format(total_users_count))
total_movies_count = len(movie_ids)
print('{0} movies'.format(total_movies_count))

# endregion

print('generating ratings matrix ...')
training_matrix = grank.generate_ratings_matrix(ratings_dataset)###########
ratings_matrix = grank.generate_ratings_matrix(ratings_dataset)

del ratings_dataset

print('generating tripartite preference graph ...')
tpg = grank.generate_tpg_alt(ratings_matrix, user_ids, movie_ids)################
print(tpg)

# print('drawing ...')
# grank.draw_tpg(tpg)

print('computing accuracy ...')
k_list = [5, 7, 9]
for k in k_list:
    print('computing NDCG@{0} ...'.format(k))
    accuracy_k = grank.compute_accuracy(tpg, ratings_matrix, user_ids, movie_ids, k)
    print('accuracy using ndcg@{0} is {1}'.format(k, accuracy_k))
