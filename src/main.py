import grank

print('loading dataset ...')
ratings_dataset = grank.load_dataset('test2.csv')
total_users_count = ratings_dataset['userId'].max()
total_movies_count = ratings_dataset['movieId'].max()

print('generating training and test datasets ...')
t = int(input('Please Provide T: '))
training_dataset, test_dataset = grank.generate_training_test_datasets(ratings_dataset, t, total_users_count)

print('generating ratings matrix ...')
ratings_matrix = grank.generate_ratings_matrix(ratings_dataset)

print('generating tripartite preference graph ...')
tpg = grank.generate_tpg(ratings_matrix, total_users_count, total_movies_count)

# print('drawing ...')
# grank.draw_tpg(tpg)

print('computing accuracy ...')
while True:
    k = int(input('Please Provide a K: '))
    print('computing NDCG@{0} ...'.format(k))
    accuracy_k = grank.compute_accuracy(tpg, ratings_matrix, ratings_dataset, total_movies_count, k)
    print('accuracy using ndcg@{0} is {1}'.format(k, accuracy_k))
