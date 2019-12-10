import grank
import datetime

start_time = datetime.datetime.now()
print('Start Time: {0}: '.format(start_time))

# region loading dataset and training/test
print('loading dataset ...')
ratings_dataset = grank.load_dataset('movielens100k_ratings.csv')

user_ids = ratings_dataset['userId'].unique().tolist()

print('generating training and test datasets ...')
t = int(input('Please Provide T: '))
training_dataset, test_dataset = grank.generate_training_test_datasets(ratings_dataset, t, user_ids)
del test_dataset

# getting user and movies list
user_ids = training_dataset['userId'].unique().tolist()  # f
movie_ids = training_dataset['movieId'].unique().tolist()  # f

total_users_count = len(user_ids)
print('{0} users'.format(total_users_count))
total_movies_count = len(movie_ids)
print('{0} movies'.format(total_movies_count))

stats = (max(user_ids), max(movie_ids))

# endregion

# region graph generation

print('generating ratings matrix ...')
training_matrix = grank.generate_ratings_matrix(training_dataset)  # f
ratings_matrix = grank.generate_ratings_matrix(ratings_dataset)

del ratings_dataset

print('generating tripartite preference graph ...')
tpg = grank.generate_tpg_alt(stats, training_matrix, user_ids, movie_ids)  # f

# endregion

# region accuracy

print('computing accuracy ...')
k_list = [3, 5, 7, 9]

accuracy = grank.compute_accuracy(stats, tpg, ratings_matrix, user_ids, movie_ids, k_list)

for k in accuracy.keys():
    print('NDCG@{0}: {1}'.format(k, accuracy[k]))

# endregion

end_time = datetime.datetime.now()
print('End Time: {0}: '.format(end_time))

print('Time to Complete: {0}'.format(str(end_time - start_time)))
