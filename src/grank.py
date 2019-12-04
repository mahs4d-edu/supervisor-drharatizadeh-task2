from os import path

import math
import pandas as pd

from mahdimatrix import MahdiGraph, MahdiMatrix


def load_dataset(file_name):
    """
    loads the dataset of movie ratings from file. the file should be placed in the "data" folder
    of the project
    :param file_name:
    :return: rating dataset
    """

    file_path = path.join(path.abspath(path.dirname(
        __file__)), '../data', file_name)
    dataset = pd.read_csv(file_path)  # , delim_whitespace=True if the old datasets were used

    # i multiply rating to two because i can work better with int than float
    dataset['rating'] = dataset['rating'] * 2
    return dataset


def save_small_dataset(ratings_dataset, size, filename):
    a = ratings_dataset[:size]
    a.to_csv(filename, index=False)


def generate_training_test_datasets(ratings_dataset, t, user_ids):
    """
    selects t ratings for each user and puts it in training dataset and others into the test dataset. users that have
    less than t+10 ratings, will be removed
    :param user_ids:
    :param ratings_dataset:
    :param t:
    :return: a tuple of (training_dataset, test_dataset)
    """

    training_dataset = []
    test_dataset = []
    for user_id in user_ids:
        user_ratings = ratings_dataset[ratings_dataset['userId'] == user_id]

        if len(user_ratings) < t + 10:
            continue

        training_samples = user_ratings.sample(n=t)
        test_samples = user_ratings.drop(training_samples.index)

        training_dataset.extend(training_samples.values.tolist())
        test_dataset.extend(test_samples.values.tolist())

    training_dataset = pd.DataFrame(training_dataset, columns=['userId', 'movieId', 'rating', 'timestamp'])
    training_dataset = training_dataset.astype(dtype={
        'userId': int,
        'movieId': int,
        'rating': int,
        'timestamp': int,
    })
    test_dataset = pd.DataFrame(test_dataset, columns=['userId', 'movieId', 'rating', 'timestamp'])
    test_dataset = test_dataset.astype(dtype={
        'userId': int,
        'movieId': int,
        'rating': int,
        'timestamp': int,
    })

    return training_dataset, test_dataset


def generate_ratings_matrix(ratings_dataset):
    """
    in this function we convert the ratings_dataset to a matrix with rows identifying users
    and the columns as movies. value of element i,j in the matrix address the rating given
    by user i to the movie j. the matrix is saved in memory as sparse matrix.
    :param ratings_dataset:
    :return: sparse rating matrix
    """
    ratings_matrix = {}
    for _, r in ratings_dataset.iterrows():
        user_id = int(r['userId'])
        movie_id = int(r['movieId'])
        rating = int(r['rating'])

        if user_id in ratings_matrix:
            ratings_matrix[user_id].append((movie_id, rating))
        else:
            ratings_matrix[user_id] = [(movie_id, rating)]

    return ratings_matrix


def get_user_index(user_id):
    return user_id - 1


def get_item_d_index(stats, movie_id):
    u, i = stats
    return u + movie_id - 1


def get_item_u_index(stats, movie_id):
    u, i = stats
    return u + i + movie_id - 1


def get_pair_index(stats, movie_id_1, movie_id_2):
    u, i = stats
    if movie_id_1 > movie_id_2:
        return u + (2 * i) + ((movie_id_1 - 1) * (i - 1)) + movie_id_2 - 1
    elif movie_id_2 > movie_id_1:
        return u + (2 * i) + ((movie_id_1 - 1) * (i - 1)) + movie_id_2 - 2
    else:
        raise ValueError('invalid pair')


def add_edge(rows, columns, data, n1, n2):
    rows.extend([n1, n2])
    columns.extend([n2, n1])
    data.extend([1, 1])


def generate_tpg_alt(stats, ratings_matrix, user_ids, movie_ids):
    """
    generates the tripartite preference graph according to the grank algorithm from rating matrix
    :param movie_ids:
    :param user_ids:
    :param ratings_matrix:
    :return: a networkx graph
    """
    m_size = int(stats[0] + stats[1] + stats[1] + (stats[1] * (stats[1] - 1)))
    print('\t> matrix size: {0} * {1}'.format(m_size, m_size))

    my_graph = MahdiGraph(m_size)

    # create A > B pairs for each possible tuple of movies
    for i in range(0, len(movie_ids)):
        movie_id_1 = movie_ids[i]
        item_d_i_index = get_item_d_index(stats, movie_id_1)
        item_u_i_index = get_item_u_index(stats, movie_id_1)

        for j in range(i + 1, len(movie_ids)):
            movie_id_2 = movie_ids[j]
            item_d_j_index = get_item_d_index(stats, movie_id_2)
            item_u_j_index = get_item_u_index(stats, movie_id_2)

            pair_1_index = get_pair_index(stats, movie_id_1, movie_id_2)
            pair_2_index = get_pair_index(stats, movie_id_2, movie_id_1)

            my_graph.add_edge(pair_1_index, item_d_i_index)
            my_graph.add_edge(pair_1_index, item_u_j_index)

            my_graph.add_edge(pair_2_index, item_d_j_index)
            my_graph.add_edge(pair_2_index, item_u_i_index)

    for user_id in ratings_matrix:
        user_ratings = ratings_matrix[user_id]

        if user_id == 0:
            continue

        print('\t> generating graph of user {0}'.format(user_id))
        user_node = get_user_index(user_id)

        # get all the movies that specific user has rated and create a node for each pair of them based on a constraint
        # that which movie he rated has a higher rate than the other. we also connect the pair nodes to their respective
        # nodes in the items section of the graph (for example "A > B" will be connected to "A_Desirable" and
        # "B_UnDesirable" respectively
        for i in range(len(user_ratings)):
            movie_id_1 = user_ratings[i][0]
            movie_rating_1 = user_ratings[i][1]
            for j in range(i + 1, len(user_ratings)):
                movie_id_2 = user_ratings[j][0]
                movie_rating_2 = user_ratings[j][1]

                if movie_rating_1 > movie_rating_2:
                    pair_node = get_pair_index(stats, movie_id_1, movie_id_2)
                elif movie_rating_2 > movie_rating_1:
                    pair_node = get_pair_index(stats, movie_id_2, movie_id_1)
                else:
                    continue

                # finally connect the user node to the pair node
                my_graph.add_edge(user_node, pair_node)

    my_graph.generate_t_matrix()
    return my_graph


def compute_pagerank_alt(tpg: MahdiGraph, user_id):
    """
    this function computes the personalized pagerank of tgp with personalized vector which sets the target users element
    one and all other nodes as zero (this version uses sparse matrix representation of tpg)
    :param tpg:
    :param user_id:
    :return: pagerank matrix as a dictionary
    """

    matrix = tpg.pagerank(0.85, get_user_index(user_id))
    return matrix


def _compute_grank(stats, pagerank: MahdiMatrix, movie_id):
    """
    computes grank of single movie from the pagerank matrix using GRank=PPR_desirable/(PPR_desirable + PPR_undesirable)
    :param pagerank:
    :param movie_id:
    :return: grank value
    """
    ppr_d = pagerank.get_item(get_item_d_index(stats, movie_id=movie_id), 0)
    ppr_u = pagerank.get_item(get_item_u_index(stats, movie_id=movie_id), 0)

    # this if statement is to prevent division by zero errors
    if ppr_d == 0 and ppr_u == 0:
        return 0

    return ppr_d / (ppr_d + ppr_u)


def get_top_k_recommendations(stats, pagerank, movie_ids, k):
    """
    computes grank for each movie and sorts them. returns the top k items of the sorted list
    :return: a list of tuples containing grank and movie id
    """

    movie_granks = []
    for movie_id in movie_ids:
        grank = _compute_grank(stats, pagerank, movie_id)
        movie_granks.append((grank, movie_id))

    movie_granks = sorted(movie_granks, key=lambda x: x[0], reverse=True)
    return movie_granks[:k]


def _get_user_rating(ratings_matrix, user_id, movie_id):
    user_ratings = ratings_matrix[user_id]
    for t in user_ratings:
        if t[0] == movie_id:
            return t[1]

    return 0


def _get_user_max_rating(ratings_matrix, user_id):
    max_rate = 0
    user_ratings = ratings_matrix[user_id]
    for t in user_ratings:
        if t[1] > max_rate:
            max_rate = t[1]

    return max_rate


def _compute_ndcg(user_id, top_k_recommendations, ratings_matrix):
    summation = 0
    alpha_sum = 0
    max_rate = _get_user_max_rating(ratings_matrix, user_id)

    for i, recommendation in enumerate(top_k_recommendations):
        rate = _get_user_rating(ratings_matrix, user_id, recommendation[1])
        summation += (2 ** rate) / math.log(i + 2, 2)
        alpha_sum += (2 ** max_rate) / math.log(i + 2, 2)

    return (1 / alpha_sum) * summation


def compute_accuracy(stats, tpg, ratings_matrix, user_ids, movie_ids, k):
    """
    computes average NDCG@k for all the users in test_dataset
    :return: average NDCG@k
    """

    ndcg_sum = 0
    for user_id in user_ids:
        print('\t>computing accuracy for user {0}'.format(user_id))

        # first we compute pagerank for this user
        pagerank = compute_pagerank_alt(tpg, user_id)

        # compute top k recommendation
        top_k_recommendations = get_top_k_recommendations(stats, pagerank, movie_ids, k)
        print(top_k_recommendations)
        # get ndcg@k of the recommendations
        ndcg = _compute_ndcg(user_id, top_k_recommendations, ratings_matrix)

        print('\t>ndcg of user {0} is {1}'.format(user_id, ndcg))
        ndcg_sum += ndcg

    return ndcg_sum / len(user_ids)
