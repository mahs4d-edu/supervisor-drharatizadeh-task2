from os import path

import numpy as np
import networkx as nx
import pandas as pd
from scipy import sparse


def load_dataset(file_name):
    """
    loads the dataset of movie ratings from file. the file should be placed in the "data" folder
    of the project
    :param file_name:
    :return: rating dataset
    """

    file_path = path.join(path.abspath(path.dirname(
        __file__)), '../data', file_name)
    dataset = pd.read_csv(file_path)
    return dataset


def generate_ratings_matrix(ratings_dataset):
    """
    in this function we convert the ratings_dataset to a matrix with rows identifying users
    and the columns as movies. value of element i,j in the matrix address the rating given
    by user i to the movie j. the matrix is saved in memory as sparse matrix.
    :param ratings_dataset:
    :return: sparse rating matrix
    """

    row = ratings_dataset['userId']
    column = ratings_dataset['movieId']
    data = ratings_dataset['rating']

    ratings_matrix = sparse.csr_matrix((data, (row, column)))
    return ratings_matrix


def generate_tpg(ratings_matrix):
    """
    generates the tripartite preference graph according to the grank algorithm from rating matrix
    :param ratings_matrix:
    :return: a networkx graph
    """

    tpg = nx.Graph()
    for user_id, user_ratings in enumerate(ratings_matrix):
        print('\t> generating graph of user {0}'.format(user_id))
        # for each user, create a node
        user_node = 'user_{0}'.format(user_id)
        if not tpg.has_node(user_node):
            tpg.add_node(user_node)

        # get all the movies that specific user has rated and create a node for each pair of them based on a constraint
        # that which movie he rated has a higher rate than the other. we also connect the pair nodes to their respective
        # nodes in the items section of the graph (for example "A > B" will be connected to "A_Desirable" and
        # "B_UnDesirable" respectively
        _, cols = user_ratings.nonzero()
        for i in range(len(cols)):
            movie_id_1 = cols[i]
            movie_rating_1 = user_ratings[0, movie_id_1]
            for j in range(i + 1, len(cols)):
                movie_id_2 = cols[j]
                movie_rating_2 = user_ratings[0, movie_id_2]
                if movie_rating_1 > movie_rating_2:
                    pair_node = 'pair_{0}>{1}'.format(movie_id_1, movie_id_2)
                    item_d_node = 'item_{0}_d'.format(movie_id_1)
                    item_u_node = 'item_{0}_u'.format(movie_id_2)
                else:
                    pair_node = 'pair_{0}>{1}'.format(movie_id_2, movie_id_1)
                    item_d_node = 'item_{0}_d'.format(movie_id_2)
                    item_u_node = 'item_{0}_u'.format(movie_id_1)

                if not tpg.has_node(item_d_node):
                    tpg.add_node(item_d_node)

                if not tpg.has_node(item_u_node):
                    tpg.add_node(item_u_node)

                if not tpg.has_node(pair_node):
                    tpg.add_node(pair_node)
                    tpg.add_edges_from([(pair_node, item_d_node), (pair_node, item_u_node)])

                # finally connect the user node to the pair node
                tpg.add_edge(user_node, pair_node)

    return tpg


def compute_pagerank(tpg, user_id):
    """
    this function computes the personalized pagerank of tgp with personalized vector which sets the target users element
    one and all other nodes as zero
    :param tpg:
    :param user_id:
    :return: pagerank matrix as a dictionary
    """

    personalization_matrix = {'user_{0}'.format(user_id): 1}
    # the matrix is generated using networkx library
    matrix = nx.pagerank(tpg, alpha=0.85, personalization=personalization_matrix)

    return matrix


def _compute_grank(pagerank, movie_id):
    """
    computes grank of single movie from the pagerank matrix using GRank=PPR_desirable/(PPR_desirable + PPR_undesirable)
    :param pagerank:
    :param movie_id:
    :return: grank value
    """

    ppr_d = pagerank.get('item_{0}_d'.format(movie_id), 0)
    ppr_u = pagerank.get('item_{0}_u'.format(movie_id), 0)

    # this if statement is to prevent division by zero errors
    if ppr_d == 0 and ppr_u == 0:
        return 0

    return ppr_d / (ppr_d + ppr_u)


def get_top_k_recommendations(pagerank, total_movies_count, k):
    """
    computes grank for each movie and sorts them. returns the top k items of the sorted list
    :param pagerank:
    :param total_movies_count:
    :param k:
    :return: a list of tuples containing grank and movie id
    """

    movie_granks = []
    for movie_id in range(1, total_movies_count + 1):
        grank = _compute_grank(pagerank, movie_id)
        movie_granks.append((grank, movie_id))

    movie_granks = sorted(movie_granks, key=lambda x: x[0], reverse=True)
    return movie_granks[:k]
