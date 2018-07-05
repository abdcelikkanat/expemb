import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random


example1 = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [4, 5], [4,6], [4,7], [5,6], [5,7], [6,7], [3,4]]
example1 = [[str(edge[0]), str(edge[1])] for edge in example1]

example2 = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [4, 5], [4,6], [4,7], [5,6], [5,7], [6,7]]
example2 = [[str(edge[0]), str(edge[1])] for edge in example2]

example3 = [[0, 1], [0, 2], [0, 3], [0, 4], [5, 6], [5, 7], [5, 8], [5, 9], [0, 5]]
example3 = [[str(edge[0]), str(edge[1])] for edge in example3]

"""
B(i): the embedding of the node i
T(j): the embedding of the node j

e_{ij} ~ Bern(p_{ij})
p_{ij} = N(T(j); B(i), I)

Results:
Karate network u icin 1 nb de ayrim fena sayilmaz ama 2 nb ve ustunde her grup tek bir noktadaa toplasiyor.
Citeseer, 150. iter icin yaklasik yuzde 22 lerde

"""

def find_distances(graph):
    N = nx.number_of_nodes(graph)

    spl = nx.shortest_path_length(graph)

    dist = np.zeros(shape=(N, N), dtype=np.int)

    for p in spl:
        for target in p[1]:
            dist[int(p[0]), int(target)] = p[1][target]

    return dist


def find_neighbors(g):
    N = g.number_of_nodes()
    nb_list = [[] for _ in range(N)]

    for node in g.nodes():

        for nb in nx.neighbors(g, node):
            if int(nb) not in nb_list[int(node)]:
                nb_list[int(node)].append(int(nb))

    return nb_list

def find_neighbors4(g):
    N = g.number_of_nodes()
    nb_list = [[] for _ in range(N)]

    for node in g.nodes():

        for nb in nx.neighbors(g, node):
            if int(nb) not in nb_list[int(node)]:
                nb_list[int(node)].append(int(nb))

            for nb_nb in nx.neighbors(g, nb):
                if int(nb_nb) not in nb_list[int(node)]:
                    nb_list[int(node)].append(int(nb_nb))

                for nb_nb_nb in nx.neighbors(g, nb_nb):
                    if int(nb_nb_nb) not in nb_list[int(node)]:
                        nb_list[int(node)].append(int(nb_nb_nb))

                    for nb_nb_nb_nb in nx.neighbors(g, nb_nb_nb):
                        if int(nb_nb_nb_nb) not in nb_list[int(node)]:
                            nb_list[int(node)].append(int(nb_nb_nb_nb))
            """ """

    return nb_list

def grad_T(g, B, T, nb_list, node):
    N = g.number_of_nodes()

    grad_sum = 0.0
    for v in range(N):
        if node in nb_list[v]:
            grad_sum += -(T[node, :] - B[v, :])
        else:
            diff = T[node, :] - B[v, :]
            prod = np.dot(diff, diff)
            grad_sum += ( +np.exp(-prod/2.0)*diff ) / ( 1.0 - np.exp(-prod/2.0) )

    return grad_sum

def grad_B(g, B, T, nb_list, node):
    N = g.number_of_nodes()

    grad_sum = 0.0
    for u in range(N):
        if node in nb_list[u]:
            grad_sum += (T[u, :] - B[node, :])
        else:
            diff = T[u, :] - B[node, :]
            prod = np.dot(diff, diff)
            grad_sum += ( -np.exp(-prod/2.0)*diff ) / ( 1.0 - np.exp(-prod/2.0) )

    return grad_sum

def compute_score(g, B, T ,nb_list, dist):

    N = g.number_of_nodes()

    var = 1.0
    #*float(dist[v, u])*float(dist[v, u])
    score = 0.0
    for v in range(N):
        for u in range(N):
            diff = T[u, :] - B[v, :]
            prod = np.dot(diff, diff)
            if u in nb_list[v]:
                score += -prod / 2.0
            else:
                score += np.log( 1.0 - np.exp(-prod/2.0) )

    return score

def draw_points(B, T, name="", g=None, base=False):
    if B.shape[1] != 2:
        raise ValueError("Dim must be 2")


    if name == "Karate":
        groundTruth = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 1, 1, 1]
        plt.figure()
        for inx in range(g.number_of_nodes()):
            if groundTruth[inx] == 0:
                plt.plot(T[inx, 0], T[inx, 1], 'r.')
                if base is True:
                    plt.plot(B[inx, 0], B[inx, 1], 'rx')
            if groundTruth[inx] == 1:
                plt.plot(T[inx, 0], T[inx, 1], 'bv')
                if base is True:
                    plt.plot(B[inx, 0], B[inx, 1], 'bx')
        plt.show()

    else:
        plt.figure()
        plt.plot(T[:4, 0], T[:4, 1], 'b.')
        plt.plot(T[4:, 0], T[4:, 1], 'r.')
        plt.show()


def run(g, dim, num_of_iters, eta):
    N = nx.number_of_nodes(g)

    # Initialize parameters
    B = np.random.normal(size=(N, dim))
    T = np.random.normal(size=(N, dim))
    B = np.load("./numpy_files/citeseer_poisson_gaussian_iter_150_B.npy")
    T = np.load("./numpy_files/citeseer_poisson_gaussian_iter_150_T.npy")

    nb_list = find_neighbors(g)

    #dist = find_distances(g)
    dist = []

    for iter in range(151, num_of_iters):
        #if iter % 10 == 0:
        #    draw_points(B, T, "Karate", g, base=True)
        for node in range(N):

            node_grad_B = grad_B(g, B, T, nb_list, node)

            for nb in nb_list[int(node)]:
                node_grad_T = grad_T(g, B, T, nb_list, nb)
                T[nb, :] += eta * node_grad_T

            B[node, :] += eta * node_grad_B


        score = compute_score(g, B, T, nb_list, dist)
        print("Iter: {} Score {}".format(iter, score))

        if iter % 50 == 0:
            np.save("./numpy_files/citeseer_gaussian_iter_{}_T".format(iter), T)
            np.save("./numpy_files/citeseer_gaussian_iter_{}_B".format(iter), B)
    #draw_points(B, T, "Karate", g, base=False)

    return B, T


#edges = example1
#g = nx.Graph()
#g.add_edges_from(edges)
g = nx.read_gml("../datasets/citeseer.gml")


B, T = run(g, dim=128, num_of_iters=1500, eta=0.001)
np.save("./numpy_files/citeseer_gaussian_son_T", T)
np.save("./numpy_files/citeseer_gaussian_son_B", B)
#draw_points(B, T, "Karate", g)
