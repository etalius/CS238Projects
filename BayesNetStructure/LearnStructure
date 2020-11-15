import sys
import numpy as np
import pandas
from scipy.special import loggamma
import matplotlib.pyplot as plt
import networkx as nx

class LearnBayes():

    def __init__(self, data, num_vars, num_samples, vars, matrix):
        self.prevM = []
        self.oldScores = []
        self.old_total_score = float("-inf")
        self.data = data
        self.num_vars = num_vars
        self.num_samples = num_samples
        self.vars = vars
        self.r_vals = data.max()
        self.r_list = list(self.r_vals)
        self.matrix = matrix

    def initG(self):
        G = nx.DiGraph()
        nums = [x for x in range(self.num_vars)]
        G.add_nodes_from(range(len(nums)))
        return G

    def calc_mjk(self, G):

        q = []
        for i in range(self.num_vars):
            preds = list(G.predecessors(i))
            q_i = 1
            for pred in preds:
                q_i *= self.r_list[pred]
            q.append(q_i)

        M = [np.zeros((q[i], self.r_list[i])) for i in range(self.num_vars)]
        P = [np.ones((q[i], self.r_list[i])) for i in range(self.num_vars)]

        updates = []
        if len(self.prevM) != 0:
            for i in range(self.num_vars):
                if M[i].shape != self.prevM[i].shape:
                    updates.append(i)

            M = self.prevM.copy()
            for i in updates:
                M[i] = np.zeros((q[i], self.r_list[i]))

        else:
            updates = list(range(self.num_vars))

        for row in range(self.num_samples):

            for i in updates:
                k = self.matrix[row, i]
                parents = list(G.predecessors(i))
                j = 1
                if len(parents) != 0:
                    dims = []
                    translate = []
                    for par in parents:
                        dims.append(self.r_list[par])
                        translate.append(self.matrix[row, par] - 1)
                    j = np.ravel_multi_index(tuple(translate), tuple(dims))
                M[i][j - 1, k - 1] += 1
        return M, P, updates

    def bayesComp(self, M, P):
        s = np.add(M, P)
        p = np.sum(loggamma(s))
        p -= np.sum(loggamma(P))
        p += np.sum(loggamma(np.sum(P, axis=1)))
        p -= np.sum(loggamma(np.sum(P, axis=1) + np.sum(M, axis=1)))
        return p

    def calculate_score(self, G):
        M, P, changes = self.calc_mjk(G)

        if self.old_total_score == float("-inf"):
            scores = [self.bayesComp(M[i], P[i]) for i in range(self.num_vars)]
            self.prevM = M
            self.old_total_score = sum(scores)
            self.oldScores = scores
            return self.old_total_score, scores, M

        else:
            scores = self.oldScores.copy()
            for i in changes:
                scores[i] = self.bayesComp(M[i], P[i])

            new_score = self.old_total_score + sum(scores[i] for i in changes) - sum(self.oldScores[i] for i in changes)
            return new_score, scores, M

    def K2(self, G):
        score = self.old_total_score
        best_scores = self.oldScores
        best_prevM = self.prevM
        for i in range(1, self.num_vars):
            print(i)
            k = i
            while True:
                y_best = float("-inf")
                j_best = 0
                if len(list(G.predecessors(i))) >= 4:
                    break
                for j in range(k):
                    if not i in G[j]:
                        G.add_edge(j, i)
                        if len(list(nx.simple_cycles(G))) != 0:
                            y_new = float("-inf")
                            scores = []
                            M = []
                        else:
                            y_new, scores, M = self.calculate_score(G)
                        if y_new > y_best:
                            y_best, j_best = y_new, j
                            best_scores = scores
                            best_prevM = M
                        G.remove_edge(j, i)
                if y_best > score:
                    score = y_best
                    self.old_total_score = y_best
                    self.oldScores = best_scores
                    self.prevM = best_prevM
                    G.add_edge(j_best, i)
                else:
                    break
        print(score)
        return G



def compute(infile, outfile):
    data = pandas.read_csv(infile)
    var_names = list(data.columns.values)
    matrix = data.to_numpy()

    num_vars = len(var_names)
    num_samples = data.shape[0]

    idx2names = {}
    for i in range(num_vars):
        idx2names[i] = var_names[i]

    problem = LearnBayes(data, num_vars, num_samples, var_names, matrix)
    G = problem.initG()
    problem.calculate_score(G)
    finalG = problem.K2(G)

    newG = nx.relabel_nodes(finalG, idx2names)


    nx.draw_networkx(newG, node_size=1000, node_color='#a7d3f0', font_size=9, edge_color='#686a6b')
    plt.show()


    write_gph(finalG, idx2names, outfile)


def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))

def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    graphfile = sys.argv[3]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()
