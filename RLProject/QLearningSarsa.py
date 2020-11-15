import sys
import numpy as np
import pandas

class Q_learn():
    def __init__(self, num_states, num_actions, gamma, alpha, lam):
        self.S = [i for i in range(num_states)]
        self.A = [i for i in range(num_actions)]
        self.Q = np.zeros((num_states, num_actions))
        self.gamma = gamma
        self.alpha = alpha
        self.N = np.zeros((num_states, num_actions))
        self.lam = lam


def compute_Q(infile, outfile, num_states, num_actions, gamma, alpha, lam):
    data = pandas.read_csv(infile)
    data_matrix = data.to_numpy()

    num_samples = data_matrix.shape[0]

    model = Q_learn(num_states, num_actions, gamma, alpha, lam)

    for x in range(10):
        for i in range(num_samples):
            row = data_matrix[i]
            s = row[0] - 1
            a = row[1] - 1
            r = row[2]
            s_prime = row[3] - 1

            s_prime_row = model.Q[s_prime]
            max_new_Q = s_prime_row.max()
            gamma, Q, alpha = model.gamma, model.Q, model.alpha

            model.N[s, a] += 1

            temp = r + gamma * max_new_Q - Q[s, a]

            for j in range(len(model.S)):
                state = model.S[j]
                for a in range(len(model.A)):
                    action = model.A[a]
                    Q[state, action] += alpha * temp * model.N[state, action]
                    model.N[state, action] *= gamma * lam


    pol = []

    for i in range(num_states):
        row = model.Q[i]
        result = row.argmax()
        pol.append(result + 1)

    df = pandas.DataFrame(pol)
    df.to_csv(outfile, index=False)



def main():
    if len(sys.argv) != 8:
        raise Exception("incorrect input")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    num_states = int(sys.argv[3])
    num_actions = int(sys.argv[4])
    gamma = float(sys.argv[5])
    alpha = float(sys.argv[6])
    lam = float(sys.argv[7])

    compute_Q(inputfilename, outputfilename, num_states, num_actions, gamma, alpha, lam)


if __name__ == '__main__':
    main()
