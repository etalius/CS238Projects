import sys
import numpy as np
import pandas

class Q_learn():
    def __init__(self, num_states, num_actions, gamma, alpha):
        self.S = [i for i in range(num_states)]
        self.A = [i for i in range(num_actions)]
        self.Q = np.zeros((num_states, num_actions))
        self.l = ()
        self.gamma = gamma
        self.alpha = alpha


def compute_Q(infile, outfile, num_states, num_actions, gamma, alpha):
    data = pandas.read_csv(infile)
    data_matrix = data.to_numpy()

    num_samples = data_matrix.shape[0]
    print(num_samples)

    model = Q_learn(num_states, num_actions, gamma, alpha)

    for x in range(10):
        print(x)
        for i in range(num_samples):
            row = data_matrix[i]
            s = row[0] - 1
            a = row[1] - 1
            r = row[2]
            s_prime = row[3] - 1

            gamma, Q, alpha = model.gamma, model.Q, model.alpha
            s_prime_row = model.Q[s_prime]
            max_new_Q = s_prime_row.max()
            model.Q[s, a] += alpha * (r + gamma * max_new_Q - Q[s, a])


    pol = []

    for i in range(num_states):
        row = model.Q[i]
        result = row.argmax()
        pol.append(result + 1)

    df = pandas.DataFrame(pol)
    df.to_csv(outfile, index=False)



def main():
    if len(sys.argv) != 7:
        raise Exception("incorrect input")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    num_states = int(sys.argv[3])
    num_actions = int(sys.argv[4])
    gamma = float(sys.argv[5])
    alpha = float(sys.argv[6])

    compute_Q(inputfilename, outputfilename, num_states, num_actions, gamma, alpha)


if __name__ == '__main__':
    main()
