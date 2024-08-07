
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from collections import Counter
from sklearn.preprocessing import OneHotEncoder


class Solution():
    # structure of the solution
    def __init__(self):
        self.best_agent = None
        self.best_fitness = None
        self.best_auc = None
        self.execution_time = None
        self.convergence_curve = {}
        self.best_model = None

class Data():
    # structure of the training data
    def __init__(self):
        self.train_X = None
        self.train_Y = None
        self.val_X = None
        self.val_Y = None


def initialize(num_agents, num_features):
    # define min and max number of features
    min_features = int(0.3 * num_features)
    max_features = int(0.6 * num_features)

    # initialize the agents with zeros
    agents = np.zeros((num_agents, num_features))

    # select random features for each agent
    for agent_no in range(num_agents):
        # find random indices
        cur_count = np.random.randint(min_features, max_features)
        temp_vec = np.random.rand(1, num_features)
        temp_idx = np.argsort(temp_vec)[0][0:cur_count]

        # select the features with the ranom indices
        agents[agent_no][temp_idx] = 1

    return agents


def sort_agents(agents, data):
    train_X, val_X, train_Y, val_Y = data.train_X, data.val_X, data.train_Y, data.val_Y
    num_agents = agents.shape[0]
    fitness = np.zeros(num_agents)
    auc = np.zeros(num_agents)
    num = np.zeros(num_agents)
    for id, agent in enumerate(agents):
        fitness[id], auc[id], num[id] = compute_fitness(agent, train_X, val_X, train_Y, val_Y)

    idx = np.argsort(-fitness)
    sorted_agents = agents[idx].copy()
    sorted_fitness = fitness[idx].copy()
    sorted_auc = auc[idx].copy()
    sorted_num = num[idx].copy()

    return sorted_agents, sorted_fitness, sorted_auc, sorted_num


def compute_auc(agent, train_X, test_X, train_Y, test_Y):

    cols = np.flatnonzero(agent)
    train_data = train_X[:, cols]
    train_label = train_Y
    test_data = test_X[:, cols]
    test_label = test_Y

    # label_counts_np = np.bincount(train_label)
    # class_weights = {0: label_counts_np[1], 1: label_counts_np[0]}
    class_weights = {0: 2, 1: 1}
    model = SVC(probability=True, class_weight=class_weights)
    model.fit(train_data, train_label)

    probabilities = model.predict_proba(test_data)[:, 1]
    auc = roc_auc_score(test_label, probabilities)

    # probabilities = model.predict_proba(test_data)
    # encoder = OneHotEncoder(sparse=False, categories='auto')
    # test_label = encoder.fit_transform(test_label.reshape(-1, 1))
    # auc = roc_auc_score(test_label, probabilities, average='macro')

    return model, auc


def compute_fitness(agent, train_X, test_X, train_Y, test_Y):

    weight_auc = 0.9

    _, auc = compute_auc(agent, train_X, test_X, train_Y, test_Y)
    weight_feat = 1 - weight_auc
    num_features = agent.shape[0]
    feat = (num_features - np.sum(agent)) / num_features
    fitness = weight_auc * auc + weight_feat * feat

    return fitness, auc, np.sum(agent)


def Conv_plot(convergence_curve):
    # plot convergence curves
    num_iter = len(convergence_curve['fitness'])
    iters = np.arange(num_iter) + 1
    fig, axes = plt.subplots(1)
    fig.tight_layout(pad=5)
    fig.suptitle('Convergence Curves')

    axes.set_title('Convergence of Fitness over Iterations')
    axes.set_xlabel('Iteration')
    axes.set_ylabel('Best Fitness')
    axes.plot(iters, convergence_curve['fitness'])

    plt.show()


def display(agents, fitness, agent_name):
    # display the population
    print('\nNumber of agents: {}'.format(agents.shape[0]))
    print('\n------------- Best Agent ---------------')
    print('Fitness: {}'.format(fitness[0]))
    print('Number of Features: {}'.format(int(np.sum(agents[0]))))
    print('----------------------------------------\n')

    for id, agent in enumerate(agents):
        print('{} {} - Fitness: {}, Number of Features: {}'.format(agent_name, id + 1, fitness[id], int(np.sum(agent))))

    print('================================================================================\n')


def BorderCheck(X,ub,lb,pop,dim):
    '''边界检查函数'''
    '''
    dim:为每个个体数据的维度大小
    X:为输入数据，维度为[pop,dim]
    ub:为个体数据上边界，维度为[dim,1]
    lb:为个体数据下边界，维度为[dim,1]
    pop:为种群数量
    '''
    for i in range(pop):
        for j in range(dim):
            if X[i,j]>ub[j]:
                X[i,j] = ub[j]
            elif X[i,j]<lb[j]:
                X[i,j] = lb[j]
    return X



