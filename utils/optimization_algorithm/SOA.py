
import numpy as np
import time
from sklearn.model_selection import train_test_split
from utils.optimization_algorithm._utilities import Solution, Data, initialize, sort_agents, display, Conv_plot,\
    compute_auc, BorderCheck
from utils.optimization_algorithm._transfer_functions import get_trans_function
from utils.altruism import generate_scc, generate_pcc, Altruism


############################# SOA #############################
def SOA(num_agents, max_iter, train_data, train_label,
        altruism_indi
        ):

    seed = 0
    np.random.seed(seed)

    trans_function_shape = 's'
    trans_function = get_trans_function(trans_function_shape)

    agent_name = 'Seagull'
    train_data, train_label = np.array(train_data), np.array(train_label)
    num_features = train_data.shape[1]


    # initialize chromosomes and Leader (the agent with the max fitness)
    seagulls = initialize(num_agents, num_features)
    Leader_agent = np.zeros((1, num_features))
    Leader_fitness = float("-inf")

    # initialize convergence curves
    convergence_curve = {}
    convergence_curve['fitness'] = np.zeros(max_iter)
    convergence_curve['auc'] = np.zeros(max_iter)
    convergence_curve['num'] = np.zeros(max_iter)

    # initialize data class
    data = Data()
    val_size = 0.3
    data.train_X, data.val_X, data.train_Y, data.val_Y = train_test_split(train_data, train_label, stratify=train_label,
                                                                          test_size=val_size, random_state=42)

    # create a solution object
    solution = Solution()

    # rank initial population
    seagulls, fitness, auc, num = sort_agents(seagulls, data)

    # start timer
    start_time = time.time()

    fc = 2  # 可调
    MS = np.zeros([num_agents, num_features])
    CS = np.zeros([num_agents, num_features])
    DS = np.zeros([num_agents, num_features])

    #Altruism prerequisite setup
    scc = generate_scc(data.train_X)  # 原论文中计算的是特征和标签的相似度。而这里计算特征间相似度，是一种无监督方式。
    pcc = generate_pcc(data.train_X)

    lb = -10 * np.ones(num_features)  # 下边界
    ub = 10 * np.ones(num_features)  # 上边界

    # main loop
    for iter_no in range(max_iter):
        print('\n================================================================================')
        print('                          Iteration - {}'.format(iter_no + 1))
        print('================================================================================\n')

        Pbest = seagulls[0, :]

        for seagull in range(num_agents):
            starting_seagulls = seagulls.copy()
            #计算Cs
            A = fc - (iter_no*(fc/max_iter))
            CS[seagull, :] = seagulls[seagull, :]*A
            #计算Ms
            rd = np.random.random()
            B = 2*(A**2)*rd
            MS[seagull, :] = B*(Pbest - seagulls[seagull, :])
            #计算Ds
            DS[seagull, :] = np.abs(CS[seagull, :] + MS[seagull, :])
            #局部搜索
            u = 1
            v = 1
            theta = np.random.random()
            r = u*np.exp(theta*v)
            x = r*np.cos(theta*2*np.pi)
            y = r*np.sin(theta*2*np.pi)
            z = r*theta
            #攻击
            seagulls[seagull, :] = x*y*z*DS[seagull, :] + Pbest

            seagulls = BorderCheck(seagulls, ub, lb, num_agents, num_features)  # 边界检测

            # Apply transformation function on the updated whale
            for j in range(num_features):
                trans_value = trans_function(seagulls[seagull, j])
                if (np.random.random() < trans_value):
                    seagulls[seagull, j] = 1
                else:
                    seagulls[seagull, j] = 0

            if np.all(seagulls[seagull] == 0):
                # 如果数组全部为0，则随机选择一个索引，将其值设置为1
                random_index = np.random.randint(0, len(seagulls[seagull]))
                seagulls[seagull][random_index] = 1

        #Altruism Operation
        altruism_seagulls = np.concatenate((starting_seagulls, seagulls), axis=0)
        seagulls = Altruism(altruism_seagulls, data.train_X, data.train_Y, data.val_X, data.val_Y,
                          scc_score=scc, pcc_score=pcc,
                          altruism_indi=altruism_indi,
                          pop_size=starting_seagulls.shape[0], alpha=0.5)

        # update final information
        seagulls, fitness, auc, num = sort_agents(seagulls, data)
        display(seagulls, fitness, agent_name)

        if fitness[0] > Leader_fitness:
            Leader_agent = seagulls[0].copy()
            Leader_fitness = fitness[0].copy()
            Leader_auc = auc[0].copy()
            Leader_num = num[0].copy()

        convergence_curve['fitness'][iter_no] = Leader_fitness
        convergence_curve['auc'][iter_no] = Leader_auc
        convergence_curve['num'][iter_no] = Leader_num

    # compute final auc
    model, Leader_auc = compute_auc(Leader_agent, data.train_X, data.val_X, data.train_Y, data.val_Y)

    print('\n================================================================================')
    print('                                    Final Result                                  ')
    print('================================================================================\n')
    print('Leader ' + agent_name + ' Dimension : {}'.format(int(np.sum(Leader_agent))))
    print('Leader ' + agent_name + ' Fitness : {}'.format(Leader_fitness))
    print('Leader ' + agent_name + ' Classification auc : {}'.format(Leader_auc))
    print('\n================================================================================\n')

    # stop timer
    end_time = time.time()
    exec_time = end_time - start_time

    # plot convergence graph
    Conv_plot(convergence_curve)

    # update attributes of solution
    solution.best_agent = Leader_agent
    solution.best_fitness = Leader_fitness
    solution.best_auc = Leader_auc
    solution.convergence_curve = convergence_curve
    solution.execution_time = exec_time
    solution.best_model = model

    return solution
