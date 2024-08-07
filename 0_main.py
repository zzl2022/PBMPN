from utils.optimization_algorithm.SOA import SOA

# 特征选择
solution = SOA(num_agents=args.num_agents, max_iter=args.max_iter, train_data=X_train, train_label=y_train,
               altruism_indi=args.altruism_indi
               )
