
from utils.utils import data_preprocess, selected_features, model_test, PasiLuukka, evaluation_metrics
from utils.optimization_algorithm.SOA import SOA
import joblib
import argparse
import pandas as pd
import numpy as np


# INPUT ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./data/radiation', help='Path to where the csv file of features')
parser.add_argument('--num_agents', type=int, default=20, help='Population size')
parser.add_argument('--max_iter', type=int, default=50, help='Maximum number of iterations to run AWOA')
parser.add_argument('--altruism_indi', type=int, default=10, help='Number of altruistic individuals')
args = parser.parse_args()

X_train, X_test, _, y_train, y_test, _ = data_preprocess(data_path=args.data_path)

# 特征选择
solution = SOA(num_agents=args.num_agents, max_iter=args.max_iter, train_data=X_train, train_label=y_train,
               altruism_indi=args.altruism_indi
               )

all_features = pd.DataFrame({'features': pd.read_csv(args.data_path+'/Training_set.csv').columns.tolist()[1:-1]})
features_selected = selected_features(all_features=all_features, best_agent=solution.best_agent)
print('features selected:\n', features_selected)

joblib.dump(solution.best_model, './result/best_model.joblib')  # 保存训练好的模型到文件
writer = pd.ExcelWriter('./result/output.xlsx', engine='xlsxwriter')
pd.DataFrame(features_selected).to_excel(writer, index=False, sheet_name='features_selected')
pd.DataFrame(solution.best_agent).to_excel(writer, index=False, sheet_name='best_agent')
pd.DataFrame(solution.convergence_curve).to_excel(writer, index=False, sheet_name='convergence_curve')
writer.save()

# 使用训练好的模型在测试集上测试
loaded_model = joblib.load('./result/best_model.joblib')
model_test(model=loaded_model, data=X_train, label=y_train, best_agent=solution.best_agent, flag='train')
model_test(model=loaded_model, data=X_test, label=y_test, best_agent=solution.best_agent, flag='test')

# 输出评价指标
best_agent = pd.read_excel('./result/output.xlsx', sheet_name='best_agent')
train_label = pd.read_excel('./result/output.xlsx', sheet_name='train_label')
train_y_prob = pd.read_excel('./result/output.xlsx', sheet_name='train_y_prob')
test_label = pd.read_excel('./result/output.xlsx', sheet_name='test_label')
test_y_prob = pd.read_excel('./result/output.xlsx', sheet_name='test_y_prob')

writer = pd.ExcelWriter('./result/metrics.xlsx', engine='xlsxwriter')
metrics_df = evaluation_metrics(true_label=train_label, pre_score=train_y_prob, agent=best_agent)
metrics_df.to_excel(writer, sheet_name='train', index=False)
metrics_df = evaluation_metrics(true_label=test_label, pre_score=test_y_prob, agent=best_agent)
metrics_df.to_excel(writer, sheet_name='test', index=False)
writer.save()
