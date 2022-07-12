from Learner.UCB.UCB_Step4 import UCB_Step4
from ContextGeneration.ContextualLearner import ContextualLearner
from ContextGeneration.ContextTreeNode import ContextTreeNode
from Model.ConfigurationParametersAverage import mergeUserClasses
from Environment import Environment
from Model.Evaluator.GraphEvaluator import GraphEvaluator
import matplotlib.pyplot as plt
import numpy as np
from Model.Evaluator.MultiClassEvaluator import MultiClassEvaluator

files = ['./Configs/config1.json', './Configs/config2.json', './Configs/config3.json', './Configs/configDump.json',
         './Configs/configuration4.json', './Configs/configuration5.json', './Configs/configuration6.json']
env = []
config_margins = []
optimal_arms = []
conv_rates = []
prod_lists = []
click_probs = []
lambdas = []
alphas = []
units_means = []
clairvoyant_opt_rew = []
actual_unit_mean = []
features_names = []

for i in range(0, len(files)):
    env.append(Environment(config_path=files[i]))
    features_names.append(env[i].classes[0].features_names)
    config = mergeUserClasses([files[i]], False)[0]
    config_margins.append(config["marginsPerPrice"])
    optimal_arms.append(config["optimalConfig"])
    conv_rates.append(config["conversionRateLevels"])
    prod_lists.append(config["productList"])
    # print("ProdList={}, Alphas={}, ConvRates={}".format(len(config["productList"]), len(config["alphas"]),
                                                      #  len(config["conversionRateLevels"])))
    click_probs.append(config["click_prob"])
    lambdas.append(config['lambda_p'])
    alphas.append(config["alphas"])
    clairvoyant_opt_rew.append(config["optimalMargin"])
    units_means.append(config["units_mean"])
    actual_unit_mean.append(config["actual_units_mean"])


n_experiments = 200
fig, axes = plt.subplots(ncols=2, nrows=len(env), sharex="all", figsize=(16, 12))

for i in range(0, len(env)):
    config_name = files[i][files[i].rfind('/') - len(files[i]) + 1:]
    print("Running config: ", config_name)
    learner = ContextualLearner(margins=config_margins[i], clickProbability=click_probs[i],
                                secondary=prod_lists[i], Lambda=lambdas[i], debug=False,
                                features_names=features_names[i], approach='ts')
    multiEvaluator = MultiClassEvaluator(config_path=files[i])
    learner_graph_margins = []
    for j in range(0, n_experiments):
        arms = learner.pull_arm()
        env[i].setPriceLevelsForContexts(arms)
        interaction = env[i].round()
        learner.update(interaction)
        learner_graph_margins.append(multiEvaluator.computeMargin_per_class(arms))

    print(learner.tree)

    x = np.linspace(0, n_experiments, n_experiments)
    axes[i, 0].plot(x, learner_graph_margins)
    # axes[i, 0].plot(x, learner_env_margins)
    axes[i, 0].set_xlabel("Time step")
    axes[i, 0].set_ylabel("Margins\n{}".format(config_name))
    axes[0, 0].set_title("Expected margins over time")

    cum_rews_graph = np.cumsum(learner_graph_margins)
    avg_cum_rews_graph = np.divide(cum_rews_graph, np.arange(1, n_experiments + 1))
    axes[i, 1].plot(x, avg_cum_rews_graph)

    axes[i, 1].set_xlabel("Time step")
    axes[i, 1].set_ylabel("Cumulative margins")
    axes[0, 1].set_title("Average reward")
    print("Optimal arm found:\n\t", learner.pull_arm(), "\nOptimal theoretical arm:\n\t", optimal_arms[i])

plt.show()

