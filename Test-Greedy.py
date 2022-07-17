from Learner.GreedyLearner import GreedyLearner
from Model.ConfigurationParametersAverage import mergeUserClasses
from Environment import Environment
from Model.Evaluator.GraphEvaluator import GraphEvaluator
from Model.Evaluator.MultiClassEvaluator import MultiClassEvaluator
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from IPython.display import clear_output
import json

# ========== PARAMETERS TO CHANGE =========
files = ['./Configs/config2.json', './Configs/config3.json', './Configs/ctx_config1.json']
n_experiments = 600

# =========================================

env = []
config_margins = []
optimal_arms = []
optimal_margins = []
conv_rates = []
prod_lists = []
click_probs = []
lambdas = []
alphas = []
units_means = []
clairvoyant_opt_rew = []
actual_unit_mean = []

for i in range(0, len(files)):
    env.append(Environment(config_path=files[i]))
    config = mergeUserClasses([files[i]], False)[0]
    config_margins.append(config["marginsPerPrice"])
    optimal_arms.append(config["optimalConfig"])
    optimal_margins.append(config["optimalMargin"])
    conv_rates.append(config["conversionRateLevels"])
    prod_lists.append(config["productList"])
    print("ProdList={}, Alphas={}, ConvRates={}".format(len(config["productList"]), len(config["alphas"]),
                                                        len(config["conversionRateLevels"])))
    click_probs.append(config["click_prob"])
    lambdas.append(config['lambda_p'])
    alphas.append(config["alphas"])
    clairvoyant_opt_rew.append(config["optimalMargin"])
    units_means.append(config["units_mean"])
    actual_unit_mean.append(config["actual_units_mean"])


fig, axes = plt.subplots(ncols=2, nrows=len(env), sharex="all", figsize=(16, 12))
plt.suptitle("Greedy Learner")

for i in range(0, len(env)):
    config_name = files[i][files[i].rfind('/') - len(files[i]) + 1:]
    print("Running config: ", config_name)
    learner = GreedyLearner()

    learner_graph_margins = np.array([])
    learner_env_margins = np.array([])
    multiEval = MultiClassEvaluator(config_path=files[i])

    for j in tqdm(range(0, n_experiments)):
        single_margin = 0
        opt_single_margin = 0
        armMargins = []
        armConvRates = []

        # compute the margin for the learner
        pulledArm = learner.pull_arm()

        for k in range(0, len(pulledArm)):
            armMargins.append(config_margins[i][k][pulledArm[k]])
            armConvRates.append(conv_rates[i][k][pulledArm[k]])

        graphEval = GraphEvaluator(products_list=prod_lists[i], click_prob_matrix=click_probs[i],
                                   lambda_prob=lambdas[i],
                                   alphas=alphas[i], conversion_rates=armConvRates, margins=armMargins,
                                   units_mean=actual_unit_mean[i], verbose=False, convert_units=False)

        env[i].setPriceLevels(pulledArm)
        interactions = env[i].round()
        single_margin = multiEval.computeMargin(pulledArm)

        env_margin = 0
        for k in range(0, len(interactions)):
            env_margin = env_margin + interactions[k].linearizeMargin(config_margins[i])
        env_margin = env_margin / len(interactions)

        learner.update(single_margin)

        learner_graph_margins = np.append(learner_graph_margins, single_margin)
        learner_env_margins = np.append(learner_env_margins, env_margin)

    opt_arms = optimal_arms[i]
    armMargins = []
    armConvRates = []
    for k in range(0, len(opt_arms)):
        armMargins.append(config_margins[i][k][opt_arms[k]])
        armConvRates.append(conv_rates[i][k][opt_arms[k]])

    graphEval = GraphEvaluator(products_list=prod_lists[i], click_prob_matrix=click_probs[i],
                               lambda_prob=lambdas[i],
                               alphas=alphas[i], conversion_rates=armConvRates, margins=armMargins,
                               units_mean=actual_unit_mean[i], verbose=False, convert_units=False)

    opt_margin = multiEval.computeMargin(opt_arms)
    optimal = np.full(n_experiments, opt_margin)

    x = np.linspace(0, n_experiments, n_experiments)
    axes[i, 0].plot(x, optimal)
    axes[i, 0].plot(x, learner_graph_margins)
    # axes[i, 0].plot(x, learner_env_margins)
    axes[i, 0].set_xlabel("Time step")
    axes[i, 0].set_ylabel("Margins\n{}".format(config_name))
    axes[0, 0].set_title("Expected margins over time")

    axes[i, 1].plot(x, optimal)
    cum_rews_graph = np.cumsum(learner_graph_margins)
    avg_cum_rews_graph = np.divide(cum_rews_graph, np.arange(1, n_experiments + 1))
    axes[i, 1].plot(x, avg_cum_rews_graph)
    cum_rews = np.cumsum(learner_env_margins)
    avg_cum_rews = np.divide(cum_rews, np.arange(1, n_experiments + 1))
    axes[i, 1].plot(x, avg_cum_rews)
    axes[i, 1].set_xlabel("Time step")
    axes[i, 1].set_ylabel("Cumulative margins")
    axes[0, 1].set_title("Average reward")
    print("Optimal arm found:\n\t", learner.pull_arm(), "\nOptimal theoretical arm:\n\t", optimal_arms[i])

plt.show()