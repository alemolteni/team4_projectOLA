from re import S

from matplotlib import units
from Learner.UCB.UCB_Step4 import UCB_Step4
from ContextGeneration.ContextualLearner import ContextualLearner
from ContextGeneration.ContextTreeNode import ContextTreeNode
from Model.ConfigurationParametersAverage import mergeUserClasses
from Environment import Environment
from Model.Evaluator.GraphEvaluator import GraphEvaluator
import matplotlib.pyplot as plt
import numpy as np
from Model.Evaluator.MultiClassEvaluator import MultiClassEvaluator
from tqdm import tqdm
from IPython.display import clear_output


#files = ['./Configs/config1.json', './Configs/config2.json', './Configs/config3.json', './Configs/configDump.json',
#         './Configs/configuration4.json', './Configs/configuration5.json', './Configs/configuration6.json']
files = ['./Configs/configuration4.json', './Configs/configuration5.json', './Configs/configuration6.json']
approach = 'ts'

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
clairvoyant_opt_context = []
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
    clairvoyant_opt_context.append(config["optimalContextual"])
    units_means.append(config["units_mean"])
    actual_unit_mean.append(config["actual_units_mean"])


n_experiments = 200
fig, axes = plt.subplots(ncols=2, nrows=len(env), sharex="all", figsize=(16, 12))

used_learners = []
for i in range(0, len(env)):
    config_name = files[i][files[i].rfind('/') - len(files[i]) + 1:]
    # print("Running config: ", config_name)
    learner = ContextualLearner(margins=config_margins[i], clickProbability=click_probs[i],
                                secondary=prod_lists[i], Lambda=lambdas[i], debug=False,
                                features_names=features_names[i], approach=approach)
    used_learners.append(learner)
    multiEvaluator = MultiClassEvaluator(config_path=files[i])
    learner_graph_margins = []
    time_first_split = 0
    environment_margins = []
    for j in tqdm(range(0, n_experiments)):
        arms = learner.pull_arm()
        env[i].setPriceLevelsForContexts(arms)
        interaction = env[i].round()
        if len(arms) > 1 and time_first_split == 0:
            time_first_split = j
            interactions_for_test = interaction
            # print("Time first split = {}".format(time_first_split))
        learner.update(interaction)
        learner_graph_margins.append(multiEvaluator.computeMargin_per_class(arms))

        round_margin = 0
        for inter in interaction:
            round_margin += inter.linearizeMargin(config_margins[i])
        environment_margins.append(round_margin/len(interaction))

    print(learner.tree)
    print("Time first split is ", time_first_split)

    non_contextual = np.full(time_first_split, clairvoyant_opt_rew[i])
    contextual = np.full(n_experiments - time_first_split, clairvoyant_opt_context[i])
    optimal_possible = np.hstack([non_contextual, contextual])

    x = np.linspace(0, n_experiments, n_experiments)
    axes[i, 0].plot(x, learner_graph_margins)
    axes[i, 0].plot(x, optimal_possible)
    axes[i, 0].plot(x, environment_margins)
    # axes[i, 0].plot(x, learner_env_margins)
    axes[i, 0].set_xlabel("Time step")
    axes[i, 0].set_ylabel("Margins\n{}".format(config_name))
    axes[0, 0].set_title("Expected margins over time")

    cum_rews_graph = np.cumsum(learner_graph_margins)
    avg_cum_rews_graph = np.divide(cum_rews_graph, np.arange(1, n_experiments + 1))
    avg_cum_rews_env = np.divide(np.cumsum(environment_margins), np.arange(1, n_experiments + 1))
    axes[i, 1].plot(x, avg_cum_rews_graph)
    axes[i, 1].plot(x, optimal_possible)
    axes[i, 1].plot(x, avg_cum_rews_env)
    axes[i, 1].set_xlabel("Time step")
    axes[i, 1].set_ylabel("Cumulative margins")
    axes[0, 1].set_title("Average reward")
    # print("Optimal arm found:\n\t", learner.pull_arm(), "\nOptimal theoretical arm:\n\t", optimal_arms[i])

plt.show()

for i in range(0, len(files)):
    print("\nReport for file {}:".format(files[i]))
    leaves = used_learners[i].tree.get_leaves()

    print("   - [ALPHAS] Estimated starting parameters are: ")
    id_class = 0
    for leaf in leaves:
        print("      - [CLASS {}] Features {} estimated alpha {}".format(id_class, leaf.split_features, np.round(leaf.learner.alphas, decimals=3)))
        id_class += 1

    print("   - [UNITS] Estimated mean number of units sold are: ")
    id_class = 0
    for leaf in leaves:
        print("      - [CLASS {}] Features {} estimated #units {}".format(id_class, leaf.split_features, np.round(leaf.learner.units_mean, decimals=3)))
        id_class += 1

    np.set_printoptions(linewidth=np.inf)
    print("   - [CONVERSION RATES] Estimated conversion rates are: ")
    id_class = 0
    for leaf in leaves:
        if approach == 'ucb':
            print("      - [CLASS {}] Features {} estimated CR \n{}".format(id_class, leaf.split_features, np.round(leaf.learner.conversion_rates, decimals=3)))
        else:
            conv_rates_exp = np.zeros((len(leaf.learner.units_mean), len(leaf.learner.margins[0])), dtype=float)
            for i in range(0,len(leaf.learner.units_mean)):
                for j in range(0,len(leaf.learner.margins[0])):
                    bought = leaf.learner.conversion_rates_distro[i][j][0]
                    not_bought = leaf.learner.conversion_rates_distro[i][j][1]
                    conv_rates_exp[i][j] = bought / (bought + not_bought)
            print("      - [CLASS {}] Features {} estimated CR \n{}".format(id_class, leaf.split_features, np.round(conv_rates_exp, decimals=3)))
        id_class += 1
