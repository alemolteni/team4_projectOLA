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
import json

np.seterr(all='raise')
# ======== PARAMETERS TO CHANGE ===========
#files = ['./Configs/config2.json', './Configs/config3.json',
#         './Configs/configuration4.json', './Configs/configuration5.json', './Configs/configuration6.json']
files = ['./Configs/ctx_config2.json', './Configs/ctx_config2.json','./Configs/ctx_config2.json','./Configs/ctx_config2.json']
approach = 'ts' # "ucb" OR "ts"
MODE = "plots" # "plots" OR "runs"
n_experiments = 400
n_runs = 10
# =========================================

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

cp_class_per_env = []

for i in range(0, len(files)):
    env.append(Environment(config_path=files[i]))

    class_cp = []
    for uc in env[i].classes:
        feat = {uc.features_names[0]: uc.features_values[0],
                uc.features_names[1]: uc.features_values[1]}
        class_cp.append([uc.raw_click, feat, uc.n_user[0]])
    cp_class_per_env.append(class_cp)

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
#print(click_probs[0])

if MODE == "plots":
    fig, axes = plt.subplots(ncols=2, nrows=len(env), sharex="all", figsize=(16, 12))

    plt.suptitle("Contextual using " + approach + " approach")
    used_learners = []
    for i in range(0, len(env)):
        config_name = files[i][files[i].rfind('/') - len(files[i]) + 1:]
        print("Running config: ", config_name)
        learner = ContextualLearner(margins=config_margins[i], clickProbability=cp_class_per_env[i],
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

        for leaf in learner.tree.get_leaves():
            print(leaf.split_features, leaf.learner.pull_arm())

        non_contextual = np.full(time_first_split, clairvoyant_opt_rew[i])
        contextual = np.full(n_experiments - time_first_split, clairvoyant_opt_context[i])
        optimal_possible = np.hstack([non_contextual, contextual])

        x = np.linspace(0, n_experiments, n_experiments)
        axes[i, 0].plot(x, learner_graph_margins)
        axes[i, 0].plot(x, optimal_possible)
        # axes[i, 0].plot(x, environment_margins)
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
else:
    # Significant number of runs computation
    from datetime import datetime
    dtime = datetime.today().strftime('%Y-%m-%d_%H-%M')

    print("\n\nStart testing all the configs for {} days for {} times".format(n_experiments, n_runs))
    datasets = []
    for i in range(0,len(files)):
        config_name = files[i][files[i].rfind('/') - len(files[i]) + 1:]
        print("\nRunning config: ", config_name)

        # Compute optimal margin evolution during time
        optimal = np.full((n_experiments), clairvoyant_opt_context[i])

        average_regrets = []
        average_expected_rewards = []
        average_env_rewards = []
        for k in tqdm(range(0,n_runs)):
            learner = ContextualLearner(margins=config_margins[i], clickProbability=cp_class_per_env[i],
                                    secondary=prod_lists[i], Lambda=lambdas[i], debug=False,
                                    features_names=features_names[i], approach=approach)
                                    
            learner_graph_margins = np.array([])
            learner_env_margins = np.array([])

            # Run one simulation
            env = Environment(config_path=files[i])
            multiEvaluator = MultiClassEvaluator(config_path=files[i])
            for j in range(0, n_experiments):
                arms = learner.pull_arm()

                ge_margin = multiEvaluator.computeMargin_per_class(arms)

                env.setPriceLevelsForContexts(arms)
                interactions = env.round()        
                env_margin = 0
                for k in range(0, len(interactions)):
                    env_margin = env_margin + interactions[k].linearizeMargin(config_margins[i])
                env_margin = env_margin / len(interactions)

                learner.update(interactions)

                learner_graph_margins = np.append(learner_graph_margins, ge_margin)
                learner_env_margins = np.append(learner_env_margins, env_margin)

            # Compute the metrics 
            average_regret = np.clip(np.subtract(optimal, learner_graph_margins), 0, None).mean()
            average_exp_rew = learner_graph_margins.mean()
            average_env_rew = learner_env_margins.mean()

            average_regrets.append(average_regret)
            average_expected_rewards.append(average_exp_rew)
            average_env_rewards.append(average_env_rew)

        # Save the metrics into a dict
        dict_metrics = {
            "averageRegrets": average_regrets,
            "averageExpectedRewards": average_expected_rewards,
            "averageEnvRewards": average_env_rewards,
            "averageOptimalReward": optimal.mean(),
            "time": dtime
        }

        # Then save the dict into a file
        file_name = "./Results/Contextual-{}_{}_{}".format(approach, dtime, config_name)
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(dict_metrics, f, ensure_ascii=False, indent=4)