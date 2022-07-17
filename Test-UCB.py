from Learner.UCB.UCB_Step3 import UCB_Step3
from Learner.UCB.UCB_Step4 import UCB_Step4
from Learner.UCB.UCB_Step5 import UCB_Step5
from Learner.UCB.UCB_SlidingWindow import UCB_SlidingWindow
from Model.ConfigurationParametersAverage import mergeUserClasses
from Environment import Environment
from Model.Evaluator.GraphEvaluator import GraphEvaluator
from Model.Evaluator.MultiClassEvaluator import MultiClassEvaluator
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from IPython.display import clear_output
import json



def choose_learner(step_num, margins, alpha, click_prob, secondary, Lambda, debug, actual_units_mean,
                   sliding_window_size=50):
    if step_num == 3:
        return UCB_Step3(margins=margins, clickProbability=click_prob, alphas=alpha,
                         secondary=secondary, Lambda=Lambda, debug=debug, units_mean=actual_units_mean)
    elif step_num == 4:
        return UCB_Step4(margins=margins, clickProbability=click_prob, secondary=secondary,
                         Lambda=Lambda, debug=debug)
    elif step_num == 5:
        return UCB_Step5(margins=margins, alphas=alpha, secondary=secondary, Lambda=Lambda,
                         debug=debug, units_mean=actual_units_mean)
    elif step_num == 6:
        return UCB_SlidingWindow(margins=margins, clickProbability=click_prob, alphas=alpha,
                                 secondary=secondary, Lambda=Lambda, debug=debug, units_mean=actual_units_mean,
                                 sliding_window_size=sliding_window_size)
    else:
        raise Exception("Invalid step number")

# ========== PARAMETERS TO CHANGE =========
# files = ['./Configs/config1.json', './Configs/config2.json', './Configs/config3.json', './Configs/configDump.json',
#          './Configs/configuration4.json', './Configs/configuration5.json']
files = ['./Configs/config2.json','./Configs/config3.json','./Configs/ctx_config1.json']

MODE = "plots" # "plots" OR "runs"
n_experiments = 400
ucb_type_step = 5  # From 3 to 5
n_runs = 40 # Only in mode "runs"
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

if MODE == "plots":
    for ucb in range(3, 7):
        fig, axes = plt.subplots(ncols=2, nrows=len(env), sharex="all", figsize=(16, 12))
        if ucb == 6:
            plt.suptitle("UCB sliding window")
        else:
            plt.suptitle("UCB step {}".format(ucb))

        for i in range(0, len(env)):
            config_name = files[i][files[i].rfind('/') - len(files[i]) + 1:]
            print("Running config: ", config_name)
            learner = choose_learner(ucb, margins=config_margins[i], alpha=alphas[i], click_prob=click_probs[i],
                                    secondary=prod_lists[i], Lambda=lambdas[i], debug=False,
                                    actual_units_mean=actual_unit_mean[i], sliding_window_size=50)

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

                learner.update(interactions)

                learner_graph_margins = np.append(learner_graph_margins, single_margin)
                learner_env_margins = np.append(learner_env_margins, env_margin)

            opt_arms = optimal_arms[i]
            armMargins = []
            armConvRates = []
            for k in range(0, len(opt_arms)):
                armMargins.append(config_margins[i][k][opt_arms[k]])
                armConvRates.append(conv_rates[i][k][opt_arms[k]])

            graphEval = GraphEvaluator(products_list=prod_lists[i], click_prob_matrix=click_probs[i], lambda_prob=lambdas[i],
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
else:
    # Significant number of runs computation mode
    from datetime import datetime
    dtime = datetime.today().strftime('%Y-%m-%d_%H-%M')


    print("\n\nStart testing all the configs for {} days for {} times".format(n_experiments, n_runs))
    datasets = []
    for i in range(0,len(files)):
        config_name = files[i][files[i].rfind('/') - len(files[i]) + 1:]
        print("\nRunning config: ", config_name)

        # Compute optimal margin evolution during time
        optimal = np.full(n_experiments, optimal_margins[i])

        average_regrets = []
        average_expected_rewards = []
        average_env_rewards = []
        for k in tqdm(range(0,n_runs)):
            learner = choose_learner(ucb_type_step, margins=config_margins[i], alpha=alphas[i], click_prob=click_probs[i],
                            secondary=prod_lists[i], Lambda=lambdas[i], debug=False,
                            actual_units_mean=actual_unit_mean[i], sliding_window_size=50)

            learner_graph_margins = np.array([])
            learner_env_margins = np.array([])

            multiEval = MultiClassEvaluator(config_path=files[i])
            # Run one simulation
            env = Environment(config_path=files[i])
            for j in range(0, n_experiments):
                single_margin = 0
                opt_single_margin = 0

                pulledArm = learner.pull_arm()

                ge_margin = multiEval.computeMargin(pulledArm, time=j)
                
                env.setPriceLevels(pulledArm)
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
        file_name = "./Results/UCB_Step{}_{}_{}".format(ucb_type_step, dtime, config_name)
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(dict_metrics, f, ensure_ascii=False, indent=4)
