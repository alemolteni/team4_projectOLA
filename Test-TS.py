from Model.ConfigurationParametersAverage import mergeUserClasses
from Environment import Environment
from Learner.TS_CR import *
from Learner.TS_Alphas import *
from Learner.TS_GW import *

from Model.Product import *
import json
import matplotlib.pyplot as plt
from Model.Evaluator.GraphEvaluator import *
from Model.Evaluator.MultiClassEvaluator import *
from tqdm import tqdm
from IPython.display import clear_output

# ======== TUNABLE PARAMETERS ===========
files = ['./Configs/config2.json','./Configs/config3.json','./Configs/ctx_config1.json']
n_experiments = 400
# Step 3 = "TS_CR" ==> Uncertain Conversion Rates
# Step 4 = "TS_Alphas" ==> Uncertain CRates, Alphas, #Units
# Step 5 = "TS_GW" ==> Uncertain CRates, Graph Weights
learner_name = "TS_CR"
n_runs = 40
MODE = "plots" # "plots" OR "runs"
# =======================================

def get_learner(name="TS_CR", margins=None, alphas=None, secondary_prod=None, click_prob=None, units_mean=None, l=None):
    if name == "TS_CR":
        return TS_CR(margins=margins, alphas=alphas, secondary_prod=secondary_prod, 
                    click_prob=click_prob, units_mean=units_mean, l=l)
    elif name == "TS_Alphas":
        return TS_Alphas(margins=margins, secondary_prod=secondary_prod, 
                        click_prob=click_prob, l=l)
    elif name == "TS_GW":
        return TS_GW(margins=margins, alphas=alphas, secondary_prod=secondary_prod, 
                     units_mean=units_mean, l=l)


# files = ['./Configs/config1.json', './Configs/config2.json']

envs = []
mc_evals = []
tsLearners = []
config_margins = []
optimal_arms = []
conv_rates = []
prod_lists = []
click_probs = []
lambdas = []
alphas = []
units_means = []
actual_units_means = []
clairvoyant_opt_rew = []
n_loops = 1

for i in range(0, len(files)):
    envs.append(Environment(config_path=files[i]))
    mc_evals.append(MultiClassEvaluator(config_path=files[i]))
    config = mergeUserClasses([files[i]], False)[0]
    l = config["lambda_p"]
    config_margins.append(config["marginsPerPrice"])
    optimal_arms.append(config["optimalConfig"])
    conv_rates.append(config["conversionRateLevels"])
    prod_lists.append(config["productList"])
    # print("ProdList={}, Alphas={}, ConvRates={}".format(len(config["productList"]),len(config["alphas"]),len(config["conversionRateLevels"])))
    click_probs.append(config["click_prob"])
    lambdas.append(config['lambda_p'])
    alphas.append(config["alphas"])
    actual_units_means.append(config["actual_units_mean"])
    clairvoyant_opt_rew.append(config["optimalMargin"])
    units_means.append(config["units_mean"])

if MODE == "plots":
    tot_ts_learner_margins = []
    tot_optimal_margins = []
    tsLearners = []
    fig, axes = plt.subplots(ncols=2, nrows=len(envs), sharex=True, figsize=(16,12))

    click_prob_convergence = []
    for i in range(0, len(envs)):
        tsLearners.append(get_learner(name=learner_name, margins=config_margins[i], alphas=alphas[i], secondary_prod=prod_lists[i],
                                    click_prob=click_probs[i], units_mean=actual_units_means[i], l=l))
        
        ts_learner_graph_margins = np.array([])
        ts_learner_env_margins = np.array([])
        actual_means = []
        
        for j in tqdm(range(0, n_experiments)):
            armMargins = []
            armConvRates = []

            # compute the margin for the TS
            pulledArm = tsLearners[i].pull_arm()
            
            envs[i].setPriceLevels(pulledArm)
            ts_interactions = envs[i].round()
            ge_margin = mc_evals[i].computeMargin(pulledArm)
                        
            env_margin = 0

            for k in range(0,len(ts_interactions)):
                # ts_env_margin = 0
                env_margin = env_margin + ts_interactions[k].linearizeMargin(config_margins[i])
            env_margin = env_margin / len(ts_interactions)

            tsLearners[i].update(ts_interactions)

            ts_learner_graph_margins = np.append(ts_learner_graph_margins, ge_margin)
            ts_learner_env_margins = np.append(ts_learner_env_margins, env_margin)

        # ========== CONVERGENCE CHECK =============
        click_dist_metric = np.abs(np.subtract(tsLearners[i].click_prob, click_probs[i])).mean()
        click_prob_convergence.append(click_dist_metric)

        # =========== PLOTTING PART ================    
        optimal = np.full((n_experiments), clairvoyant_opt_rew[i])

        config_name = files[i][files[i].rfind('/')-len(files[i])+1:]

        x = np.linspace(0, n_experiments, n_experiments)
        axes[i,0].plot(x, optimal)
        axes[i,0].plot(x, ts_learner_graph_margins)
        # axes[i,0].plot(x, ts_learner_env_margins)
        axes[i,0].set_xlabel("Time step")
        axes[i,0].set_ylabel("Margins difference\n{}".format(config_name))
        axes[0,0].set_title("Difference between margins of BruteForce and TS")

        axes[i,1].plot(x,optimal)
        cum_rews = np.cumsum(ts_learner_graph_margins)
        avg_cum_rews = np.divide(cum_rews, np.arange(1,n_experiments+1))
        axes[i,1].plot(x, avg_cum_rews)
        cum_rews = np.cumsum(ts_learner_env_margins)
        avg_cum_rews = np.divide(cum_rews, np.arange(1,n_experiments+1))
        axes[i,1].plot(x, avg_cum_rews)
        axes[i,1].set_xlabel("Time step")
        axes[i,1].set_ylabel("Margin")
        axes[0,1].set_title("Average reward: Clairvoyant vs TS")
        # print("Optimal arm found:\n\t", tsLearners[i].pull_arm(), "\nOptimal theoretical arm:\n\t", optimal_arms[i])

    clear_output(wait=True)
    plt.show()

    print("\nClick convergence metrics: {}".format(click_prob_convergence))

    for i in range(0, len(files)):
        print("{} \n",np.abs(np.subtract(tsLearners[i].click_prob, click_probs[i])))

else: 
    # Significant number of runs computation
    from datetime import datetime
    dtime = datetime.today().strftime('%Y-%m-%d_%H-%M')

    # ======= PARAMETERS TO CHANGE ========
    n_experiments = 400
    n_runs = 40
    # =====================================

    print("\n\nStart testing all the configs for {} days for {} times".format(n_experiments, n_runs))
    datasets = []
    for i in range(0,len(files)):
        config_name = files[i][files[i].rfind('/') - len(files[i]) + 1:]
        print("\nRunning config: ", config_name)

        # Compute optimal margin evolution during time
        optimal = np.full((n_experiments), clairvoyant_opt_rew[i])

        average_regrets = []
        average_expected_rewards = []
        average_env_rewards = []
        for k in tqdm(range(0,n_runs)):
            learner = get_learner(name=learner_name, margins=config_margins[i], alphas=alphas[i], secondary_prod=prod_lists[i],
                            click_prob=click_probs[i], units_mean=actual_units_means[i], l=l)

            learner_graph_margins = np.array([])
            learner_env_margins = np.array([])

            # Run one simulation
            env = Environment(config_path=files[i])
            for j in range(0, n_experiments):
                single_margin = 0
                opt_single_margin = 0

                pulledArm = learner.pull_arm()

                ge_margin = mc_evals[i].computeMargin(pulledArm)
                
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
        file_name = "./Results/{}_{}_{}".format(learner_name, dtime, config_name)
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(dict_metrics, f, ensure_ascii=False, indent=4)