import torch
#torch.set_num_threads(8)
from game import gametable, TableType, GeneneratorSpecs
from AlgoPPO_test import Algorithm_PPO
import numpy as np
import pandas as pd
from environment import EnvSpecs, EnvType
import os.path
import random

path = ''
folder = "Results_TEST/"            #Folder where to save the results
if not os.path.isdir(folder):
    os.mkdir(path + folder)
path = path+folder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Choose a set of instances to test:

# - Taillard benchmark set ('TaiBenchmarkSet')
# - Taillard generated set ('TaiGeneratedSet')
# - To generate new Taillard instances ('TaillardGenerator')

# - Gaussian Set ('GaussianSet')
# - Poisson set ('PoissonSet')
# - To generate new Gaussian instances ('GaussianGenerator')
# - To generate new Poisson instances ('PoissonGenerator')

set_to_test = 'TaiBenchmarkSet'

path_model, nep_model = 'SavedModel/', 25000                              # Load model

if set_to_test == 'TaiBenchmarkSet':
    probs_size = [(15, 15), (20, 15), (20, 20), (30, 15), (30, 20)]       # Choose problem size (J, M)
    repetitions = 10                                                      # Number of instances of each size
    jobsspecs = {
        GeneneratorSpecs.Problems: probs_size,
        GeneneratorSpecs.Repetitions: repetitions,
    }
    problems, costs = gametable.table(TableType.benchmark, jobsspecs)

elif set_to_test == 'TaiGeneratedSet':
    probs_size = [(6,6), (10,10), (15,10), (15, 15), (20, 10), (20, 20)]  # Choose problem size (J, M)
    repetitions = 100                                                     # Number of instances of each size
    jobsspecs = {
        GeneneratorSpecs.Category: 'Taillard',
        GeneneratorSpecs.Problems: probs_size,
        GeneneratorSpecs.Repetitions: repetitions,
        GeneneratorSpecs.Seed : 100,
    }
    problems, costs = gametable.table(TableType.reader, jobsspecs)

elif set_to_test == 'TaillardGenerator':
    probs_size = [(15,10), (20,10)]                                      # Choose problem size (J, M)
    repetitions = 100                                                    # Number of instances of each size
    seed = 100                                                           # Random seed
    params = {'lb': 1, 'ub': 99}                                         # Minimum and maximum value of processing times
    path_set = ''                                                        # Path where to save the new generated instances

    jobsspecs = {
        GeneneratorSpecs.Category: 'Taillard',
        GeneneratorSpecs.Problems: probs_size,
        GeneneratorSpecs.Repetitions: repetitions,
        GeneneratorSpecs.DistParams: params,
        GeneneratorSpecs.Seed: seed,
        GeneneratorSpecs.Path: path_set,
    }
    problems, costs = gametable.table(TableType.taillard_generator, jobsspecs)

elif set_to_test == 'GaussianSet':
    probs_size =  [(30,25),(35,30),(40,35),(45,40),(50,45)]              # Choose problem size (J, M)
    repetitions = 100                                                    # Number of instances of each size

    jobsspecs = {
        GeneneratorSpecs.Category: 'Gaussian',
        GeneneratorSpecs.Problems: probs_size,
        GeneneratorSpecs.Repetitions: repetitions,
    }

    problems, costs = gametable.table(TableType.reader, jobsspecs)

elif set_to_test == 'PoissonSet':
    probs_size = [(30,25),(35,30),(40,35),(45,40),(50,45)]               # Choose problem size (J, M)
    repetitions = 100                                                    # Number of instances of each size

    jobsspecs = {
        GeneneratorSpecs.Category: 'Poisson',
        GeneneratorSpecs.Problems: probs_size,
        GeneneratorSpecs.Repetitions: repetitions,
    }
    problems, costs = gametable.table(TableType.reader, jobsspecs)

elif set_to_test == 'GaussianGenerator':
    probs_size = [(15,10), (20,10)]                                      # Choose problem size (J, M)
    repetitions = 100                                                    # Number of instances of each size
    seed = 100                                                           # Random seed
    prob = 0.7                                                           # Probability of assigning a machine to a job
    params = {'mu': 100, 'sigma': 10}                                    # Params of the Gaussian Distribution
    path_set = ''                                                        # Path where to save the new generated instances

    jobsspecs = {
        GeneneratorSpecs.Problems: probs_size,
        GeneneratorSpecs.Repetitions: repetitions,
        GeneneratorSpecs.Distribution: random.gauss,
        GeneneratorSpecs.Probability: prob,
        GeneneratorSpecs.DistParams: params,
        GeneneratorSpecs.Seed: seed,
        GeneneratorSpecs.Path: path_set,
    }
    problems, costs = gametable.table(TableType.random_jobs, jobsspecs)

elif set_to_test == 'PoissonGenerator':
    probs_size = [(15,10), (20,10)]                                     # Choose problem size (J, M)
    repetitions = 100                                                   # Number of instances of each size
    seed = 100                                                          # Random seed
    prob = 0.7                                                          # Probability of assigning a machine to a job
    params = {'lam': 100}                                               # Params of the Poisson Distribution
    path_set = ''                                                       # Path where to save the new generated instances

    jobsspecs = {
        GeneneratorSpecs.Problems: probs_size,
        GeneneratorSpecs.Repetitions: repetitions,
        GeneneratorSpecs.Distribution: np.random.poisson,
        GeneneratorSpecs.Probability: 0.7,
        GeneneratorSpecs.DistParams: params,
        GeneneratorSpecs.Seed: seed,
        GeneneratorSpecs.Path: path_set,
    }
    problems, costs = gametable.table(TableType.random_jobs, jobsspecs)

nmachines = 11
D_in = 3 #(1, njobs, 2) #batch, seq_lenght, num_features
specs_lstm_policy = (nmachines*10,1,1) #hidden_dim, layer_dim, layer2_dim
specs_lstm_value = (nmachines*10,1,1)

specs_ff_value = [
     ("relu", nmachines*10),
    ("relu", nmachines*5),
    ("relu", nmachines),
    ("linear", 1)
 ]

criterion = "mse"
optimizer_value = "adam"
optimizer_policy = "adam"
optspecs_value = { "lr" : 1e-4} #, "momentum": 0.1, "nesterov": False }
optspecs_policy = { "lr" : 1e-4} #, "momentum": 0.1, "nesterov": False }
scheduler_value = None # "multiplicative"
schedspecs = None #{"factor":0.85}

environment_specs = {
    EnvSpecs.type : EnvType.job_shop_PPO_flexible,
    EnvSpecs.statusdimension : D_in,
    EnvSpecs.rewardimension : 1,
    EnvSpecs.costs : costs.copy(),
    EnvSpecs.prize : 1300,
    EnvSpecs.penalty : -1000,
    EnvSpecs.problems : problems
}

algo = Algorithm_PPO(
                environment_specs = environment_specs,
                D_in = D_in,
                specs_lstm_policy = specs_lstm_policy,
                specs_lstm_value = specs_lstm_value,
                specs_ff_value = specs_ff_value,
                criterion = criterion,
                optimizer_value = optimizer_value,
                optimizer_policy = optimizer_policy,
                optspecs_value = optspecs_value,
                optspecs_policy = optspecs_policy,
                scheduler_value = scheduler_value,
                schedspecs = schedspecs,
                seed = 100,
                path = path,
                path_model = path_model,
                nep_model = nep_model,
                device = device)

stats_test, stats_test_best, stats_test_mean = algo.test(probs_size, repetitions, costs, greedy_flag=True)

makespans = [-stat["final_objective"] for stat in stats_test if stat["is_final"] == 1]
times = [stat["time"] for stat in stats_test]

mean_makespans = [np.mean(makespans[i*repetitions:i*repetitions+repetitions]) for i in range(len(probs_size))]
mean_times = [np.mean(times[i*repetitions:i*repetitions+repetitions]) for i in range(len(probs_size))]

std_makespans = [np.std(makespans[i * repetitions:i * repetitions + repetitions]) for
                          i in range(len(probs_size))]
std_times = [np.std(times[i * repetitions:i * repetitions + repetitions]) for i in
               range(len(probs_size))]

max_makespans = [np.max(makespans[i * repetitions:i * repetitions + repetitions]) for i
                        in range(len(probs_size))]
max_times = [np.max(times[i * repetitions:i * repetitions + repetitions]) for i in
             range(len(probs_size))]

min_makespans = [np.min(makespans[i * repetitions:i * repetitions + repetitions]) for i
                        in range(len(probs_size))]
min_times = [np.min(times[i * repetitions:i * repetitions + repetitions]) for i in
             range(len(probs_size))]

data_all, data_agg = np.array([]), np.array([])

for p in range(len(probs_size)):
    for rep in range(repetitions):
        row3 = np.array([probs_size[p], rep, makespans[p*repetitions+rep], times[p*repetitions+rep]], dtype=object)
        data_all = np.append(data_all, row3)
for i in range(len(probs_size)):
    prob = probs_size[i]
    row4 = np.array([prob, mean_makespans[i], std_makespans[i], max_makespans[i],
            min_makespans[i], mean_times[i], std_times[i], max_times[i], min_times[i]], dtype=object)
    data_agg = np.append(data_agg, row4)

columns3 = ['Prob', 'Rep', 'Obj values', 'Times']
columns4 = ['Prob', 'Mean makespan', 'Std makespan', 'Max makespan', 'Min makespan', 'Mean time', 'Std time', 'Max time',
            'Min time']

data_all = data_all.reshape(-1, 4)
data_agg = data_agg.reshape(-1, 9)

data_all = pd.DataFrame(data_all, columns=columns3)
data_agg = pd.DataFrame(data_agg, columns=columns4)

writer = pd.ExcelWriter(path + 'Statistics_TEST_' + set_to_test + '.xlsx', engine='xlsxwriter')

data_all.to_excel(writer, sheet_name='Test total')
data_agg.to_excel(writer, sheet_name='Test aggregated')
writer.save()

print('MEAN MAKESPANS:', mean_makespans)

