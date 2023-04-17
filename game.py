from enum import Enum
import random
import numpy as np
from uniform_gen import uni_instance_gen

class TableType(Enum):
    random_jobs = 1
    random_generator = 2
    taillard_generator = 3
    reader = 4
    benchmark = 5

class GeneneratorSpecs(Enum):
    Problems = 1
    Probability = 2
    Seed = 3
    Repetitions = 4
    Distribution = 5
    DistParams = 6
    Path = 7
    Category = 8

class gametable():

    def table(type=TableType.benchmark, specs=None):
        if type == TableType.random_jobs:
            return gametable.random_jobs(specs)
        elif type == TableType.taillard_generator:
            return gametable.taillard_generator(specs)
        elif type == TableType.reader:
            return gametable.instances_reader(specs)
        elif type == TableType.benchmark:
            return gametable.benchmark_generator(specs)

    def benchmark_generator(specs=None):

        probs = specs.get(GeneneratorSpecs.Problems)
        repetitions = specs.get(GeneneratorSpecs.Repetitions)

        if repetitions is None:
            repetitions = 10

        data = {(j,m): [] for (j,m) in probs}
        for (j,m) in probs:
            data_prob = np.load('./TaillardBenchmark/tai{}x{}.npy'.format(j, m))
            data[(j,m)] = data_prob

        problems = {}
        costs_probs = {}
        for p in probs:
            njobs = p[0]
            jobs = [[[] for _ in range(njobs)] for _ in range(repetitions)]
            costs = [[{} for _ in range(njobs)] for _ in range(repetitions)]

            data_prob = data[p]
            for rep in range(repetitions):
                for j in range(njobs):
                    machs_j = list(data_prob[rep][1][j])
                    jobs[rep][j] = machs_j
                    for im in range(len(machs_j)):
                            m = machs_j[im]
                            cost = data_prob[rep][0][j][im]
                            costs[rep][j][m] = float(cost)

            problems[p] = jobs
            costs_probs[p] = costs

        return problems, costs_probs

    def instances_reader(specs=None):

        category = specs.get(GeneneratorSpecs.Category)
        probs = specs.get(GeneneratorSpecs.Problems)
        seed = specs.get(GeneneratorSpecs.Seed)
        repetitions = specs.get(GeneneratorSpecs.Repetitions)

        if repetitions is None:
            repetitions = 100

        data = {(j,m): [] for (j,m) in probs}
        for (j,m) in probs:

           if category == 'Taillard':   data_prob = np.load('TaillardGeneratedSet/GeneratedTai{}_{}_seed{}.npy'.format(j, m, seed))
           elif category == 'Gaussian': data_prob = np.load('GaussianSet/GaussianSet{}_{}.npy'.format(j, m), allow_pickle = True)
           elif category == 'Poisson':  data_prob = np.load('PoissonSet/PoissonSet{}_{}.npy'.format(j, m), allow_pickle = True)

           data[(j,m)] = data_prob

        problems = {}
        costs_probs = {}
        for p in probs:
            njobs = p[0]
            jobs = [[[] for _ in range(njobs)] for _ in range(repetitions)]
            costs = [[{} for _ in range(njobs)] for _ in range(repetitions)]

            data_prob = data[p]
            for rep in range(repetitions):
                for j in range(njobs):
                    machs_j = list(data_prob[rep][1][j])
                    jobs[rep][j] = machs_j
                    for im in range(len(machs_j)):
                            m = machs_j[im]
                            cost = data_prob[rep][0][j][im]
                            costs[rep][j][m] = float(cost)

            problems[p] = jobs
            costs_probs[p] = costs

        return problems, costs_probs

    def taillard_generator(specs=None):

        probs = specs.get(GeneneratorSpecs.Problems)
        seed = specs.get(GeneneratorSpecs.Seed)
        repetitions = specs.get(GeneneratorSpecs.Repetitions)
        distparams = specs.get(GeneneratorSpecs.DistParams)
        path = specs.get(GeneneratorSpecs.Path)

        if distparams is None:
            distparams = {"lb": 1, "ub": 99}
        if repetitions is None:
            repetitions = 100

        lb, ub = distparams['lb'], distparams['ub']

        data = {(j,m): [] for (j,m) in probs}
        for (j,m) in probs:

            np.random.seed(seed)
            data_prob = [uni_instance_gen(n_j=j, n_m=m, low=lb, high=ub) for _ in range(repetitions)]
            data_prob = [[[list(e) for e in p] for p in rep] for rep in data_prob]
            np.save(path + 'Taillard_generated_data{}_{}_Seed{}.npy'.format(j, m, seed), data_prob)
            data[(j,m)] = data_prob

        problems = {}
        costs_probs = {}
        for p in probs:
            njobs = p[0]
            jobs = [[[] for _ in range(njobs)] for _ in range(repetitions)]
            costs = [[{} for _ in range(njobs)] for _ in range(repetitions)]

            data_prob = data[p]
            for rep in range(repetitions):
                for j in range(njobs):
                    machs_j = data_prob[rep][1][j]
                    jobs[rep][j] = machs_j
                    for im in range(len(machs_j)):
                            m = machs_j[im]
                            cost = data_prob[rep][0][j][im]
                            costs[rep][j][m] = float(cost)

            problems[p] = jobs
            costs_probs[p] = costs

        return problems, costs_probs

    def random_jobs(specs=None):

        probs = specs.get(GeneneratorSpecs.Problems)
        pr = specs[GeneneratorSpecs.Probability]
        seed = specs[GeneneratorSpecs.Seed]
        repetitions = specs.get(GeneneratorSpecs.Repetitions)
        distribution = specs.get(GeneneratorSpecs.Distribution)
        distparams = specs.get(GeneneratorSpecs.DistParams)
        path = specs.get(GeneneratorSpecs.Path)

        if repetitions is None:
            repetitions = len(probs)
        if distribution is None:
            distribution = random.randint
        if distparams is None:
            distparams = {"lb": 1, "ub": 100}

        random.seed(a=seed)
        np.random.seed(0)

        problems = {}
        costs_probs = {}
        for p in probs:
            njobs = p[0]
            nmachines = p[1]
            jobs = [[] for _ in range(njobs)]
            costs = [[{} for _ in range(njobs)] for _ in range(repetitions)]

            machines = [i + 1 for i in range(nmachines)]
            for j in range(njobs):
                c = 0
                machs = random.sample(machines, len(machines))
                for m in machs:
                    if c == 0:
                        jobs[j].append(m)
                        c += 1

                        for rep in range(repetitions):
                            cost = gametable.custrand(distribution, distparams)
                            costs[rep][j][m] = round(cost, 2) + 0.0
                    else:
                        chance = random.random()
                        if chance <= pr:
                            jobs[j].append(m)

                            for rep in range(repetitions):
                                cost = gametable.custrand(distribution, distparams)
                                costs[rep][j][m] = round(cost, 2) + 0.0

            problems[p] = [jobs for _ in range(repetitions)]
            costs_probs[p] = costs
            data_prob = [[[[d[key] for key in d] for d in costs[rep]], jobs] for rep in range(repetitions)]
            if distribution == random.gauss:
                np.save(path + 'Gaussian_generated_data{}_{}_Seed{}.npy'.format(p[0], p[1], seed), np.array(data_prob, dtype=object))
            elif distribution == np.random.poisson:
                np.save(path + 'Poisson_generated_data{}_{}_Seed{}.npy'.format(p[0], p[1], seed), np.array(data_prob, dtype=object))

        return problems, costs_probs

    def custrand(distribution, distparams):
        if distribution == random.uniform or distribution == random.randint:
            return distribution(a=distparams["a"], b=distparams["b"])
        elif distribution == random.gauss:
            return max(0.1, distribution(mu=distparams["mu"], sigma=distparams["sigma"]))
        elif distribution == np.random.poisson:
            return max(0.1, float(distribution(lam=distparams["lam"], size=1)))



