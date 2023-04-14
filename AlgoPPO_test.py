from model import Model
from data_manager import data_manager
from environment import EnvSpecs, EnvType, job_shop_PPO_flexible
import torch
#torch.set_num_threads(8)
import random
import time
import numpy as np
import logging

np.random.seed(0)
random.seed(0)

class Algorithm_PPO():
    def __init__(self,
                 environment_specs,
                 D_in,
                 specs_lstm_policy,
                 specs_lstm_value,
                 specs_ff_value,
                 criterion="mse",
                 optimizer_value="adam",
                 optimizer_policy="adam",
                 optspecs_value=None,
                 optspecs_policy=None,
                 scheduler_value="multiplicative",
                 schedspecs=None,
                 memorylength=None,
                 memorypath=None,
                 seed=None,
                 stop_function=None,
                 path = '',
                 path_model = '',
                 nep_model = None,
                 device = torch.device('cpu')
                 ):

        if optspecs_value is None:
            optspecs_value = {"lr": 1e-4}
        if optspecs_policy is None:
            optspecs_policy = {"lr": 1e-4}

        self._model_policy = Model(D_in, specs_lstm_policy, device = device, LSTMpolicyflag=True, seed=seed)
        self._model_value = Model(D_in, specs_lstm_value, device = device, LSTMvalueflag=True, specs_ff=specs_ff_value,
                                      seed=seed)
        self._model_value.set_loss(criterion)
        self._data = data_manager(stacklength=memorylength, seed=seed)

        if nep_model is not None: self.load_models_only(nep_model, test_flag=False, path_model = path_model)

        self._model_value.set_optimizer(name=optimizer_value, options=optspecs_value)
        self._model_policy.set_optimizer(name=optimizer_policy, options=optspecs_policy)
        self._model_value.set_scheduler(name=scheduler_value, options=schedspecs)

        if memorypath != None:
            self._data.memoryrecoverCSV(memorypath)
        self._stop_function = stop_function
        if stop_function == None:
            self._stop_function = Algorithm_PPO._no_stop
        self._envspecs = environment_specs
        self._buildenvironment(self._envspecs)

        self.path = path
        self.path_model = path_model
        self.device = device

    def _no_stop(elem):
        return False

    def _buildenvironment(self, envspecs):
        if envspecs[EnvSpecs.type] == EnvType.job_shop_PPO_flexible:
            self.env = job_shop_PPO_flexible(envspecs)

    def load_models_only(self, nep_model = None, test_flag = False, path_model = ''):
        modelpath_policy = path_model + "model_policy" + str(nep_model) + ".txt"
        modelpath_value = path_model + "model_value" + str(nep_model) + ".txt"

        checkpoint = torch.load(modelpath_policy)
        self._model_policy.coremdl.load_state_dict(checkpoint['model_state_dict'])
        # self._model_policy.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        checkpoint = torch.load(modelpath_value)
        self._model_value.coremdl.load_state_dict(checkpoint['model_state_dict'])
        # self._model_value.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if test_flag:
            self._model_policy.coremdl.eval()
            self._model_value.coremdl.eval()
        else:
            self._model_policy.coremdl.train()
            self._model_value.coremdl.train()

    def test(self, probs_size, repetitions, testcosts_probs, n_traj= 10, greedy_flag = True):

        len_test = len(probs_size)*repetitions
        if greedy_flag:
            n_traj = 1

        stats_episode_best = []
        stats_episode_mean = []
        stats = []

        for episode in range(len_test):

            ip = int(episode//repetitions)
            irep = episode%repetitions
            p = probs_size[ip]
            print('p;',p, 'irep:',irep)
            self.env.set_instance(irep, p, testcosts_probs[p])

            timesteps = p[0]*p[1]
            #timesteps = sum([len(job) for job in problems[p]])

            #logging.info("Inst." + str(episode) + ")")
            #print("Inst." + str(episode) + ")")

            st0 = self.env.initial_state()
            mask0 = self.env.initial_mask(st0)

            stat_episode_best = {"counts": 0,
                    "final_objective": 0,
                    "cumulative_reward": [],
                    "prob": p,
                    "rep": episode,
                    "irep": irep}

            stat_episode_mean = {"counts": 0,
                    "final_objective": 0,
                    "cumulative_reward": [],
                    "prob": p,
                    "rep": episode,
                    "irep": irep}

            for t in range(n_traj):

                stat = {"counts": 0,
                        "final_objective": 0,
                        "cumulative_reward": [],
                        "prob": p,
                        "rep": episode,
                        "irep": irep
                        }

                cumRM = [0]
                mask = mask0
                st = st0
                RM = 0  # cumulative reward

                feasible = True
                final = False
                sol = []

                start = time.time()

                for i in range(timesteps):
                    if not final and feasible:
                        insts = self.env.instances(st, mask)
                        mask_onehot = [0 if len(elem) == 1 else 1 for elem in insts]

                        probs_torch = self._model_policy.coremdl([insts, mask_onehot])

                        probs = probs_torch  # lunghezza maschera
                        probs = probs.detach().cpu().numpy()

                        if greedy_flag:
                            at1_indx = np.argmax(probs)
                        else:
#                            random.seed(0)
                            at1_indx = np.random.choice(a=np.arange(len(mask_onehot)), p=probs)

                        indx = [i for i in range(len(mask)) if mask[i][0] == at1_indx]
                        at1 = mask[indx[0]]

                        sol.append(at1)
                        st, rt, final, mask, feasible, inst = self.env.output(st, at1, insts)

                        rt0 = rt
                        RM += rt0
                        cumRM.append(RM)

                end = time.time()
                stat["cumulative_reward"] = cumRM
                stat["counts"] = len(cumRM)
                stat["final_objective"] = RM
                stat["is_final"] = 1 if feasible else 0
                stat["solution"] = sol
                stat["time"] = end - start
                stats.append(stat)

                if (episode+1) % repetitions == 0:
                    ip = int(episode // repetitions)
                    p = probs_size[ip]
                    print('MEAN MAKESPAN of size ({},{}): '.format(p[0],p[1]), np.mean([-stat['final_objective'] for stat in stats[-repetitions:]]))

            i_best = n_traj*episode + np.argmax([stat["final_objective"] for stat in stats[-n_traj:]]) #best trajectory
            stat_episode_best["counts"] = stats[i_best]['counts']
            stat_episode_best["final_objective"] = stats[i_best]['final_objective']
            stat_episode_best["cumulative_reward"] = stats[i_best]['cumulative_reward'],
            stat_episode_best["is_final"] = stats[i_best]["is_final"]
            stat_episode_best["solution"] = stats[i_best]["solution"],
            stat_episode_best["time"] = stats[i_best]["time"]

            stat_episode_mean["counts"] = np.mean([stat['counts'] for stat in stats[-n_traj:]])
            stat_episode_mean["final_objective"] = np.mean([stat['final_objective'] for stat in stats[-n_traj:]])
            stat_episode_mean["time"] = np.mean([stat['time'] for stat in stats[-n_traj:]])

            stats_episode_best.append(stat_episode_best)
            stats_episode_mean.append(stat_episode_mean)

        return stats, stats_episode_best, stats_episode_mean