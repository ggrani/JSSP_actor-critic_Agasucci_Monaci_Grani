import copy
from enum import Enum
# from sys import float_info as fi
import numpy as np
from math import ceil, floor
import os

class EnvType(Enum):
    min_path = 1
    min_path_compressed = 2
    job_shop_scheduling = 3
    js_LSTM = 4
    train_dispatching = 5
    bnb1 = 6
    min_path_PPO = 7
    job_shop_PPO = 8
    js_LSTM_PPO = 9
    job_shop_PPO_flexible = 10
    job_shop_PPO_random = 11
    job_shop_TD = 12

class EnvSpecs(Enum):
    # always present
    type = 1
    statusdimension = 2
    actiondimension = 3
    rewardimension = 4
    costs = 5
    prize = 6
    penalty = 7
    # may vary
    # min_path and min_path_TRPO
    edges = 8
    finalpoint = 9
    startingpoint = 10
    # bnb
    As = 11
    bs = 12
    cs = 13
    N = 14
    #job_shop_scheduling
    operations = 15
    #js_LSTM
    jobs = 16
    #js_LSTM_flexible
    problems = 17

class RewType(Enum):
    linear = 1

class environment():
    def __init__(self, envspecs):
        self._type = envspecs.get(EnvSpecs.type)
        self._statusdimension = envspecs.get(EnvSpecs.statusdimension)
        self._actiondimension = envspecs.get(EnvSpecs.actiondimension)
        self._rewarddimension = envspecs.get(EnvSpecs.rewardimension)
        self._costs = envspecs.get(EnvSpecs.costs)
        self._prize = envspecs.get(EnvSpecs.prize)
        self._penalty = envspecs.get(EnvSpecs.penalty)
        return

    def initial_state(self):
        pass
    def initial_mask(self, st0):
        pass
    def set_instance(self, rep):
        pass
    def instances(self, st, mask):
        pass
    def last_states(self):
        pass
    def output(self, st, at1):
        pass
    def prize(self):
        return self._prize
    def penalty(self):
        return self._penalty
    def linear_costs(self):
        return self._costs

class job_shop_TD(environment):
    def __init__(self, envspecs):
        super(job_shop_TD, self).__init__(envspecs)
        self._problems = envspecs.get(EnvSpecs.problems)  #lista di tuple, operation=(job, macchina)

    def set_problem(self, p):
         self._jobs = self._problems[p]
         self._operations = [[(i, j, self._jobs[i][j]) for j in range(len(self._jobs[i]))] for i
                             in range(len(self._jobs))]  # (i,j,m) [[],[]]
         self._machines = set([m for job in self._jobs for m in job])

      #   print('jobs', self._jobs)

    def set_instance(self, irep, p=0, testcosts=None):
        self.set_problem(p)
        if testcosts is None:
            self.costs = self._costs[p]
            self._linear_reward = [{key: -self.costs[irep][i][key] + 0.0 for key in self.costs[irep][i].keys()} for i in range(len(self.costs[0]))]
            self._proc_times = [list(elem.values()) for elem in self.costs[irep]]
        #    print('proc_times', self._proc_times)

        else:
            self._linear_reward = [{key: -testcosts[irep][i][key] + 0.0 for key in testcosts[irep][i].keys()} for i in range(len(testcosts[0]))]
            self._proc_times = [list(elem.values()) for elem in testcosts[irep]]

    def initial_state(self):  #critical path schedule
        st0 = [[sum(self._proc_times[i][:j]) for j in range(len(self._jobs[i]))] for i in range(len(self._jobs))] #istante iniziale o -1 se non allocato
        schedule0 = {machine: [(i,j,m,st0[i][j],st0[i][j]+self._proc_times[i][j]) for job in self._operations for (i,j,m) in job if m == machine] for machine in self._machines}

        for k in schedule0:
            schedule0[k].sort(key=lambda x: x[3])

     #   print('st0',st0)
     #   print('schedule0',schedule0)
        self.trui0, self.overallocated0, self.time_viol0, self.time_windows0 = self.measures(schedule0)

        return st0, schedule0

    #def initial_mask(self,st0):  #la maschera è una lista di tuple a quattro elementi: [numero, job, macchina, primo istante in cui si può allocare]
    # if len(diz[w]) == 0:
    #         if w[1]-w[0] >= self.proc_times[i][j]:
    #             if w == windows[w_idx-2]:
    #                 w_next = windows[:w_idx-1]
    #                 if diz[w_next] == [(i,j,m,s,C)]: #or C-s
    #                     earlier_time = w_viol[0] - self._proc_times[i][j] #todo??
    #             else:
    #                 earlier_time = w[0]
    #         else:
    #             if w == windows[w_idx - 2]:
    #                 w_next = windows[:w_idx - 1]
    #                 if diz[w_next] == [(i, j, m, s, C)]:
    #                     if w_viol[0] - w[0] >= self._proc_times[i][j]:
    #                         earlier_time = w_viol[0] - self._proc_times[i][j]

    def action_selection(self, schedule):
        mask = []

        for machine in schedule: #{m: (i,j,m,st,completion time)}
            operations = schedule[machine]

            time_int = list(set([s for (i,j,m,s,C) in operations]+[C for (i,j,m,s,C) in operations]))
            time_int.sort()

            diz = {(time_int[i],time_int[i+1]): [] for i in range(len(time_int)-1)}
            for (i,j,m,s,C) in operations:
                for (sd, Cd) in diz:
                    if sd<=s<Cd or sd<C<=Cd or (s <= sd and C >= Cd):
                        diz[(sd,Cd)] += [(i,j,m,s,C)]

            windows = [(time_int[i],time_int[i+1]) for i in range(len(time_int)-1)]
            windows_viol = [w for w in diz if len(diz[w])>=2]

            if windows_viol == []: continue

            w_viol, w_idx = windows_viol[0], windows.index(windows_viol[0])      #first violation e il suo indice
            makespan_m = max([C for (i,j,m,s,C) in operations])     #todo se sono ordinate puoi cambiarlo

          #  earlier = {(i,j): None for (i,j,m,s,C) in diz[w_viol]}
          #  later = {(i,j): makespan_m for (i,j,m,s,C) in diz[w_viol]} #initialize
            costs_of_move = {}  #todo pesi?
            earlier, later = {}, {}     #key: op, value: earlier_time (later_time)
            earlier_time, later_time = None, makespan_m     #initialize

            for (i,j,m,s,C) in diz[w_viol]: #operations involved in the violated window
               # print((i,j,m,s,C))
                if s > 0: #first earlier time
                    if diz[windows[w_idx - 1]] == [(i, j, m, s, C)] and diz[windows[w_idx - 2]] == []:
                        w = windows[w_idx - 2] #vuota
                        if w_viol[0] - w[0] >= self._proc_times[i][j]: #todo verifica w_viol[0] == windows[w_idx - 1][1]
                            earlier_time = w_viol[0] - self._proc_times[i][j]
                    else:
                        for w in windows[:w_idx]:  # todo while? or break reverse?
                            if diz[w] == [] and w[1] - w[0] >= self._proc_times[i][j]:
                                earlier_time = w[1] - self._proc_times[i][j]

                #first later time
                if C == makespan_m:
                    #if [diz[w] == [(i, j, m, s, C)] for w in windows[w_idx + 1:]]:
                    later_time = max([op[-1] for wi in windows[w_idx:] for op in diz[wi] if op != (i, j, m, s, C)])

                elif diz[windows[w_idx + 1]] == [(i, j, m, s, C)] and diz[windows[w_idx + 2]] == []:
                    w = windows[w_idx + 2] #vuota
                    if w[1] - w_viol[1] >= self._proc_times[i][j]:
                        later_time = w_viol[1]
                else:
                    for w in windows[w_idx+1:][::-1]: #todo? #break
                        if len(diz[w]) == 0 and w[1] - w[0] >= self._proc_times[i][j]:
                            later_time = w[0]

                earlier[(i,j)], later[(i,j)] = earlier_time, later_time
                if earlier_time == None:
                    s_new = later_time
                    distance = later_time - C #todo or s ?
                else:
                    distance = min(s - earlier_time,later_time - C)
                    i_min = np.argmin([s - earlier_time,later_time - C])
                    s_new = [earlier_time,later_time][i_min]

                costs_of_move[(i,j,m,s_new,s_new+self._proc_times[i][j])] = len(self._operations[i][j+1:]) + self._proc_times[i][j] / (w_viol[1] - w_viol[0]) + distance #todo pesi #todo from C or s? #when earlier time non esiste or viceversa? #both random?

        #    print('costs_of_move',costs_of_move)
            op = min(costs_of_move, key=costs_of_move.get)
            mask.append(op)

       # print('mask',mask)
        return mask

    def reward(self, schedule, rdf_flag = False):
        trui = 0
        makespan = max([op[-1] for m in schedule for op in schedule[m]])

        for k in schedule:
            schedule[k].sort(key=lambda x: x[3])

        for machine in schedule: #{m: (i,j,m,st,completion time}
            operations = schedule[machine]
            trui_m = makespan
            for o in range(len(operations)-1):
                (i,j,m,s,C) = operations[o]
                (i2, j2, m2, s2, C2) = operations[o+1]
                if s2 < C:
                    trui_m += C-s2
                    if not rdf_flag:
                        return 0.001 #?
            trui += trui_m
        return trui/self.trui0  #RDF

    def slack_measures(self, st):

      #  print('st', st) #todo!!!
        slacks = [[st[i][j+1]-(st[i][j] + self._proc_times[i][j]) for (i,j,m) in job[:-1]] for job in self._operations]
        min_slacks = [min(elem) for elem in slacks if elem != []] #todo??
        avg_slacks = [np.mean(elem) for elem in slacks if elem != []]

        return np.mean(min_slacks), np.std(min_slacks), np.mean(avg_slacks), np.std(avg_slacks)

    def measures(self, schedule): #todo verificare schedula ordinata
        trui = 0
        time_viol = {m: 0 for m in self._machines}
        overallocated = []

        time_int_tot = list(set([s for machine in schedule for (i, j, m, s, C) in schedule[machine]] + [C for machine in schedule for (i, j, m, s, C) in schedule[machine]]))
        time_int_tot.sort()
        windows_int = [time_int_tot[i+1]-time_int_tot[i] for i in range(len(time_int_tot)-1)]
        time_w = max(windows_int)

        makespan = max([op[-1] for m in schedule for op in schedule[m]])
        n_windows = ceil(makespan/time_w)
        time_windows = {(i*time_w,(i+1)*time_w): 1 for i in range(n_windows-2)}
        time_windows[((n_windows-1)*time_w,makespan)] = 1

        for machine in schedule: #{m: (i,j,m,st,completion time}
            overall = False
            operations = schedule[machine]
            #trui_m = makespan #max([C for (i,j,m,s,C) in operations]) #initialize #todo? - 0? #se ordinata puoi prendere C dell'ultima operazione

            time_int = list(set([s for (i,j,m,s,C) in operations]+[C for (i,j,m,s,C) in operations]))
            time_int.sort()
            diz = {(time_int[i],time_int[i+1]): 0 for i in range(len(time_int)-1)}

            for (i,j,m,s,C) in operations:
                for (sd, Cd) in diz:
                    if sd<=s<Cd or sd<C<=Cd or (s <= sd and C >= Cd):
                        overall = True
                        diz[(sd,Cd)] += 1

                # if s2 < C: # se sono sovrapposte
                #     overall = True
                #     trui_m += C-s2            #m1 = 0-7: job 1   2-7: job2   3.5-8: job 3               rui(m1,2) = 1    rui(m1,3)=2   rui(m1,3.5) = 3   trui(m1) = n_times*1

            if time_int[0] != 0.0:
                diz[(0.0, time_int[0])] = 1
            if time_int[-1] != makespan:
                diz[(time_int[-1],makespan)] = 1

            for key in diz:
                if diz[key] == 0:
                    diz[key] = 1

            trui_m = sum([(key[1]-key[0])*diz[key] for key in diz])

            for (wi,wf) in time_windows:
                for (i,f) in diz:
                    if diz[(i,f)] >= 2:
                        if wi <= i < wf or wi < f <= wf and (i <= wi and f >= wf):
                            time_windows[(wi,wf)] += 1

            time_viol[machine] = sum([key[1]-key[0] for key in diz if diz[key] >= 2 ])    #+= C-s2
            if overall: overallocated.append(m)
            trui += trui_m
            return trui, overallocated, time_viol, time_windows

    def features(self, st, schedule):

        mean_min_slack, std_min_slack, mean_avg_slack, std_avg_slack = self.slack_measures(st)
        trui, overallocated, time_viol, time_windows = self.measures(schedule)

        rdf = trui/self.trui0 #= RDF?
        overall_idx = len(overallocated)/len(self.overallocated0)
        perc_time_viol = sum(time_viol.values())/max([op[-1] for m in schedule for op in schedule[m]]) #todo schedule period?
        #sum(time_viol[m] for m in time_viol)

        w_viol = [w for w in time_windows if time_windows[w] >= 2]
        perc_window_viol = len(w_viol)/len(time_windows)

        if len(w_viol)>0:
            w_earl, idx_w_earl = w_viol[0], list(time_windows.keys()).index(w_viol[0])
            windows_after_w9 = list(time_windows.keys())[idx_w_earl:idx_w_earl+9]
            windows_viol_after_w9 = [w for w in windows_after_w9 if w in w_viol]

            perc_window_viol9 = len(windows_viol_after_w9)/len(windows_after_w9)
            first_viol_idx = (len(time_windows)-idx_w_earl)/len(time_windows)

        else: perc_window_viol9, first_viol_idx = 0, 0

        return [mean_min_slack, std_min_slack, mean_avg_slack, std_avg_slack, rdf, overall_idx, perc_time_viol, perc_window_viol, perc_window_viol9, first_viol_idx]

    # def completion_time(self, op, st):  # tempo di completamento  (no preemption)
    #     return st[op] + self._proc_times[op]

    def reschedule(self, schedule, st, op): #todo! #rischedula le altre

        (i,j,m,s,C) = op
        for (ii,jj,mm) in self._operations[i][j:]: #per ogni operazione successiva (nel job) a quella allocata
            ss = s + sum(self._proc_times[ii][j:jj])
            CC = ss + self._proc_times[ii][jj]
            st[ii][jj] = ss

            #Aggiorniamo la schedula in base a nuovi tempi
            for l in range(len(schedule[mm])): #per ogni operazione nella stessa macchina
                    (iii, jjj, mmm, sss, CCC) = schedule[mm][l]
                    if (ii,jj,mm) == (iii,jjj,mmm):
                        schedule[mm][l] = (ii,jj,mm,ss,CC)
                        #print((ii,jj,mm,ss,CC), (iii,jjj,mmm,sss,CCC))
        return schedule, st

    def instances(self, st, mask, schedule):
        insts = {}
        states = {}
        schedules = {}
        #at = [-1 for _ in range(len(self._operations))]
        for (i,j,m,s,C) in mask:
          #  at1 = copy.deepcopy(at)
          #  at1[i] = s
            st1 = copy.deepcopy(st)
            schedule1 = copy.deepcopy(schedule) #todo verifica se necessario

            schedule1, st1 = self.reschedule(schedule1, st1, (i,j,m,s,C))

            states[(i,j,m,s,C)] = st1
            schedules[(i,j,m,s,C)] = schedule1
            insts[(i,j,m,s,C)] = self.features(st1, schedule1)

        self._last_states = states
        self._schedules = schedules
        self._insts = insts
        return insts

    def last_states(self): #todo?
        return self._last_states

    def output(self, st, at1, last_states=None, insts=None, schedules = None):
        if last_states is not None:
            self._last_states = last_states
        if insts is not None:
            self._insts = insts
        if schedules is not None:
            self._schedules = schedules

        st1 = self._last_states[at1] # at1 in questo caso è una tupla di qattri elementi: (nop,job, macchina, istante inizio), si potrebbe anche ridurre a (nop, istante inzio) con qualche modifica nel codice
        schedule1 = self._schedules[at1]
        rt = - self.reward(schedule1, rdf_flag = False)
        mask = self.action_selection(schedule1)

        final = True #(-1 not in st1)  # quando tutte le operazioni sono state già allocate
        feasible = not rt == - 0.001 #True todo????
        inst = self._insts[at1]
        return st1, rt, final, mask, feasible, inst, schedule1
#
# class job_shop_TD(environment):
#     def __init__(self, envspecs):
#         super(job_shop_TD, self).__init__(envspecs)
#         self._problems = envspecs.get(EnvSpecs.problems)
#
#     def set_problem(self, p):
#          self._jobs = self._problems[p]
#         # self._operations = [[(i,j,self._jobs[i][j]) for j in range(len(self._jobs[i]))] for i in range(len(self._jobs))] #job, n op nel job, macchina
#         # self._prec = [[() if j == 0 else self._operations[i][j - 1] for j in range(len(self._operations[i]))] for i in range(len(self._operations))]
#
#     def initial_state(self):  #critical path schedule
#         st0 = [[sum(self._proc_times[i][:j]) for j in range(len(self._jobs[i]))] for i in range(len(self._jobs))]
#         return st0
#
#     def initial_mask(self, st0):  # la maschera è una lista di tuple a quattro elementi: [job, n_op, macchina, primo istante in cui si può allocare] #volendo togliere la macchina
#         m0 = [(i, 0, self._jobs[i][0], 0) for i in range(len(self._jobs))]
#         return m0
#
#     def set_instance(self, irep, p=0, testcosts=None):
#         self.set_problem(p)
#         if testcosts is None:
#             self.costs = self._costs[p]
#             self._linear_reward = [{key: -self.costs[irep][i][key] + 0.0 for key in self.costs[irep][i].keys()} for i in range(len(self.costs[0]))]
#             self._proc_times = [list(elem.values()) for elem in self.costs[irep]]
#         else:
#             self._linear_reward = [{key: -testcosts[irep][i][key] + 0.0 for key in testcosts[irep][i].keys()} for i in range(len(testcosts[0]))]
#             self._proc_times = [list(elem.values()) for elem in testcosts[irep]]
#
#     def completion_time(self, job, op, st):  # tempo di completamento  (no preemption)
#         C = float(st[job][op] + self._proc_times[job][op] if op != -1 else 0)
#         return C
#
#     def instances(self, st, mask):
#        # states = {}
#       #  at = [[-1 for _ in i] for i in self._jobs]
#
#         jobs0 = [[(m, t) for m, t in list(zip(self._jobs[i], self._proc_times[i]))] for i in range(len(self._jobs))] #jobs come tupla (macchina,proc_time)
#         jobs1 = [[jobs0[i][j] for j in range(len(st[i])) if st[i][j] == -1] for i in range(len(st))] #jobs filtrati (ancora da assegnare)
#         jobs1 = [[jobs1[i][j] + (elem[-1],) if j == 0 else jobs1[i][j] + (-1,) for j in range(len(jobs1[i])) for elem in mask if elem[0] == i] for i in range(len(jobs1))]
#         jobs1 = [job + [(-1, -1, -1)] for job in jobs1]  # fine sequenza
#
#         insts = jobs1
# #        self._last_states = states
#         self._insts = insts
#         return insts
#     #
#     # def last_states(self):
#     #     return self._last_states
#
#     def output(self, st, at1, insts=None):
#         if insts is not None:
#             self._insts = insts
#
#         if self.check_violation():
#             rt = -0.001
#         else:
#
#
#
#         self.allocated = [(i,j,self._jobs[i][j]) for i in range(len(st)) for j in range(len(st[i])) if st[i][j] >= -1 + (1e-6)]
#         st1 = [elem.copy() for elem in st]
#         st1[at1[0]][at1[1]] = at1[-1]
#         C = self.completion_time(at1[0], at1[1], st1)
#         rt = -C
#         if len(self.allocated) > 0:
#           pippo = [self.completion_time(all[0], all[1], st) for all in self.allocated]
#           rt = min(-(C - max(pippo)),0)
#        # else:
#        #   rt = -C
#
#         self.allocated.append(at1[:-1])
#         jobs2 = [[(i,j,self._jobs[i][j]) for j in range(len(st1[i])) if st1[i][j] == -1] for i in range(len(st1))]
#         jobs2 = [job + [(-1, -1)] for job in jobs2] #fine sequenza
#         mask = [jobs2[i][0] for i in range(len(jobs2)) if jobs2[i] != [(-1,-1)]]
#         last_k = [[all for all in self.allocated if all[2] == m[2]] for m in mask]
#         indexes = [np.argmax([st1[e[0]][e[1]] for e in elem]) if len(elem) > 0 else -1 for elem in last_k]
#         last_k = {mask[i]: last_k[i][indexes[i]] if indexes[i] != -1 else (-1,-1,-1) for i in range(len(mask))}
#         stimes = {m: max(self.completion_time(m[0],m[1]-1,st1),self.completion_time(last_k[m][0],last_k[m][1],st1)) for m in mask}
#         mask = [m+(stimes[m],) for m in mask]
#
#         final = len(mask) == 0  # quando tutte le operazioni sono state già allocate
#         feasible = True
#         inst = self._insts
#         print(mask)
#         return st1, rt, final, mask, feasible, inst

class job_shop_PPO_random(environment):
    def __init__(self, envspecs):
        super(job_shop_PPO_random, self).__init__(envspecs)
        self._problems = envspecs.get(EnvSpecs.problems)

    def set_problem(self, ep):
         self._jobs = self._problems[ep]
         self._operations = [[(i,j,self._jobs[i][j]) for j in range(len(self._jobs[i]))] for i in range(len(self._jobs))] #job, n op nel job, macchina
         self._prec = [[() if j == 0 else self._operations[i][j - 1] for j in range(len(self._operations[i]))] for i in range(len(self._operations))]

    def initial_state(self):  # lista di jobs con istanti di tempo inizializzati a -1
        st0 = [[-1 for _ in job] for job in self._jobs]
        return st0

    def initial_mask(self, st0):  # la maschera è una lista di tuple a quattro elementi: [job, n_op, macchina, primo istante in cui si può allocare] #volendo togliere la macchina
        m0 = [(i, 0, self._jobs[i][0], 0) for i in range(len(self._jobs))]
        return m0

    def set_instance(self, ep, testcosts=None):
        self.set_problem(ep)
        if testcosts is None:
            self.costs = self._costs[ep]
            self._linear_reward = [{key: -self.costs[i][key] + 0.0 for key in self.costs[i].keys()} for i in range(len(self.costs))]
            self._proc_times = [list(elem.values()) for elem in self.costs]
        else:
            self._linear_reward = [{key: -testcosts[i][key] + 0.0 for key in testcosts[i].keys()} for i in range(len(testcosts))]
            self._proc_times = [list(elem.values()) for elem in testcosts]

    def completion_time(self, job, op, st):  # tempo di completamento  (no preemption)
        return st[job][op] + self._proc_times[job][op]

    def instances(self, st, mask):
        states = {}
        at = [[-1 for _ in i] for i in self._jobs]
        jobs0 = [[(m, t) for m, t in list(zip(self._jobs[i], self._proc_times[i]))] for i in range(len(self._jobs))] #jobs come tupla (macchina,proc_time)
        jobs1 = [[jobs0[i][j] for j in range(len(st[i])) if st[i][j] == -1] for i in range(len(st))] #jobs filtrati
        jobs1 = [job + [(-1, -1)] for job in jobs1] #fine sequenza
        for m in mask:
            at1 = copy.deepcopy(at)
            at1[m[0]][m[1]] = m[-1]
            st1 = copy.deepcopy(st)
            st1[m[0]][m[1]] = m[-1]
            states[m] = st1
        insts = jobs1
        self._last_states = states
        self._insts = insts
        return insts

    def last_states(self):
        return self._last_states

    def output(self, st, at1, last_states=None, insts=None):
        if last_states is not None:
            self._last_states = last_states
        if insts is not None:
            self._insts = insts
        allocated = [self._operations[i][j] for i in range(len(st)) for j in range(len(st[i])) if st[i][j] >= -1 + (1e-6)]
        st1 = self._last_states[at1]
        C = self.completion_time(at1[0], at1[1], st1)
        rt = 0
        if len(allocated) > 0:
            pippo = [self.completion_time(all[0], all[1], st) for all in allocated]
            if C > max(pippo):
                rt = -(C - max(pippo))
        else:
            rt = -C
        allocated.append(at1[:-1])
        allocated1 = np.array(allocated)  # trasformiamo allocated in un array, per sfruttare la funzione np.where
        mask = []
        for i in range(len(self._operations)):
          for j in range(len(self._operations[i])):
            if (self._operations[i][j] not in allocated):  # per tutte le operazioni  non già allocate
                index = np.where(allocated1[:, 2] == self._operations[i][j][2])[0]  # indici in allocated delle operazioni allocate con la stessa macchina
                if index.size > 0: # se c'è un elemento in allocated che usa la stessa macchina di quell'operazione
                    e=[st1[allocated[ind][0]][allocated[ind][1]] for ind in index]
                    a=np.argmax(e)
                    index=index[a] # indice in allocated dell'ultima operazione allocata con la stessa macchina
                    if self._prec[i][j] == ():  # e se questa operazione non ha precedenti
                        mask.append(self._operations[i][j] + (self.completion_time(allocated[int(index)][0],allocated[int(index)][1],st1),))  # aggiungiamola alla maschera ma il primo istante possibile sarà quello dell'ultima operazione su quella macchina
                    if self._prec[i][j] in allocated:  # se l'operazione invece ha precedenti in allocated
                        maxtime = max(self.completion_time(allocated[int(index)][0],allocated[int(index)][1], st1),self.completion_time(self._prec[i][j][0],self._prec[i][j][1],st1))  # (prendo il tempo di completamento maggiore tra quello del precedente e quello dell'ultima op sulla stessa macchina)
                        mask.append(self._operations[i][j] + (maxtime,))
                else:  # se non c'è alcuna operazione già allocata che usa la stessa macchina
                    if self._prec[i][j] == ():
                        mask.append(self._operations[i][j] + (0,))  # disponibile istante 0
                    if self._prec[i][j] in allocated:
                        mask.append(self._operations[i][j] + (self.completion_time(self._prec[i][j][0], self._prec[i][j][1],st1),))  # disponibile al tempo di completamento del precedente
        final = len(mask) == 0  # quando tutte le operazioni sono state già allocate
        feasible = True
        inst = self._insts
        return st1, rt, final, mask, feasible, inst

class job_shop_PPO_flexible(environment):
    def __init__(self, envspecs):
        super(job_shop_PPO_flexible, self).__init__(envspecs)
        self._problems = envspecs.get(EnvSpecs.problems)

    def set_problem(self, p, irep):
         self._jobs = self._problems[p][irep]
        # self._operations = [[(i,j,self._jobs[i][j]) for j in range(len(self._jobs[i]))] for i in range(len(self._jobs))] #job, n op nel job, macchina
        # self._prec = [[() if j == 0 else self._operations[i][j - 1] for j in range(len(self._operations[i]))] for i in range(len(self._operations))]

    def initial_state(self):  # lista di jobs con istanti di tempo inizializzati a -1
        st0 = [[-1 for _ in job] for job in self._jobs]
        return st0

    def initial_mask(self, st0):  # la maschera è una lista di tuple a quattro elementi: [job, n_op, macchina, primo istante in cui si può allocare] #volendo togliere la macchina
        m0 = [(i, 0, self._jobs[i][0], 0) for i in range(len(self._jobs))]
        return m0

    def set_instance(self, irep, p=0, testcosts=None):
        self.set_problem(p, irep)
        if testcosts is None:
            self.costs = self._costs[p]
            self._linear_reward = [{key: -self.costs[irep][i][key] + 0.0 for key in self.costs[irep][i].keys()} for i in range(len(self.costs[0]))]
            self._proc_times = [list(elem.values()) for elem in self.costs[irep]]
        else:
            self._linear_reward = [{key: -testcosts[irep][i][key] + 0.0 for key in testcosts[irep][i].keys()} for i in range(len(testcosts[0]))]
            self._proc_times = [list(elem.values()) for elem in testcosts[irep]]

    def completion_time(self, job, op, st):  # tempo di completamento  (no preemption)
        C = float(st[job][op] + self._proc_times[job][op] if op != -1 else 0)
        return C

    def instances(self, st, mask):
       # states = {}
      #  at = [[-1 for _ in i] for i in self._jobs]

        jobs0 = [[(m, t) for m, t in list(zip(self._jobs[i], self._proc_times[i]))] for i in range(len(self._jobs))] #jobs come tupla (macchina,proc_time)
        jobs1 = [[jobs0[i][j] for j in range(len(st[i])) if st[i][j] == -1] for i in range(len(st))] #jobs filtrati (ancora da assegnare)
        jobs1 = [[jobs1[i][j] + (elem[-1],) if j == 0 else jobs1[i][j] + (-1,) for j in range(len(jobs1[i])) for elem in mask if elem[0] == i] for i in range(len(jobs1))]
        jobs1 = [job + [(-1, -1, -1)] for job in jobs1]  # fine sequenza

        insts = jobs1
#        self._last_states = states
        self._insts = insts
        return insts
    #
    # def last_states(self):
    #     return self._last_states

    def output(self, st, at1, insts=None):
        if insts is not None:
            self._insts = insts

        self.allocated = [(i,j,self._jobs[i][j]) for i in range(len(st)) for j in range(len(st[i])) if st[i][j] >= -1 + (1e-6)]
        st1 = [elem.copy() for elem in st]
        st1[at1[0]][at1[1]] = at1[-1]
        C = self.completion_time(at1[0], at1[1], st1)
        rt = -C
        if len(self.allocated) > 0:
          pippo = [self.completion_time(all[0], all[1], st) for all in self.allocated]
          rt = min(-(C - max(pippo)),0)
       # else:
       #   rt = -C

        self.allocated.append(at1[:-1])
        jobs2 = [[(i,j,self._jobs[i][j]) for j in range(len(st1[i])) if st1[i][j] == -1] for i in range(len(st1))]
        jobs2 = [job + [(-1, -1)] for job in jobs2] #fine sequenza
        mask = [jobs2[i][0] for i in range(len(jobs2)) if jobs2[i] != [(-1,-1)]]
        last_k = [[all for all in self.allocated if all[2] == m[2]] for m in mask]
        indexes = [np.argmax([st1[e[0]][e[1]] for e in elem]) if len(elem) > 0 else -1 for elem in last_k]
        last_k = {mask[i]: last_k[i][indexes[i]] if indexes[i] != -1 else (-1,-1,-1) for i in range(len(mask))}
        stimes = {m: max(self.completion_time(m[0],m[1]-1,st1),self.completion_time(last_k[m][0],last_k[m][1],st1)) for m in mask}
        mask = [m+(stimes[m],) for m in mask]

        final = len(mask) == 0  # quando tutte le operazioni sono state già allocate
        feasible = True
        inst = self._insts
        return st1, rt, final, mask, feasible, inst
#
#
# class js_LSTM_TRPO_flexible(environment):
#     def __init__(self, envspecs):
#         super(js_LSTM_TRPO_flexible, self).__init__(envspecs)
#         self._problems = envspecs.get(EnvSpecs.problems)
#
#     def set_problem(self, p):
#          self._jobs = self._problems[p]
#          self._operations = [[(i,j,self._jobs[i][j]) for j in range(len(self._jobs[i]))] for i in range(len(self._jobs))] #job, n op nel job, macchina
#          self._prec = [[() if j == 0 else self._operations[i][j - 1] for j in range(len(self._operations[i]))] for i in range(len(self._operations))]
#         # self.costs = self._costs[p]
#
#     def initial_state(self):  # lista di jobs con istanti di tempo inizializzati a -1
#         st0 = [[-1 for _ in job] for job in self._jobs]
#         return st0
#
#     def initial_mask(self, st0):  # la maschera è una lista di tuple a quattro elementi: [job, n_op, macchina, primo istante in cui si può allocare] #volendo togliere la macchina
#         m0 = [(i, 0, self._jobs[i][0], 0) for i in range(len(self._jobs))]
#         return m0
#
#     def set_instance(self, irep, p=0, testcosts=None):
#         self.set_problem(p)
#         if testcosts is None:
#             self.costs = self._costs[p]
#             self._linear_reward = [{key: -self.costs[irep][i][key] + 0.0 for key in self.costs[irep][i].keys()} for i in range(len(self.costs[0]))]
#             self._proc_times = [list(elem.values()) for elem in self.costs[irep]]
#         else:
#             self._linear_reward = [{key: -testcosts[irep][i][key] + 0.0 for key in testcosts[irep][i].keys()} for i in range(len(testcosts[0]))]
#             self._proc_times = [list(elem.values()) for elem in testcosts[irep]]
#
#     def completion_time(self, job, op, st):  # tempo di completamento  (no preemption)
#         return st[job][op] + self._proc_times[job][op]
#
#     def instances(self, st, mask):
#         states = {}
#         at = [[-1 for _ in i] for i in self._jobs]
#         jobs0 = [[(m, t) for m, t in list(zip(self._jobs[i], self._proc_times[i]))] for i in range(len(self._jobs))] #jobs come tupla (macchina,proc_time)
#         jobs1 = [[jobs0[i][j] for j in range(len(st[i])) if st[i][j] == -1] for i in range(len(st))] #jobs filtrati
#         jobs1 = [[jobs1[i][j] + (elem[-1],) if j == 0 else jobs1[i][j] + (-1,) for j in range(len(jobs1[i])) for elem in mask
#              if elem[0] == i] for i in range(len(jobs1))]
#         jobs1 = [job + [(-1, -1, -1)] for job in jobs1] #fine sequenza
#         for m in mask:
#             #jobs2 = copy.deepcopy(jobs1)
#             at1 = at.copy()
#            # at1 = at
#             at1[m[0]][m[1]] = m[-1]
#             st1 = [elem.copy() for elem in st]
#             #st1 = st
#             st1[m[0]][m[1]] = m[-1]
#             states[m] = st1
#         insts = jobs1
#         self._last_states = states
#         self._insts = insts
#         return insts
#
#     def last_states(self):
#         return self._last_states
#
#     def output(self, st, at1, last_states=None, insts=None):
#         if last_states is not None:
#             self._last_states = last_states
#         if insts is not None:
#             self._insts = insts
#         allocated = [self._operations[i][j] for i in range(len(st)) for j in range(len(st[i])) if st[i][j] >= -1 + (1e-6)]
#         st1 = self._last_states[at1]
#         C = self.completion_time(at1[0], at1[1], st1)
#         rt = 0
#         if len(allocated) > 0:
#             pippo = [self.completion_time(all[0], all[1], st) for all in allocated]
#             if C > max(pippo):
#                 rt = -(C - max(pippo))
#         else:
#             rt = -C
#         allocated.append(at1[:-1])
#         allocated1 = np.array(allocated)  # trasformiamo allocated in un array, per sfruttare la funzione np.where
#         mask = []
#         for i in range(len(self._operations)):
#           for j in range(len(self._operations[i])):
#             if (self._operations[i][j] not in allocated):  # per tutte le operazioni  non già allocate
#                 index = np.where(allocated1[:, 2] == self._operations[i][j][2])[0]  # indici in allocated delle operazioni allocate con la stessa macchina
#                 if index.size > 0: # se c'è un elemento in allocated che usa la stessa macchina di quell'operazione
#                     e=[st1[allocated[ind][0]][allocated[ind][1]] for ind in index]
#                     a=np.argmax(e)
#                     index=index[a] # indice in allocated dell'ultima operazione allocata con la stessa macchina
#                     if self._prec[i][j] == ():  # e se questa operazione non ha precedenti
#                         mask.append(self._operations[i][j] + (self.completion_time(allocated[int(index)][0],allocated[int(index)][1],st1),))  # aggiungiamola alla maschera ma il primo istante possibile sarà quello dell'ultima operazione su quella macchina
#                     if self._prec[i][j] in allocated:  # se l'operazione invece ha precedenti in allocated
#                         maxtime = max(self.completion_time(allocated[int(index)][0],allocated[int(index)][1], st1),self.completion_time(self._prec[i][j][0],self._prec[i][j][1],st1))  # (prendo il tempo di completamento maggiore tra quello del precedente e quello dell'ultima op sulla stessa macchina)
#                         mask.append(self._operations[i][j] + (maxtime,))
#                 else:  # se non c'è alcuna operazione già allocata che usa la stessa macchina
#                     if self._prec[i][j] == ():
#                         mask.append(self._operations[i][j] + (0,))  # disponibile istante 0
#                     if self._prec[i][j] in allocated:
#                         mask.append(self._operations[i][j] + (self.completion_time(self._prec[i][j][0], self._prec[i][j][1],st1),))  # disponibile al tempo di completamento del precedente
#         final = len(mask) == 0  # quando tutte le operazioni sono state già allocate
#         feasible = True
#         inst = self._insts
#         return st1, rt, final, mask, feasible, inst

class job_shop_TRPO(environment):
    def __init__(self, envspecs):
        super(job_shop_TRPO, self).__init__(envspecs)
        self._jobs = envspecs.get(EnvSpecs.jobs)

        self._operations = [[(i,j,self._jobs[i][j]) for j in range(len(self._jobs[i]))] for i in range(len(self._jobs))] #job, n op nel job, macchina
        self._prec = [[() if j == 0 else self._operations[i][j - 1] for j in range(len(self._operations[i]))] for i in range(len(self._operations))]

    def initial_state(self):  # lista di jobs con istanti di tempo inizializzati a -1
        st0 = [[-1 for _ in job] for job in self._jobs]
        return st0

    def initial_mask(self, st0):  # la maschera è una lista di tuple a quattro elementi: [job, n_op, macchina, primo istante in cui si può allocare] #volendo togliere la macchina
        m0 = [(i, 0, self._jobs[i][0], 0) for i in range(len(self._jobs))]
        return m0

    def set_instance(self, rep, testcosts=None):
        if testcosts is None:
            self._linear_reward = [{key: -self._costs[rep][i][key] + 0.0 for key in self._costs[rep][i].keys()} for i in range(len(self._costs[0]))]
            self._proc_times = [list(elem.values()) for elem in self._costs[rep]]
        else:
            self._linear_reward = [{key: -testcosts[rep][i][key] + 0.0 for key in testcosts[rep][i].keys()} for i in range(len(testcosts[0]))]
            self._proc_times = [list(elem.values()) for elem in testcosts[rep]]

    def completion_time(self, job, op, st):  # tempo di completamento  (no preemption)
        return st[job][op] + self._proc_times[job][op]

    def instances(self, st, mask):
        states = {}
        at = [[-1 for _ in i] for i in self._jobs]
        jobs0 = [[(m, t) for m, t in list(zip(self._jobs[i], self._proc_times[i]))] for i in range(len(self._jobs))] #jobs come tupla (macchina,proc_time)
        jobs1 = [[jobs0[i][j] for j in range(len(st[i])) if st[i][j] == -1] for i in range(len(st))] #jobs filtrati
        jobs1 = [job + [(-1, -1)] for job in jobs1] #fine sequenza
        for m in mask:
            at1 = copy.deepcopy(at)
            at1[m[0]][m[1]] = m[-1]
            st1 = copy.deepcopy(st)
            st1[m[0]][m[1]] = m[-1]
            states[m] = st1
        insts = jobs1
        self._last_states = states
        self._insts = insts
        return insts

    def last_states(self):
        return self._last_states

    def output(self, st, at1, last_states=None, insts=None):
        if last_states is not None:
            self._last_states = last_states
        if insts is not None:
            self._insts = insts
        allocated = [self._operations[i][j] for i in range(len(st)) for j in range(len(st[i])) if st[i][j] >= -1 + (1e-6)]
        st1 = self._last_states[at1]
        C = self.completion_time(at1[0], at1[1], st1)
        rt = 0
        if len(allocated) > 0:
            pippo = [self.completion_time(all[0], all[1], st) for all in allocated]
            if C > max(pippo):
                rt = -(C - max(pippo))
        else:
            rt = -C
        allocated.append(at1[:-1])
        allocated1 = np.array(allocated)  # trasformiamo allocated in un array, per sfruttare la funzione np.where
        mask = []
        for i in range(len(self._operations)):
          for j in range(len(self._operations[i])):
            if (self._operations[i][j] not in allocated):  # per tutte le operazioni  non già allocate
                index = np.where(allocated1[:, 2] == self._operations[i][j][2])[0]  # indici in allocated delle operazioni allocate con la stessa macchina
                if index.size > 0: # se c'è un elemento in allocated che usa la stessa macchina di quell'operazione
                    e=[st1[allocated[ind][0]][allocated[ind][1]] for ind in index]
                    a=np.argmax(e)
                    index=index[a] # indice in allocated dell'ultima operazione allocata con la stessa macchina
                    if self._prec[i][j] == ():  # e se questa operazione non ha precedenti
                        mask.append(self._operations[i][j] + (self.completion_time(allocated[int(index)][0],allocated[int(index)][1],st1),))  # aggiungiamola alla maschera ma il primo istante possibile sarà quello dell'ultima operazione su quella macchina
                    if self._prec[i][j] in allocated:  # se l'operazione invece ha precedenti in allocated
                        maxtime = max(self.completion_time(allocated[int(index)][0],allocated[int(index)][1], st1),self.completion_time(self._prec[i][j][0],self._prec[i][j][1],st1))  # (prendo il tempo di completamento maggiore tra quello del precedente e quello dell'ultima op sulla stessa macchina)
                        mask.append(self._operations[i][j] + (maxtime,))
                else:  # se non c'è alcuna operazione già allocata che usa la stessa macchina
                    if self._prec[i][j] == ():
                        mask.append(self._operations[i][j] + (0,))  # disponibile istante 0
                    if self._prec[i][j] in allocated:
                        mask.append(self._operations[i][j] + (self.completion_time(self._prec[i][j][0], self._prec[i][j][1],st1),))  # disponibile al tempo di completamento del precedente
        final = len(mask) == 0  # quando tutte le operazioni sono state già allocate
        feasible = True
        inst = self._insts
        return st1, rt, final, mask, feasible, inst

class js_LSTM(environment):
    def __init__(self, envspecs):
        super(js_LSTM, self).__init__(envspecs)
        self._jobs = envspecs.get(EnvSpecs.jobs)

        self._operations = [[(i,j,self._jobs[i][j]) for j in range(len(self._jobs[i]))] for i in range(len(self._jobs))] #job,n op nel job,macchina
        self._prec = [[() if j == 0 else self._operations[i][j - 1] for j in range(len(self._operations[i]))] for i in range(len(self._operations))]

    def initial_state(self):  # lista di tante componenti quante operazioni, ma con istanti di tempo, inizializziamo a -1
        st0 = [[-1 for _ in range(len(self._operations[i]))] for i in range(len(self._operations))]
        return st0

    def initial_mask(self, st0):  # la maschera è una lista di tuple a quattro elementi: [job, n_op, macchina, primo istante in cui si può allocare]
        m0 = [self._operations[i][0] + (0,) for i in range(len(self._operations))]
        return m0

    def set_instance(self, rep, testcosts=None):
        if testcosts is None:
            self._linear_reward = [{key: -self._costs[rep][i][key] + 0.0 for key in self._costs[rep][i].keys()} for i in range(len(self._costs[0]))]
            self._proc_times = [list(self._costs[rep][i].values()) for i in range(len(self._costs[rep]))]
        else:
            self._linear_reward = [{key: -testcosts[rep][i][key] + 0.0 for key in testcosts[rep][i].keys()} for i in range(len(testcosts[0]))]
            self._proc_times = [list(testcosts[rep][i].values()) for i in range(len(testcosts[rep]))]

    def completion_time(self, job, op, st):  # tempo di completamento  (no preemption)
        return st[job][op] + self._proc_times[job][op]

    def instances(self, st, mask):
        insts = {}
        states = {}
        at = [[-1 for _ in range(len(self._operations[i]))] for i in range(len(self._operations))] #potrei cambiare relativa a jobs
        jobs0 = [[(m, t) for m, t in list(zip(self._jobs[i], self._proc_times[i]))] for i in range(len(self._jobs))]
        jobs = [[jobs0[i][j] for j in range(len(st[i])) if st[i][j] == -1] for i in range(len(st))]
        jobs = [job + [(-1, -1)] for job in jobs]
        for m in mask:
            jobs1 = copy.deepcopy(jobs)
            at1 = copy.deepcopy(at)
            at1[m[0]][m[1]] = m[-1]
            st1 = copy.deepcopy(st)
            st1[m[0]][m[1]] = m[-1]
            states[m] = st1
            op=(m[2],jobs0[m[0]][m[1]][1])
            jobs1[m[0]].remove(op)
            insts[m] = jobs1
        self._last_states = states
        self._insts = insts
        return insts

    def last_states(self):
        return self._last_states

    def output(self, st, at1, last_states=None, insts=None):
        if last_states is not None:
            self._last_states = last_states
        if insts is not None:
            self._insts = insts
        allocated = [self._operations[i][j] for i in range(len(st)) for j in range(len(st[i])) if st[i][j] >= -1 + (1e-6)]
        st1 = self._last_states[at1]
        C = self.completion_time(at1[0], at1[1], st1)
        rt = 0
        if len(allocated) > 0:
            pippo = [self.completion_time(all[0], all[1], st) for all in allocated]
            if C > max(pippo):
                rt = -(C - max(pippo))
        else:
            rt = -C
        allocated.append(at1[:-1])
        allocated1 = np.array(allocated)  # trasformiamo allocated in un array, per sfruttare la funzione np.where
        mask = []
        for i in range(len(self._operations)):
          for j in range(len(self._operations[i])):
            if (self._operations[i][j] not in allocated):  # per tutte le operazioni  non già allocate
                index = np.where(allocated1[:, 2] == self._operations[i][j][2])[0]  # indici in allocated delle operazioni allocate con la stessa macchina
                if index.size > 0: # se c'è un elemento in allocated che usa la stessa macchina di quell'operazione
                    e=[st1[allocated[ind][0]][allocated[ind][1]] for ind in index]
                    a=np.argmax(e)
                    index=index[a] # indice in allocated dell'ultima operazione allocata con la stessa macchina
                    if self._prec[i][j] == ():  # e se questa operazione non ha precedenti
                        mask.append(self._operations[i][j] + (self.completion_time(allocated[int(index)][0],allocated[int(index)][1],st1),))  # aggiungiamola alla maschera ma il primo istante possibile sarà quello dell'ultima operazione su quella macchina
                    if self._prec[i][j] in allocated:  # se l'operazione invece ha precedenti in allocated
                        maxtime = max(self.completion_time(allocated[int(index)][0],allocated[int(index)][1], st1),self.completion_time(self._prec[i][j][0],self._prec[i][j][1],st1))  # (prendo il tempo di completamento maggiore tra quello del precedente e quello dell'ultima op sulla stessa macchina)
                        mask.append(self._operations[i][j] + (maxtime,))
                else:  # se non c'è alcuna operazione già allocata che usa la stessa macchina
                    if self._prec[i][j] == ():
                        mask.append(self._operations[i][j] + (0,))  # disponibile istante 0
                    if self._prec[i][j] in allocated:
                        mask.append(self._operations[i][j] + (self.completion_time(self._prec[i][j][0], self._prec[i][j][1],st1),))  # disponibile al tempo di completamento del precedente
        final = len(mask) == 0  # quando tutte le operazioni sono state già allocate
        feasible = True
        inst = self._insts[at1]
        return st1, rt, final, mask, feasible, inst


class job_shop_TRPO(environment):
    def __init__(self, envspecs):
        super(job_shop_TRPO, self).__init__(envspecs)
        self._operations = envspecs.get(EnvSpecs.operations)  #lista di tuple, operation=(job, macchina)

        self._operations = [(i,)+self._operations[i] for i in range(len(self._operations))]  #lista di tuple, operation=(numero, job, macchina)
        self._prec = [() if i == 0 or self._operations[i][1] != self._operations[i - 1][1] else self._operations[i-1] for i in range(len(self._operations))]
        #lista dei precedenti [numero, job, macchina], se un'operazione non ha precedenti -> ()

    def initial_state(self):  #lista di tante componenti quanto operazioni, ma con istanti di tempo, inizializziamo a -1
        st0 = [-1 for _ in range(len(self._operations))]
        return st0

    def initial_mask(self,st0):  #la maschera è una lista di tuple a quattro elementi: [numero, job, macchina, primo istante in cui si può allocare] #potrei mettere anche solo numero e istante con qualche modifiche
        m0 = [self._operations[i] + (0,) for i in range(len(self._prec)) if self._prec[i] == ()]
        return m0

    def set_instance(self, rep, testcosts=None):
        if testcosts is None:
            self._linear_reward = {key: -self._costs[rep][key] + 0.0 for key in self._costs[rep].keys()}
            self._proc_times = list(self._costs[rep].values())
        else:
            self._linear_reward = {key: -testcosts[rep][key] + 0.0 for key in testcosts[rep].keys()}
            self._proc_times = list(testcosts[rep].values())

    def completion_time(self, op, st):  # tempo di completamento  (no preemption)
        return st[op] + self._proc_times[op]

    def instances(self, st, mask):
        states = {}
        at = [-1 for _ in range(len(self._operations))]
        for m in mask:
            at1 = copy.deepcopy(at)
            at1[m[0]] = m[-1]
            st1 = copy.deepcopy(st)
            st1[m[0]] = m[-1]
            states[m] = st1
        insts = st + self._proc_times
        self._last_states = states
        self._insts = insts
        return insts

    def last_states(self):
        return self._last_states

    def output(self, st, at1, last_states=None, insts=None):
        if last_states is not None:
            self._last_states = last_states
        if insts is not None:
            self._insts = insts
        allocated = [self._operations[i] for i in range(len(st)) if st[i] >= -1 + (1e-8)]
        #allocated = set(allocated)  #per come definito, non dovrebbe essere necessario
        st1 = self._last_states[at1] # at1 in questo caso è una tupla di qattri elementi: (nop,job, macchina, istante inizio), si potrebbe anche ridurre a (nop, istante inzio) con qualche modifica nel codice
        C = self.completion_time(at1[0], st1)
        rt=0
        if len(allocated)>0:
            Cs = [self.completion_time(all[0], st) for all in allocated]
            if C > max(Cs):
                rt = -(C - max(Cs))
        else:
            rt=-C
        allocated.append(at1[:-1])
        #print("allocated",allocated)
        allocated1 = np.array(allocated)  # trasformiamo allocated in un array, per sfruttare la funzione np.where
        mask = []
        for i in range(len(self._operations)):
             # if len(allocated) > 0:
                if (self._operations[i] not in allocated):  # per tutte le operazioni  non già allocate
                        index = np.where(allocated1[:, 2] == self._operations[i][2])[0]  # indici in allocated delle operazioni allocate con la stessa macchina
                        if index.size > 0:  # se c'è un elemento in allocated che usa la stessa macchina di quell'operazione
                      # se c'è un elemento in allocated che usa la stessa macchina di quell'operazione
                           e = [st1[allocated[ind][0]] for ind in index]
                           a = np.argmax(e)
                           index = index[a] # indice in allocated dell'ultima operazione allocata con la stessa macchina
                           if self._prec[i] == ():  # e se questa operazione non ha precedenti
                                mask.append(self._operations[i] + (self.completion_time(allocated[int(index)][0],st1),))  # aggiungiamola alla maschera ma il primo istante possibile sarà quello dell'ultima operazione su quella macchina
                           if self._prec[i] in allocated: #se l'operazione invece ha precedenti in allocated
                                maxtime = max(self.completion_time(allocated[int(index)][0], st1),self.completion_time(self._prec[i][0], st1))  # (prendo il tempo di completamento maggiore tra quello del precedente e quello dell'ultima op sulla stessa macchina)
                                mask.append(self._operations[i] + (maxtime,))

                        else:  # se non c'è alcuna operazione già allocata che usa la stessa macchina
                            if self._prec[i] == ():
                                mask.append(self._operations[i] + (0,)) # disponibile istante 0
                            if self._prec[i] in allocated:
                                mask.append(self._operations[i] + (self.completion_time(self._prec[i][0], st1),))  # disponibile al tempo di completamento del precedente
        #print("mask",mask)
        final = (-1 not in st1)  # quando tutte le operazioni sono state già allocate
        feasible = True
        inst = self._insts   #potrei anche non usarlo
        #print("at1", at1, "st1", st1)
        return st1, rt, final, mask, feasible, inst

class min_path_TRPO(environment):
    def __init__(self, envspecs):
        super(min_path_TRPO, self).__init__(envspecs)
        self._edges = envspecs.get(EnvSpecs.edges)
        self._startingpoint = envspecs.get(EnvSpecs.startingpoint)
        self._finalpoint = envspecs.get(EnvSpecs.finalpoint)
        self._nodes = [i for i in range(self._startingpoint, self._finalpoint+1)]
        self._neighbors = [[] for _ in self._nodes]
        for i in self._nodes:
            count = 0
            for (h, k) in self._edges:
                if h == i:
                    self._neighbors[i].append(count)
                count += 1

    def initial_state(self):
        return [0 for edge in self._edges]

    def initial_mask(self, st0):
        return self._neighbors[0]

    def set_instance(self, rep, testcosts=None):
        if testcosts is None:
            self._linear_reward = {key: -self._costs[rep][key] + 0.0 for key in self._costs[rep].keys()}
            self._costlist = list(self._costs[rep].values())
        else:
            self._linear_reward = {key: -testcosts[rep][key] + 0.0 for key in testcosts[rep].keys()}
            self._costlist = list(testcosts[rep].values())

    def instances(self, st, mask):
        insts = {}
        states = {}
        at = [0 for elem in range(len(self._edges))]
        for m in mask:
             at1 = copy.deepcopy(at)
             at1[m] = 1
             st1 = copy.deepcopy(st)
             st1[m] = 1
             states[m] = st1
        insts = st + self._costlist
        self._last_states = states
        self._insts = insts
        return insts

    def last_states(self):
        return self._last_states

    def output(self,st, at1, last_states=None, insts = None):
        if last_states is not None:
            self._last_states = last_states
        if insts is not None:
            self._insts = insts
        visited = [0]
        for i in range(len(st)):
            if st[i] >= 1e-8:
                visited.append(self._edges[i][1]) #self._edges[i] = (h,k) self._edges[i][1] = k
        visited = set(visited)
       # print('last states', self._last_states)
        st1 = self._last_states[at1]
        rt = -self._costlist[at1]
        nextpoint = self._edges[at1][1]
        mask = []
        for neigh in self._neighbors[nextpoint]:
            (i,j) = self._edges[neigh]
            if j not in visited:###
                mask.append(neigh)
        final = nextpoint == self._finalpoint
        feasible = len(mask) > 0 or final
        inst = self._insts
        #print("at1", at1, "st1", st1)
        return st1, rt, final, mask, feasible, inst

class job_shop_scheduling(environment):
    def __init__(self, envspecs):
        super(job_shop_scheduling, self).__init__(envspecs)
        self._operations = envspecs.get(EnvSpecs.operations)  #lista di tuple, operation=(job, macchina)

        self._operations = [(i,)+self._operations[i] for i in range(len(self._operations))]  #lista di tuple, operation=(numero, job, macchina)
        self._prec = [() if i == 0 or self._operations[i][1] != self._operations[i - 1][1] else self._operations[i-1] for i in range(len(self._operations))]
        #lista dei precedenti [numero, job, macchina], se un'operazione non ha precedenti -> ()

    def initial_state(self):  #lista di tante componenti quanto operazioni, ma con istanti di tempo, inizializziamo a -1
        st0 = [-1 for _ in range(len(self._operations))]
        return st0

    def initial_mask(self,st0):  #la maschera è una lista di tuple a quattro elementi: [numero, job, macchina, primo istante in cui si può allocare]
        m0 = [self._operations[i] + (0,) for i in range(len(self._prec)) if self._prec[i] == ()]
        return m0

    def set_instance(self, rep, testcosts=None):
        if testcosts is None:
            self._linear_reward = {key: -self._costs[rep][key] + 0.0 for key in self._costs[rep].keys()}
            self._proc_times = list(self._costs[rep].values())
        else:
            self._linear_reward = {key: -testcosts[rep][key] + 0.0 for key in testcosts[rep].keys()}
            self._proc_times = list(testcosts[rep].values())

    def completion_time(self, op, st):  # tempo di completamento  (no preemption)
        return st[op] + self._proc_times[op]

    def instances(self, st, mask):
        insts = {}
        states = {}
        at = [-1 for _ in range(len(self._operations))]
        for m in mask:
            at1 = copy.deepcopy(at)
            at1[m[0]] = m[-1]
            st1 = copy.deepcopy(st)
            st1[m[0]] = m[-1]
            states[m] = st1
            insts[m] = st + at1 + st1 + self._proc_times
        self._last_states = states
        self._insts = insts
        return insts

    def last_states(self):
        return self._last_states

    def output(self, st, at1, last_states=None, insts=None):
        if last_states is not None:
            self._last_states = last_states
        if insts is not None:
            self._insts = insts
        allocated = [self._operations[i] for i in range(len(st)) if st[i] >= -1 + (1e-6)]
        #allocated = set(allocated)  #per come definito, non dovrebbe essere necessario
        st1 = self._last_states[at1] # at1 in questo caso è una tupla di qattri elementi: (nop,job, macchina, istante inizio), si potrebbe anche ridurre a (nop, istante inzio) con qualche modifica nel codice
        C = self.completion_time(at1[0], st1)
        rt=0
        if len(allocated)>0:
            pippo = [self.completion_time(all[0], st) for all in allocated]
            if C > max(pippo):
                rt = -(C - max(pippo))
        else:
            rt=-C
        allocated.append(at1[:-1])
        allocated1 = np.array(allocated)  # trasformiamo allocated in un array, per sfruttare la funzione np.where
        mask = []
        for i in range(len(self._operations)):
                if (self._operations[i] not in allocated):  # per tutte le operazioni  non già allocate
                        index = np.where(allocated1[:, 2] == self._operations[i][2])[0]  # indici in allocated delle operazioni allocate con la stessa macchina
                        if index.size > 0:  # se c'è un elemento in allocated che usa la stessa macchina di quell'operazione
                      # se c'è un elemento in allocated che usa la stessa macchina di quell'operazione
                           e = [st1[allocated[ind][0]] for ind in index]
                           a = np.argmax(e)
                           index = index[a] # indice in allocated dell'ultima operazione allocata con la stessa macchina
                           if self._prec[i] == ():  # e se questa operazione non ha precedenti
                                mask.append(self._operations[i] + (self.completion_time(allocated[int(index)][0],st1),))  # aggiungiamola alla maschera ma il primo istante possibile sarà quello dell'ultima operazione su quella macchina
                           if self._prec[i] in allocated: #se l'operazione invece ha precedenti in allocated
                                maxtime = max(self.completion_time(allocated[int(index)][0], st1),self.completion_time(self._prec[i][0], st1))  # (prendo il tempo di completamento maggiore tra quello del precedente e quello dell'ultima op sulla stessa macchina)
                                mask.append(self._operations[i] + (maxtime,))
                        else:  # se non c'è alcuna operazione già allocata che usa la stessa macchina
                            if self._prec[i] == ():
                                mask.append(self._operations[i] + (0,)) # disponibile istante 0
                            if self._prec[i] in allocated:
                                mask.append(self._operations[i] + (self.completion_time(self._prec[i][0], st1),))  # disponibile al tempo di completamento del precedente
        final = (-1 not in st1)  # quando tutte le operazioni sono state già allocate
        feasible = True
        inst = self._insts[at1]
        return st1, rt, final, mask, feasible, inst

class min_path(environment):
    def __init__(self, envspecs):
        super(min_path, self).__init__(envspecs) #chiama il metodo init di environment con specifiche
        self._edges = envspecs.get(EnvSpecs.edges)
        self._startingpoint = envspecs.get(EnvSpecs.startingpoint)
        self._finalpoint = envspecs.get(EnvSpecs.finalpoint) #lunghezza grafo - 1
        self._nodes = [i for i in range(self._startingpoint, self._finalpoint+1)]
        self._neighbors = [[] for _ in self._nodes]
        for i in self._nodes:
            count = 0 #così do un ordine sequenziale agli archi
            for (h, k) in self._edges:
                if h == i:   #considero la stella uscente di i
                    self._neighbors[i].append(count)
                count += 1

    def initial_state(self):
        return [0 for edge in self._edges]

    def initial_mask(self, st0):
        return self._neighbors[0]

    def set_instance(self, rep, testcosts=None):
        if testcosts is None:
            self._linear_reward = {key: -self._costs[rep][key] + 0.0 for key in self._costs[rep].keys()} #key è l'arco
            self._costlist = list(self._costs[rep].values())#prendiamo i valori dei costi del training della rep-esima ripetizione
        else:
            self._linear_reward = {key: -testcosts[rep][key] + 0.0 for key in testcosts[rep].keys()}
            self._costlist = list(testcosts[rep].values())  #prendiamo i valori dei costi del test della rep-esima ripetizione

    def instances(self,st, mask):
        insts = {} #inizializzo dizionario delle istanze
        states = {} ##inizializzo dizionario degli stati
        at = [0 for elem in range(len(self._edges))] #azione al tempo t one-hot
        for m in mask:  #indica il numero dell'arco
            #(i, j) = self._edges[m]
            at1 = copy.deepcopy(at)
            at1[m] = 1  #nella posizione m-esima che indica l'arco m
            st1 = copy.deepcopy(st)
            st1[m] = 1  #nella posizione m-esima che indica l'arco m, se compiamo questa azione lo stato successivo sarà questo
            states[m] = st1 #aggiungiamo al dizionario all'elemento m, lo stato successivo
            insts[m] = st + at1 + st1 + self._costlist #aggiungiamo al dizionario all'elemento m, l'istanza relativa
        self._last_states = states #memorizziamoci gli stati in last states
        self._insts = insts #memorizziamoci le instanze in insts
        # print(insts)
        return insts

    def last_states(self):
        return self._last_states

    def output(self,st, at1, last_states=None, insts = None):  #qui l'azione non è one-hot ma un numero
        if last_states is not None:
            self._last_states = last_states
        if insts is not None:
            self._insts = insts
        visited = [0] #lista dei nodi visitati
        for i in range(len(st)):
            if st[i] >= 1e-8:
              #  print(st)
                visited.append(self._edges[i][1]) #self._edges[i] = (h,k) self._edges[i][1] = k
        visited = set(visited) #costruisce l'insieme (toglie doppioni)
        st1 = self._last_states[at1] #stato successivo relativo all'azione at1 scelta
        rt = -self._costlist[at1] #reward relativo all'azione scelta (arco selezionato)
        nextpoint = self._edges[at1][1] #nodo successivo (testa dell'arco selezionato)
        mask = [] #inizializziamo la maschera
        for neigh in self._neighbors[nextpoint]:
            (i,j) = self._edges[neigh] #prendiamo il neigh-esimo arco
            if j not in visited:### #se il nodo non è stato già visitato, aggiungiamo l'arco alla maschera
                mask.append(neigh)
        final = nextpoint == self._finalpoint #se nodo successivo è il finale
        feasible = len(mask) > 0 or final #ammissibile se esiste un arco da scegliere non già visitato o se fine
        inst = self._insts[at1] #istanza relativa all'azione at1 scelta
        return st1, rt, final, mask, feasible, inst


class testinstance():
    def __init__(self,c,A,b):
        self.A = A
        self.c = c
        self.b = b

class bnb1(environment):
    def __init__(self, envspecs):
        super(bnb1, self).__init__(envspecs)
        self._As = envspecs[EnvSpecs.As]
        self._bs = envspecs[EnvSpecs.As]
        self._cs = envspecs[EnvSpecs.As]
        self._N = envspecs[EnvSpecs.N]


    def initial_state(self):
        return [0 for i in range(self._N*2)]
    def initial_mask(self, st0):
        return [i for i in range(self._N * 2)]
    def set_instance(self, rep, testcosts=None):
        if testcosts is None:
            self._c = self._cs[rep]
            self._A = self._As[rep]
            self._b = self._bs[rep]
        else:
            self._c = testcosts[rep].c
            self._A = testcosts[rep].A
            self._b = testcosts[rep].b


    def instances(self, st, mask):
        insts = {}
        states = {}
        at = self.initial_state().copy()
        for m in mask:
            # (i, j) = self._edges[m]
            at1 = copy.deepcopy(at)
            at1[m] = 1
            st1 = copy.deepcopy(st)
            st1[m] = 1
            states[m] = st1
            insts[m] = st + at1 + st1 + self._c + sum(self._A[i] for i in range(len(self._A))) + self._b
        self._last_states = states
        self._insts = insts
        return insts
    def last_states(self):
        return self._last_states
    def output(self,st, at1, last_states = None, insts = None, bound = None):
        if last_states is not None:
            self._last_states = last_states
        if insts is not None:
            self._insts = insts
        st1 = self._last_states[at1]
        inst = self._insts[at1]

        m1 = [i for i in range(self._N) if st1[i] == 0]
        m2 = [i for i in range(self._N, self._N*2) if st1[i] == 0]
        nodesnotfixed = list(set(m1).intersection(set(m2)))
        mask = nodesnotfixed + [m + self._N for m in nodesnotfixed]
        rt = 0
        if at1 < self._N :
            fixedone = [i for i in  range(self._N) if st1[i] == 1]
            AA = [sum(self._A[j][f] for f in fixedone) for j in range(len(self._b))]
            rt =  -self._c[at1] - self._penalty*sum(AA[j] <= self._b[j] for j in range(len(self._b))) + self._penalty*sum(AA[j]-self._A[j][at1] <= self._b[j] for j in range(len(self._b)))
        final = len(nodesnotfixed) == 1
        feasible = True

        return st1, rt, final, mask, feasible, inst
