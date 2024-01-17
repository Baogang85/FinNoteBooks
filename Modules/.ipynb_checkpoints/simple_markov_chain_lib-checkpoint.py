from bisect import bisect_left
from random import random

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix

class MarkovChain:

    def __init__(self, markov_table, init_dist = None):

        self.running_state = None
        self.steps = 0
        self.visits = {state: 0 for state in markov_table}
        size = len(markov_table)

        # Set up transition probs
        self._states = {state: self._partial_sums(dist) for state, dist in markov_table.items()}
        for state, dist in self._states.items():
            if not np.isclose(dist[-1][0], 1.0):
                msg = "State {} transitions do not add up to 1.0".format(state)
                raise ValueError(msg)
        self._probs_state = np.array([0]* size)

        #Adjacency Matrix
        data, rows, cols = [], [], []
        for row, dist in markov_table.items():
            col, pval = zip(*[(s,p) for s, p in dist.items() if p > 0])
            rows += [row] * len(col)
            cols += col
            data += pval

        # check order
        enum = {state: i for i, state in enumerate(self._states)}
        rows = [enum[r] for r in rows]
        cols = [enum[c] for c in cols]
        self._adj = csr_matrix((data, (rows, cols)), shape=(size, size))

        # communication classes
        classes = {'Closed' : [], 'Open':[]}
        g = nx.MultiDiGraph(self._adj)
        scc = list(nx.strongly_connected_components(g))
        g = nx.condensation(g)

        for n in g:
            if g.out_degree(n) == 0:
                classes["Closed"].append(scc[n])
            else:
                classes["Open"].append(scc[n])
        self.communication_classes = classes

        # set initial state
        self._init_dist = None
        if init_dist is not None:
            self.init_dist = init_dist

    def __len__(self):
        return len(self._states)

    def _partial_sums(self, dist):

        states, probs = zip(*[(s,p)for s, p in dist.items() if p > 0])
        probs = np.cumsum(probs)

        return list(zip(probs, states))

    @property
    def init_dist(self):
        return self._init_dist

    @init_dist.setter
    def init_dist(self, dist):
        if not np.isclose(sum(dist.values()), 1.0):
            msg = "The transition probabilites of init_dist must add up to 1.0"
            raise ValueError(msg)
        self._init_dist = dist
        self._state0 = self._partial_sums(dist)
        self.running_state = None

    @property
    def probs_matrix(self):
        return self._adj.toarray()

    @property
    def probs_state(self):
        init_dist = np.array([self.init_dist.get(state, 0.0) for state in self._states])
        probs = init_dist @ (self._adj ** self.steps)
        return dict(zip(self._states, probs))

    @property
    def eigenvavalues(self):
        return list(np.sort(np.linalg.eigvals(self.probs_matrix)))

    def _next_state(self, state):
        return state[bisect_left(state, (random(), ))][1]

    def start(self):
        self.steps = 0
        for state in self._states:
            self.visits[state] = 0

        self.running_state = self._next_state(self._state0)
        self.visits[self.running_state] = 1

    def move(self):
        transitions_probs = self._states[self.running_state]
        self.running_state = self._next_state(transitions_probs)
        self.steps += 1
        self.visits[self.running_state] += 1
