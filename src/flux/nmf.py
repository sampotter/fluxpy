import numpy as np


class NmfBuilder:

    def __init__(self, A):
        self.A = A
        m, n = A.shape
        self.m = m
        self.n = n
        self.w = [np.ones(m)/np.sqrt(m)]
        self.h = [np.ones(n)/np.sqrt(n)]
        self.R = self.A - sum(np.outer(w, h) for w, h in zip(self.w, self.h))

    @property
    def res(self):
        return np.linalg.norm(self.R, 'fro')

    def step(self):
        for j, (wj, hj) in enumerate(zip(self.w, self.h)):
            Aj = self.R + np.outer(wj, hj) # definitely better not to explicitly form this sum
            hj[...] = np.maximum(0, Aj.T@wj)
            wj[...] = np.maximum(0, Aj@hj)
            wj[...] = wj/np.linalg.norm(wj)
            self.R = Aj - np.outer(wj, hj)

    def solve(self, tol=1e-7):
        prev_res = self.res
        print(prev_res)
        self.step()
        while abs(self.res - prev_res) > tol:
            prev_res = self.res
            print(prev_res)
            self.step()

    def add_term(self):
        # TODO: come up with a smarter way to initialize this...
        self.w.append(np.ones(self.m)/np.sqrt(self.m))
        self.h.append(np.ones(self.n)/np.sqrt(self.n))
        self.R = self.A - sum(np.outer(w, h) for w, h in zip(self.w, self.h))

    @property
    def k(self):
        return len(self.w)

    @property
    def W(self):
        return np.array(self.w).reshape(self.m, self.k)

    @property
    def H(self):
        return np.array(self.h).reshape(self.n, self.k)
