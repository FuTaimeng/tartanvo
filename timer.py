import time

class Timer:
    def __init__(self):
        self.st = {}
        self.tot_dt = {}
        self.count = {}

    def clear(self, name=None):
        if name is None:
            for key in self.tot_dt:
                self.tot_dt[key] = 0
                self.count[key] = 0
        elif isinstance(name, list) or isinstance(name, tuple):
            for nm in name:
                self.tot_dt[nm] = 0
                self.count[nm] = 0
        else:
            self.tot_dt[name] = 0
            self.count[name] = 0

    def tic(self, name='default'):
        self.st[name] = time.perf_counter()
        if name not in self.tot_dt:
            self.tot_dt[name] = 0
            self.count[name] = 0
    
    def toc(self, name='default'):
        t = time.perf_counter()
        dt = t - self.st[name]
        self.st[name] = t
        if name not in self.tot_dt:
            self.tot_dt[name] = 0
            self.count[name] = 0
        self.tot_dt[name] += dt
        self.count[name] += 1
        return dt

    def avg(self, name='default'):
        if name not in self.tot_dt:
            return 0
        if self.count[name] == 0:
            return 0
        return self.tot_dt[name] / self.count[name]

    def tot(self, name='default'):
        if name not in self.tot_dt:
            return 0
        return self.tot_dt[name]
        