class CUSUM:
    def __init__(self, M, eps, h):
        """
        initialize the relevant variables
        """
        self.M = M
        self.eps = eps
        self.h = h
        self.t = 0
        self.reference = 0
        self.g_plus = 0
        self.g_minus = 0

    def update(self, sample, n_sample):
        """
        takes a id sample and return True if a detection was flagged
        """
        self.t += n_sample
        if self.t > 0:
            self.reference = (self.reference * (self.t - n_sample) + sample * n_sample) / self.t

        if self.t <= self.M:
            return False
        else:
            s_plus = (sample - self.reference) - self.eps
            s_minus = -(sample - self.reference) - self.eps
            self.g_plus = max(0, self.g_plus + s_plus)
            self.g_minus = max(0, self.g_minus + s_minus)
            return self.g_plus > self.h or self.g_minus > self.h

    def reset(self):
        """
        reset all the relevant variables
        """
        self.g_plus = 0
        self.g_minus = 0
        self.t = 0
        self.reference = 0
