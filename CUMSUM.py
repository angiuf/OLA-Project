class CUMSUM:
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

    def update(self, sample: float):
        """
        takes a id sample and return True if a detection was flagged
        """
        self.t += 1
        if self.t <= self.M:
            self.reference += sample / self.M
            return False
        else:
            self.reference += (self.reference * (self.t - 1) + sample) / self.t
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
