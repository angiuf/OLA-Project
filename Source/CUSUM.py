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

    def update(self, sample):
        """
        takes a id sample and return True if a detection was flagged
        """
        for i in range(len(sample)):

            self.t += 1
            if self.t <= self.M:
                self.reference += sample[i]/self.M
            else:
                self.reference = (self.reference * (self.t - 1) + sample[i]) / self.t
                s_plus = (sample[i] - self.reference) - self.eps
                s_minus = -(sample[i] - self.reference) - self.eps
                self.g_plus = max(0, self.g_plus + s_plus)
                self.g_minus = max(0, self.g_minus + s_minus)
                if self.g_plus > self.h or self.g_minus > self.h:
                    return True

        return False

    def reset(self):
        """
        reset all the relevant variables
        """
        self.g_plus = 0
        self.g_minus = 0
        self.t = 0
        self.reference = 0
