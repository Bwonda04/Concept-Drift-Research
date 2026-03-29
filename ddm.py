class DDM:
    def __init__(self):
        # error rate tracking
        self.n = 0        # number of instances seen
        self.p = 0        # current error rate (running mean)
        self.s = 0        # current standard deviation

        # minimum values seen so far
        self.p_min = float('inf')
        self.s_min = float('inf')

        # warning tracking
        self.warning_index = None  # where warning started in the stream

    def update(self, error):
        """
        Update DDM statistics with a new instance.

        Parameters:
        - error (int): 1 if prediction was wrong, 0 if correct

        Returns:
        - str: 'drift', 'warning', or 'stable'
        """
        self.n += 1
        self.p = self.p + (error - self.p) / self.n
        self.s = (self.p * (1 - self.p) / self.n) ** 0.5

        # update minimums
        if self.p + self.s <= self.p_min + self.s_min:
            self.p_min = self.p
            self.s_min = self.s

        # check thresholds
        if self.p + self.s > self.p_min + 3 * self.s_min:
            # drift confirmed — full reset
            self.n = 0
            self.p = 0
            self.s = 0
            self.p_min = float('inf')
            self.s_min = float('inf')
            self.warning_index = None
            return 'drift'

        elif self.p + self.s > self.p_min + 2 * self.s_min:
            # warning zone — record where it started
            if self.warning_index is None:
                self.warning_index = self.n
            return 'warning'

        else:
            return 'stable'
        
    def reset(self):
        """
        Fully reset DDM to initial state.
        Called externally when needed.
        """
        self.n = 0
        self.p = 0
        self.s = 0
        self.p_min = float('inf')
        self.s_min = float('inf')
        self.warning_index = None