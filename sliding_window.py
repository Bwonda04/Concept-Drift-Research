"""
Sliding Window implementation for concept drift project.

Purpose:
- Maintain a fixed-size window of the most recent data points.
- Used as the training set for the classifier.
- Supports reset when drift is detected (triggered by DDM).

"""

class SlidingWindow:
    def __init__(self, max_size):
        """
        Initialize the sliding window.

        Parameters:
        - max_size (int): Maximum number of instances the window can hold
        """
        self.max_size = max_size
        self.X = []  # list of feature vectors
        self.y = []  # list of labels

    def add_instance(self, x, y):
        """
        Add a new data point to the window.

        If the window exceeds max_size, remove the oldest instance.

        Parameters:
        - x: feature vector (e.g., numpy array)
        - y: label
        """
        self.X.append(x)
        self.y.append(y)

        # If window is too large, remove oldest item
        if len(self.X) > self.max_size:
            self.X.pop(0)
            self.y.pop(0)

    def get_window(self):
        """
        Return the current window data.

        Returns:
        - X: list of feature vectors
        - y: list of labels
        """
        return self.X, self.y

    def reset(self):
        """
        Clear the entire window.

        This should be called when drift is detected by DDM.
        """
        self.X = []
        self.y = []

    def size(self):
        """
        Get current number of elements in the window.

        Returns:
        - int: number of elements
        """
        return len(self.X)

    def is_full(self):
        """
        Check if window has reached max capacity.

        Returns:
        - bool
        """
        return len(self.X) == self.max_size

    def shrink(self, new_size):
        """
        Shrink the window instead of full reset. Is optional and keeps only the most recent 'new_size' elements.

        Parameters:
        - new_size (int): new size after shrinking
        """
        if new_size < len(self.X):
            self.X = self.X[-new_size:]
            self.y = self.y[-new_size:]
