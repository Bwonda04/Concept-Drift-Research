import numpy as np
import matplotlib.pyplot as plt

class StreamEvaluator:
    """
    Evaluator for online data streams.
    Tracks accuracy over time, warning triggers, and drift detections.
    """
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.total_instances = 0
        self.total_correct = 0
        
        # Accuracy tracking
        self.prequential_accuracies = []  # Overall running accuracy
        self.window_accuracies = []       # Moving window accuracy

        # Index tracking for events
        self.warning_indices = []
        self.drift_indices = []

        # Internal buffer for windowed calculation
        self._window_results = [] # 1 for correct, 0 for wrong

    def update(self, prediction, actual):
        """
        Record a prediction result.
        """
        self.total_instances += 1
        is_correct = 1 if (prediction == actual) else 0
        
        if is_correct:
            self.total_correct += 1

        # Update windowed accuracy buffer
        self._window_results.append(is_correct)
        if len(self._window_results) > self.window_size:
            self._window_results.pop(0)

        # Record accuracies for plotting
        self.prequential_accuracies.append(self.total_correct / self.total_instances)
        self.window_accuracies.append(sum(self._window_results) / len(self._window_results))

        return bool(is_correct)

    def record_warning(self):
        """Record that a warning was triggered at the current instance."""
        self.warning_indices.append(self.total_instances)

    def record_drift(self):
        """Record that drift was detected at the current instance."""
        self.drift_indices.append(self.total_instances)

    def get_prequential_accuracy(self):
        return self.prequential_accuracies[-1] if self.prequential_accuracies else 0.0

    def get_window_accuracy(self):
        return self.window_accuracies[-1] if self.window_accuracies else 0.0

    def plot_results(self, save_path="accuracy_plot.png"):
        """
        Generate a plot showing accuracies, warnings, and drifts.
        """
        plt.figure(figsize=(12, 6))
        
        # Plot accuracies
        plt.plot(self.window_accuracies, label=f'Windowed Accuracy (size={self.window_size})', color='blue', alpha=0.7)
        plt.plot(self.prequential_accuracies, label='Prequential Accuracy', color='green', linestyle='--')

        # Plot vertical lines for events
        for idx in self.warning_indices:
            plt.axvline(x=idx, color='orange', linestyle=':', alpha=0.5, label='Warning' if idx == self.warning_indices[0] else "")
        
        for idx in self.drift_indices:
            plt.axvline(x=idx, color='red', linestyle='-', alpha=0.8, label='Drift' if idx == self.drift_indices[0] else "")

        plt.title('Model Performance and Drift Detection')
        plt.xlabel('Instances')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        
        # Avoid duplicate labels in legend if many events occur
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
        plt.show()

    def get_report(self):
        return {
            "total_instances": self.total_instances,
            "final_prequential_accuracy": self.get_prequential_accuracy(),
            "final_window_accuracy": self.get_window_accuracy(),
            "warning_count": len(self.warning_indices),
            "drift_count": len(self.drift_indices)
        }
