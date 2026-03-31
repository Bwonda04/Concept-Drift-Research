from sklearn.tree import DecisionTreeClassifier
import numpy as np

from data_loader import load_elec2
from ddm import DDM
from sliding_window import SlidingWindow
from evaluator import StreamEvaluator

def run_experiment(X, y, use_ddm=True, window_size=500, min_train=200, dataset_name='ELEC2'):
    window = SlidingWindow(max_size=window_size)
    ddm = DDM()
    evaluator = StreamEvaluator(window_size=1000)
    clf = DecisionTreeClassifier()

    print(f"\nRunning {'Adaptive (DDM)' if use_ddm else 'Baseline (No DDM)'} on {dataset_name}...")

    for i in range(len(X) - 1):
        # add current instance to window
        window.add_instance(X[i], y[i])

        # wait until minimum training size is reached
        if window.size() < min_train:
            continue

        # only retrain every 50 instances or if classifier not yet fitted
        if i % 50 == 0 or not hasattr(clf, 'tree_'):
            X_win, y_win = window.get_window()
            clf.fit(X_win, y_win)

        # predict the next instance
        prediction = clf.predict([X[i + 1]])[0]
        actual = y[i + 1]

        # update evaluator
        evaluator.update(prediction, actual)

        if use_ddm:
            error = 1 if prediction != actual else 0
            status = ddm.update(error)

            if status == 'warning':
                evaluator.record_warning()

            elif status == 'drift':
                evaluator.record_drift()
                # keep only data from warning index onward
                if ddm.warning_index is not None:
                    X_new, y_new = window.get_from_index(ddm.warning_index)
                else:
                    X_new, y_new = [], []
                window.reset()
                for x, label in zip(X_new, y_new):
                    window.add_instance(x, label)
                ddm.reset()

    # results
    report = evaluator.get_report()
    print(f"\n--- Results: {'Adaptive' if use_ddm else 'Baseline'} on {dataset_name} ---")
    print(f"Total Instances:        {report['total_instances']}")
    print(f"Final Prequential Acc:  {report['final_prequential_accuracy']:.4f}")
    print(f"Final Window Acc:       {report['final_window_accuracy']:.4f}")
    print(f"Warnings Triggered:     {report['warning_count']}")
    print(f"Drifts Detected:        {report['drift_count']}")

    label = 'adaptive' if use_ddm else 'baseline'
    evaluator.plot_results(save_path=f"{dataset_name}_{label}_plot.png")

    return report


if __name__ == '__main__':
    # load data
    X, y = load_elec2('elec.csv')

    # run adaptive (DDM)
    adaptive_report = run_experiment(X, y, use_ddm=True, dataset_name='ELEC2')

    # run baseline (no DDM)
    baseline_report = run_experiment(X, y, use_ddm=False, dataset_name='ELEC2')

    # final comparison
    print("\n--- Final Comparison ---")
    print(f"Adaptive Final Accuracy:  {adaptive_report['final_prequential_accuracy']:.4f}")
    print(f"Baseline Final Accuracy:  {baseline_report['final_prequential_accuracy']:.4f}")