import numpy as np
from fis_controller import control_pendulum as control_fis
from genetic_fuzzy_controller import optimize_fuzzy_rules
from neuro_fuzzy_controller import NeuroFuzzyController

samples = 100
angles = np.random.uniform(-50, 50, samples)
velocities = np.random.uniform(-10, 10, samples)

genetic_fuzzy_params = optimize_fuzzy_rules()
neuro_fuzzy = NeuroFuzzyController()


def control_genetic_fuzzy(angle, velocity):
    return control_fis(angle, velocity)


def test_system(control_function, angles, velocities):
    results = []
    for angle, velocity in zip(angles, velocities):
        result = control_function(angle, velocity)
        results.append(result)
    return results


results_fis = test_system(control_fis, angles, velocities)
results_genetic_fuzzy = test_system(control_genetic_fuzzy, angles, velocities)
results_neuro_fuzzy = test_system(neuro_fuzzy.control_pendulum, angles, velocities)


def calculate_metrics(results):
    error = np.abs(results - np.zeros_like(results))
    mean_error = np.mean(error)
    std_dev = np.std(error)
    return {
        "mean_error": mean_error,
        "std_dev": std_dev
    }


metrics_fis = calculate_metrics(np.array(results_fis))
metrics_genetic_fuzzy = calculate_metrics(np.array(results_genetic_fuzzy))
metrics_neuro_fuzzy = calculate_metrics(np.array(results_neuro_fuzzy))

print("Métricas FIS:", metrics_fis)
print("Métricas Genético-Fuzzy:", metrics_genetic_fuzzy)
print("Métricas Neuro-Fuzzy:", metrics_neuro_fuzzy)
