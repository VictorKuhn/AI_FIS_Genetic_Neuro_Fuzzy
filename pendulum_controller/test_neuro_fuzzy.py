from neuro_fuzzy_controller import NeuroFuzzyController, generate_simulated_data


def test():
    neuro_fuzzy = NeuroFuzzyController()
    X_test, _ = generate_simulated_data(samples=10)

    for i in range(len(X_test)):
        angle, angular_velocity = X_test[i]
        action = neuro_fuzzy.control_pendulum(angle, angular_velocity)
        print(f"Test {i+1}: Angle: {angle}, Angular Velocity: {angular_velocity}, Action: {action}")


if __name__ == '__main__':
    test()