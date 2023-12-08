import numpy as np
from skfuzzy import control as ctrl
from keras.models import Sequential
from keras.layers import Dense
from fis_controller import pendulum_control


class NeuroFuzzyController:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(8, input_dim=2, activation='relu'))
        self.model.add(Dense(3, activation='relu'))
        self.model.add(Dense(1, activation='linear'))

        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def train(self, X_train, y_train, epochs=100):
        self.model.fit(X_train, y_train, epochs=epochs, verbose=0)

    def control_pendulum(self, angle, angular_velocity):
        pendulum_simulator = ctrl.ControlSystemSimulation(pendulum_control)
        pendulum_simulator.input['angle'] = angle
        pendulum_simulator.input['angular_velocity'] = angular_velocity
        pendulum_simulator.compute()
        fuzzy_output = pendulum_simulator.output['action']

        neural_output = self.model.predict(np.array([[angle, angular_velocity]]))[0][0]
        return (fuzzy_output + neural_output) / 2


def generate_simulated_data(samples=1000):
    angles = np.random.uniform(-50, 50, samples)
    velocities = np.random.uniform(-10, 10, samples)
    actions = []

    for angle, velocity in zip(angles, velocities):
        pendulum_simulator = ctrl.ControlSystemSimulation(pendulum_control)
        pendulum_simulator.input['angle'] = angle
        pendulum_simulator.input['angular_velocity'] = velocity
        pendulum_simulator.compute()
        actions.append(pendulum_simulator.output['action'])

    return np.column_stack((angles, velocities)), np.array(actions)


neuro_fuzzy = NeuroFuzzyController()
X_train, y_train = generate_simulated_data()
neuro_fuzzy.train(X_train, y_train)