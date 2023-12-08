import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

angle = ctrl.Antecedent(np.arange(-50, 51, 1), 'angle')
angular_velocity = ctrl.Antecedent(np.arange(-10, 11, 1), 'angular_velocity')
position = ctrl.Antecedent(np.arange(-10, 11, 1), 'position')
linear_velocity = ctrl.Antecedent(np.arange(-10, 11, 1), 'linear_velocity')
action = ctrl.Consequent(np.arange(-100, 101, 1), 'action')

angle.automf(3, names=['left', 'vertical', 'right'])
angular_velocity.automf(3, names=['moving_left', 'stopped', 'moving_right'])
position.automf(3, names=['left', 'center', 'right'])
linear_velocity.automf(3, names=['moving_left', 'stopped', 'moving_right'])

action['strong_left'] = fuzz.trimf(action.universe, [-100, -50, 0])
action['left'] = fuzz.trimf(action.universe, [-50, 0, 50])
action['none'] = fuzz.trimf(action.universe, [-10, 0, 10])
action['right'] = fuzz.trimf(action.universe, [-50, 0, 50])
action['strong_right'] = fuzz.trimf(action.universe, [0, 50, 100])


rules = [
    ctrl.Rule(angle['left'] & angular_velocity['moving_left'], action['strong_left']),
    ctrl.Rule(angle['left'] & angular_velocity['stopped'], action['left']),
    ctrl.Rule(angle['left'] & angular_velocity['moving_right'], action['none']),
    ctrl.Rule(angle['vertical'] & angular_velocity['moving_left'], action['left']),
    ctrl.Rule(angle['vertical'] & angular_velocity['stopped'], action['none']),
    ctrl.Rule(angle['vertical'] & angular_velocity['moving_right'], action['right']),
    ctrl.Rule(angle['right'] & angular_velocity['moving_left'], action['none']),
    ctrl.Rule(angle['right'] & angular_velocity['stopped'], action['right']),
    ctrl.Rule(angle['right'] & angular_velocity['moving_right'], action['strong_right']),
]


pendulum_control = ctrl.ControlSystem(rules)
pendulum_simulator = ctrl.ControlSystemSimulation(pendulum_control)


def control_pendulum(angle_val, angular_velocity_val):
    pendulum_simulator.input['angle'] = angle_val
    pendulum_simulator.input['angular_velocity'] = angular_velocity_val
    pendulum_simulator.compute()
    return pendulum_simulator.output['action']


if __name__ == '__main__':
    print(control_pendulum(10, 5, -5, 2))
