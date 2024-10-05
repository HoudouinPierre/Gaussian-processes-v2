import numpy as np
import random
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from ucimlrepo import fetch_ucirepo


def power_sinus_1D_sampler(space_size, N, function_parameters):
    power = function_parameters["Power"]
    X, Y  = [], []
    for _ in range(N):
        x = np.random.rand(1) * space_size
        y = np.sin(2 * np.pi * x[0]**power)
        X.append(x)
        Y.append(y)
    X, Y = np.array(X), np.array(Y)
    return X, Y


def heaviside_1D_sampler(space_size, N, function_parameters):
    x_gaps       = function_parameters["x gaps"      ]
    steps_value  = function_parameters["steps value" ]
    sigma_noises = function_parameters["sigma noises"]
    X, Y = [], []
    for _ in range(N):
        x = np.random.rand(1) * space_size
        y = 0
        for i, (step_value, sigma_noise) in enumerate(zip(steps_value, sigma_noises)):
            if x_gaps[i] <= x[0] <= x_gaps[i+1]:
                y = step_value + np.random.normal(loc=0, scale=sigma_noise)
        X.append(x)
        Y.append(y)
    X, Y = np.array(X), np.array(Y)
    return X, Y


def multimodal_sinus_1D_sampler(space_size, N, function_parameters):
    x_gaps       = function_parameters["x gaps"      ]
    speeds_value = function_parameters["speeds value"]
    X, Y         = [], []
    for _ in range(N):
        x = np.random.rand(1) * space_size
        y = 0
        for i, speed_value in enumerate(speeds_value):
            if x_gaps[i] <= x[0] <= x_gaps[i + 1]:
                y = np.sin(2 * np.pi * speed_value * x[0])
        X.append(x)
        Y.append(y)
    X, Y = np.array(X), np.array(Y)
    return X, Y


def noisy_sinus_1D_sampler(space_size, N, function_parameters):
    sigma_noise = function_parameters["sigma noise"]
    X, Y        = [], []
    for _ in range(N):
        x = np.random.rand(1) * space_size
        y = np.sin(2 * np.pi * x[0]) + np.random.normal(loc=0, scale=sigma_noise)
        X.append(x)
        Y.append(y)
    X, Y = np.array(X), np.array(Y)
    return X, Y


def sinus_times_x_1D_sampler(space_size, N, _):
    X, Y = [], []
    for _ in range(N):
        x = np.random.rand(1) * space_size
        y = x[0] * np.sin(2 * np.pi * x[0])
        X.append(x)
        Y.append(y)
    X, Y = np.array(X), np.array(Y)
    return X, Y


def sinus_cardinal_1D_sampler(space_size, N, _):
    X, Y = [], []
    for _ in range(N):
        x = np.random.rand(1) * space_size
        y = np.sin(2 * np.pi * x[0]) / x[0]
        X.append(x)
        Y.append(y)
    X, Y = np.array(X), np.array(Y)
    return X, Y


def slide_function_sampler(space_size, N, _):
    X, Y = [], []
    for _ in range(N):
        x = np.random.rand(1) * space_size
        y = np.log(1 + x[0]) / 5 + (np.sin(x[0]**2 / np.pi**2) - np.cos(np.sqrt(x[0] / np.pi))) * np.sin(x[0])
        X.append(x)
        Y.append(y)
    X, Y = np.array(X), np.array(Y)
    return X, Y


def two_bumps_sampler(space_size, N, _):
    X, Y = [], []
    for _ in range(N):
        x  = np.random.rand(1) * space_size
        x1 = 2 * x[0] - 1
        y  = -(0.7 * x1 + np.sin(5 * x1 + 1) + 0.1 * np.sin(10 * x1))
        X.append(x)
        Y.append(y)
    X, Y = np.array(X), np.array(Y)
    return X, Y


def power_sinus_2D_sampler(space_size, N, function_parameters):
    power = function_parameters["Power"]
    X, Y  = [], []
    for _ in range(N):
        x = np.random.rand(2) * space_size
        y = np.sin(2 * np.pi * np.sum(x)**power)
        X.append(x)
        Y.append(y)
    X, Y = np.array(X), np.array(Y)
    return X, Y


def heaviside_2D_sampler(space_size, N, function_parameters):
    x_gaps       = function_parameters["x gaps"      ]
    steps_value  = function_parameters["steps value" ]
    sigma_noises = function_parameters["sigma noises"]
    X, Y = [], []
    for _ in range(N):
        x = np.random.rand(2) * space_size
        y = 0
        for i, (step_value, sigma_noise) in enumerate(zip(steps_value, sigma_noises)):
            if x_gaps[i] <= np.sum(x) <= x_gaps[i+1]:
                y = step_value + np.random.normal(loc=0, scale=sigma_noise)
        X.append(x)
        Y.append(y)
    X, Y = np.array(X), np.array(Y)
    return X, Y


def multimodal_sinus_2D_sampler(space_size, N, function_parameters):
    x_gaps       = function_parameters["x gaps"      ]
    speeds_value = function_parameters["speeds value"]
    X, Y         = [], []
    for _ in range(N):
        x = np.random.rand(2) * space_size
        y = 0
        for i, speed_value in enumerate(speeds_value):
            if x_gaps[i] <= x[0] <= x_gaps[i + 1]:
                y = np.sin(2 * np.pi * speed_value * np.sum(x))
        X.append(x)
        Y.append(y)
    X, Y = np.array(X), np.array(Y)
    return X, Y


def noisy_sinus_2D_sampler(space_size, N, function_parameters):
    sigma_noise = function_parameters["sigma noise"]
    X, Y        = [], []
    for _ in range(N):
        x = np.random.rand(2) * space_size
        y = np.sin(2 * np.pi * np.sum(x)) + np.random.normal(loc=0, scale=sigma_noise)
        X.append(x)
        Y.append(y)
    X, Y = np.array(X), np.array(Y)
    return X, Y


def sinus_times_x_2D_sampler(space_size, N, _):
    X, Y = [], []
    for _ in range(N):
        x = np.random.rand(2) * space_size
        y = np.sum(x) * np.sin(2 * np.pi * np.sum(x))
        X.append(x)
        Y.append(y)
    X, Y = np.array(X), np.array(Y)
    return X, Y


def sinus_cardinal_2D_sampler(space_size, N, _):
    X, Y = [], []
    for _ in range(N):
        x = np.random.rand(2) * space_size
        y = np.sin(2 * np.pi * np.sum(x)) / np.sum(x)
        X.append(x)
        Y.append(y)
    X, Y = np.array(X), np.array(Y)
    return X, Y


def gaussian_process_trajectory_sampler(space_size, N, function_parameters):
    sigma0           = function_parameters["sigma0"]
    theta            = function_parameters["theta" ]
    kernel           = C(constant_value=sigma0) * RBF(length_scale=theta)
    gaussian_process = GaussianProcessRegressor(kernel=kernel)
    X                = []
    for _ in range(N):
        x = np.random.rand(1) * space_size
        X.append(x)
    X                = np.array(X).reshape(-1, 1)
    samples          = gaussian_process.sample_y(X, n_samples=1)
    X                = [np.array(x) for x in X]
    Y                = np.array(samples.T[0])
    return X, Y


def uci_dataset_sampler(uci_dataset_id):
    uci_dataset = fetch_ucirepo(id=uci_dataset_id)
    X           = np.array(uci_dataset.data.features)
    Y           = np.array(uci_dataset.data.targets).T[0]
    return X, Y


def morokoff_caflisch_1_sampler(space_size, N, function_parameters):
    d    = function_parameters["Dimension"]
    X, Y = [], []
    for _ in range(N):
        x = np.random.rand(d) * space_size
        y = (1 + 1 / d)**d * np.prod(x)**(1 / d)
        X.append(x)
        Y.append(y)
    X, Y = np.array(X), np.array(Y)
    return X, Y


def morokoff_caflisch_2_sampler(space_size, N, function_parameters):
    d    = function_parameters["Dimension"]
    X, Y = [], []
    for _ in range(N):
        x = np.random.rand(d) * space_size
        y = np.prod(d * np.ones(d) - x) / (d - 0.5)**d
        X.append(x)
        Y.append(y)
    X, Y = np.array(X), np.array(Y)
    return X, Y


def branin_sampler(space_size, N, _):
    X, Y = [], []
    for _ in range(N):
        x      = np.random.rand(2) * space_size
        x1, x2 = 15 * x[0] - 5, 15 * x[1]
        y      = (x2 - 5.1 * x1**2 / (4 * np.pi**2) + 5 * x1 / np.pi - 6)**2 + (10 - 10 / (8 * np.pi)) * np.cos(x1) + 10
        X.append(x)
        Y.append(y)
    X, Y = np.array(X), np.array(Y)
    return X, Y


def branin_hoo_sampler(space_size, N, _):
    X, Y = [], []
    for _ in range(N):
        x      = np.random.rand(2) * space_size
        x1, x2 = 15 * x[0] - 5, 15 * x[1]
        y      = ((x2 - 5.1 * x1**2 / (4 * np.pi**2) + 5 * x1 / np.pi - 6)**2 + (10 - 10 / (8 * np.pi)) * np.cos(x1) - 44.81) / 51.95
        X.append(x)
        Y.append(y)
    X, Y = np.array(X), np.array(Y)
    return X, Y


def goldstein_sampler(space_size, N, _):
    X, Y = [], []
    for _ in range(N):
        x      = np.random.rand(2) * space_size
        x1, x2 = 4 * x[0] - 2, 4 * x[1] - 2
        y      = (1 + (x1 + x2 + 1)**2 * (19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2)) * (30 + (2 * x1 - 3 * x2)**2 * (18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2))
        X.append(x)
        Y.append(y)
    X, Y = np.array(X), np.array(Y)
    return X, Y


def goldstein_price_sampler(space_size, N, _):
    X, Y = [], []
    for _ in range(N):
        x      = np.random.rand(2) * space_size
        x1, x2 = 4 * x[0] - 2, 4 * x[1] - 2
        y      = (np.log((1 + (x1 + x2 + 1)**2 * (19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2)) * (30 + (2 * x1 - 3 * x2)**2 * (18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2))) - 8.693) / 2.427
        X.append(x)
        Y.append(y)
    X, Y = np.array(X), np.array(Y)
    return X, Y


def rosenbrock4_sampler(space_size, N, _):
    X, Y = [], []
    for _ in range(N):
        x              = np.random.rand(4) * space_size
        x1, x2, x3, x4 = 15 * x[0] - 5, 15 * x[1] - 5, 15 * x[2] - 5, 15 * x[3] - 5
        y              = (((100 * (x2 - x1**2)**2 + (1- x1)**2) + (100 * (x3 - x2**2)**2 + (1- x2)**2) + (100 * (x4 - x3**2)**2 + (1- x3)**2)) - 3.827 * 1e5) / (3.755 * 1e5)
        X.append(x)
        Y.append(y)
    X, Y = np.array(X), np.array(Y)
    return X, Y


def hartman4_sampler(space_size, N, _):
    c    = [1, 1.2, 3, 3.2]
    a    = [[10, 0.05, 3, 17], [3, 10, 3.5, 8], [17, 17, 1.7, 0.05], [3.5, 0.1, 10, 10], [1.7, 8, 17, 0.1], [8, 14, 8, 14]]
    p    = [[0.1312, 0.2329, 0.2348, 0.4047], [0.1696, 0.4135, 0.1451, 0.8828], [0.5569, 0.8307, 0.3522, 0.8732], [0.0124, 0.3736, 0.2883, 0.5743], [0.8283, 0.1004, 0.3047, 0.1091], [0.5886, 0.9991, 0.6650, 0.0381]]
    X, Y = [], []
    for _ in range(N):
        x = np.random.rand(4) * space_size
        y = (1.1 - np.sum(np.array([c[i] * np.exp(- np.sum(np.array([a[j][i] * (x[j] - p[j][i])**2 for j in range(4)]))) for i in range(4)]))) / 0.839
        X.append(x)
        Y.append(y)
    X, Y = np.array(X), np.array(Y)
    return X, Y


def hartman6_sampler(space_size, N, _):
    c    = [1, 1.2, 3, 3.2]
    a    = [[10, 0.05, 3, 17], [3, 10, 3.5, 8], [17, 17, 1.7, 0.05], [3.5, 0.1, 10, 10], [1.7, 8, 17, 0.1], [8, 14, 8, 14]]
    p    = [[0.1312, 0.2329, 0.2348, 0.4047], [0.1696, 0.4135, 0.1451, 0.8828], [0.5569, 0.8307, 0.3522, 0.8732], [0.0124, 0.3736, 0.2883, 0.5743], [0.8283, 0.1004, 0.3047, 0.1091], [0.5886, 0.9991, 0.6650, 0.0381]]
    X, Y = [], []
    for _ in range(N):
        x = np.random.rand(6) * space_size
        y = - (2.58 + np.sum(np.array([c[i] * np.exp(- np.sum(np.array([a[j][i] * (x[j] - p[j][i])**2 for j in range(6)]))) for i in range(4)]))) / 1.94
        X.append(x)
        Y.append(y)
    X, Y = np.array(X), np.array(Y)
    return X, Y


def sphere6_sampler(space_size, N, _):
    X, Y = [], []
    for _ in range(N):
        x = np.random.rand(6) * space_size
        y = (np.sum(np.array([x[j]**2 * 2**(j+1) for j in range(6)])) - 1745) / 899
        X.append(x)
        Y.append(y)
    X, Y = np.array(X), np.array(Y)
    return X, Y


def iooss_1_sampler(space_size, N, _):
    X, Y = [], []
    for _ in range(N):
        x      = np.random.rand(2) * space_size
        x1, x2 = 2 * x[0] - 1, 2 * x[1] - 1
        y      = np.exp(x1) / 5 - x2 / 5 + x2**6 / 3 + 4 * x2**4 - 4 * x2**2 + 7 * x1**2 / 10 + x1**4 + 3 / (4 * x1**2 + 4 * x2**2 + 1)
        X.append(x)
        Y.append(y)
    X, Y = np.array(X), np.array(Y)
    return X, Y


def iooss_2_sampler(space_size, N, _):
    X, Y = [], []
    for _ in range(N):
        x      = np.random.rand(2) * space_size
        x1, x2 = x[0], x[1]
        y      = np.cos(5 + 3 / 2 * x1) + np.sin(5 + 3 / 2 * x1) + 0.01 * (5 + 3 / 2 * x1) * (5 + 3 / 2 * x2)
        X.append(x)
        Y.append(y)
    X, Y = np.array(X), np.array(Y)
    return X, Y


def wing_weight_sampler(_, N, __):
    X, Y = [], []
    for _ in range(N):
        x                                       = np.array([150, 220, 6, -10, 16, 0.5, 0.08, 2.5, 1700, 0.025]) + np.array([50, 80, 4, 20, 29, 0.5, 0.1, 3.5, 800, 0.055]) * np.random.rand(10)
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]
        y                                       = 0.036 * x1**0.758 * x2**0.0035 * (x3 / np.cos(x4)**2)**0.6 * x5**0.006 * x6**0.04 * (100 * x7 / np.cos(x4))**(-0.3) * (x8 * x9)**0.49 + x1 * x10
        X.append(x)
        Y.append(y)
    X, Y = np.array(X), np.array(Y)
    return X, Y


def sobol_sampler(space_size, N, function_parameters):
    a    = function_parameters["a"]
    X, Y = [], []
    for _ in range(N):
        x = np.random.rand(len(a)) * space_size
        y = np.prod(np.array([(np.abs(4 * x[k] - 2) + a[k]) / (1 + a[k]) for k in range(len(a))]))
        X.append(x)
        Y.append(y)
    X, Y = np.array(X), np.array(Y)
    return X, Y


def two_input_toy_sampler(space_size, N, _):
    X, Y = [], []
    for _ in range(N):
        x      = np.random.rand(2) * space_size
        x1, x2 = x[0], x[1]
        y      = (1 - np.exp(- 1 / (2 * x2))) * (2300 * x1**3 + 1900 * x1**2 + 2092 * x1 + 60) / (100 * x1**3 + 500 * x1**2 + 4 * x1 + 20)
        X.append(x)
        Y.append(y)
    X, Y = np.array(X), np.array(Y)
    return X, Y


def highly_non_linear_sampler(space_size, N, _):
    X, Y = [], []
    for _ in range(N):
        x      = np.random.rand(2) * space_size
        x1, x2 = x[0], x[1]
        y      = np.cos(6 * (x1 - 0.5)) + 3.1 * np.abs(x1 - 0.7) + 2 * (x1 - 0.5) + 7 * np.sin(1 / (0.5 * x1 + 0.31)) + 0.5 * x2
        X.append(x)
        Y.append(y)
    X, Y = np.array(X), np.array(Y)
    return X, Y


def mystery_sampler(space_size, N, _):
    X, Y = [], []
    for _ in range(N):
        x      = np.random.rand(2) * space_size
        x1, x2 = 5 * x[0], 5 * x[1]
        y      = 2 + 0.01 * (x2 - x1**2)**2 + (1 - x1) + 2 * (2 - x2)**2 + 7 * np.sin(0.5 * x1) * np.sin(0.7 * x1 * x2)
        X.append(x)
        Y.append(y)
    X, Y = np.array(X), np.array(Y)
    return X, Y


def six_hump_camelback_sampler(space_size, N, _):
    X, Y = [], []
    for _ in range(N):
        x      = np.random.rand(2) * space_size
        x1, x2 = 4 * x[0] - 2, 2 * x[1] - 1
        y      = (4 - 2.1 * x1**2 + x1**4 / 3) * x1**2 + x1 * x2 + (-4 + 4 * x2**2) * x2**2
        X.append(x)
        Y.append(y)
    X, Y = np.array(X), np.array(Y)
    return X, Y


def ishigami_sampler(space_size, N, _):
    X, Y = [], []
    for _ in range(N):
        x          = np.random.rand(3) * space_size
        x1, x2, x3 = x[0], x[1], x[2]
        y          = np.sin(-np.pi + 2 * np.pi * x1) + 7 * np.sin(-np.pi + 2 * np.pi * x2)**2 + 0.1 * (-np.pi + 2 * np.pi * x3)**4 * np.sin(-np.pi + 2 * np.pi * x1)
        X.append(x)
        Y.append(y)
    X, Y = np.array(X), np.array(Y)
    return X, Y


def sort_samples(X, Y):
    X_sorted, Y_sorted, xmax_sorted = [], [], -np.inf
    for _ in range(len(X)):
        xmin_sorted, xmin_sorted_x, xmin_sorted_y = np.inf, None, None
        for x, y in zip(X, Y):
            if xmin_sorted >= np.sum(x) > xmax_sorted:
                xmin_sorted   = np.sum(x)
                xmin_sorted_x = x
                xmin_sorted_y = y
        X_sorted.append(xmin_sorted_x)
        Y_sorted.append(xmin_sorted_y)
        xmax_sorted = xmin_sorted
    return X_sorted, Y_sorted


def sampler(N, function_name, sampler_parameters):
    space_size = sampler_parameters["Space size"]
    X, Y       = [], []
    if function_name == "Power sinus 1D":
        X, Y = power_sinus_1D_sampler             (space_size, N, sampler_parameters[function_name])
    if function_name == "Heaviside 1D":
        X, Y =  heaviside_1D_sampler              (space_size, N, sampler_parameters[function_name])
    if function_name == "Multimodal sinus 1D":
        X, Y =  multimodal_sinus_1D_sampler       (space_size, N, sampler_parameters[function_name])
    if function_name == "Noisy sinus 1D":
        X, Y =  noisy_sinus_1D_sampler            (space_size, N, sampler_parameters[function_name])
    if function_name == "Sinus times x 1D":
        X, Y =  sinus_times_x_1D_sampler          (space_size, N, sampler_parameters[function_name])
    if function_name == "Sinus cardinal 1D":
        X, Y =  sinus_cardinal_1D_sampler         (space_size, N, sampler_parameters[function_name])
    if function_name == "Slide function":
        X, Y =  slide_function_sampler            (space_size, N, sampler_parameters[function_name])
    if function_name == "Two bumps":
        X, Y =  two_bumps_sampler                 (space_size, N, sampler_parameters[function_name])
    if function_name == "Power sinus 2D":
        X, Y = power_sinus_2D_sampler             (space_size, N, sampler_parameters[function_name])
    if function_name == "Heaviside 2D":
        X, Y =  heaviside_2D_sampler              (space_size, N, sampler_parameters[function_name])
    if function_name == "Multimodal sinus 2D":
        X, Y =  multimodal_sinus_2D_sampler       (space_size, N, sampler_parameters[function_name])
    if function_name == "Noisy sinus 2D":
        X, Y =  noisy_sinus_2D_sampler            (space_size, N, sampler_parameters[function_name])
    if function_name == "Sinus times x 2D":
        X, Y =  sinus_times_x_2D_sampler          (space_size, N, sampler_parameters[function_name])
    if function_name == "Sinus cardinal 2D":
        X, Y =  sinus_cardinal_2D_sampler         (space_size, N, sampler_parameters[function_name])
    if function_name == "Gaussian process trajectory":
        X, Y = gaussian_process_trajectory_sampler(space_size, N, sampler_parameters[function_name])
    if function_name == "Concrete compressive strength":
        X, Y = uci_dataset_sampler(uci_dataset_id=165)
    if function_name == "Energy efficiency":
        X, Y = uci_dataset_sampler(uci_dataset_id=242)
    if function_name == "Auto mpg":
        X, Y = uci_dataset_sampler(uci_dataset_id=9)
    if function_name == "Combined cycle power plant":
        X, Y = uci_dataset_sampler(uci_dataset_id=294)
    if function_name == "Airfoil self-noise":
        X, Y = uci_dataset_sampler(uci_dataset_id=291)
    if function_name == "Morokoff & caflisch 1":
        X, Y = morokoff_caflisch_1_sampler        (space_size, N, sampler_parameters[function_name])
    if function_name == "Morokoff & caflisch 2":
        X, Y = morokoff_caflisch_2_sampler        (space_size, N, sampler_parameters[function_name])
    if function_name == "Branin":
        X, Y = branin_sampler                     (space_size, N, sampler_parameters[function_name])
    if function_name == "Branin hoo":
        X, Y = branin_hoo_sampler                 (space_size, N, sampler_parameters[function_name])
    if function_name == "Goldstein":
        X, Y = goldstein_sampler                  (space_size, N, sampler_parameters[function_name])
    if function_name == "Goldstein price":
        X, Y = goldstein_price_sampler            (space_size, N, sampler_parameters[function_name])
    if function_name == "Rosenbrock4":
        X, Y = rosenbrock4_sampler                (space_size, N, sampler_parameters[function_name])
    if function_name == "Hartman4":
        X, Y = hartman4_sampler                   (space_size, N, sampler_parameters[function_name])
    if function_name == "Hartman6":
        X, Y = hartman6_sampler                   (space_size, N, sampler_parameters[function_name])
    if function_name == "Sphere6":
        X, Y = sphere6_sampler                    (space_size, N, sampler_parameters[function_name])
    if function_name == "Iooss1":
        X, Y = iooss_1_sampler                    (space_size, N, sampler_parameters[function_name])
    if function_name == "Iooss2":
        X, Y = iooss_2_sampler                    (space_size, N, sampler_parameters[function_name])
    if function_name == "Wing weight":
        X, Y = wing_weight_sampler                (space_size, N, sampler_parameters[function_name])
    if function_name == "Sobol":
        X, Y = sobol_sampler                      (space_size, N, sampler_parameters[function_name])
    if function_name == "Two input toy":
        X, Y = two_input_toy_sampler              (space_size, N, sampler_parameters[function_name])
    if function_name == "Highly non linear":
        X, Y = highly_non_linear_sampler          (space_size, N, sampler_parameters[function_name])
    if function_name == "Mystery":
        X, Y = mystery_sampler                    (space_size, N, sampler_parameters[function_name])
    if function_name == "Six hump camelback":
        X, Y = six_hump_camelback_sampler         (space_size, N, sampler_parameters[function_name])
    if function_name == "Ishigami":
        X, Y = ishigami_sampler                   (space_size, N, sampler_parameters[function_name])
    if len(X[0]) == 1:
        X, Y = sort_samples(X, Y)
    return np.array(X), np.array(Y)


def extract_K_samples(X, Y, K):
    extracted_X, extracted_Y = [], []
    extracted_samples_index = random.sample(range(len(X)), K)
    for extracted_sample_index in extracted_samples_index:
        extracted_X.append(X[extracted_sample_index])
        extracted_Y.append(Y[extracted_sample_index])
    return np.array(extracted_X), np.array(extracted_Y)


def f(X_mesh, Y_mesh, function_name, sampler_parameters):
    Z_mesh = np.zeros(X_mesh.shape)
    for i in range(len(Z_mesh)):
        for j in range(len(Z_mesh[i])):
            x, y = np.array([X_mesh[j][i], Y_mesh[j][i]]), 0
            if function_name == "Power sinus 2D":
                print(sampler_parameters.keys())
                power        = sampler_parameters[function_name]["Power"]
                y            = np.sin(2 * np.pi * np.sum(x)**power)
            if function_name == "Heaviside 2D":
                x_gaps       = sampler_parameters[function_name]["x gaps"]
                steps_value  = sampler_parameters[function_name]["steps value"]
                sigma_noises = sampler_parameters[function_name]["sigma noises"]
                for k, (step_value, sigma_noise) in enumerate(zip(steps_value, sigma_noises)):
                    if x_gaps[k] <= np.sum(x) <= x_gaps[k + 1]:
                        y    = step_value + np.random.normal(loc=0, scale=sigma_noise)
            if function_name == "Multimodal sinus 2D":
                x_gaps       = sampler_parameters[function_name]["x gaps"]
                speeds_value = sampler_parameters[function_name]["speeds value"]
                for k, speed_value in enumerate(speeds_value):
                    if x_gaps[k] <= x[0] <= x_gaps[k + 1]:
                        y    = np.sin(2 * np.pi * speed_value * np.sum(x))
            if function_name == "Noisy sinus 2D":
                sigma_noise  = sampler_parameters[function_name]["sigma noise"]
                y            = np.sin(2 * np.pi * np.sum(x)) + np.random.normal(loc=0, scale=sigma_noise)
            if function_name == "Sinus times x 2D":
                y = np.sum(x) * np.sin(2 * np.pi * np.sum(x))
            if function_name == "Sinus cardinal 2D":
                y = np.sin(2 * np.pi * np.sum(x)) / np.sum(x)
            if function_name == "Morokoff & caflisch 1":
                y            = (1 + 1 / 2)**2 * np.prod(x)**(1 / 2)
            if function_name == "Morokoff & caflisch 2":
                y            = np.prod(2 * np.ones(2) - x) / (2 - 0.5)**2
            if function_name == "Branin":
                x1, x2 = 15 * x[0] - 5, 15 * x[1]
                y      = (x2 - 5.1 * x1 ** 2 / (4 * np.pi ** 2) + 5 * x1 / np.pi - 6) ** 2 + (10 - 10 / (8 * np.pi)) * np.cos(x1) + 10
            if function_name == "Branin hoo":
                x1, x2 = 15 * x[0] - 5, 15 * x[1]
                y      = ((x2 - 5.1 * x1 ** 2 / (4 * np.pi ** 2) + 5 * x1 / np.pi - 6) ** 2 + (10 - 10 / (8 * np.pi)) * np.cos(x1) - 44.81) / 51.95
            if function_name == "Goldstein":
                x1, x2 = 4 * x[0] - 2, 4 * x[1] - 2
                y      = (1 + (x1 + x2 + 1) ** 2 * (19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2)) * (30 + (2 * x1 - 3 * x2) ** 2 * (18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2))
            if function_name == "Goldstein price":
                x1, x2 = 4 * x[0] - 2, 4 * x[1] - 2
                y      = (np.log((1 + (x1 + x2 + 1) ** 2 * (19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2)) * (30 + (2 * x1 - 3 * x2) ** 2 * (18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2))) - 8.693) / 2.427
            if function_name == "Iooss1":
                x1, x2 = 2 * x[0] - 1, 2 * x[1] - 1
                y      = np.exp(x1) / 5 - x2 / 5 + x2 ** 6 / 3 + 4 * x2 ** 4 - 4 * x2 ** 2 + 7 * x1 ** 2 / 10 + x1 ** 4 + 3 / (4 * x1 ** 2 + 4 * x2 ** 2 + 1)
            if function_name == "Iooss2":
                x1, x2 = x[0], x[1]
                y      = np.cos(5 + 3 / 2 * x1) + np.sin(5 + 3 / 2 * x1) + 0.01 * (5 + 3 / 2 * x1) * (5 + 3 / 2 * x2)
            if function_name == "Sobol":
                a = sampler_parameters[function_name]["a"]
                y = np.prod(np.array([(np.abs(4 * x[k] - 2) + a[k]) / (1 + a[k]) for k in range(len(a))]))
            if function_name == "Two input toy":
                x1, x2 = x[0], x[1]
                y      = (1 - np.exp(- 1 / (2 * x2))) * (2300 * x1 ** 3 + 1900 * x1 ** 2 + 2092 * x1 + 60) / (100 * x1 ** 3 + 500 * x1 ** 2 + 4 * x1 + 20)
            if function_name == "Highly non linear":
                x1, x2 = x[0], x[1]
                y      = np.cos(6 * (x1 - 0.5)) + 3.1 * np.abs(x1 - 0.7) + 2 * (x1 - 0.5) + 7 * np.sin(1 / (0.5 * x1 + 0.31)) + 0.5 * x2
            if function_name == "Mystery":
                x1, x2 = 5 * x[0], 5 * x[1]
                y      = 2 + 0.01 * (x2 - x1 ** 2) ** 2 + (1 - x1) + 2 * (2 - x2) ** 2 + 7 * np.sin(0.5 * x1) * np.sin(0.7 * x1 * x2)
            if function_name == "Six hump camelback":
                x1, x2 = 4 * x[0] - 2, 2 * x[1] - 1
                y      = (4 - 2.1 * x1 ** 2 + x1 ** 4 / 3) * x1 ** 2 + x1 * x2 + (-4 + 4 * x2 ** 2) * x2 ** 2
            Z_mesh[j][i] = y
    return Z_mesh
