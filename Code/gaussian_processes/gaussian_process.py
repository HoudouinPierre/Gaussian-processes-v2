import numpy as np
import torch
from torch import tensor, ones
import Code.gaussian_processes.num_core_kernel as nck
import time


def compute_mean_square_error(gaussian_process_model, train_X, train_Y, validation_X, validation_Y):
    posterior_mean, posterior_variance = gaussian_process_model.predict(train_X, train_Y, validation_X)
    mean_square_error                  = np.dot((validation_Y - posterior_mean), (validation_Y - posterior_mean)) / len(validation_Y)
    return mean_square_error


def compute_determination_coefficient(gaussian_process_model, train_X, train_Y, validation_X, validation_Y):
    mean_square_error         = compute_mean_square_error(gaussian_process_model, train_X, train_Y, validation_X, validation_Y)
    explained_variance        = np.mean((validation_Y - np.mean(validation_Y))**2)
    determination_coefficient = 1 - mean_square_error / explained_variance
    return determination_coefficient


def compute_predictive_variance_adequacy(gaussian_process_model, train_X, train_Y, validation_X, validation_Y):
    posterior_mean, posterior_variance = gaussian_process_model.predict(train_X, train_Y, validation_X)
    predictive_variance_adequacy       = np.mean((validation_Y - posterior_mean)**2 / posterior_variance)
    return predictive_variance_adequacy


class GaussianProcessModel:
    def __init__(self, train_X, train_Y, parameters):
        self._train_X                       = train_X
        self._train_Y                       = train_Y
        self._mean_initialization_method    = parameters["Mean initialization method"        ]
        self._kernel_initialization_method  = parameters["Kernel initialization method"      ]
        self._kernel_optimization_criterion = parameters["Kernel optimization criterion"     ]
        self._default_mean_param            = parameters["Default mean parameter"            ]
        self._default_kernel_param          = parameters["Default kernel parameter"          ]
        self._kernel_optimization_bounds    = parameters["Kernel optimization bounds"        ]
        self._nugget                        = parameters["Nugget"                            ]
        self._matern_parameter              = parameters["Matern parameter"                  ]
        self._prior_mean_parameter          = None
        self._mean_function                 = None
        self._kernel_function               = None
        self._gaussian_process_model        = None
        self._time                          = None
        self.optimization_infos             = None

    def _initialize_prior_mean_parameter(self):
        if self._mean_initialization_method == "Average":
            self._prior_mean_parameter = tensor([np.mean(self._train_Y)])
        if self._mean_initialization_method == "Fixed":
            self._prior_mean_parameter = tensor([self._default_mean_param])

    def _create_mean_function(self):
        def mean_function(x, meanparam):
            return meanparam[0] * ones((x.shape[0], 1))
        self._mean_function = mean_function

    def _create_kernel_function(self):
        def kernel_function(x, y, covparam, pairwise=False):
            p      = self._matern_parameter
            nugget = self._nugget
            return nck.maternp_covariance(x, y, p, covparam, nugget, pairwise)
        self._kernel_function = kernel_function

    def _create_gaussian_process_model(self):
        self._gaussian_process_model = nck.Model(self._mean_function, self._kernel_function, self._prior_mean_parameter)

    def _optimize_gaussian_process_model_hyperparameters(self):
        self._gaussian_process_model, self.optimization_infos = nck.select_parameters(self._kernel_initialization_method, self._kernel_optimization_criterion, self._default_kernel_param, self._kernel_optimization_bounds, self._gaussian_process_model, self._train_X, self._train_Y)

    def train(self):
        start = time.time()
        self._initialize_prior_mean_parameter                ()
        self._create_mean_function                           ()
        self._create_kernel_function                         ()
        self._create_gaussian_process_model                  ()
        self._optimize_gaussian_process_model_hyperparameters()
        finish                                                    = time.time()
        self._time                                                = finish - start

    def predict(self, unobserved_X):
        posterior_mean, posterior_variance = self._gaussian_process_model.predict(self._train_X, self._train_Y, unobserved_X)
        return posterior_mean, posterior_variance

    def get_hyperparameters(self):
        gaussian_process_model_parameters = {"Time"          : self._time                                                                                  ,
                                             "p"             : self._matern_parameter                                                                      ,
                                             "mu"            : self._prior_mean_parameter.item()                                                           ,
                                             "sigma"         : np.exp(0.5 * self._gaussian_process_model.covparam[0].item())                               ,
                                             "Length scales" : [np.exp(-length_scale.item()) for length_scale in self._gaussian_process_model.covparam[1:]],
                                             "Nugget"        : self._nugget}
        return gaussian_process_model_parameters

    def set_hyperparameters(self, sigma2, rho):
        covparam                     = torch.tensor(np.concatenate((np.log(np.array([sigma2])), -np.log(np.array(rho)))))
        self._create_kernel_function()
        self._gaussian_process_model = nck.Model(self._mean_function, self._kernel_function, self._prior_mean_parameter, covparam=covparam)

    def get_gaussian_process_model_covparam(self):
        return self._gaussian_process_model.covparam

    def get_kernel_optimization_criterion(self):
        return self._kernel_optimization_criterion

    def compute_covariance_matrix(self, X, Y, covparam=None):
        if covparam is not None:
            return self._kernel_function(X, Y, covparam)
        else:
            return self._kernel_function(X, Y, self._gaussian_process_model.covparam)

    def predict_for_plot(self, X_mesh, Y_mesh):
        Z_mesh_mean, Z_mesh_std = np.zeros(X_mesh.shape), np.zeros(X_mesh.shape)
        for i in range(len(Z_mesh_mean)):
            for j in range(len(Z_mesh_mean[i])):
                x                 = np.array([[X_mesh[j][i], Y_mesh[j][i]]])
                mean, variance    = self.predict(x)
                Z_mesh_mean[i][j] = mean[0]
                Z_mesh_std [i][j] = np.sqrt(variance[0])
        return Z_mesh_mean, Z_mesh_std
