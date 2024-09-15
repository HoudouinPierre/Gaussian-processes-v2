import numpy as np
import torch
import gaussian_process_gpmp.num as gnp
import gaussian_process_gpmp.kernel as kernel
import gaussian_process_gpmp.core as core
import time


def mu0_initial_guess(Y_observed):
    return np.mean(Y_observed)


class GaussianProcess:

    def __init__(self, X_observed, Y_observed, mu0, p, initialization_method, optimization_method, custom_covparam0, bounds, nugget):
        self._X_observed                       = X_observed
        self._Y_observed                       = Y_observed
        self._mu0                              = mu0
        self._p                                = p
        self._initialization_method            = initialization_method
        self._optimization_method              = optimization_method
        self._custom_covparam0                 = custom_covparam0
        self._bounds                           = bounds
        self._nugget                           = nugget
        self._meanparam                        = gnp.array([mu0])
        self._prior_mean_function              = None
        self._kernel_function                  = None
        self._gaussian_process_model           = None
        self._optimization_infos               = None
        self._gaussian_process_hyperparameters = None
        self._time                             = None

    def _create_prior_mean_function(self):
        def prior_mean_function(x, meanparam):
            return meanparam[0] * gnp.ones((x.shape[0], 1))
        self._prior_mean_function = prior_mean_function

    def _create_kernel_function(self):
        def kernel_function(x, y, covparam, pairwise=False):
            p      = self._p
            nugget = self._nugget
            return kernel.maternp_covariance(x, y, p, covparam, pairwise, nugget)
        self._kernel_function = kernel_function

    def _create_gaussian_process(self):
        self._gaussian_process_model = core.Model(self._prior_mean_function, self._kernel_function, self._meanparam, meantype="parameterized")

    def _train_gaussian_process(self):
        start                                                  = time.time()
        self._gaussian_process_model, self._optimization_infos = kernel.select_parameters(self._initialization_method, self._optimization_method, self._custom_covparam0, self._bounds, self._gaussian_process_model, self._X_observed, self._Y_observed)
        finish                                                 = time.time()
        self._time                                             = finish - start

    def _get_gaussian_process_hyperparameters(self):
        self._gaussian_process_hyperparameters = {"Time"  : self._time,
                                                  "mu0"   : self._mu0 ,
                                                  "sigma0": np.exp(0.5 * self._gaussian_process_model.covparam[0].item()),
                                                  "Noise" : self._nugget}
        for length_scale_index, length_scale in enumerate(self._gaussian_process_model.covparam[1:]):
            self._gaussian_process_hyperparameters["Length scale " + str(length_scale_index)] = np.exp(-length_scale.item())

    def create_and_train_gaussian_process(self):
        self._create_prior_mean_function          ()
        self._create_kernel_function              ()
        self._create_gaussian_process             ()
        self._train_gaussian_process              ()
        self._get_gaussian_process_hyperparameters()

    def optimization_infos(self):
        return self._optimization_infos

    def hyperparameters(self):
        return self._gaussian_process_hyperparameters

    def covparam(self):
        return self._gaussian_process_model.covparam

    def optimization_method(self):
        return self._optimization_method

    def set_gaussian_process_hyperparameters(self, sigma0, length_scales):
        covparam                     = torch.tensor(np.concatenate((np.array([2 * np.log(sigma0)]), np.array([-np.log(length_scale) for length_scale in length_scales]))))
        self._gaussian_process_model = core.Model(self._prior_mean_function, self._kernel_function, self._meanparam, covparam=covparam, meantype="parameterized")
        self._get_gaussian_process_hyperparameters()

    def predict(self, X_unobserved):
        posterior_mean, posterior_variance = self._gaussian_process_model.predict(self._X_observed, self._Y_observed, X_unobserved)
        return posterior_mean, posterior_variance

    def predict_for_plot(self, X_mesh, Y_mesh):
        Z_mesh_mean, Z_mesh_std = np.zeros(X_mesh.shape), np.zeros(X_mesh.shape)
        for i in range(len(Z_mesh_mean)):
            for j in range(len(Z_mesh_mean[i])):
                x                 = np.array([[X_mesh[j][i], Y_mesh[j][i]]])
                mean, variance    = self.predict(x)
                Z_mesh_mean[i][j] = mean[0]
                Z_mesh_std [i][j] = np.sqrt(variance[0])
        return Z_mesh_mean, Z_mesh_std

    def compute_covariance_matrix(self, X, Y, covparam=None):
        if covparam is not None:
            return self._kernel_function(X, Y, covparam)
        else:
            return self._kernel_function(X, Y, self._gaussian_process_model.covparam)
