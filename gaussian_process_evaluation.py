import numpy as np
import torch
import gaussian_process_gpmp.num as gnp
from scipy.linalg import sqrtm
from scipy.stats import chi2, norm, gamma
from scipy.special import erfinv


def negative_log_likelihood_one_parameter(gaussian_process, xi, zi, first_parameter, first_parameter_index):
    covparam                         = torch.tensor([0.0 for _ in range(len(gaussian_process.covparam()))])
    covparam.copy_(gaussian_process.covparam())
    covparam[first_parameter_index]  = torch.tensor(first_parameter)
    K                                = gaussian_process.compute_covariance_matrix(xi, xi, covparam)
    n                                = K.shape[0]
    Kinv_zi, C                       = gnp.cholesky_solve(K, zi)
    norm2                            = gnp.einsum("i..., i...", zi, Kinv_zi)
    ldetK                            = 2.0 * gnp.sum(gnp.log(gnp.diag(C)))
    L                                = 0.5 * (n * gnp.log(2.0 * gnp.pi) + ldetK + norm2)
    return L.reshape(())


def negative_log_likelihood_two_parameters(gaussian_process, xi, zi, first_parameter, first_parameter_index, second_parameter, second_parameter_index):
    covparam                         = torch.tensor([0.0 for _ in range(len(gaussian_process.covparam()))])
    covparam.copy_(gaussian_process.covparam())
    covparam[first_parameter_index ] = torch.tensor(first_parameter )
    covparam[second_parameter_index] = torch.tensor(second_parameter)
    K                                = gaussian_process.compute_covariance_matrix(xi, xi, covparam)
    n                                = K.shape[0]
    Kinv_zi, C                       = gnp.cholesky_solve(K, zi)
    norm2                            = gnp.einsum("i..., i...", zi, Kinv_zi)
    ldetK                            = 2.0 * gnp.sum(gnp.log(gnp.diag(C)))
    L                                = 0.5 * (n * gnp.log(2.0 * gnp.pi) + ldetK + norm2)
    return L.reshape(())


def compute_negative_log_likelihood_evolution(gaussian_process, xi, zi, range_log_sigma2, range_log_inv_rho, n_optimization_functions_plot):
    if len(xi[0]) == 1:  # 2D heatmap plot for likelihood
        log_sigma2s, log_inv_rhos         = np.meshgrid(np.linspace(range_log_sigma2[0], range_log_sigma2[1], n_optimization_functions_plot), np.linspace(range_log_inv_rho[0], range_log_inv_rho[1], n_optimization_functions_plot))
        first_parameter_index             = 0
        second_parameter_index            = 1
        negative_log_likelihoods          = np.array([[negative_log_likelihood_two_parameters(gaussian_process, xi, torch.from_numpy(zi), log_sigma2s[i][j], first_parameter_index, log_inv_rhos[i][j], second_parameter_index) for j in range(n_optimization_functions_plot)] for i in range(n_optimization_functions_plot)])
        negative_log_likelihood_evolution = {"Log sigma2s": log_sigma2s, "Log inv rhos": log_inv_rhos, "Negative log likelihoods": negative_log_likelihoods}
    else:  # 1D plot along each hyperparameter
        negative_log_likelihood_evolution = {}
        for hyperparameter_index in range(1 + len(xi[0])):
            abscissas                     = np.linspace(range_log_sigma2[0], range_log_sigma2[1], n_optimization_functions_plot) if hyperparameter_index == 0 else np.linspace(range_log_inv_rho[0], range_log_inv_rho[1], n_optimization_functions_plot)
            negative_log_likelihoods      = np.array([negative_log_likelihood_one_parameter(gaussian_process, xi, torch.from_numpy(zi), abscissa, hyperparameter_index) for abscissa in abscissas])
            if hyperparameter_index == 0:
                negative_log_likelihood_evolution["Log sigma2"                                  ] = {"Abscissas": abscissas, "Negative log likelihoods": negative_log_likelihoods}
            else:
                negative_log_likelihood_evolution["Log inv rho " + str(hyperparameter_index - 1)] = {"Abscissas": abscissas, "Negative log likelihoods": negative_log_likelihoods}
    return negative_log_likelihood_evolution


def profiled_negative_log_likelihood_one_parameter(gaussian_process, xi, zi, first_parameter, first_parameter_index):
    covparam                             = torch.tensor([0.0 for _ in range(len(gaussian_process.covparam()))])
    covparam.copy_(gaussian_process.covparam())
    covparam[0]                          = torch.tensor([0.0])
    covparam[first_parameter_index + 1]  = torch.tensor(first_parameter)
    K                                    = gaussian_process.compute_covariance_matrix(xi, xi, covparam)
    n                                    = K.shape[0]
    Kinv_zi, C                           = gnp.cholesky_solve(K, zi)
    norm2                                = gnp.einsum("i..., i...", zi, Kinv_zi)
    ldetK                                = 2.0 * gnp.sum(gnp.log(gnp.diag(C)))
    sigma2                               = norm2 / n
    L                                    = 0.5 * (n * gnp.log(2.0 * gnp.pi) + ldetK + norm2 / sigma2 + n * gnp.log(sigma2))
    return L.reshape(())


def profiled_negative_log_likelihood_two_parameters(gaussian_process, xi, zi, first_parameter, first_parameter_index, second_parameter, second_parameter_index):
    covparam                             = torch.tensor([0.0 for _ in range(len(gaussian_process.covparam()))])
    covparam.copy_(gaussian_process.covparam())
    covparam[0]                          = torch.tensor([0.0])
    covparam[first_parameter_index  + 1] = torch.tensor(first_parameter )
    covparam[second_parameter_index + 1] = torch.tensor(second_parameter)
    K                                    = gaussian_process.compute_covariance_matrix(xi, xi, covparam)
    n                                    = K.shape[0]
    Kinv_zi, C                           = gnp.cholesky_solve(K, zi)
    norm2                                = gnp.einsum("i..., i...", zi, Kinv_zi)
    ldetK                                = 2.0 * gnp.sum(gnp.log(gnp.diag(C)))
    sigma2                               = norm2 / n
    L                                    = 0.5 * (n * gnp.log(2.0 * gnp.pi) + ldetK + norm2 / sigma2 + n * gnp.log(sigma2))
    return L.reshape(())


def profiled_pva_negative_log_likelihood_one_parameter(gaussian_process, xi, zi, first_parameter, first_parameter_index):
    covparam                             = torch.tensor([0.0 for _ in range(len(gaussian_process.covparam()))])
    covparam.copy_(gaussian_process.covparam())
    covparam[0]                          = torch.tensor([0.0])
    covparam[first_parameter_index + 1]  = torch.tensor(first_parameter)
    K                                    = gaussian_process.compute_covariance_matrix(xi, xi, covparam)
    n                                    = K.shape[0]
    Kinv_zi, C                           = gnp.cholesky_solve(K, zi)
    K_inv                                = gnp.inv(K)
    K_inv_diag_inv                       = gnp.inv(gnp.diag(gnp.diag(K_inv)))
    pva_matrix                           = torch.mm(K_inv, torch.mm(K_inv_diag_inv, K_inv))
    norm2                                = gnp.einsum("i..., i...", zi, Kinv_zi)
    pva_norm2                            = gnp.einsum("i..., i...", zi, torch.mv(pva_matrix, zi))
    ldetK                                = 2.0 * gnp.sum(gnp.log(gnp.diag(C)))
    sigma2                               = pva_norm2 / n
    L                                    = 0.5 * (n * gnp.log(2.0 * gnp.pi) + ldetK + norm2 / sigma2 + n * gnp.log(sigma2))
    return L.reshape(())


def profiled_pva_negative_log_likelihood_two_parameters(gaussian_process, xi, zi, first_parameter, first_parameter_index, second_parameter, second_parameter_index):
    covparam                             = torch.tensor([0.0 for _ in range(len(gaussian_process.covparam()))])
    covparam.copy_(gaussian_process.covparam())
    covparam[0]                          = torch.tensor([0.0])
    covparam[first_parameter_index  + 1] = torch.tensor(first_parameter )
    covparam[second_parameter_index + 1] = torch.tensor(second_parameter)
    K                                    = gaussian_process.compute_covariance_matrix(xi, xi, covparam)
    n                                    = K.shape[0]
    Kinv_zi, C                           = gnp.cholesky_solve(K, zi)
    K_inv                                = gnp.inv(K)
    K_inv_diag_inv                       = gnp.inv(gnp.diag(gnp.diag(K_inv)))
    pva_matrix                           = torch.mm(K_inv, torch.mm(K_inv_diag_inv, K_inv))
    norm2                                = gnp.einsum("i..., i...", zi, Kinv_zi)
    pva_norm2                            = gnp.einsum("i..., i...", zi, torch.mv(pva_matrix, zi))
    ldetK                                = 2.0 * gnp.sum(gnp.log(gnp.diag(C)))
    sigma2                               = pva_norm2 / n
    L                                    = 0.5 * (n * gnp.log(2.0 * gnp.pi) + ldetK + norm2 / sigma2 + n * gnp.log(sigma2))
    return L.reshape(())


def mean_square_error_one_parameter(gaussian_process, xi, zi, first_parameter, first_parameter_index):
    covparam                             = torch.tensor([0.0 for _ in range(len(gaussian_process.covparam()))])
    covparam.copy_(gaussian_process.covparam())
    covparam[0]                          = torch.tensor([0.0])
    covparam[first_parameter_index + 1]  = torch.tensor(first_parameter)
    K                                    = gaussian_process.compute_covariance_matrix(xi, xi, covparam)
    K_inv                                = gnp.inv(K)
    K_inv_diag_inv                       = gnp.inv(gnp.diag(gnp.diag(K_inv)))
    mse_matrix                           = torch.mm(K_inv, torch.mm(torch.mm(K_inv_diag_inv, K_inv_diag_inv), K_inv))
    mse_norm2                            = gnp.einsum("i..., i...", zi, torch.mv(mse_matrix, zi))
    return mse_norm2.reshape(())


def mean_square_error_two_parameters(gaussian_process, xi, zi, first_parameter, first_parameter_index, second_parameter, second_parameter_index):
    covparam                             = torch.tensor([0.0 for _ in range(len(gaussian_process.covparam()))])
    covparam.copy_(gaussian_process.covparam())
    covparam[0]                          = torch.tensor([0.0])
    covparam[first_parameter_index  + 1] = torch.tensor(first_parameter )
    covparam[second_parameter_index + 1] = torch.tensor(second_parameter)
    K                                    = gaussian_process.compute_covariance_matrix(xi, xi, covparam)
    K_inv                                = gnp.inv(K)
    K_inv_diag_inv                       = gnp.inv(gnp.diag(gnp.diag(K_inv)))
    mse_matrix                           = torch.mm(K_inv, torch.mm(torch.mm(K_inv_diag_inv, K_inv_diag_inv), K_inv))
    mse_norm2                            = gnp.einsum("i..., i...", zi, torch.mv(mse_matrix, zi))
    return mse_norm2.reshape(())


def compute_profiled_quantity_evolution(gaussian_process, xi, zi, range_log_inv_rho, n_optimization_functions_plot, profiled_quantity_metric):
    if len(xi[0]) == 1:  # 1D plot
        log_inv_rhos                 = np.linspace(range_log_inv_rho[0], range_log_inv_rho[1], n_optimization_functions_plot)
        first_parameter_index        = 0
        profiled_quantity            = None
        if profiled_quantity_metric == "Profiled negative log likelihood":
            profiled_quantity        = np.array([profiled_negative_log_likelihood_one_parameter    (gaussian_process, xi, torch.from_numpy(zi), log_inv_rho, first_parameter_index) for log_inv_rho in log_inv_rhos])
        if profiled_quantity_metric == "Profiled PVA negative log likelihood":
            profiled_quantity        = np.array([profiled_pva_negative_log_likelihood_one_parameter(gaussian_process, xi, torch.from_numpy(zi), log_inv_rho, first_parameter_index) for log_inv_rho in log_inv_rhos])
        if profiled_quantity_metric == "Mean square error":
            profiled_quantity        = np.array([mean_square_error_one_parameter                   (gaussian_process, xi, torch.from_numpy(zi), log_inv_rho, first_parameter_index) for log_inv_rho in log_inv_rhos])
        profiled_quantity_evolution  = {"Log inv rhos": log_inv_rhos, profiled_quantity_metric + "s": profiled_quantity}
    elif len(xi[0]) == 2:  # 2D plot
        log_inv_rhos1, log_inv_rhos2 = np.meshgrid(np.linspace(range_log_inv_rho[0], range_log_inv_rho[1], n_optimization_functions_plot), np.linspace(range_log_inv_rho[0], range_log_inv_rho[1], n_optimization_functions_plot))
        first_parameter_index        = 0
        second_parameter_index       = 1
        profiled_quantity            = None
        if profiled_quantity_metric == "Profiled negative log likelihood":
            profiled_quantity        = np.array([[profiled_negative_log_likelihood_two_parameters    (gaussian_process, xi, torch.from_numpy(zi), log_inv_rhos1[i][j], first_parameter_index, log_inv_rhos2[i][j], second_parameter_index) for j in range(n_optimization_functions_plot)] for i in range(n_optimization_functions_plot)])
        if profiled_quantity_metric == "Profiled PVA negative log likelihood":
            profiled_quantity        = np.array([[profiled_pva_negative_log_likelihood_two_parameters(gaussian_process, xi, torch.from_numpy(zi), log_inv_rhos1[i][j], first_parameter_index, log_inv_rhos2[i][j], second_parameter_index) for j in range(n_optimization_functions_plot)] for i in range(n_optimization_functions_plot)])
        if profiled_quantity_metric == "Mean square error":
            profiled_quantity        = np.array([[mean_square_error_two_parameters                   (gaussian_process, xi, torch.from_numpy(zi), log_inv_rhos1[i][j], first_parameter_index, log_inv_rhos2[i][j], second_parameter_index) for j in range(n_optimization_functions_plot)] for i in range(n_optimization_functions_plot)])
        profiled_quantity_evolution  = {"Log inv rhos 1": log_inv_rhos1, "Log inv rhos 2": log_inv_rhos2, profiled_quantity_metric + "s": profiled_quantity}
    else:  # 1D plot along each hyperparameter
        profiled_quantity_evolution  = {}
        for hyperparameter_index in range(len(xi[0])):
            log_inv_rhos             = np.linspace(range_log_inv_rho[0], range_log_inv_rho[1], n_optimization_functions_plot)
            profiled_quantity        = None
            if profiled_quantity_metric == "Profiled negative log likelihood":
                profiled_quantity        = np.array([profiled_negative_log_likelihood_one_parameter    (gaussian_process, xi, torch.from_numpy(zi), log_inv_rho, hyperparameter_index) for log_inv_rho in log_inv_rhos])
            if profiled_quantity_metric == "Profiled PVA negative log likelihood":
                profiled_quantity        = np.array([profiled_pva_negative_log_likelihood_one_parameter(gaussian_process, xi, torch.from_numpy(zi), log_inv_rho, hyperparameter_index) for log_inv_rho in log_inv_rhos])
            if profiled_quantity_metric == "Mean square error":
                profiled_quantity        = np.array([mean_square_error_one_parameter                   (gaussian_process, xi, torch.from_numpy(zi), log_inv_rho, hyperparameter_index) for log_inv_rho in log_inv_rhos])
            profiled_quantity_evolution["Log inv rho " + str(hyperparameter_index)] = {"Log inv rhos": log_inv_rhos, profiled_quantity_metric + "s": profiled_quantity}
    return profiled_quantity_evolution


class GaussianProcessEvaluation:

    def __init__(self, gaussian_process, X_observed, Y_observed, X_test, Y_test, X_function, Y_function, n_alpha=1000, range_log_sigma2=(-5, 5), range_log_inv_rho=(-5, 5), n_optimization_functions_plot=100):
        self._gaussian_process              = gaussian_process
        self._X_observed                    = X_observed
        self._Y_observed                    = Y_observed
        self._X_test                        = X_test
        self._Y_test                        = Y_test
        self._X_function                    = X_function
        self._Y_function                    = Y_function
        self._n_alpha                       = n_alpha
        self._range_log_sigma2              = range_log_sigma2
        self._range_log_inv_rho             = range_log_inv_rho
        self._n_optimization_functions_plot = n_optimization_functions_plot
        self._evaluation                    = None
        self._LOO_mean                      = None
        self._LOO_variance                  = None
        self._Mean_train                    = None
        self._Mean_test                     = None
        self._Mu_train                      = None
        self._Sigma_train_log_det           = None
        self._Sigma_posterior_log_det       = None
        self._Sigma_train_root_inv          = None
        self._Sigma_train_inv               = None
        self._Sigma_posterior               = None
        self._loo_reliability_vector        = None
        self._loo_performance_vector        = None
        self._Sigma_posterior_root_inv      = None
        self._Sigma_train_inv_diag_root_inv = None
        self._lambdas_train, self._P_train  = None, None
        self._Mu_posterior                  = None
        self._Sigma_posterior_diag_root_inv = None
        self._lambdas_test, self._P_test    = None, None

    def _compute_useful_matrices(self):
        mu0                                 = self._gaussian_process.hyperparameters()["mu0"]
        Sigma_train                         = self._gaussian_process.compute_covariance_matrix(self._X_observed, self._X_observed)
        Sigma_test                          = self._gaussian_process.compute_covariance_matrix(self._X_test    , self._X_test    )
        Sigma_train_test                    = self._gaussian_process.compute_covariance_matrix(self._X_observed, self._X_test    )
        self._Sigma_train_log_det           = np.log(np.linalg.det(Sigma_train))
        self._Sigma_train_root_inv          = np.linalg.inv(sqrtm(Sigma_train))
        self._Sigma_train_inv               = np.linalg.inv(      Sigma_train)
        self._Sigma_posterior               = Sigma_test - np.dot(Sigma_train_test.T, np.dot(self._Sigma_train_inv, Sigma_train_test))
        self._Sigma_posterior_log_det       = np.log(np.linalg.det(self._Sigma_posterior))
        Mu_test                             = mu0 * np.ones(self._Y_test.shape)
        self._Mean_train                    = np.sum(self._Y_observed) / len(self._Y_observed) * np.ones(self._Y_observed.shape)
        self._Mean_test                     = np.sum(self._Y_test    ) / len(self._Y_test    ) * np.ones(self._Y_test.shape    )
        self._Mu_train                      = mu0 * np.ones(self._Y_observed.shape)
        self._loo_reliability_vector        = np.dot(self._Sigma_train_inv, (self._Y_observed - self._Mu_train))
        self._LOO_mean                      = self._Y_observed - self._loo_reliability_vector / np.diag(self._Sigma_train_inv)
        self._LOO_variance                  = 1 / np.diag(self._Sigma_train_inv)
        self._Sigma_train_inv_diag_inv      = np.linalg.inv(np.diag(np.diag(self._Sigma_train_inv)))
        self._Sigma_train_inv_diag_root_inv = sqrtm(self._Sigma_train_inv_diag_inv)
        self._lambdas_train, self._P_train  = np.linalg.eig(np.dot(self._Sigma_train_inv_diag_root_inv, np.dot(self._Sigma_train_inv, self._Sigma_train_inv_diag_root_inv)))
        self._Mu_posterior                  = Mu_test + np.dot(Sigma_train_test.T, self._loo_reliability_vector)
        self._Sigma_posterior_root_inv      = np.linalg.inv(sqrtm(self._Sigma_posterior))
        self._Sigma_posterior_diag_root_inv = np.linalg.inv(sqrtm(np.diag(np.diag(self._Sigma_posterior))))
        self._lambdas_test, self._P_test    = np.linalg.eig(np.dot(self._Sigma_posterior_diag_root_inv, np.dot(self._Sigma_posterior, self._Sigma_posterior_diag_root_inv)))
        self._loo_performance_vector        = np.dot(self._Sigma_train_inv_diag_inv, self._loo_reliability_vector)

    def _negative_log_likelihood_evolution(self):
        self._evaluation["Optimization functions evolution"]["Negative log likelihood evolution"             ] = compute_negative_log_likelihood_evolution(self._gaussian_process, self._X_observed, self._Y_observed - self._Mu_train, self._range_log_sigma2, self._range_log_inv_rho, self._n_optimization_functions_plot)

    def _profiled_negative_log_likelihood_evolution(self):
        profiled_quantity_metric                                                                               = "Profiled negative log likelihood"
        self._evaluation["Optimization functions evolution"]["Profiled negative log likelihood evolution"    ] = compute_profiled_quantity_evolution      (self._gaussian_process, self._X_observed, self._Y_observed - self._Mu_train, self._range_log_inv_rho, self._n_optimization_functions_plot, profiled_quantity_metric)

    def _profiled_pva_negative_log_likelihood_evolution(self):
        profiled_quantity_metric                                                                               = "Profiled PVA negative log likelihood"
        self._evaluation["Optimization functions evolution"]["Profiled PVA negative log likelihood evolution"] = compute_profiled_quantity_evolution      (self._gaussian_process, self._X_observed, self._Y_observed - self._Mu_train, self._range_log_inv_rho, self._n_optimization_functions_plot, profiled_quantity_metric)

    def _mean_square_error_evolution(self):
        profiled_quantity_metric                                                                               = "Mean square error"
        self._evaluation["Optimization functions evolution"]["Mean square error evolution"                   ] = compute_profiled_quantity_evolution      (self._gaussian_process, self._X_observed, self._Y_observed - self._Mu_train, self._range_log_inv_rho, self._n_optimization_functions_plot, profiled_quantity_metric)

    def _function_prediction(self):
        posterior_mean, posterior_variance      = self._gaussian_process.predict(self._X_function)
        function_prediction                     = {"X observed"         : self._X_observed  ,
                                                   "Y observed"         : self._Y_observed  ,
                                                   "LOO mean"           : self._LOO_mean    ,
                                                   "LOO variance"       : self._LOO_variance,
                                                   "X test"             : self._X_test      ,
                                                   "Y test"             : self._Y_test      ,
                                                   "X function"         : self._X_function  ,
                                                   "Y function"         : self._Y_function  ,
                                                   "Posterior mean"     : posterior_mean    ,
                                                   "Posterior variance" : posterior_variance}
        self._evaluation["Function prediction"] = function_prediction

    def _predictions(self):
        predictions_train               = np.array([self._Y_observed[i] - self._loo_reliability_vector[i] / self._Sigma_train_inv[i][i] for i in range(len(self._Y_observed))])
        std_train                       = np.array([np.sqrt(1 / self._Sigma_train_inv[i][i])                                            for i in range(len(self._Y_observed))])
        predictions_test, variance_test = self._gaussian_process.predict(self._X_test)
        std_test                        = np.sqrt(variance_test)
        self._evaluation["Train"]["Predictions"] = {"True value"      : self._Y_observed, "Predicted value" : predictions_train, "Std" : std_train}
        self._evaluation["Test" ]["Predictions"] = {"True value"      : self._Y_test    , "Predicted value" : predictions_test , "Std" : std_test}

    def _hyperparameters(self):
        self._evaluation["Hyperparameters"   ] = self._gaussian_process.hyperparameters()

    def _optimization_infos(self):
        self._evaluation["Optimization infos"] = self._gaussian_process.optimization_infos()

    def _optimization_functions_evolution(self):
        self._evaluation["Optimization functions evolution"] = {}
        self._negative_log_likelihood_evolution             ()
        self._profiled_negative_log_likelihood_evolution    ()
        self._profiled_pva_negative_log_likelihood_evolution()
        self._mean_square_error_evolution                   ()

    def _reliability(self):
        alphas                             = np.array([1 / self._n_alpha * i for i in range(self._n_alpha)])
        q_alphas                           = np.array([np.sqrt(2) * erfinv(alpha) for alpha in alphas])
        normalized_prediction_errors_train = np.dot(self._Sigma_train_inv_diag_root_inv, self._loo_reliability_vector)
        normalized_prediction_errors_test  = np.dot(self._Sigma_posterior_diag_root_inv, self._Y_test - self._Mu_posterior)
        PVA_train                          = np.dot(normalized_prediction_errors_train.T, normalized_prediction_errors_train) / len(normalized_prediction_errors_train)
        PVA_test                           = np.dot(normalized_prediction_errors_test .T, normalized_prediction_errors_test ) / len(normalized_prediction_errors_test )
        quantile_PVA_train                 = gamma.cdf(PVA_train, len(normalized_prediction_errors_train) / 2, 0, 2 / len(normalized_prediction_errors_train))
        quantile_PVA_test                  = gamma.cdf(PVA_test , len(normalized_prediction_errors_test ) / 2, 0, 2 / len(normalized_prediction_errors_test ))
        ECP_train                          = np.array([sum(abs(normalized_prediction_errors_train) < q_alpha) / len(self._Y_observed) for q_alpha in q_alphas])
        ECP_test                           = np.array([sum(abs(normalized_prediction_errors_test) < q_alpha) / len(self._Y_test) for q_alpha in q_alphas])
        self._evaluation["Train"]["Reliability"] = {"alphas" : alphas   , "q alphas"     : q_alphas          , "Normalized prediction errors" : normalized_prediction_errors_train,
                                                    "PVA"    : PVA_train, "Quantile PVA" : quantile_PVA_train, "ECP"                          : ECP_train}
        self._evaluation["Test" ]["Reliability"] = {"alphas" : alphas   , "q alphas"     : q_alphas          , "Normalized prediction errors" : normalized_prediction_errors_test ,
                                                    "PVA"    : PVA_test , "Quantile PVA" : quantile_PVA_test , "ECP"                          : ECP_test}

    def _gaussianity_train(self):
        Y_observed_whitened           = np.dot(self._Sigma_train_root_inv, (self._Y_observed - self._Mu_train))
        mahalanobis_distance          = np.dot(Y_observed_whitened.T, Y_observed_whitened)
        log_likelihood                = len(self._Y_observed) / 2 * np.log(2 * np.pi) - 0.5 * mahalanobis_distance - 0.5 * self._Sigma_train_log_det
        quantile_mahalanobis_distance = 1 - chi2.cdf(mahalanobis_distance, df=len(self._Y_observed))
        equivalent_normal_observation = np.sqrt(2) * erfinv(quantile_mahalanobis_distance)
        alphas                        = np.array([1 / self._n_alpha * i for i in range(self._n_alpha)])
        q_alphas                      = np.array([np.sqrt(2) * erfinv(alpha) for alpha in alphas])
        normalized_prediction_errors  = np.dot(self._Sigma_train_inv_diag_root_inv, self._loo_reliability_vector)
        PVA                           = np.dot(normalized_prediction_errors.T, normalized_prediction_errors) / len(normalized_prediction_errors)
        quantile_PVA                  = -1 # we need the generalized chi 2 package
        ECP                           = np.array([sum(                       abs(normalized_prediction_errors  ) < q_alpha)  / len(self._Y_observed) for q_alpha in q_alphas])
        uncorrelated_errors           = np.dot(self._P_train.T, normalized_prediction_errors) / np.sqrt(self._lambdas_train)
        PMIW                          = np.sum(self._Sigma_train_inv_diag_root_inv) / len(self._Y_observed)
        IAE                           = np.sum(np.array([abs(empirical_alpha - alpha) for empirical_alpha, alpha in zip(ECP, alphas)]))
        alphas, ECP                   = np.append(alphas, 1), np.append(ECP, 1)
        self._evaluation["Train"]["Gaussianity"] = {"Y whitened"                    : Y_observed_whitened          ,
                                                    "Mahalanobis distance"          : mahalanobis_distance         ,
                                                    "Quantile Mahalanobis distance" : quantile_mahalanobis_distance,
                                                    "Log likelihood"                : log_likelihood               ,
                                                    "Equivalent normal observation" : equivalent_normal_observation,
                                                    "alphas"                        : alphas                       ,
                                                    "q alphas"                      : q_alphas                     ,
                                                    "ECP"                           : ECP                          ,
                                                    "PVA"                           : PVA                          ,
                                                    "Quantile PVA"                  : quantile_PVA                 ,
                                                    "Lambdas"                       : self._lambdas_train          ,
                                                    "P"                             : self._P_train                ,
                                                    "Uncorrelated errors"           : uncorrelated_errors          ,
                                                    "PMIW"                          : PMIW                         ,
                                                    "IAE"                           : IAE}

    def _gaussianity_test(self):
        Y_test_whitened               = np.dot(self._Sigma_posterior_root_inv, (self._Y_test - self._Mu_posterior))
        mahalanobis_distance          = np.dot(Y_test_whitened.T, Y_test_whitened)
        log_likelihood                = len(self._Y_test) / 2 * np.log(2 * np.pi) - 0.5 * mahalanobis_distance - 0.5 * self._Sigma_posterior_log_det
        quantile_mahalanobis_distance = chi2.cdf(mahalanobis_distance, df=len(self._Y_test))
        equivalent_normal_observation = np.sqrt(2) * erfinv(quantile_mahalanobis_distance)
        alphas                        = np.array([1 / self._n_alpha * i for i in range(self._n_alpha)])
        q_alphas                      = np.array([np.sqrt(2) * erfinv(alpha) for alpha in alphas])
        normalized_prediction_errors  = np.dot(self._Sigma_posterior_diag_root_inv, (self._Y_test - self._Mu_posterior))
        PVA                           = np.dot(normalized_prediction_errors.T, normalized_prediction_errors) / len(normalized_prediction_errors)
        quantile_PVA                  = -1 # we need the generalized chi 2 package
        ECP                           = np.array([sum(                      abs(normalized_prediction_errors  ) < q_alpha)  / len(self._Y_test) for q_alpha in q_alphas])
        uncorrelated_errors           = np.dot(self._P_test.T, normalized_prediction_errors) / np.sqrt(self._lambdas_test)
        PMIW                          = np.sum(self._Sigma_posterior_diag_root_inv) / len(self._Y_observed)
        IAE                           = np.sum(np.array([abs(empirical_alpha - alpha) for empirical_alpha, alpha in zip(ECP, alphas)]))
        alphas, ECP                   = np.append(alphas, 1), np.append(ECP, 1)
        self._evaluation["Test"]["Gaussianity"] = {"Y whitened"                    : Y_test_whitened              ,
                                                   "Mahalanobis distance"          : mahalanobis_distance         ,
                                                   "Quantile Mahalanobis distance" : quantile_mahalanobis_distance,
                                                   "Log likelihood"                : log_likelihood               ,
                                                   "Equivalent normal observation" : equivalent_normal_observation,
                                                   "alphas"                        : alphas                       ,
                                                   "q alphas"                      : q_alphas                     ,
                                                   "ECP"                           : ECP                          ,
                                                   "PVA"                           : PVA                          ,
                                                   "Quantile PVA"                  : quantile_PVA                 ,
                                                   "Lambdas"                       : self._lambdas_test           ,
                                                   "P"                             : self._P_test                 ,
                                                   "Uncorrelated errors"           : uncorrelated_errors          ,
                                                   "PMIW"                          : PMIW                         ,
                                                   "IAE"                           : IAE}

    def _reliability_metrics(self):
        n_train    = len(self._evaluation["Train"]["Gaussianity"]["Lambdas"])
        PVA_train  =     self._evaluation["Train"]["Gaussianity"]["PVA"    ]
        ECP_train  =     self._evaluation["Train"]["Gaussianity"]["ECP"    ]
        PMIW_train =     self._evaluation["Train"]["Gaussianity"]["PMIW"   ]
        IAE_train  =     self._evaluation["Train"]["Gaussianity"]["IAE"    ]
        n_test     = len(self._evaluation["Test" ]["Gaussianity"]["Lambdas"])
        PVA_test   =     self._evaluation["Test" ]["Gaussianity"]["PVA"    ]
        ECP_test   =     self._evaluation["Test" ]["Gaussianity"]["ECP"    ]
        PMIW_test  =     self._evaluation["Test" ]["Gaussianity"]["PMIW"   ]
        IAE_test   =     self._evaluation["Test" ]["Gaussianity"]["IAE"    ]
        self._evaluation["Train"]["Reliability metrics"] = {"N" : n_train, "PVA" : PVA_train, "ECP" : ECP_train, "PMIW" : PMIW_train, "IAE" : IAE_train}
        self._evaluation["Test" ]["Reliability metrics"] = {"N" : n_test , "PVA" : PVA_test , "ECP" : ECP_test , "PMIW" : PMIW_test , "IAE" : IAE_test }

    def _performance_metrics(self):
        MSE_train           = np.dot(self._loo_performance_vector.T, self._loo_performance_vector)                   / len(self._Y_observed)
        Q2_train            = 1 - np.dot(self._loo_performance_vector.T, self._loo_performance_vector)               / np.dot((self._Y_observed - self._Mean_train).T, (self._Y_observed - self._Mean_train))
        MSE_test            = np.dot((self._Y_test - self._Mu_posterior).T, (self._Y_test - self._Mu_posterior))     / len(self._Y_test)
        Q2_test             = 1 - np.dot((self._Y_test - self._Mu_posterior).T, (self._Y_test - self._Mu_posterior)) / np.dot((self._Y_test     - self._Mean_test ).T, (self._Y_test     - self._Mean_test ))
        self._evaluation["Train"]["Performance metrics"] = {"MSE" : MSE_train, "Q2" : Q2_train}
        self._evaluation["Test" ]["Performance metrics"] = {"MSE" : MSE_test , "Q2" : Q2_test }

    def _hybrid_metrics(self):
        alphas                                                          = np.array([1 / self._n_alpha * i for i in range(self._n_alpha)])
        NLPD_train, NLPD_test, IS_train, CRPS_train, CRPS_test, IS_test = 0, 0, [0 for _ in alphas], 0, 0, [0 for _ in alphas]
        for i in range(len(self._Y_observed)):
            posterior_mean = self._Y_observed[i] - self._loo_reliability_vector[i] / self._Sigma_train_inv[i][i]
            posterior_std  = 1 / np.sqrt(self._Sigma_train_inv[i][i])
            NLPD_train     = NLPD_train + (np.log(posterior_std) + 0.5 * ((self._Y_observed[i] - posterior_mean) / posterior_std)**2) / len(self._Y_observed)
            CRPS_train     = CRPS_train + posterior_std * ((self._Y_observed[i] - posterior_mean) / posterior_std * (2 * norm.cdf((self._Y_observed[i] - posterior_mean) / posterior_std) - 1) + 2 * norm.pdf((self._Y_observed[i] - posterior_mean) / posterior_std) - 1 / np.sqrt(np.pi)) / len(self._Y_observed)
            z              = (self._Y_observed[i] - posterior_mean) / posterior_std
            for alpha_index, alpha in enumerate(alphas):
                one_minus_alpha_on_2  ,   one_minus_one_minus_alpha_on_2 = (1 - alpha) / 2                          , 1 - (1 - alpha) / 2
                q_one_minus_alpha_on_2, q_one_minus_one_minus_alpha_on_2 = np.sqrt(2) * erfinv(one_minus_alpha_on_2), np.sqrt(2) * erfinv(one_minus_one_minus_alpha_on_2)
                IS_train[alpha_index]                                    = IS_train[alpha_index] + posterior_std * (q_one_minus_one_minus_alpha_on_2 - q_one_minus_alpha_on_2 + 1 / one_minus_alpha_on_2 * ((q_one_minus_alpha_on_2 - z) * (z <= q_one_minus_alpha_on_2) + (z - q_one_minus_one_minus_alpha_on_2) * (z >= q_one_minus_one_minus_alpha_on_2))) / len(self._Y_observed)
        for i in range(len(self._Y_test)):
            posterior_mean = self._Mu_posterior[i]
            posterior_std  = 1 / self._Sigma_posterior_diag_root_inv[i][i]
            NLPD_test      = NLPD_test  + (np.log(posterior_std) + 0.5 * ((self._Y_test[i]     - posterior_mean) / posterior_std) ** 2) / len(self._Y_test)
            CRPS_test      = CRPS_test  + posterior_std * ((self._Y_test[i]     - posterior_mean) / posterior_std * (2 * norm.cdf((self._Y_test[i]     - posterior_mean) / posterior_std) - 1) + 2 * norm.pdf((self._Y_test[i]     - posterior_mean) / posterior_std) - 1 / np.sqrt(np.pi)) / len(self._Y_test)
            z              = (self._Y_test[i] - posterior_mean) / posterior_std
            for alpha_index, alpha in enumerate(alphas):
                one_minus_alpha_on_2  ,   one_minus_one_minus_alpha_on_2 = (1 - alpha) / 2                          , 1 - (1 - alpha) / 2
                q_one_minus_alpha_on_2, q_one_minus_one_minus_alpha_on_2 = np.sqrt(2) * erfinv(one_minus_alpha_on_2), np.sqrt(2) * erfinv(one_minus_one_minus_alpha_on_2)
                IS_test[alpha_index]                                     = IS_test[alpha_index]  + posterior_std * (q_one_minus_one_minus_alpha_on_2 - q_one_minus_alpha_on_2 + 1 / one_minus_alpha_on_2 * ((q_one_minus_alpha_on_2 - z) * (z <= q_one_minus_alpha_on_2) + (z - q_one_minus_one_minus_alpha_on_2) * (z >= q_one_minus_one_minus_alpha_on_2))) / len(self._Y_test)
        self._evaluation["Train"]["Hybrid metrics"] = {"NLPD" : NLPD_train, "CRPS" : CRPS_train, "IS" : IS_train}
        self._evaluation["Test" ]["Hybrid metrics"] = {"NLPD" : NLPD_test , "CRPS" : CRPS_test , "IS" : IS_test }

    def evaluation(self):
        self._evaluation = {"Train" : {}, "Test" : {}}
        self._compute_useful_matrices         ()
        self._function_prediction             ()
        self._predictions                     ()
        self._hyperparameters                 ()
        self._optimization_infos              ()
        self._optimization_functions_evolution()
        self._reliability                     ()
        self._gaussianity_train               ()
        self._gaussianity_test                ()
        self._reliability_metrics             ()
        self._performance_metrics             ()
        self._hybrid_metrics                  ()
        return self._evaluation
