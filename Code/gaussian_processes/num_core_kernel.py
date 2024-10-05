import time
import math
import warnings
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from torch import tensor, einsum, matmul, zeros, ones, any, eye, diag, pi, concatenate, finfo, float64
from torch.linalg import inv, qr
from scipy.special import gammaln
from scipy.optimize import minimize


## -- Utils

def asarray(x):
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, (int, float)):
        return torch.tensor([x])
    else:
        return torch.asarray(x)


def scalar_safe(f):
    def f_(x):
        return f(x) if torch.is_tensor(x) else f(torch.tensor(x))
    return f_


def axis_to_dim(f):
    def f_(x, axis=None, **kwargs):
        if axis is None:
            return f(x, **kwargs)
        else:
            return f(x, dim=axis, **kwargs)

    return f_


log       = scalar_safe(torch.log)
exp       = scalar_safe(torch.exp)
sqrt      = scalar_safe(torch.sqrt)
torch_sum = axis_to_dim(torch.sum)


## -- kernels


def custom_sqrt(x):
    arbitrary_value, mask = 1.0, x == 0.0
    x_copy                = torch.where(mask, arbitrary_value, x)
    res                   = torch.where(mask, 0.0, sqrt(x_copy))
    return res


def cdist(x, y):
    distances     = custom_sqrt(((x ** 2).sum(1).view(-1, 1) + (x ** 2).sum(1).view(-1, 1).t() - 2.0 * torch.mm(x, x.t())).clamp(min=0.0)) if x is y else custom_sqrt(((x ** 2).sum(1).view(-1, 1) + (y**2).sum(1).view(1, -1) - 2.0 * torch.mm(x, y.t())).clamp(min=0.0))
    if x is y:
        distances = distances.masked_fill(torch.eye(distances.size(0), dtype=torch.bool, device=x.device), 0.0)
    return distances


def scaled_distance(loginvrho, x, y):
    d = cdist(exp(loginvrho) * x, exp(loginvrho) * x) if x is y else cdist(exp(loginvrho) * x, exp(loginvrho) * y)
    return d


def maternp_kernel(p: int, h):
    gln                     = [asarray(gammaln(i)) for i in range(2 * p + 2)]
    h                       = torch.where(torch.isinf(h), torch.full_like(h, finfo(float64).max / 1000.0), h)
    c                       = 2.0 * math.sqrt(p + 0.5)
    twoch                   = 2.0 * c * h
    polynomial              = ones(h.shape)
    for i in range(p):
        exp_log_combination = exp(gln[p + 1] - gln[2 * p + 1] + gln[p + i + 1] - gln[i + 1] - gln[p - i + 1])
        twoch_pow           = twoch ** (p - i)
        polynomial         += exp_log_combination * twoch_pow
    return exp(-c * h) * polynomial


def maternp_covariance_ii_or_tt(x, p, param, nugget, pairwise=False):
    sigma2    = exp(param[0])
    loginvrho = param[1:]
    if pairwise:
        K     = sigma2 * ones((x.shape[0],))
    else:
        K     = scaled_distance(loginvrho, x, x)
        K     = sigma2 * (maternp_kernel(p, K) + nugget * eye(K.shape[0]))
    return K


def maternp_covariance_it(x, y, p, param, pairwise=False):
    sigma2    = exp(param[0])
    loginvrho = param[1:]
    if pairwise:
        K = None
    else:
        K     = scaled_distance(loginvrho, x, y)
    K         = sigma2 * maternp_kernel(p, K)
    return K


def maternp_covariance(x, y, p, param, nugget, pairwise=False):
    return maternp_covariance_ii_or_tt(x, p, param, nugget, pairwise) if (y is x or y is None) else maternp_covariance_it(x, y, p, param, pairwise)


## -- Parameters initialization


def norm_k_sqrd(mean, meanparam, covariance, xi, zi, covparam):
    K              = covariance(xi, xi, covparam)
    P              = mean(xi, meanparam)
    n, q           = P.shape
    [Q, _]         = qr(P, "complete")
    W              = Q[:, q:n]
    Wzi            = matmul(W.T, zi)
    G              = matmul(W.T, matmul(K, W))
    try:
        C          = torch.linalg.cholesky(G)
        WKWinv_Wzi = torch.cholesky_solve(Wzi.reshape(-1, 1), C, upper=False)
    except:
        return tensor(float("inf"), requires_grad=True)
    norm_sqrd      = einsum("i..., i...", Wzi, WKWinv_Wzi)
    return norm_sqrd


def anisotropic_parameters_initial_guess(mean, meanparam, covariance, xi, zi):
    xi_        = asarray(xi)
    zi_        = asarray(zi).reshape(-1, 1)
    n, d       = xi_.shape[0], xi_.shape[1]
    delta      = torch.max(xi_, dim=0).values - torch.min(xi_, dim=0).values
    rho        = exp(gammaln(d / 2 + 1) / d) / (pi**0.5) * delta
    covparam   = concatenate((tensor([math.log(1.0)]), -log(rho)))
    sigma2_GLS = 1.0 / n * norm_k_sqrd(mean, meanparam, covariance, xi_, zi_, covparam)
    covparam0  = concatenate((log(sigma2_GLS), -log(rho)))
    return covparam0


def moment_based_parameters_initial_guess(xi, zi):
    sigma2    = np.std(zi)**2
    rho       = []
    for dimension in range(len(xi[0])):
        rho.append(np.std(xi.T[dimension]))
    covparam0 = np.concatenate((np.log(np.array([sigma2])), -np.log(np.array(rho))))
    return covparam0


def custom_parameters_initial_guess(custom_covparam0):
    sigma2, rho = custom_covparam0["sigma2"], custom_covparam0["rho"]
    covparam0   = np.concatenate((np.log(np.array([sigma2])), -np.log(np.array(rho))))
    return covparam0


def covparam_initialization(initialization_method, custom_covparam0, mean, meanparam, covariance, xi, zi):
    covparam0     = None
    if initialization_method == "GPMP":
        covparam0 = anisotropic_parameters_initial_guess(mean, meanparam, covariance, xi, zi)
    if initialization_method == "Moment-based":
        covparam0 = moment_based_parameters_initial_guess(xi, zi)
    if initialization_method == "Custom":
        covparam0 = custom_parameters_initial_guess(custom_covparam0)
    return covparam0


## -- Optimization criterion


def grad(f):
    def f_grad(x):
        x         = x.detach().clone().requires_grad_(True) if torch.is_tensor(x) else torch.tensor(x, requires_grad=True)
        gradients = torch.autograd.grad(f(x), x, allow_unused=True)[0]
        return gradients
    return f_grad


class jax:
    @staticmethod
    def jit(f):
        return f


def negative_log_restricted_likelihood(mean, meanparam, covariance, covparam, xi, zi):
    K              = covariance(xi, xi, covparam)
    P              = mean(xi, meanparam)
    n, q           = P.shape
    [Q, _]         = qr(P, "complete")
    W              = Q[:, q:n]
    Wzi            = matmul(W.T, zi)
    G              = matmul(W.T, matmul(K, W))
    try:
        C          = torch.linalg.cholesky(G)
        WKWinv_Wzi = torch.cholesky_solve(Wzi.reshape(-1, 1), C, upper=False)
    except:
        return tensor(float("inf"), requires_grad=True)
    norm2          = einsum("i..., i...", Wzi, WKWinv_Wzi)
    ldetWKW        = 2.0 * torch_sum(log(diag(C)))
    L              = 0.5 * ((n - q) * log(2.0 * pi) + ldetWKW + norm2)
    return L.reshape(())


def negative_log_likelihood_zero_mean(covariance, covparam, xi, zi):
    K           = covariance(xi, xi, covparam)
    n           = K.shape[0]
    try:
        C       = torch.linalg.cholesky(K)
        Kinv_zi = torch.cholesky_solve(zi.reshape(-1, 1), C, upper=False)
    except:
        return tensor(float("inf"), requires_grad=True)
    norm2       = einsum("i..., i...", zi, Kinv_zi)
    ldetK       = 2.0 * torch_sum(log(diag(C)))
    L           = 0.5 * (n * log(2.0 * pi) + ldetK + norm2)
    return L.reshape(())


def profiled_negative_log_likelihood_zero_mean(covariance, covparam, xi, zi):
    K           = covariance(xi, xi, torch.cat((torch.tensor([0]), covparam)))
    n           = K.shape[0]
    try:
        C       = torch.linalg.cholesky(K)
        Kinv_zi = torch.cholesky_solve(zi.reshape(-1, 1), C, upper=False)
    except:
        return tensor(float("inf"), requires_grad=True)
    norm2       = einsum("i..., i...", zi, Kinv_zi)
    ldetK       = 2.0 * torch_sum(log(diag(C)))
    sigma2      = norm2 / n
    L           = 0.5 * (n * log(2.0 * pi) + ldetK + norm2 / sigma2 + n * log(sigma2))
    return L.reshape(())


def profiled_pva_negative_log_likelihood_zero_mean(covariance, covparam, xi, zi):
    K              = covariance(xi, xi, torch.cat((torch.tensor([0]), covparam)))
    n              = K.shape[0]
    try:
        C          = torch.linalg.cholesky(K)
        Kinv_zi    = torch.cholesky_solve(zi.reshape(-1, 1), C, upper=False)
    except:
        return tensor(float("inf"), requires_grad=True)
    K_inv          = inv(K)
    K_inv_diag_inv = inv(diag(diag(K_inv)))
    pva_matrix     = torch.mm(K_inv, torch.mm(K_inv_diag_inv, K_inv))
    norm2          = einsum("i..., i...", zi, Kinv_zi)
    pva_norm2      = einsum("i..., i...", zi, torch.mv(pva_matrix, zi))
    ldetK          = 2.0 * torch_sum(log(diag(C)))
    sigma2         = pva_norm2 / n
    L              = 0.5 * (n * log(2.0 * pi) + ldetK + norm2 / sigma2 + n * log(sigma2))
    return L.reshape(())


def mean_square_error(covariance, covparam, xi, zi):
    K              = covariance(xi, xi, torch.cat((torch.tensor([0]), covparam)))
    K_inv          = inv(K)
    K_inv_diag_inv = inv(diag(diag(K_inv)))
    mse_matrix     = torch.mm(K_inv, torch.mm(torch.mm(K_inv_diag_inv, K_inv_diag_inv), K_inv))
    mse_norm2      = einsum("i..., i...", zi, torch.mv(mse_matrix, zi))
    return mse_norm2.reshape(())


def make_selection_criterion_with_gradient(optimization_criterion, mean, meanparam, covariance, xi, zi):
    xi_, zi_, crit_ = asarray(xi), asarray(zi), None
    if optimization_criterion == "REML" or "REML - PVA":
        def crit_(covparam):
            return negative_log_restricted_likelihood(mean, meanparam, covariance, covparam, xi_, zi_)
    if optimization_criterion == "MLE" or "MLE - PVA":
        def crit_(covparam):
            return negative_log_likelihood_zero_mean(covariance, covparam, xi_, zi_)
    if optimization_criterion == "PMLE":
        def crit_(covparam):
            return profiled_negative_log_likelihood_zero_mean(covariance, covparam, xi_, zi_)
    if optimization_criterion == "PMLE - PVA":
        def crit_(covparam):
            return profiled_pva_negative_log_likelihood_zero_mean(covariance, covparam, xi_, zi_)
    if optimization_criterion == "MSE - PVA":
        def crit_(covparam):
            return mean_square_error(covariance, covparam, xi_, zi_)
    crit_jit = jax.jit(crit_)
    dcrit    = jax.jit(grad(crit_jit))
    return crit_jit, dcrit


## -- Hyperparameters optimization


def compute_transformed_bounds(bounds):
    if bounds is not None:
        sigma2_transformed_bounds = np.log(np.array([bounds["sigma2"]]))
        rho_transformed_bounds    = -np.log(np.array([bounds["rho"][i][::-1] for i in range(len(bounds["rho"]))]))
        transformed_bounds        = np.concatenate((sigma2_transformed_bounds, rho_transformed_bounds))
    else:
        transformed_bounds        = None
    return transformed_bounds


def autoselect_parameters(p0, criterion, gradient, bounds, method="SLSQP"):
    tic               = time.time()
    history_params    = []
    history_criterion = []
    best_criterion    = float("inf")
    best_params       = None
    options           = {"disp": False}
    if method == "L-BFGS-B":
        options.update({"maxcor": 20, "ftol": 1e-6, "gtol": 1e-5, "eps": 1e-8, "maxfun": 15000, "maxiter": 15000, "iprint": -1, "maxls": 40, "finite_diff_rel_step": None})
    elif method == "SLSQP":
        options.update({"ftol": 1e-6, "eps": 1e-8, "maxiter": 15000, "finite_diff_rel_step": None})
    else:
        raise ValueError("Optimization method not implemented.")

    def record_history(p, criterion_value):
        nonlocal best_criterion, best_params
        history_params   .append(p.copy())
        history_criterion.append(criterion_value)
        if criterion_value < best_criterion:
            best_criterion = criterion_value
            best_params    = p.copy()

    def crit_asnumpy(p):
        v = criterion(asarray(p))
        J = v.detach().item()
        record_history(p, J)
        return J

    def gradient_asnumpy(p):
        g = gradient(asarray(p))
        if g is None:
            return zeros(p.shape)
        else:
            return g
    r = minimize(crit_asnumpy, p0, args=(), method=method, jac=gradient_asnumpy, bounds=bounds, tol=None, callback=None, options=options)
    if r.fun > best_criterion:
        r.x                   = best_params
        r.fun                 = best_criterion
        r.best_value_returned = False
    else:
        r.best_value_returned = True
    r.history_params      = history_params
    r.history_criterion   = history_criterion
    r.initial_params      = p0
    r.final_params        = r.x
    r.selection_criterion = criterion
    r.total_time          = time.time() - tic
    optimization_infos    = {"Hyperparameters history" : history_params, "Criterion history" : history_criterion, "Number of training iterations" : len(history_criterion)}
    return r.x, optimization_infos


## -- sigma adaptation


def compute_log_sigma2_pmle(covariance, covparam, xi, zi):
    K          = covariance(xi, xi, torch.from_numpy(covparam))
    n          = K.shape[0]
    C          = torch.linalg.cholesky(K)
    Kinv_zi    = torch.cholesky_solve(zi.reshape(-1, 1), C, upper=False)
    norm2      = einsum("i..., i...", zi, Kinv_zi)
    log_sigma2 = log(norm2 / n)
    return log_sigma2.reshape(())


def compute_log_sigma2_pva(covariance, covparam, xi, zi):
    K              = covariance(xi, xi, torch.from_numpy(covparam))
    n              = K.shape[0]
    K_inv          = inv(K)
    K_inv_diag_inv = inv(diag(diag(K_inv)))
    pva_matrix     = torch.mm(K_inv, torch.mm(K_inv_diag_inv, K_inv))
    pva_norm2      = einsum("i..., i...", zi, torch.mv(pva_matrix, zi))
    log_sigma2     = log(pva_norm2 / n)
    return log_sigma2.reshape(())


def compute_completed_log_sigma2(optimization_criterion, covariance, covparam, xi, zi):
    if optimization_criterion == "REML - PVA":
        return compute_log_sigma2_pva    (covariance, covparam, xi, asarray(zi))
    if optimization_criterion == "MLE - PVA":
        return compute_log_sigma2_pva    (covariance, covparam, xi, asarray(zi))
    if optimization_criterion == "PMLE":
        return compute_log_sigma2_pmle   (covariance, covparam, xi, asarray(zi))
    if optimization_criterion == "PMLE - PVA":
        return compute_log_sigma2_pva    (covariance, covparam, xi, asarray(zi))
    if optimization_criterion == "MSE - PVA" :
        return compute_log_sigma2_pva    (covariance, covparam, xi, asarray(zi))


## -- Hyperparameters selection


def select_parameters(initialization_method, optimization_criterion, custom_covparam0, bounds, model, xi, zi):
    zi_prior_mean                             = model.mean(xi, model.meanparam).reshape(-1)
    centered_zi                               = zi if optimization_criterion == "REMl" else zi - zi_prior_mean.numpy()
    covparam0                                 = covparam_initialization(initialization_method, custom_covparam0, model.mean, model.meanparam, model.covariance, xi, centered_zi) if len(xi) > 1 else custom_parameters_initial_guess(custom_covparam0)
    crit, dcrit                               = make_selection_criterion_with_gradient(optimization_criterion  , model.mean, model.meanparam, model.covariance, xi, centered_zi) if len(xi) > 1 else (None, None)
    transformed_bounds                        = compute_transformed_bounds(bounds)
    covparam, optimization_infos              = covparam0, {"Hyperparameters history" : [[]], "Criterion history" : [], "Number of training iterations" : 0}
    if optimization_criterion in ["REML", "MLE"] and len(xi) > 1:
        covparam, optimization_infos          = autoselect_parameters(           covparam0, crit, dcrit, transformed_bounds)
    if optimization_criterion in ["REML - PVA", "MLE - PVA"] and len(xi) > 1:
        covparam, optimization_infos          = autoselect_parameters(           covparam0, crit, dcrit, transformed_bounds)
        log_sigma2                            = compute_completed_log_sigma2(optimization_criterion, model.covariance, covparam, xi, centered_zi)
        covparam[0]                           = log_sigma2.item()
    if optimization_criterion in ["PMLE", "PMLE - PVA", "MSE - PVA"] and len(xi) > 1:
        restricted_covparam0                  = covparam0[1:]
        covparam, optimization_infos          = autoselect_parameters(restricted_covparam0, crit, dcrit, transformed_bounds)
        covparam                              = np.concatenate([np.array([0]), covparam])
        log_sigma2                            = compute_completed_log_sigma2(optimization_criterion, model.covariance, covparam, xi, centered_zi)
        covparam[0]                           = log_sigma2.item()
    model.covparam                            = asarray(covparam)
    return model, optimization_infos


## -- Gaussian process model


class Model:

    def __init__(self, mean, covariance, meanparam, covparam=None):
        self.meanparam  = meanparam
        self.covparam   = covparam
        self.mean       = mean
        self.covariance = covariance

    def kriging_predictor_with_zero_mean(self, xi, xt, return_type=0):
        Kii = self.covariance(xi, xi, self.covparam)
        Kit = self.covariance(xi, xt, self.covparam)
        lambda_t, zt_posterior_variance = torch.linalg.solve(Kii, Kit), None
        if return_type == 0:
            zt_prior_variance     = self.covariance(xt, None, self.covparam, pairwise=True)
            zt_posterior_variance = zt_prior_variance - einsum("i..., i...", lambda_t, Kit)
        elif return_type == 1:
            zt_prior_variance     = self.covariance(xt, None, self.covparam, pairwise=False)
            zt_posterior_variance = zt_prior_variance - matmul(lambda_t.T, Kit)
        return lambda_t, zt_posterior_variance

    def predict(self, xi, zi, xt):
        xi_, zi_, xt_                    = asarray(xi), asarray(zi), asarray(xt)
        zi_                              = zi_ - self.mean(xi_, self.meanparam).reshape(-1)
        lambda_t, zt_posterior_variance_ = self.kriging_predictor_with_zero_mean(xi_, xt_)
        if any(zt_posterior_variance_ < 0.0):
            warnings.warn("In predict: negative variances detected. Consider using jitter.", RuntimeWarning)
        zt_posterior_variance = (torch.maximum(tensor(zt_posterior_variance_) if not torch.is_tensor(zt_posterior_variance_) else zt_posterior_variance_, tensor(0.0) if not torch.is_tensor(0.0) else 0.0)).numpy()
        zt_posterior_mean     = (einsum("i..., i...", lambda_t, zi_) + self.mean(xt_, self.meanparam).reshape(-1)).numpy()
        return zt_posterior_mean, zt_posterior_variance
