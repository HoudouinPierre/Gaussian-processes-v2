import time
import numpy as np
import num as gnp
from scipy.optimize import minimize, OptimizeWarning
from math import log, sqrt


## -- kernels


def exponential_kernel(h):
    """Exponential kernel.

    The exponential kernel is defined as:

    .. math::

        k(h) = \exp(-h)

    where `h` represents the distances between points.

    Parameters
    ----------
    h : gnp.array, shape (n,)
        An array of distances between points.

    Returns
    -------
    gnp.array, shape (n,)
        An array of the exponential kernel values corresponding to the input distances.
    """
    return gnp.exp(-h)


def matern32_kernel(h):
    """Matérn 3/2 kernel.

    The Matérn 3/2 kernel is defined as:

    .. math::

        K(h) = (1 + 2\\sqrt{3/2}h) \\exp(-2\\sqrt{3/2}h)

    where `h` represents the distances between points.

    Parameters
    ----------
    h : gnp.array, shape (n,)
        An array of distances between points.

    Returns
    -------
    gnp.array, shape (n,)
        An array of the Matérn 3/2 kernel values corresponding to the input distances.
    """
    nu = 3.0 / 2.0
    c = 2.0 * sqrt(nu)
    t = c * h

    return (1.0 + t) * gnp.exp(-t)


def compute_gammaln(up_to_p):
    """Compute gammaln values."""
    return [gnp.asarray(gnp.gammaln(i)) for i in range(2 * up_to_p + 2)]


gln = []
pmax = -1


def maternp_kernel(p: int, h):
    """Matérn kernel with half-integer regularity nu = p + 1/2.

    The Matérn kernel is defined as in Stein 1999, page 50:

    .. math::

        K(h) = \\frac{1}{\\Gamma(\\nu) 2^{\\nu - 1}} (\\sqrt{2 \\nu} h)^\\nu K_\\nu(\\sqrt{2 \\nu} h)

    Where:
    - :math:`h` represents the distances between points.
    - :math:`\\nu` is the regularity of the kernel.
    - :math:`K_\\nu` is the modified Bessel function of the second kind of order :math:`\\nu`.

    The implementation provided in this function uses this half-integer simplification.

    In the particular case of half-integer regularity (nu = p + 1/2), the Matérn kernel
    simplifies to a product of an exponential term and a polynomial term (Watson 1922,
    A treatise on the theory of Bessel functions, pp. 80, Abramowitz and Stegun 1965, pp. 374-379, 443-444):

    .. math::

        K(h) = \\exp(-2\\sqrt{\\nu}h) \\frac{\\Gamma(p+1)}{\\Gamma(2p+1)} \\sum_{i=0}^{p} \\frac{(p+i)!}{i!(p-i)!} (4\\sqrt{\\nu}h)^{p-i}
    """
    global gln, pmax

    # Check if p exceeds pmax and compute gammaln cache if needed
    if p > pmax:
        gln = compute_gammaln(p)
        pmax = p

    h = gnp.inftobigf(h)
    c = 2.0 * sqrt(p + 0.5)
    twoch = 2.0 * c * h
    polynomial = gnp.ones(h.shape)

    for i in range(p):
        exp_log_combination = gnp.exp(
            gln[p + 1] - gln[2 * p + 1] + gln[p + i + 1] - gln[i + 1] - gln[p - i + 1]
        )
        twoch_pow = twoch ** (p - i)
        polynomial += exp_log_combination * twoch_pow

    return gnp.exp(-c * h) * polynomial


def maternp_covariance_ii_or_tt(x, p, param, pairwise=False, nugget=1e-10):
    """Covariance between observations or predictands at x.

    The covariance matrix is computed using the Matérn kernel with half-integer regularity:

    .. math::

        K_{ij} = \\sigma^2  K(h_{ij}) + \\epsilon \\delta_{ij}

    where `K(h_{ij})` is the Matérn kernel value for the distance `h_{ij}` between points `x_i` and `x_j`,
    `sigma^2` is the variance, `delta_{ij}` is the Kronecker delta, and `epsilon` is a small positive constant.

    Parameters
    ----------
    x : gnp.array, shape (nx, d)
        Observation points.
    p : int
        Half-integer regularity nu = p + 1/2.
    param : gnp.array, shape (1 + d,)
        sigma2 and range parameters.
    pairwise : bool, optional
        Whether to return a covariance matrix k(x_i, x_j),
        for i and j = 1 ... nx, if pairwise is False, or a covariance
        vector k(x_i, x_i) if pairwise is True. Default is False.

    Returns
    -------
    K : gnp.array
        Covariance matrix (nx, nx) or covariance vector if pairwise is True.
    """
    sigma2 = gnp.exp(param[0])
    loginvrho = param[1:]
    #nugget = 10.0 * sigma2 * gnp.finfo(gnp.float64).eps

    if pairwise:
        # return a vector of covariances
        K = sigma2 * gnp.ones((x.shape[0],))  # nx x 0
    else:
        # return a covariance matrix
        K = gnp.scaled_distance(loginvrho, x, x)  # nx x nx
        K = sigma2 * (maternp_kernel(p, K) + nugget * gnp.eye(K.shape[0]))
    return K


def maternp_covariance_it(x, y, p, param, pairwise=False):
    """Covariance between observations and prediction points.

    Parameters
    ----------
    x : ndarray, shape (nx, d)
        Observation points.
    y : ndarray, shape (ny, d)
        Observation points.
    p : int
        Half-integer regularity nu = p + 1/2.
    param : ndarray, shape (1 + d,)
        log(sigma2) and log(1/range) parameters.
    pairwise : bool, optional
        Whether to return a covariance matrix k(x_i, y_j),
        for i in 1 ... nx and j in 1 ... ny, if pairwise is False,
        or a covariance vector k(x_i, y_i) if pairwise is True. Default is False.

    Returns
    -------
    K : ndarray
        Covariance matrix (nx, ny) or covariance vector if pairwise is True.
    """
    sigma2 = gnp.exp(param[0])
    loginvrho = param[1:]

    if pairwise:
        # return a vector of distances
        K = gnp.scaled_distance_elementwise(loginvrho, x, y)  # nx x 0
    else:
        # return a distance matrix
        K = gnp.scaled_distance(loginvrho, x, y)  # nx x ny

    K = sigma2 * maternp_kernel(p, K)

    return K


def maternp_covariance(x, y, p, param, pairwise=False, nugget=1e-10):
    """Matérn covariance function with half-integer regularity nu = p + 1/2.

    The kernel is defined in terms of the Euclidean distance, between
    pairs of input points. For the Matérn kernel, the distance measure
    is scaled by a length scale parameter, which controls how much
    influence distant points have on each other. The kernel has two
    hyperparameters: the length scale and a smoothness parameter,
    which is typically an integer or half-integer value.

    Parameters
    ----------
    x : ndarray, shape (nx, d)
        Observation points.
    y : ndarray, shape (ny, d) or None
        Prediction points. If None, it is assumed that y is x.
    p : int
        Half-integer regularity nu = p + 1/2.
    param : ndarray, shape (1 + d)
        Covariance parameters
        [log(sigma2) log(1/rho_1) log(1/rho_2) ...].
    pairwise : bool, optional
        If True, return a covariance vector k(x_i, y_i). If False,
        return a covariance matrix k(x_i, y_j) for i in the range 1 to nx
        and j in the range 1 to ny. Default is False.

    Returns
    -------
    K : ndarray
        Covariance matrix (nx, ny) or covariance vector if pairwise is True.

    Notes
    -----
    An isotropic covariance is obtained if param = [log(sigma2) log(1/rho)]
    (only one length scale parameter).
    """
    if y is x or y is None:
        return maternp_covariance_ii_or_tt(x, p, param, pairwise, nugget)
    else:
        return maternp_covariance_it(x, y, p, param, pairwise)


## -- parameters


def anisotropic_parameters_initial_guess_zero_mean(model, xi, zi):
    """Anisotropic initialization strategy with zero mean.

    See anisotropic_parameters_initial_guess
    """
    xi_ = gnp.asarray(xi)
    zi_ = gnp.asarray(zi).reshape(-1, 1)
    n = xi_.shape[0]
    d = xi_.shape[1]

    delta = gnp.max(xi_, axis=0) - gnp.min(xi_, axis=0)
    rho = gnp.exp(gnp.gammaln(d / 2 + 1) / d) / (gnp.pi**0.5) * delta
    covparam = gnp.concatenate((gnp.array([log(1.0)]), -gnp.log(rho)))
    sigma2_GLS = 1.0 / n * model.norm_k_sqrd_with_zero_mean(xi_, zi_, covparam)

    return gnp.concatenate((gnp.log(sigma2_GLS), -gnp.log(rho)))


def anisotropic_parameters_initial_guess_constant_mean(model, xi, zi):
    """Anisotropic initialization strategy with a parameterized constant mean.

    This function provides initial parameter guesses for an
    anisotropic Gaussian process with a parameterized constant mean.

    Parameters
    ----------
    model : object
        The Gaussian process model object.
    xi : array_like, shape (n, d)
        Input data points used for fitting the GP model, where `n` is
        the number of points and `d` is the dimensionality.
    zi : array_like, shape (n, )
        Output (response) values corresponding to the input data points xi.

    Returns
    -------
    mean_GLS : float
        The generalized least squares (GLS) estimator of the
        mean. Computed as:

        .. math::

            m_{GLS} = \frac{\mathbf{1}^T K^{-1} \mathbf{z}}{\mathbf{1}^T K^{-1} \mathbf{1}}

    concatenated parameters : array_like
        An array containing the initialized :math:`\sigma^2_{GLS}` and :math:`\rho` values.
        The estimator :math:`\sigma^2_{GLS}` is given by:

        .. math::

            \sigma^2_{GLS} = \frac{1}{n} \mathbf{z}^T K^{-1} \mathbf{z}
    """
    xi_ = gnp.asarray(xi)
    zi_ = gnp.asarray(zi).reshape((-1, 1))  # Ensure zi_ is a column vector
    n = xi_.shape[0]
    d = xi_.shape[1]

    delta = gnp.max(xi_, axis=0) - gnp.min(xi_, axis=0)
    rho = gnp.exp(gnp.gammaln(d / 2 + 1) / d) / (gnp.pi**0.5) * delta

    covparam = gnp.concatenate((gnp.array([gnp.log(1.0)]), -gnp.log(rho)))
    zTKinvz, Kinv1, Kinvz = model.k_inverses(xi_, zi_, covparam)

    mean_GLS = gnp.sum(Kinvz) / gnp.sum(Kinv1)
    sigma2_GLS = (1.0 / n) * zTKinvz

    return mean_GLS.reshape(1), gnp.concatenate((gnp.log(sigma2_GLS), -gnp.log(rho)))


def anisotropic_parameters_initial_guess(model, xi, zi):
    """Anisotropic initialization strategy for parameters of a Gaussian process model.

    Given the observed data points and their values, this function
    computes an initial guess for the anisotropic parameters. The
    guess for :math:`\\sigma^2` is initialized using the Generalized
    Least Squares (GLS) estimate as described below.

    Parameters
    ----------
    model : object
        An instance of a Gaussian process model.
    xi : ndarray, shape (n, d)
        Locations of the observed data points.
    zi : ndarray, shape (n,)
        Observed values at the data points.

    Returns
    -------
    initial_params : ndarray
        Initial guess for the anisotropic parameters, comprising the
        estimate for :math:`\\sigma^2_{GLS}` followed by the estimates
        for the anisotropic lengthscales.

    Notes
    -----
    The GLS estimate for :math:`\\sigma^2` is given by:

    .. math::

        \\sigma^2_{GLS} = \\frac{1}{n} \\mathbf{z}^T \\mathbf{K}^{-1} \\mathbf{z}

    Where:

    * :math:`n` is the number of data points.
    * :math:`\\mathbf{z}` is the vector of observed data.
    * :math:`\\mathbf{K}` is the covariance matrix associated with the data locations.

    Additionally, the function uses a relation (not from the
    reference) between :math:`\\rho` and the volume of a ball in
    dimension :math:`d` for initialization:

    .. math::

        V_d(R) = \\frac{\\pi^{d/2} R^d}{\\Gamma(d/2+1)}

    Where :math:`R` is defined as :math:`\\rho / 2`.

    .. [1] Basak, S., Petit, S., Bect, J., & Vazquez, E. (2021).
       Numerical issues in maximum likelihood parameter estimation for
       Gaussian process interpolation. arXiv:2101.09747.
    """

    xi_ = gnp.asarray(xi)
    zi_ = gnp.asarray(zi).reshape(-1, 1)
    n = xi_.shape[0]
    d = xi_.shape[1]

    delta = gnp.max(xi_, axis=0) - gnp.min(xi_, axis=0)
    rho = gnp.exp(gnp.gammaln(d / 2 + 1) / d) / (gnp.pi**0.5) * delta

    covparam = gnp.concatenate((gnp.array([log(1.0)]), -gnp.log(rho)))
    sigma2_GLS = 1.0 / n * model.norm_k_sqrd(xi_, zi_, covparam)
    covparam0  = gnp.concatenate((gnp.log(sigma2_GLS), -gnp.log(rho)))

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


def make_selection_criterion_with_gradient(selection_criterion, xi, zi, parameterized_mean=False, meanparam_len=1):
    """Make selection criterion function with gradient.

    Parameters
    ----------
    selection_criterion : function
        Selection criterion function.
    xi : ndarray, shape (n, d)
        Locations of the observed data points.
    zi : ndarray, shape (n,)
        Observed values at the data points.
    parameterized_mean : bool, optional
        Whether to use mean parameter in the selection criterion.
    meanparam_len : int, optional
        Length of the mean parameter, used only if parameterized_mean is True.

    Returns
    -------
    crit_jit : function
        Selection criterion function with gradient.
    dcrit : function
        Gradient of the selection criterion function.
    """
    xi_ = gnp.asarray(xi)
    zi_ = gnp.asarray(zi)

    if parameterized_mean:
        # make a selection criterion with mean parameter
        def crit_(param):
            meanparam = param[:meanparam_len]
            covparam = param[meanparam_len:]
            l = selection_criterion(meanparam, covparam, xi_, zi_)
            return l

    else:
        # make a selection criterion without mean parameter
        def crit_(covparam):
            l = selection_criterion(covparam, xi_, zi_)
            return l

    crit_jit = gnp.jax.jit(crit_)
    dcrit = gnp.jax.jit(gnp.grad(crit_jit))

    return crit_jit, dcrit


def autoselect_parameters(p0, criterion, gradient, bounds=None, method="SLSQP"):
    """Optimize parameters using a provided criterion and gradient function.

    This function automatically optimizes the parameters of a given model based on
    a specified criterion function and its gradient. Different optimization methods
    can be used based on the `method` argument.

    Parameters
    ----------
    p0 : ndarray
        Initial guess of the parameters.
    criterion : function
        Function that computes the selection criterion for a given parameter set.
        It should take an ndarray of parameters and return a scalar value.
    gradient : function
        Function that computes the gradient of the selection criterion with respect
        to the parameters. It should take an ndarray of parameters and return an ndarray
        of the same shape.
    bounds : sequence of tuple, optional
        A sequence of (min, max) pairs specifying bounds for each parameter.
        Use None to indicate no bounds. Default is None.
    silent : bool, optional
        If True, suppresses optimization output messages. Default is True.
    info : bool, optional
        If True, returns additional information about the optimization process.
        Default is False.
    method : str, optional
        Optimization method to use. Supported methods are 'L-BFGS-B' and 'SLSQP'.
        Default is 'SLSQP'.
    method_options : dict, optional, default {}
        User options for the optimization method.

    Returns
    -------
    best : ndarray
        Array of optimized parameters.
    r : OptimizeResult, optional
        A dictionary of optimization information (only if `info=True`). This includes
        details like the initial parameters, final parameters, selection criterion function,
        and total time taken for optimization.

    Notes
    -----
    The function uses the `minimize` method from `scipy.optimize` for optimization.
    Depending on the backend (`gnp._gpmp_backend_`), different preparations are made
    for the criterion and gradient functions to ensure compatibility.
    """
    tic               = time.time()
    history_params    = []
    history_criterion = []
    best_criterion    = float("inf")
    best_params       = None

    def record_history(p, criterion_value):
        nonlocal best_criterion, best_params
        history_params   .append(p.copy())
        history_criterion.append(criterion_value)
        if criterion_value < best_criterion:
            best_criterion = criterion_value
            best_params    = p.copy()

    def crit_asnumpy(p):
        v = criterion(gnp.asarray(p))
        J = v.detach().item()
        record_history(p, J)
        return J

    def gradient_asnumpy(p):
        g = gradient(gnp.asarray(p))
        if g is None:
            return gnp.zeros(p.shape)
        else:
            return g

    options = {"disp": False}
    if method == "L-BFGS-B":
        options.update({"maxcor": 20, "ftol": 1e-6, "gtol": 1e-5, "eps": 1e-8, "maxfun": 15000, "maxiter": 15000, "iprint": -1, "maxls": 40, "finite_diff_rel_step": None})
    elif method == "SLSQP":
        options.update({"ftol": 1e-6, "eps": 1e-8, "maxiter": 15000, "finite_diff_rel_step": None})
    else:
        raise ValueError("Optimization method not implemented.")
    print(bounds)
    # Perform the minimization
    r = minimize(crit_asnumpy, p0, args=(), method=method, jac=gradient_asnumpy, bounds=bounds, tol=None, callback=None, options=options)

    # Ensure that the best values are returned
    if r.fun > best_criterion:
        r.x                   = best_params
        r.fun                 = best_criterion
        r.best_value_returned = False
    else:
        r.best_value_returned = True

    # Set additional information about the optimization process
    r.history_params      = history_params
    r.history_criterion   = history_criterion
    r.initial_params      = p0
    r.final_params        = r.x
    r.selection_criterion = criterion
    r.total_time          = time.time() - tic
    optimization_infos    = {"Hyperparameters history" : history_params, "Criterion history" : history_criterion}
    return r.x, optimization_infos


def compute_transformed_bounds(bounds):
    if bounds is not None:
        sigma2_transformed_bounds = np.log(np.array([bounds["sigma2"]]))
        rho_transformed_bounds    = -np.log(np.array([bounds["rho"][i][::-1] for i in range(len(bounds["rho"]))]))
        transformed_bounds        = np.concatenate((sigma2_transformed_bounds, rho_transformed_bounds))
    else:
        transformed_bounds        = None
    return transformed_bounds


def covparam_initialization(initialization_method, custom_covparam0, model, xi, zi):
    covparam0 = None
    if initialization_method == "GPMP":
        covparam0 = anisotropic_parameters_initial_guess(model, xi, zi)
    if initialization_method == "Moment-based":
        covparam0 = moment_based_parameters_initial_guess(xi, zi)
    if initialization_method == "Custom":
        covparam0 = custom_parameters_initial_guess(custom_covparam0)
    return covparam0

def retrieve_selection_criterion(optimization_method, model, xi, zi):
    if optimization_method == "REML"      :
        return make_selection_criterion_with_gradient(model.negative_log_restricted_likelihood            , xi, zi)
    if optimization_method == "MLE"       :
        return make_selection_criterion_with_gradient(model.negative_log_likelihood_zero_mean             , xi, zi)
    if optimization_method == "REML - PVA":
        return make_selection_criterion_with_gradient(model.negative_log_restricted_likelihood            , xi, zi)
    if optimization_method == "MLE - PVA" :
        return make_selection_criterion_with_gradient(model.negative_log_likelihood_zero_mean             , xi, zi)
    if optimization_method == "PMLE"      :
        return make_selection_criterion_with_gradient(model.profiled_negative_log_likelihood_zero_mean    , xi, zi)
    if optimization_method == "PMLE - PVA":
        return make_selection_criterion_with_gradient(model.profiled_pva_negative_log_likelihood_zero_mean, xi, zi)
    if optimization_method == "MSE - PVA" :
        return make_selection_criterion_with_gradient(model.mean_square_error                             , xi, zi)


def compute_completed_log_sigma2(optimization_method, model, covparam, xi, zi):
    if optimization_method == "REML - PVA":
        return model.compute_log_sigma2_pva    (covparam, xi, gnp.asarray(zi))
    if optimization_method == "MLE - PVA":
        return model.compute_log_sigma2_pva    (covparam, xi, gnp.asarray(zi))
    if optimization_method == "PMLE":
        return model.compute_log_sigma2_pmle   (covparam, xi, gnp.asarray(zi))
    if optimization_method == "PMLE - PVA":
        return model.compute_log_sigma2_pva    (covparam, xi, gnp.asarray(zi))
    if optimization_method == "MSE - PVA" :
        return model.compute_log_sigma2_pva    (covparam, xi, gnp.asarray(zi))


def select_parameters(initialization_method, optimization_method, custom_covparam0, bounds, model, xi, zi):
    zi_prior_mean                             = model.mean(xi, model.meanparam).reshape(-1)
    centered_zi                               = zi if optimization_method == "REMl" else zi - zi_prior_mean.numpy()
    covparam0                                 = covparam_initialization(initialization_method, custom_covparam0, model, xi, centered_zi)
    transformed_bounds                        = compute_transformed_bounds(bounds)
    crit, dcrit                               = retrieve_selection_criterion(optimization_method, model, xi, centered_zi)
    covparam, optimization_infos              = None, None
    if optimization_method in ["REML", "MLE"]:
        covparam, optimization_infos          = autoselect_parameters(           covparam0, crit, dcrit, transformed_bounds)
    if optimization_method in ["REML - PVA", "MLE - PVA"]:
        covparam, optimization_infos          = autoselect_parameters(           covparam0, crit, dcrit, transformed_bounds)
        log_sigma2                            = compute_completed_log_sigma2(optimization_method, model, covparam, xi, centered_zi)
        covparam[0]                           = log_sigma2.item()
    if optimization_method in ["PMLE", "PMLE - PVA", "MSE - PVA"]:
        restricted_covparam0                  = covparam0[1:]  # remove scale parameter
        covparam, optimization_infos          = autoselect_parameters(restricted_covparam0, crit, dcrit, transformed_bounds)
        covparam                              = np.concatenate([np.array([0]), covparam])
        log_sigma2                            = compute_completed_log_sigma2(optimization_method, model, covparam, xi, centered_zi)
        covparam[0]                           = log_sigma2.item()
    model.covparam                            = gnp.asarray(covparam)
    return model, optimization_infos


def update_parameters_with_reml(model, xi, zi, info=False):
    """Update model parameters with Restricted Maximum Likelihood (REML).

    Parameters
    ----------
    model : object
        Gaussian process model object.
    xi : ndarray, shape (n, d)
        Locations of the observed data points.
    zi : ndarray, shape (n,)
        Observed values at the data points.
    info : bool, optional
        If True, returns additional information. Default is False.

    Returns
    -------
    model : object
        Updated Gaussian process model object with optimized parameters.
    info_ret : dict, optional
        Additional information about the optimization (if info=True).
    """
    tic = time.time()

    covparam0 = model.covparam

    nlrl, dnlrl = make_selection_criterion_with_gradient(
        model.negative_log_restricted_likelihood, xi, zi
    )

    covparam_reml, info_ret = autoselect_parameters(
        covparam0, nlrl, dnlrl, silent=True, info=True
    )

    model.covparam = covparam_reml

    if info:
        info_ret["covparam0"] = covparam0
        info_ret["covparam"] = covparam_reml
        info_ret["selection_criterion"] = nlrl
        info_ret["time"] = time.time() - tic
        return model, info_ret
    else:
        return model
    # --- end if
