import os
import importlib
import numpy


_gpmp_backend_ = os.environ.get("GPMP_BACKEND")


def set_backend_env_var(backend):
    global _gpmp_backend_
    os.environ["GPMP_BACKEND"] = backend
    _gpmp_backend_ = backend


# Automatically set the backend if not already set in the environment.
if _gpmp_backend_ is None:
    if importlib.util.find_spec("torch") is not None:
        set_backend_env_var("torch")
    elif importlib.util.find_spec("jax") is not None:
        set_backend_env_var("jax")
    else:
        set_backend_env_var("numpy")

print(f"Using backend: {_gpmp_backend_}")


# -----------------------------------------------------
#
#                      NUMPY
#
# -----------------------------------------------------
if _gpmp_backend_ == "numpy":
    from numpy import array, empty

    from numpy import (
        copy,
        reshape,
        where,
        any,
        isscalar,
        isnan,
        isinf,
        isfinite,
        unique,
        hstack,
        vstack,
        stack,
        tile,
        concatenate,
        empty,
        empty_like,
        zeros,
        ones,
        full,
        full_like,
        eye,
        diag,
        arange,
        linspace,
        logspace,
        meshgrid,
        abs,
        sqrt,
        exp,
        log,
        log10,
        sum,
        mean,
        std,
        cov,
        sort,
        min,
        max,
        argmin,
        argmax,
        minimum,
        maximum,
        einsum,
        matmul,
        inner,
        all,
        logical_not,
        logical_and,
        logical_or,
    )
    from numpy.linalg import norm, cond, qr, svd, inv
    from numpy.random import rand, randn, choice
    from numpy import pi, inf
    from numpy import finfo, float64
    from scipy.special import gammaln
    from scipy.linalg import solve, cholesky, cho_factor, cho_solve
    from scipy.spatial.distance import cdist
    from scipy.stats import norm as normal
    from scipy.stats import multivariate_normal as scipy_mvnormal

    eps = finfo(float64).eps
    fmax = numpy.finfo(numpy.float64).max

    def set_elem1(x, index, v):
        x[index] = v
        return x

    def set_row2(A, index, x):
        A[index, :] = x
        return A

    def set_col2(A, index, x):
        A[:, index] = x
        return A

    def set_col3(A, index, x):
        A[:, :, index] = x
        return A

    def asarray(x):
        if isinstance(x, numpy.ndarray):
            return x
        elif isinstance(x, (int, float)):
            return numpy.array([x])
        else:
            return numpy.asarray(x)

    def asdouble(x):
        return x.astype(float64)

    def to_np(x):
        return x

    def to_scalar(x):
        return x.item()

    def isarray(x):
        return isinstance(x, numpy.ndarray)

    def inftobigf(a, bigf=fmax / 1000.0):
        a = where(numpy.isinf(a), numpy.full_like(a, bigf), a)
        return a

    def grad(f):
        return None

    class jax:
        @staticmethod
        def jit(f, *args, **kwargs):
            return f

    def scaled_distance(loginvrho, x, y):
        invrho = exp(loginvrho)
        xs = invrho * x
        ys = invrho * y
        return cdist(xs, ys)

    def cholesky_inv(A):
        # FIXME: slow!
        # n = A.shape[0]
        # C, lower = cho_factor(A)
        # Ainv = cho_solve((C, lower), eye(n))
        return inv(A)

    def cholesky_solve(A, b):
        C, lower = cho_factor(A)
        return cho_solve((C, lower), b), C

    class multivariate_normal:
        @staticmethod
        def rvs(mean=0.0, cov=1.0, n=1):
            # Check if cov is a scalar or 1x1 array, and use norm if so
            if isscalar(cov) or (isinstance(cov, numpy.ndarray) and cov.size == 1):
                return normal.rvs(mean, numpy.sqrt(cov), size=n)

            # For dxd covariance matrix, use multivariate_normal
            d = cov.shape[0]  # Dimensionality from the covariance matrix
            mean_array = numpy.full(d, mean)  # Expand mean to an array
            return scipy_mvnormal.rvs(mean=mean_array, cov=cov, size=n)

        @staticmethod
        def logpdf(x, mean=0.0, cov=1.0):
            # Check if cov is a scalar or 1x1 array, and use norm if so
            if numpy.isscalar(cov) or (
                isinstance(cov, numpy.ndarray) and cov.size == 1
            ):
                return normal.logpdf(x, mean, numpy.sqrt(cov))

            # For dxd covariance matrix, use multivariate_normal
            d = x.shape[-1] if x.ndim > 1 else 1  # Infer dimensionality from x
            mean_array = numpy.full(d, mean)  # Expand mean to an array
            return scipy_mvnormal.logpdf(x, mean=mean_array, cov=cov)

        @staticmethod
        def cdf(x, mean=0.0, cov=1.0):
            # Check if cov is a scalar or 1x1 array, and use norm for the univariate case
            if isscalar(cov) or (isinstance(cov, numpy.ndarray) and cov.size == 1):
                return normal.cdf(x, mean, sqrt(cov))

            # For dxd covariance matrix, use multivariate_normal
            if isinstance(mean, (float, int)):
                d = cov.shape[0]  # Dimensionality from the covariance matrix
                mean = full(d, mean)  # Expand mean to an array
            return scipy_mvnormal.cdf(x, mean=mean, cov=cov)


# -----------------------------------------------------
#
#                      TORCH
#
# -----------------------------------------------------
elif _gpmp_backend_ == "torch":
    import torch

    torch.set_default_dtype(torch.float64)

    from torch import tensor, is_tensor

    from torch import (
        reshape,
        where,
        any,
        isnan,
        isinf,
        isfinite,
        hstack,
        vstack,
        stack,
        tile,
        concatenate,
        empty,
        empty_like,
        zeros,
        ones,
        full,
        eye,
        diag,
        arange,
        linspace,
        logspace,
        meshgrid,
        abs,
        cov,
        argmax,
        argmin,
        einsum,
        matmul,
        inner,
        logical_not,
        logical_and,
        logical_or,
    )
    from torch.linalg import cond, qr, inv
    from torch import rand, randn
    from torch import pi, inf
    from torch import finfo, float64
    from torch.distributions.multivariate_normal import MultivariateNormal
    from torch.distributions.normal import Normal
    from scipy.stats import multivariate_normal as scipy_mvnormal
    from scipy.special import gammaln

    eps = finfo(float64).eps
    fmax = finfo(float64).max

    def copy(x):
        return x.clone()

    def set_elem1(x, index, v):
        x[index] = v
        return x

    def set_row2(A, index, x):
        A[index, :] = x
        return A

    def set_col2(A, index, x):
        A[:, index] = x
        return A

    def set_col3(A, index, x):
        A[:, :, index] = x
        return A

    def array(x: list):
        return tensor(x)

    def asarray(x):
        if isinstance(x, torch.Tensor):
            return x
        elif isinstance(x, (int, float)):
            return tensor([x])
        else:
            return torch.asarray(x)

    def asdouble(x):
        return x.to(torch.double)

    def to_np(x):
        return x.numpy()

    def to_scalar(x):
        return x.item()

    def isarray(x):
        return torch.is_tensor(x)

    def isscalar(x):
        if torch.is_tensor(x):
            return x.reshape(-1).size()[0] == 1
        elif isinstance(x, (int, float)):
            return True
        else:
            return False

    def scalar_safe(f):
        def f_(x):
            if torch.is_tensor(x):
                return f(x)
            else:
                return f(torch.tensor(x))

        return f_

    log = scalar_safe(torch.log)
    log10 = scalar_safe(torch.log10)
    exp = scalar_safe(torch.exp)
    sqrt = scalar_safe(torch.sqrt)

    def axis_to_dim(f):
        def f_(x, axis=None, **kwargs):
            if axis is None:
                return f(x, **kwargs)
            else:
                return f(x, dim=axis, **kwargs)

        return f_

    all = axis_to_dim(torch.all)
    unique = axis_to_dim(torch.unique)
    sum = axis_to_dim(torch.sum)
    mean = axis_to_dim(torch.mean)
    std = axis_to_dim(torch.std)
    var = axis_to_dim(torch.var)

    def norm(x, axis=None, ord=2):
        return torch.norm(x, dim=axis, p=ord)

    def min(x, axis=0):
        m = torch.min(x, dim=axis)
        return m.values

    def max(x, axis=0):
        m = torch.max(x, dim=axis)
        return m.values

    def maximum(x1, x2):
        if not torch.is_tensor(x1):
            x1 = tensor(x1)
        if not torch.is_tensor(x2):
            x2 = tensor(x2)
        return torch.maximum(x1, x2)

    def minimum(x1, x2):
        if not torch.is_tensor(x1):
            x1 = tensor(x1)
        if not torch.is_tensor(x2):
            x2 = tensor(x2)
        return torch.minimum(x1, x2)

    def sort(x, axis=-1):
        xsorted = torch.sort(x, dim=axis)
        return xsorted.values

    def inftobigf(a, bigf=fmax / 1000.0):
        a = torch.where(torch.isinf(a), torch.full_like(a, bigf), a)
        return a

    def svd(A, full_matrices=True, hermitian=True):
        return torch.linalg.svd(A, full_matrices)

    def solve(A, B, overwrite_a=True, overwrite_b=True, assume_a="gen", sym_pos=False):
        return torch.linalg.solve(A, B)

    def grad(f):
        def f_grad(x):
            if not torch.is_tensor(x):
                x = torch.tensor(x, requires_grad=True)
            else:
                x = x.detach().clone().requires_grad_(True)

            y = f(x)
            gradients = torch.autograd.grad(y, x, allow_unused=True)[0]
            return gradients

        return f_grad

    class jax:
        @staticmethod
        def jit(f, *args, **kwargs):
            return f

    def custom_sqrt(x):
        arbitrary_value = 1.0
        mask = x == 0.0
        x_copy = torch.where(mask, arbitrary_value, x)

        res = torch.where(mask, 0.0, sqrt(x_copy))

        return res

    def cdist(x, y, zero_diagonal=True):
        if x is y:
            # use view method: requires contiguous tensor
            x_norm = (x**2).sum(1).view(-1, 1)
            distances = x_norm + x_norm.t() - 2.0 * torch.mm(x, x.t())
        else:
            x_norm = (x**2).sum(1).view(-1, 1)
            y_norm = (y**2).sum(1).view(1, -1)
            distances = x_norm + y_norm - 2.0 * torch.mm(x, y.t())

        distances = custom_sqrt(distances.clamp(min=0.0))

        if zero_diagonal and x is y:
            mask = torch.eye(distances.size(0), dtype=torch.bool, device=x.device)
            distances = distances.masked_fill(mask, 0.0)

        return distances

    def scaled_distance(loginvrho, x, y):
        invrho = exp(loginvrho)
        xs = invrho * x

        if x is y:
            d = cdist(xs, xs)
        else:
            ys = invrho * y
            d = cdist(xs, ys)

        return d

    def cholesky(A, lower=False, overwrite_a=False, check_finite=True):
        # NOTE: in cholesky(), overwrite_a and check_finite
        # are kept for consistency with Scipy but silently ignored.
        return torch.linalg.cholesky(A, upper=not (lower))

    def cho_factor(A, lower=False, overwrite_a=False, check_finite=True):
        # torch.linalg does not have cho_factor(), use cholesky() instead.
        return torch.linalg.cholesky(A, upper=not (lower))

    def cholesky_solve(A, b):
        C = torch.linalg.cholesky(A)
        return torch.cholesky_solve(b.reshape(-1, 1), C, upper=False), C

    def cholesky_inv(A):
        C = torch.linalg.cholesky(A)
        return torch.cholesky_inverse(C)

    class normal:
        @staticmethod
        def cdf(x, loc=0.0, scale=1.0):
            d = Normal(loc, scale)
            return d.cdf(x)

        @staticmethod
        def pdf(x, loc=0.0, scale=1.0):
            t = (x - loc) / scale
            return 1 / sqrt(2 * pi) * exp(-0.5 * t**2)

    class multivariate_normal:
        @staticmethod
        def rvs(mean=0.0, cov=1.0, n=1):
            # Check if cov is a scalar or 1x1 array, and use Normal if so
            if (
                torch.is_tensor(cov)
                and cov.ndim == 0
                or (cov.ndim == 2 and cov.shape[0] == 1 and cov.shape[1] == 1)
            ):
                distribution = Normal(torch.tensor(mean), cov.sqrt())
                return distribution.sample((n,))

            # For dxd covariance matrix, use MultivariateNormal
            d = cov.shape[0]  # Dimensionality from the covariance matrix
            mean_tensor = torch.full((d,), mean)  # Expand mean to a tensor
            distribution = MultivariateNormal(mean_tensor, covariance_matrix=cov)
            return distribution.sample((n,))

        @staticmethod
        def logpdf(x, mean=0.0, cov=1.0):
            # Check if cov is a scalar or 1x1 array, and use Normal if so
            if (
                torch.is_tensor(cov)
                and cov.ndim == 0
                or (cov.ndim == 2 and cov.shape[0] == 1 and cov.shape[1] == 1)
            ):
                distribution = Normal(torch.tensor(mean), cov.sqrt())
                return distribution.log_prob(x.squeeze(-1))

            # For dxd covariance matrix, use MultivariateNormal
            d = x.shape[-1]  # Infer dimensionality from x
            mean_tensor = torch.full((d,), mean)  # Expand mean to a tensor
            distribution = MultivariateNormal(mean_tensor, covariance_matrix=cov)
            return distribution.log_prob(x)

        @staticmethod
        def cdf(x, mean=0.0, cov=1.0):
            # Convert inputs to NumPy arrays if they are PyTorch tensors
            if torch.is_tensor(x):
                x = x.numpy()
            if torch.is_tensor(mean):
                mean = mean.numpy()
            if torch.is_tensor(cov):
                cov = cov.numpy()

            # Check if cov is a scalar or 1x1 array, and use norm for the univariate case
            if (
                isscalar(cov)
                or (isinstance(cov, numpy.ndarray) and cov.size == 1)
                or (torch.is_tensor(cov) and cov.size == 1)
            ):
                return Normal.cdf(x, mean, numpy.sqrt(cov))

            # For dxd covariance matrix, use multivariate_normal
            if isscalar(mean):
                d = cov.shape[0]  # Dimensionality from the covariance matrix
                mean = full(d, mean)  # Expand mean to an array
            return scipy_mvnormal.cdf(x, mean, cov)
