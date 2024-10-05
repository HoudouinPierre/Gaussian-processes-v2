mean_initialization_method    = "Average" # Average or Fixed
kernel_initialization_method  = "GPMP"    # GPMP, Moment-based, Custom
kernel_optimization_criterion = "MLE"     # REML, MLE, REML - PVA, MLE - PVA, PMLE, PMLE - PVA, MSE - PVA
default_mean_parameter        = 0
default_kernel_parameter      = {"sigma2" : 1, "rho" : [1                     for _ in range(1)]}
kernel_optimization_bounds    = {"sigma2" : [1e-4, 1e4], "rho" : [[1e-4, 1e4] for _ in range(1)]} if 0 else None
nugget                        = 1e-8
matern_parameter              = 5


gaussian_process_model_parameters = {"Mean initialization method"    : mean_initialization_method   ,
                                     "Kernel initialization method"  : kernel_initialization_method ,
                                     "Kernel optimization criterion" : kernel_optimization_criterion,
                                     "Default mean parameter"        : default_mean_parameter       ,
                                     "Default kernel parameter"      : default_kernel_parameter     ,
                                     "Kernel optimization bounds"    : kernel_optimization_bounds   ,
                                     "Nugget"                        : nugget                       ,
                                     "Matern parameter"              : matern_parameter}
