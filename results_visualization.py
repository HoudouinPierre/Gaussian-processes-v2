import os
import numpy as np
from scipy.stats import norm, chi2, gamma
import matplotlib.pyplot as pl
import plotly.graph_objects as go
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import plotly.express as px
from plotly.subplots import make_subplots
from dataset_generation import f


FIGSIZE = (20, 13)


class ResultsVisualization:

    def __init__(self, function_name, function_parameters,  gaussian_process, evaluation, optimization_method, alpha, library, save_path):
        self._function_name       = function_name
        self._function_parameters = function_parameters
        self._gaussian_process    = gaussian_process
        self._evaluation          = evaluation
        self._optimization_method = optimization_method
        self._alpha               = alpha
        self._library             = library
        self._save_path           = save_path

    @staticmethod
    def _index_of_closest_element(x, t):
        best_index, best_distance = 0, np.inf
        for index, y in enumerate(t):
            if np.abs(y - x) < best_distance:
                best_index, best_distance = index, np.abs(y - x)
        return best_index

    def create_new_folder_for_usecase(self):
        if not os.path.exists(self._save_path + self._function_name):
            os.mkdir(self._save_path + self._function_name)

    def plot_1D_function_prediction(self):
        X_observed     =         self._evaluation["Function prediction"]["X observed"        ]
        Y_observed     =         self._evaluation["Function prediction"]["Y observed"        ]
        LOO_mean       =         self._evaluation["Function prediction"]["LOO mean"          ]
        LOO_variance   =         self._evaluation["Function prediction"]["LOO variance"      ]
        X_test         =         self._evaluation["Function prediction"]["X test"            ]
        Y_test         =         self._evaluation["Function prediction"]["Y test"            ]
        X_function     =         self._evaluation["Function prediction"]["X function"        ]
        Y_function     =         self._evaluation["Function prediction"]["Y function"        ]
        posterior_mean =         self._evaluation["Function prediction"]["Posterior mean"    ]
        posterior_std  = np.sqrt(self._evaluation["Function prediction"]["Posterior variance"])
        q_alpha        =         self._evaluation["Train"]["Gaussianity"]["q alphas"][int(len(self._evaluation["Train"]["Gaussianity"]["q alphas"]) * self._alpha)]
        if self._library == "matplotlib":
            pl.figure             (figsize=FIGSIZE)
            pl.clf                ()
            pl.scatter            (X_observed.ravel(), Y_observed    , label="Samples" , color="red"            , marker="x", s=200)
            pl.scatter            (X_test    .ravel(), Y_test        , label="Test"    , color="blue"           , marker="o", s=200)
            LOO_plot = pl.scatter (X_observed.ravel(), LOO_mean      , label="LOO Mean", c=np.sqrt(LOO_variance), marker="s", s=200, cmap='viridis', alpha=0.25)
            cbar     = pl.colorbar(LOO_plot)
            cbar.set_label        ("LOO std", fontsize=20)
            pl.plot               (X_function.ravel(), Y_function    , label="f"      , color="green", linewidth=1, linestyle="--")
            pl.plot               (X_function.ravel(), posterior_mean, label="GP"     , color="blue" , linewidth=1)
            pl.fill_between       (X_function.ravel(), posterior_mean - q_alpha * posterior_std, posterior_mean + q_alpha * posterior_std, alpha=0.4, color="lightskyblue", label=str(int(np.round(100*self._alpha))) + "% CI")
            pl.xticks             (fontsize=16)
            pl.yticks             (fontsize=16)
            pl.grid               ()
            pl.legend             (fontsize=24)
            pl.xlabel             ('x', fontsize=20)
            pl.ylabel             ('f(x)', fontsize=20)
            pl.savefig            (self._save_path + self._function_name + "/1.1 - Function prediction.png")
            pl.show               ()
        if self._library == "plotly":
            figure = go.Figure  ()
            figure.add_trace    (go.Scatter(x=X_observed.ravel(), y=Y_observed    , mode='markers', name='Samples' , marker_color="red" , marker_size=10, marker_symbol="x"     ))
            figure.add_trace    (go.Scatter(x=X_test    .ravel(), y=Y_test        , mode='markers', name='Test'    , marker_color="blue", marker_size=10, marker_symbol="circle"))
            figure.add_trace    (go.Scatter(x=X_observed.ravel(), y=LOO_mean      , mode='markers', name='LOO Mean', marker=dict(size=10, color=LOO_variance, colorscale='Viridis', showscale=True, colorbar=dict(title='LOO Variance')), marker_symbol="square"))
            figure.add_trace    (go.Scatter(x=X_function.ravel(), y=Y_function    , mode="lines"  , name="f"       , line=dict(width=1, color="green", dash="dash")))
            figure.add_trace    (go.Scatter(x=X_function.ravel(), y=posterior_mean, mode='lines'  , name='GP'      , line=dict(width=1, color="blue" )))
            figure.add_trace    (go.Scatter(x=np.concatenate([X_function.ravel(), X_function.ravel()[::-1]]), y=np.concatenate([posterior_mean - q_alpha * posterior_std, (posterior_mean + q_alpha * posterior_std)[::-1]]), fill='toself', fillcolor='rgba(0,0,0,0.2)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name=str(int(np.round(100*self._alpha))) + "% CI"))
            figure.update_xaxes (tickprefix="<b>", ticksuffix="<b><br>")
            figure.update_yaxes (tickprefix="<b>", ticksuffix="<b><br>")
            figure.update_layout(legend=dict(x=0.8, y=0.95, xanchor='left', yanchor='top'), xaxis_title="x", yaxis_title="f(x)", width=1100, height=700)
            figure.show         ()

    def plot_2D_function_prediction_3D(self):
        X_observed              = self._evaluation["Function prediction"]["X observed"        ]
        Y_observed              = self._evaluation["Function prediction"]["Y observed"        ]
        X_test                  = self._evaluation["Function prediction"]["X test"            ]
        Y_test                  = self._evaluation["Function prediction"]["Y test"            ]
        X_function              = self._evaluation["Function prediction"]["X function"        ]
        X1_function             = np.unique  (X_function[:, 0])
        X2_function             = np.unique  (X_function[:, 1])
        X_mesh, Y_mesh          = np.meshgrid(X1_function, X2_function)
        Z_mesh                  = f(X_mesh, Y_mesh, self._function_name, self._function_parameters)
        Z_mesh_mean, Z_mesh_std = self._gaussian_process.predict_for_plot(X_mesh, Y_mesh)
        X1_observed             = X_observed[:, 0]
        X2_observed             = X_observed[:, 1]
        X1_test                 = X_test    [:, 0]
        X2_test                 = X_test    [:, 1]
        if self._library == "matplotlib":
            figure = pl.figure         (figsize=FIGSIZE)
            axes   = figure.add_subplot(221, projection='3d')
            axes.plot_surface          (X_mesh, Y_mesh, Z_mesh, cmap='viridis')
            axes.scatter               (X1_observed, X2_observed, Y_observed, color="red", marker="*", s=200, label='Samples')
            axes.scatter               (X1_test, X2_test, Y_test, color="blue", marker="o", s=200, label='Test')
            axes.set_xlabel            ('x1', fontsize=20)
            axes.set_ylabel            ('x2', fontsize=20)
            axes.set_zlabel            ('f(x1,x2)', fontsize=20)
            axes.set_title             ("Function", fontsize=24)
            axes   = figure.add_subplot(222, projection='3d')
            axes.plot_surface          (X_mesh, Y_mesh, Z_mesh_mean, cmap='viridis')
            axes.scatter               (X1_observed, X2_observed, Y_observed, color="red", marker="*", s=200, label='Samples')
            axes.scatter               (X1_test, X2_test, Y_test, color="blue", marker="o", s=200, label='Test')
            axes.set_xlabel            ('x1', fontsize=20)
            axes.set_ylabel            ('x2', fontsize=20)
            axes.set_zlabel            ('f(x1,x2)', fontsize=20)
            axes.set_title             ("Mean", fontsize=24)
            axes   = figure.add_subplot(224, projection='3d')
            axes.plot_surface          (X_mesh, Y_mesh, Z_mesh_std, cmap='viridis')
            axes.set_xlabel            ('x1', fontsize=20)
            axes.set_ylabel            ('x2', fontsize=20)
            axes.set_zlabel            ('f(x1,x2)', fontsize=20)
            axes.set_title             ("Std", fontsize=24)
            pl.savefig                 (self._save_path + self._function_name + "/1.1 - Function prediction.png")
            pl.show                    ()
        if self._library == "plotly":
            figure = make_subplots(rows=2, cols=2, specs=[[{'type': 'surface'}, {'type': 'surface'}], [None, {'type': 'surface'}]], subplot_titles=("Function", "Mean", "Std"))
            figure.add_trace      (go.Surface  (z=Z_mesh     , x=X_mesh     , y=Y_mesh    , colorscale='Viridis', showscale=False), row=1, col=1)
            figure.add_trace      (go.Scatter3d(x=X1_observed, y=X2_observed, z=Y_observed, mode='markers', marker=dict(color='red' , size=10, symbol='x'), name='Samples'), row=1, col=1)
            figure.add_trace      (go.Scatter3d(x=X1_test    , y=X2_test    , z=Y_test    , mode='markers', marker=dict(color='blue', size=10, symbol='circle'  ), name='Test'   ), row=1, col=1)
            figure.add_trace      (go.Surface  (z=Z_mesh_mean, x=X_mesh     , y=Y_mesh    , colorscale='Viridis', showscale=False), row=1, col=2)
            figure.add_trace      (go.Scatter3d(x=X1_observed, y=X2_observed, z=Y_observed, mode='markers', marker=dict(color='red' , size=10, symbol='x'), name='Samples'), row=1, col=2)
            figure.add_trace      (go.Scatter3d(x=X1_test    , y=X2_test    , z=Y_test    , mode='markers', marker=dict(color='blue', size=10, symbol='circle'  ), name='Test'   ), row=1, col=2)
            figure.add_trace      (go.Surface  (z=Z_mesh_std , x=X_mesh     , y=Y_mesh    , colorscale='Viridis', showscale=False), row=2, col=2)
            figure.update_layout  (scene =dict(xaxis_title='x1', yaxis_title='x2', zaxis_title='f(x1,x2)'), scene2=dict(xaxis_title='x1', yaxis_title='x2', zaxis_title='y'), scene3=dict(xaxis_title='x1', yaxis_title='x2', zaxis_title='y'), height=800, width=800)
            figure.show           ()

    def plot_2D_function_prediction_heatmap(self):
        X_observed              = self._evaluation["Function prediction"]["X observed"        ]
        X_test                  = self._evaluation["Function prediction"]["X test"            ]
        X_function              = self._evaluation["Function prediction"]["X function"        ]
        X1_function             = np.unique  (X_function[:, 0])
        X2_function             = np.unique  (X_function[:, 1])
        X_mesh, Y_mesh          = np.meshgrid(X1_function, X2_function)
        Z_mesh                  = f(X_mesh, Y_mesh, self._function_name, self._function_parameters)
        Z_mesh_mean, Z_mesh_std = self._gaussian_process.predict_for_plot(X_mesh, Y_mesh)
        X1_observed             = X_observed[:, 0]
        X2_observed             = X_observed[:, 1]
        X1_test                 = X_test    [:, 0]
        X2_test                 = X_test    [:, 1]
        if self._library == "matplotlib":
            figure = pl.figure         (figsize=FIGSIZE)
            axes   = figure.add_subplot(221)
            c1     = axes.imshow       (Z_mesh, extent=(min(X1_function), max(X1_function), min(X2_function), max(X2_function)), origin='lower', cmap='viridis', interpolation='nearest')
            axes.scatter               (X1_observed, X2_observed, color="red", marker="x", s=200, label='Samples')
            axes.scatter               (X1_test    , X2_test    , color="blue", marker="o", s=200, label='Test')
            axes.set_xlabel            ('x1', fontsize=20)
            axes.set_ylabel            ('x2', fontsize=20)
            axes.set_title             ("Function", fontsize=24)
            figure.colorbar            (c1, ax=axes)
            axes   = figure.add_subplot(222)
            c2     = axes.imshow       (Z_mesh_mean, extent=(min(X1_function), max(X1_function), min(X2_function), max(X2_function)), origin='lower', cmap='viridis', interpolation='nearest')
            axes.scatter               (X1_observed, X2_observed, color="red", marker="x", s=200, label='Samples')
            axes.scatter               (X1_test    , X2_test    , color="blue", marker="o", s=200, label='Test')
            axes.set_xlabel            ('x1', fontsize=20)
            axes.set_ylabel            ('x2', fontsize=20)
            axes.set_title             ("Mean", fontsize=24)
            figure.colorbar            (c2, ax=axes)
            axes   = figure.add_subplot(224)
            c3     = axes.imshow       (Z_mesh_std, extent=(min(X1_function), max(X1_function), min(X2_function), max(X2_function)), origin='lower', cmap='viridis', interpolation='nearest')
            axes.set_xlabel            ('x1', fontsize=20)
            axes.set_ylabel            ('x2', fontsize=20)
            axes.set_title             ("Std", fontsize=24)
            figure.colorbar            (c3, ax=axes)
            pl.savefig                 (self._save_path + self._function_name + "/1.1 - Function prediction.png")
            pl.show                    ()
        if self._library == "plotly":
            figure = make_subplots(rows=2, cols=2, subplot_titles=("Function", "Mean", "Std", "Std"), specs=[[{"type": "heatmap"}, {"type": "heatmap"}], [None, {"type": "heatmap"}]], column_widths=[0.5, 0.5], row_heights=[0.5, 0.5])
            figure.add_trace      (go.Heatmap(z=Z_mesh     , x=X1_function, y=X2_function , colorscale='Viridis'    , colorbar=dict(title='Function', len=0.3, x=0.25)), row=1, col=1)
            figure.add_trace      (go.Scatter(x=X1_observed, y=X2_observed, mode='markers', marker=dict(color='red' , size=10, symbol='x')  , name='Samples')       , row=1, col=1)
            figure.add_trace      (go.Scatter(x=X1_test    , y=X2_test    , mode='markers', marker=dict(color='blue', size=10, symbol='circle'), name='Test')          , row=1, col=1)
            figure.add_trace      (go.Heatmap(z=Z_mesh_mean, x=X1_function, y=X2_function , colorscale='Viridis'    , showscale=False)                                 , row=1, col=2)
            figure.add_trace      (go.Scatter(x=X1_observed, y=X2_observed, mode='markers', marker=dict(color='red', size=10, symbol='x'), name='Samples')          , row=1, col=2)
            figure.add_trace      (go.Scatter(x=X1_test    , y=X2_test    , mode='markers', marker=dict(color='blue', size=10, symbol='circle'), name='Test')          , row=1, col=2)
            figure.add_trace      (go.Heatmap(z=Z_mesh_std , x=X1_function, y=X2_function , colorscale='Viridis', colorbar=dict(title='Std', len=0.3, x=1.03)), row=2, col=2)
            figure.update_layout  (scene=dict(xaxis_title='x1', yaxis_title='x2'), scene2=dict(xaxis_title='x1', yaxis_title='x2', zaxis_title='f(x1,x2)'), scene3=dict(xaxis_title='x1', yaxis_title='x2'), height=800, width=800)
            figure.update_xaxes   (title_text="x1", row=1, col=1)
            figure.update_yaxes   (title_text="x2", row=1, col=1)
            figure.update_xaxes   (title_text="x1", row=1, col=2)
            figure.update_yaxes   (title_text="x2", row=1, col=2)
            figure.update_xaxes   (title_text="x1", row=2, col=2)
            figure.update_yaxes   (title_text="x2", row=2, col=2)
            figure.show           ()

    def plot_predictions(self, dataset):
        true_value      = self._evaluation[dataset]["Predictions"        ]["True value"     ]
        predicted_value = self._evaluation[dataset]["Predictions"        ]["Predicted value"]
        std             = self._evaluation[dataset]["Predictions"        ]["Std"            ]
        MSE             = self._evaluation[dataset]["Performance metrics"]["MSE"            ]
        if self._library == "matplotlib":
            pl.figure             (figsize=FIGSIZE)
            pl.clf                ()
            pl.errorbar           (true_value, predicted_value, yerr=std, fmt='o', ecolor='gray', elinewidth=1, capsize=3, color='white', alpha=0.3)
            scatter = pl.scatter  (true_value, predicted_value, c=std, cmap="viridis", label='Predictions')
            pl.plot               ([min(true_value), max(true_value)], [min(true_value), max(true_value)], color='green', linestyle='--', label="Reference")
            pl.xticks             (fontsize=16)
            pl.yticks             (fontsize=16)
            pl.grid               ()
            pl.legend             (fontsize=24)
            pl.xlabel             ('Truth', fontsize=20)
            pl.ylabel             ('Prediction', fontsize=20)
            colorbar = pl.colorbar(scatter)
            colorbar.set_label    ("Std", fontsize=24)
            pl.title              (dataset + " set - MSE = " + str(np.round(MSE, 2)), fontsize=24)
            pl.savefig            (self._save_path + self._function_name + "/1.2 - Predictions " + dataset + ".png")
            pl.show               ()
        if self._library == "plotly":
            figure = go.Figure  ()
            figure.add_trace    (go.Scatter(x=true_value, y=predicted_value, mode='markers', name='Predicted values', marker=dict(color=std, colorscale='Viridis', size=10, colorbar=dict(title="Std")), error_y=dict(type='data', array=std, visible=True, color='gray')))
            figure.add_trace    (go.Scatter(x=[min(true_value), max(true_value)], y=[min(true_value), max(true_value)], mode='lines', name='Reference', line=dict(color='green', dash='dash')))
            figure.update_xaxes (tickprefix="<b>", ticksuffix="<b><br>")
            figure.update_yaxes (tickprefix="<b>", ticksuffix="<b><br>")
            figure.update_layout(legend=dict(x=0.8, y=0.95, xanchor='left', yanchor='top'), xaxis_title="Truth", yaxis_title="Prediction", title=dataset + " set - MSE = " + str(np.round(MSE, 2)), width=1100, height=700)
            figure.show         ()

    def plot_hyperparameters_history(self):
        hyperparameters_history = np.array(self._evaluation["Optimization infos"]["Hyperparameters history"]).T
        titles                  = ["log(sigma2)"] + ["-log(rho{})".format(plot_index) for plot_index in range(len(hyperparameters_history) - 1)] if self._gaussian_process.optimization_method() in ["REML", "MLE", "REML - PVA", "MLE - PVA"] else ["-log(rho{})".format(plot_index) for plot_index in range(len(hyperparameters_history))]
        if self._library == "matplotlib":
            fig, axs = pl.subplots(int(np.ceil(len(hyperparameters_history) / 2)), 2, figsize=FIGSIZE)
            if int(np.ceil(len(hyperparameters_history) / 2)) == 1:
                axs = [axs]
            for plot_index in range(len(hyperparameters_history)):
                axs[int(np.floor(plot_index / 2))][plot_index % 2].plot([i + 1 for i in range(len(hyperparameters_history[0]))], hyperparameters_history[plot_index])
                axs[int(np.floor(plot_index / 2))][plot_index % 2].set_title(titles[plot_index])
                axs[int(np.floor(plot_index / 2))][plot_index % 2].set_xlabel("Iterations")
                axs[int(np.floor(plot_index / 2))][plot_index % 2].set_ylabel("HP value")
                axs[int(np.floor(plot_index / 2))][plot_index % 2].grid(True)
            pl.tight_layout()
            pl.savefig     (self._save_path + self._function_name + "/2.1 - Hyperparameters history.png")
            pl.show        ()
        if self._library == "plotly":
            figure = make_subplots(rows=int(np.ceil(len(hyperparameters_history) / 2)), cols=2, subplot_titles=titles)
            for plot_index in range(len(hyperparameters_history)):
                figure.add_trace   (go.Scatter(x=[i + 1 for i in range(len(hyperparameters_history[0]))], y=hyperparameters_history[plot_index], mode='lines', name=titles[plot_index]), row=1 + int(np.floor(plot_index) / 2), col=1 + plot_index % 2)
                figure.update_yaxes(title_text="HP value", row=1 + int(np.floor(plot_index / 2)), col=1 + plot_index % 2, showgrid=True)
            figure.update_xaxes    (title_text="Iterations", showgrid=True)
            figure.update_layout   (height=800, width=1100)
            figure.show            ()

    def plot_criterion_history(self):
        criterion_history = self._evaluation["Optimization infos"]["Criterion history"]
        criterion         = "Log " + self._optimization_method.split(" - ")[0]
        if self._library == "matplotlib":
            pl.figure(figsize=FIGSIZE)
            pl.clf    ()
            pl.plot   ([i + 1 for i in range(len(criterion_history))], criterion_history, label="Criterion history", color="black", linewidth=1)
            pl.xticks (fontsize=16)
            pl.yticks (fontsize=16)
            pl.grid   ()
            pl.legend (fontsize=24)
            pl.xlabel ("Iterations", fontsize=20)
            pl.ylabel (criterion, fontsize=20)
            pl.savefig(self._save_path + self._function_name + "/2.2 - Criterion history.png")
            pl.show   ()
        if self._library == "plotly":
            figure = go.Figure  ()
            figure.add_trace    (go.Scatter(x=[i + 1 for i in range(len(criterion_history))], y=criterion_history, mode='lines', name="Criterion history", line=dict(width=1, color="black" )))
            figure.update_xaxes (tickprefix="<b>", ticksuffix="<b><br>")
            figure.update_yaxes (tickprefix="<b>", ticksuffix="<b><br>")
            figure.update_layout(xaxis_title="Iterations", yaxis_title=criterion, width=1100, height=700)
            figure.show         ()

    def plot_negative_log_likelihood_evolution_2D_3D(self, log_scale):
        log_sigma2s              = self._evaluation["Optimization functions evolution"]["Negative log likelihood evolution"]["Log sigma2s"             ]
        log_inv_rhos             = self._evaluation["Optimization functions evolution"]["Negative log likelihood evolution"]["Log inv rhos"            ]
        negative_log_likelihoods = self._evaluation["Optimization functions evolution"]["Negative log likelihood evolution"]["Negative log likelihoods"]
        hyperparameters_0        = self._evaluation["Optimization infos"              ]["Hyperparameters history"          ][0 ]
        hyperparameters_f        = self._evaluation["Optimization infos"              ]["Hyperparameters history"          ][-1]
        if self._library == "matplotlib":
            figure = pl.figure         (figsize=FIGSIZE)
            axes   = figure.add_subplot(111, projection='3d')
            axes.set_xlabel            ("log(sigma2)" , fontsize=20)
            axes.set_ylabel            ("-log(rho)", fontsize=20)
            if log_scale:
                axes.plot_surface      (log_sigma2s, log_inv_rhos, np.log(negative_log_likelihoods), cmap='viridis')
                axes.set_zlabel        ("Log negative log likelihood", fontsize=20)
                if self._optimization_method in ["MLE", "MLE - PVA"]:
                    axes.scatter       ([hyperparameters_0[0]], [hyperparameters_0[1]], [0], color="blue", marker="d", s=200, label='Start' )
                    axes.scatter       ([hyperparameters_f[0]], [hyperparameters_f[1]], [0], color="red" , marker="d", s=200, label='Finish')
                pl.savefig             (self._save_path + self._function_name + "/2.3.1 - Log negative log likelihood.png")
            else:
                axes.plot_surface      (log_sigma2s, log_inv_rhos, negative_log_likelihoods, cmap='viridis')
                axes.set_zlabel        ("Negative log likelihood", fontsize=20)
                if self._optimization_method in ["MLE", "MLE - PVA"]:
                    axes.scatter       ([hyperparameters_0[0]], [hyperparameters_0[1]], [0], color="blue", marker="d", s=200, label='Start' )
                    axes.scatter       ([hyperparameters_f[0]], [hyperparameters_f[1]], [0], color="red" , marker="d", s=200, label='Finish')
                pl.savefig             (self._save_path + self._function_name + "/2.3.1 - Negative log likelihood.png")
            pl.show                    ()
        if self._library == "plotly":
            if log_scale:
                figure = go.Figure  (data=[go.Surface(z=np.log(negative_log_likelihoods), x=log_sigma2s, y=log_inv_rhos, colorscale='Viridis')])
                figure.update_layout(scene=dict(xaxis_title="log(sigma2)", yaxis_title="-log(rho)", zaxis_title="Log negative log likelihood"), width=1100, height=700)
            else:
                figure = go.Figure  (data=[go.Surface(z=negative_log_likelihoods, x=log_sigma2s, y=log_inv_rhos, colorscale='Viridis')])
                figure.update_layout(scene=dict(xaxis_title="log(sigma2)", yaxis_title="-log(rho)", zaxis_title="Negative log likelihood"), width=1100, height=700)
            if self._optimization_method in ["MLE", "MLE - PVA"]:
                figure.add_trace    (go.Scatter3d(x=[hyperparameters_0[0]], y=[hyperparameters_0[1]], z=[0], mode='markers', marker=dict(size=10, color="blue", symbol="diamond"), name='Start' ))
                figure.add_trace    (go.Scatter3d(x=[hyperparameters_f[0]], y=[hyperparameters_f[1]], z=[0], mode='markers', marker=dict(size=10, color="red" , symbol="diamond"), name='Finish'))
            figure.show             ()

    def plot_negative_log_likelihood_evolution_2D_heatmap(self, log_scale):
        log_sigma2s              = self._evaluation["Optimization functions evolution"]["Negative log likelihood evolution"]["Log sigma2s"             ]
        log_inv_rhos             = self._evaluation["Optimization functions evolution"]["Negative log likelihood evolution"]["Log inv rhos"            ]
        negative_log_likelihoods = self._evaluation["Optimization functions evolution"]["Negative log likelihood evolution"]["Negative log likelihoods"]
        hyperparameters_0        = self._evaluation["Optimization infos"              ]["Hyperparameters history"          ][0 ]
        hyperparameters_f        = self._evaluation["Optimization infos"              ]["Hyperparameters history"          ][-1]
        if self._library == "matplotlib":
            pl.figure     (figsize=FIGSIZE)
            pl.clf        ()
            if log_scale:
                pl.imshow (np.log(np.maximum(1e-10, negative_log_likelihoods)), extent=(np.min(log_sigma2s), np.max(log_sigma2s), np.min(log_inv_rhos), np.max(log_inv_rhos)), origin='lower', cmap='viridis', interpolation='nearest')
            else:
                pl.imshow (negative_log_likelihoods, extent=(np.min(log_sigma2s), np.max(log_sigma2s), np.min(log_inv_rhos), np.max(log_inv_rhos)), origin='lower', cmap='viridis', interpolation='nearest')
            if self._optimization_method in ["MLE", "MLE - PVA"]:
                pl.scatter([hyperparameters_0[0]], [hyperparameters_0[1]], color="blue", marker="d", s=200, label='Start' )
                pl.scatter([hyperparameters_f[0]], [hyperparameters_f[1]], color="red" , marker="d", s=200, label='Finish')
            pl.colorbar   ()
            pl.xticks     (fontsize=16)
            pl.yticks     (fontsize=16)
            pl.xlabel     ('log(sigma2)')
            pl.ylabel     ('-log(rho)')
            if log_scale:
                pl.savefig(self._save_path + self._function_name + "/2.3.1 - Log negative log likelihood.png")
            else:
                pl.savefig(self._save_path + self._function_name + "/2.3.1 - Negative log likelihood.png")
            pl.show       ()
        if self._library == "plotly":
            if log_scale:
                figure = px.imshow  (np.log(np.maximum(1e-10, negative_log_likelihoods)), x=log_sigma2s[0], y=log_inv_rhos.T[0], labels={'x': "log(sigma2)", 'y': "-log(rho)", 'color': 'Log negative log likelihood'})
                figure.update_layout(scene=dict(xaxis_title='log(sigma2)', yaxis_title='-log(rho)', zaxis_title='Log negative log likelihood'), width=800, height=800)
            else:
                figure = px.imshow  (negative_log_likelihoods, x=log_sigma2s[0], y=log_inv_rhos.T[0], labels={'x': "log(sigma2)", 'y': "-log(rho)", 'color': 'Negative log likelihood'})
                figure.update_layout(scene=dict(xaxis_title='log(sigma2)', yaxis_title='-log(rho)', zaxis_title='Negative log likelihood'), width=800, height=800)
            if self._optimization_method in ["MLE", "MLE - PVA"]:
                figure.add_trace    (go.Scatter(x=[hyperparameters_0[0]], y=[hyperparameters_0[1]], mode='markers', marker=dict(color='blue', symbol="diamond", size=10), name="Start" ))
                figure.add_trace    (go.Scatter(x=[hyperparameters_f[0]], y=[hyperparameters_f[1]], mode='markers', marker=dict(color='red' , symbol="diamond", size=10), name="Finish"))
            figure.update_yaxes     (autorange='reversed')
            figure.show             ()

    def plot_negative_log_likelihood_evolution_nD(self, log_scale):
        negative_log_likelihood_evolution = self._evaluation["Optimization functions evolution"]["Negative log likelihood evolution"]
        hyperparameters_0                 = self._evaluation["Optimization infos"              ]["Hyperparameters history"          ][0 ]
        hyperparameters_f                 = self._evaluation["Optimization infos"              ]["Hyperparameters history"          ][-1]
        titles                            = ["log(sigma2)"] + ["-log(rho{})".format(plot_index) for plot_index in range(len(negative_log_likelihood_evolution) - 1)]
        if self._library == "matplotlib":
            fig, axs = pl.subplots(int(np.ceil(len(negative_log_likelihood_evolution) / 2)), 2, figsize=FIGSIZE)
            for plot_index in range(len(negative_log_likelihood_evolution)):
                key = list(negative_log_likelihood_evolution.keys())[plot_index]
                if log_scale:
                    axs[int(np.floor(plot_index / 2))][plot_index % 2].plot(negative_log_likelihood_evolution[key]["Abscissas"], np.log(np.maximum(1e-10, negative_log_likelihood_evolution[key]["Negative log likelihoods"])))
                    if self._optimization_method in ["MLE", "MLE - PVA"]:
                        axs[int(np.floor(plot_index / 2))][plot_index % 2].scatter   ([hyperparameters_0[plot_index]], [np.log(negative_log_likelihood_evolution[key]["Negative log likelihoods"][self._index_of_closest_element(hyperparameters_0[plot_index], negative_log_likelihood_evolution[key]["Abscissas"])])], color="blue", marker="d", s=200, label='Start' )
                        axs[int(np.floor(plot_index / 2))][plot_index % 2].scatter   ([hyperparameters_f[plot_index]], [np.log(negative_log_likelihood_evolution[key]["Negative log likelihoods"][self._index_of_closest_element(hyperparameters_f[plot_index], negative_log_likelihood_evolution[key]["Abscissas"])])], color="red" , marker="d", s=200, label='Finish')
                    axs    [int(np.floor(plot_index / 2))][plot_index % 2].set_ylabel("Log negative log likelihood")
                else:
                    axs    [int(np.floor(plot_index / 2))][plot_index % 2].plot(negative_log_likelihood_evolution[key]["Abscissas"], negative_log_likelihood_evolution[key]["Negative log likelihoods"])
                    if self._optimization_method in ["MLE", "MLE - PVA"]:
                        axs[int(np.floor(plot_index / 2))][plot_index % 2].scatter   ([hyperparameters_0[plot_index]], [negative_log_likelihood_evolution[key]["Negative log likelihoods"][self._index_of_closest_element(hyperparameters_0[plot_index], negative_log_likelihood_evolution[key]["Abscissas"])]], color="blue", marker="d", s=200, label='Start' )
                        axs[int(np.floor(plot_index / 2))][plot_index % 2].scatter   ([hyperparameters_f[plot_index]], [negative_log_likelihood_evolution[key]["Negative log likelihoods"][self._index_of_closest_element(hyperparameters_f[plot_index], negative_log_likelihood_evolution[key]["Abscissas"])]], color="red" , marker="d", s=200, label='Finish')
                    axs    [int(np.floor(plot_index / 2))][plot_index % 2].set_ylabel("Negative log likelihood")
                axs        [int(np.floor(plot_index / 2))][plot_index % 2].set_title (titles[plot_index])
                axs        [int(np.floor(plot_index / 2))][plot_index % 2].set_xlabel("HP value")
                axs        [int(np.floor(plot_index / 2))][plot_index % 2].grid      (True)
            pl.tight_layout()
            if log_scale:
                pl.savefig(self._save_path + self._function_name + "/2.3.1 - Log negative log likelihood.png")
            else:
                pl.savefig(self._save_path + self._function_name + "/2.3.1 - Negative log likelihood.png")
            pl.show()
        if self._library == "plotly":
            figure = make_subplots(rows=int(np.ceil(len(negative_log_likelihood_evolution) / 2)), cols=2, subplot_titles=titles)
            for plot_index in range(len(negative_log_likelihood_evolution)):
                key = list(negative_log_likelihood_evolution.keys())[plot_index]
                if log_scale:
                    figure    .add_trace(go.Scatter(x=negative_log_likelihood_evolution[key]["Abscissas"], y=np.log(np.maximum(1e-10, negative_log_likelihood_evolution[key]["Negative log likelihoods"])), mode='lines', name=titles[plot_index]), row=1 + int(np.floor(plot_index) / 2), col=1 + plot_index % 2)
                    if self._optimization_method in ["MLE", "MLE - PVA"]:
                        figure.add_trace(go.Scatter(x=[hyperparameters_0[plot_index]], y=[np.log(negative_log_likelihood_evolution[key]["Negative log likelihoods"][self._index_of_closest_element(hyperparameters_0[plot_index], negative_log_likelihood_evolution[key]["Abscissas"])])], mode='markers', marker=dict(color='blue', symbol="diamond", size=10), name="Start" ), row=1 + int(np.floor(plot_index / 2)), col=1 + plot_index % 2)
                        figure.add_trace(go.Scatter(x=[hyperparameters_f[plot_index]], y=[np.log(negative_log_likelihood_evolution[key]["Negative log likelihoods"][self._index_of_closest_element(hyperparameters_f[plot_index], negative_log_likelihood_evolution[key]["Abscissas"])])], mode='markers', marker=dict(color='red' , symbol="diamond", size=10), name="Finish"), row=1 + int(np.floor(plot_index / 2)), col=1 + plot_index % 2)
                    figure.update_yaxes (title_text="Log negative log likelihood", row=1 + int(np.floor(plot_index / 2)), col=1 + plot_index % 2, showgrid=True)
                else:
                    figure    .add_trace(go.Scatter(x=negative_log_likelihood_evolution[key]["Abscissas"], y=negative_log_likelihood_evolution[key]["Negative log likelihoods"], mode='lines', name=titles[plot_index]), row=1 + int(np.floor(plot_index) / 2), col=1 + plot_index % 2)
                    if self._optimization_method in ["MLE", "MLE - PVA"]:
                        figure.add_trace(go.Scatter(x=[hyperparameters_0[plot_index]], y=[negative_log_likelihood_evolution[key]["Negative log likelihoods"][self._index_of_closest_element(hyperparameters_0[plot_index], negative_log_likelihood_evolution[key]["Abscissas"])]], mode='markers', marker=dict(color='blue', symbol="diamond", size=10), name="Start" ), row=1 + int(np.floor(plot_index / 2)), col=1 + plot_index % 2)
                        figure.add_trace(go.Scatter(x=[hyperparameters_f[plot_index]], y=[negative_log_likelihood_evolution[key]["Negative log likelihoods"][self._index_of_closest_element(hyperparameters_f[plot_index], negative_log_likelihood_evolution[key]["Abscissas"])]], mode='markers', marker=dict(color='red' , symbol="diamond", size=10), name="Finish"), row=1 + int(np.floor(plot_index / 2)), col=1 + plot_index % 2)
                    figure.update_yaxes (title_text="Negative log likelihood", row=1 + int(np.floor(plot_index / 2)), col=1 + plot_index % 2, showgrid=True)
            figure.update_xaxes (title_text="HP value", showgrid=True)
            figure.update_layout(height=800, width=1100)
            figure.show         ()

    def plot_profiled_negative_log_likelihood_evolution_1D(self, log_scale):
        log_inv_rhos                      = self._evaluation["Optimization functions evolution"]["Profiled negative log likelihood evolution"]["Log inv rhos"                     ]
        profiled_negative_log_likelihoods = self._evaluation["Optimization functions evolution"]["Profiled negative log likelihood evolution"]["Profiled negative log likelihoods"]
        hyperparameters_0                 = self._evaluation["Optimization infos"              ]["Hyperparameters history"                   ][0 ]
        hyperparameters_f                 = self._evaluation["Optimization infos"              ]["Hyperparameters history"                   ][-1]
        if self._library == "matplotlib":
            pl.figure         (figsize=FIGSIZE)
            pl.clf            ()
            if log_scale:
                pl.plot       (log_inv_rhos, np.log(np.maximum(1e-10, profiled_negative_log_likelihoods)), color="black", linewidth=1)
                if self._optimization_method == "PMLE":
                    pl.scatter([hyperparameters_0[0]], [np.log(profiled_negative_log_likelihoods[self._index_of_closest_element(hyperparameters_0[0], log_inv_rhos)])], color="blue", marker="d", s=200, label='Start' )
                    pl.scatter([hyperparameters_f[0]], [np.log(profiled_negative_log_likelihoods[self._index_of_closest_element(hyperparameters_f[0], log_inv_rhos)])], color="red" , marker="d", s=200, label='Finish')
                pl.ylabel     ("Log profiled negative log likelihood", fontsize=20)
            else:
                pl.plot       (log_inv_rhos, profiled_negative_log_likelihoods, color="black", linewidth=1)
                if self._optimization_method == "PMLE":
                    pl.scatter([hyperparameters_0[0]], [profiled_negative_log_likelihoods[self._index_of_closest_element(hyperparameters_0[0], log_inv_rhos)]], color="blue", marker="d", s=200, label='Start' )
                    pl.scatter([hyperparameters_f[0]], [profiled_negative_log_likelihoods[self._index_of_closest_element(hyperparameters_f[0], log_inv_rhos)]], color="red" , marker="d", s=200, label='Finish')
                pl.ylabel     ("Profiled negative log likelihood", fontsize=20)
            pl.xticks         (fontsize=16)
            pl.yticks         (fontsize=16)
            pl.grid           ()
            pl.xlabel         ("-log(rho)", fontsize=20)
            if log_scale:
                pl.savefig    (self._save_path + self._function_name + "/2.3.2 - Log profiled negative log likelihood.png")
            else:
                pl.savefig    (self._save_path + self._function_name + "/2.3.2 - Profiled negative log likelihood.png")
            pl.show           ()
        if self._library == "plotly":
            figure = go.Figure()
            if log_scale:
                figure    .add_trace(go.Scatter(x=log_inv_rhos, y=np.log(np.maximum(1e-10, profiled_negative_log_likelihoods)), mode='lines', line=dict(width=1, color="black" )))
                if self._optimization_method == "PMLE":
                    figure.add_trace(go.Scatter(x=[hyperparameters_0[0]], y=[np.log(profiled_negative_log_likelihoods[self._index_of_closest_element(hyperparameters_0[0], log_inv_rhos)])], mode='markers', marker=dict(color='blue', symbol="diamond", size=10), name="Start" ))
                    figure.add_trace(go.Scatter(x=[hyperparameters_f[0]], y=[np.log(profiled_negative_log_likelihoods[self._index_of_closest_element(hyperparameters_f[0], log_inv_rhos)])], mode='markers', marker=dict(color='red' , symbol="diamond", size=10), name="Finish"))
                figure.update_layout(xaxis_title="-log(rho)", yaxis_title="Log profiled negative log likelihood", width=1100, height=700)
            else:
                figure    .add_trace(go.Scatter(x=log_inv_rhos, y=profiled_negative_log_likelihoods, mode='lines', line=dict(width=1, color="black" )))
                if self._optimization_method == "PMLE":
                    figure.add_trace(go.Scatter(x=[hyperparameters_0[0]], y=[profiled_negative_log_likelihoods[self._index_of_closest_element(hyperparameters_0[0], log_inv_rhos)]], mode='markers', marker=dict(color='blue', symbol="diamond", size=10), name="Start"))
                    figure.add_trace(go.Scatter(x=[hyperparameters_f[0]], y=[profiled_negative_log_likelihoods[self._index_of_closest_element(hyperparameters_f[0], log_inv_rhos)]], mode='markers', marker=dict(color='red', symbol="diamond", size=10), name="Finish"))
                figure.update_layout(xaxis_title="-log(rho)", yaxis_title="Profiled negative log likelihood", width=1100, height=700)
            figure.update_xaxes(tickprefix="<b>", ticksuffix="<b><br>")
            figure.update_yaxes(tickprefix="<b>", ticksuffix="<b><br>")
            figure.show        ()

    def plot_profiled_negative_log_likelihood_evolution_2D_3D(self, log_scale):
        log_inv_rhos1                     = self._evaluation["Optimization functions evolution"]["Profiled negative log likelihood evolution"]["Log inv rhos 1"                   ]
        log_inv_rhos2                     = self._evaluation["Optimization functions evolution"]["Profiled negative log likelihood evolution"]["Log inv rhos 2"                   ]
        profiled_negative_log_likelihoods = self._evaluation["Optimization functions evolution"]["Profiled negative log likelihood evolution"]["Profiled negative log likelihoods"]
        hyperparameters_0                 = self._evaluation["Optimization infos"              ]["Hyperparameters history"                   ][0 ]
        hyperparameters_f                 = self._evaluation["Optimization infos"              ]["Hyperparameters history"                   ][-1]
        if self._library == "matplotlib":
            figure = pl.figure         (figsize=FIGSIZE)
            axes   = figure.add_subplot(111, projection='3d')
            axes.set_xlabel            ("-log(rho 1)", fontsize=20)
            axes.set_ylabel            ("-log(rho 2)", fontsize=20)
            if log_scale:
                axes.plot_surface      (log_inv_rhos1, log_inv_rhos2, np.log(np.maximum(1e-10, profiled_negative_log_likelihoods)), cmap='viridis')
                if self._optimization_method == "PMLE":
                    axes.scatter       ([hyperparameters_0[0]], [hyperparameters_0[1]], [0], color="blue", marker="d", s=200, label='Start' )
                    axes.scatter       ([hyperparameters_f[0]], [hyperparameters_f[1]], [0], color="red" , marker="d", s=200, label='Finish')
                axes.set_zlabel        ("Log profiled negative log likelihood", fontsize=20)
                pl.savefig             (self._save_path + self._function_name + "/2.3.2 - Log profiled negative log likelihood.png")
            else:
                axes.plot_surface      (log_inv_rhos1, log_inv_rhos2, profiled_negative_log_likelihoods, cmap='viridis')
                if self._optimization_method == "PMLE":
                    axes.scatter       ([hyperparameters_0[0]], [hyperparameters_0[1]], [0], color="blue", marker="d", s=200, label='Start' )
                    axes.scatter       ([hyperparameters_f[0]], [hyperparameters_f[1]], [0], color="red" , marker="d", s=200, label='Finish')
                axes.set_zlabel        ("Profiled negative log likelihood", fontsize=20)
                pl.savefig             (self._save_path + self._function_name + "/2.3.2 - Profiled negative log likelihood.png")
            pl.show                    ()
        if self._library == "plotly":
            if log_scale:
                figure = go.Figure  (data=[go.Surface(z=np.log(np.maximum(1e-10, profiled_negative_log_likelihoods)), x=log_inv_rhos1, y=log_inv_rhos2, colorscale='Viridis')])
                figure.update_layout(scene=dict(xaxis_title="-log(rho 1)", yaxis_title="-log(rho 2)", zaxis_title="Log profiled negative log likelihood"), width=1100, height=700)
            else:
                figure = go.Figure  (data=[go.Surface(z=profiled_negative_log_likelihoods, x=log_inv_rhos1, y=log_inv_rhos2, colorscale='Viridis')])
                figure.update_layout(scene=dict(xaxis_title="-log(rho 1)", yaxis_title="-log(rho 2)", zaxis_title="Profiled negative log likelihood"), width=1100, height=700)
            if self._optimization_method == "PMLE":
                figure.add_trace    (go.Scatter3d(x=[hyperparameters_0[0]], y=[hyperparameters_0[1]], z=[0], mode='markers', marker=dict(size=10, color="blue", symbol="diamond"), name='Start' ))
                figure.add_trace    (go.Scatter3d(x=[hyperparameters_f[0]], y=[hyperparameters_f[1]], z=[0], mode='markers', marker=dict(size=10, color="red" , symbol="diamond"), name='Finish'))
            figure.show             ()

    def plot_profiled_negative_log_likelihood_evolution_2D_heatmap(self, log_scale):
        log_inv_rhos1                     = self._evaluation["Optimization functions evolution"]["Profiled negative log likelihood evolution"]["Log inv rhos 1"                   ]
        log_inv_rhos2                     = self._evaluation["Optimization functions evolution"]["Profiled negative log likelihood evolution"]["Log inv rhos 2"                   ]
        profiled_negative_log_likelihoods = self._evaluation["Optimization functions evolution"]["Profiled negative log likelihood evolution"]["Profiled negative log likelihoods"]
        hyperparameters_0                 = self._evaluation["Optimization infos"              ]["Hyperparameters history"                   ][0 ]
        hyperparameters_f                 = self._evaluation["Optimization infos"              ]["Hyperparameters history"                   ][-1]
        if self._library == "matplotlib":
            pl.figure     (figsize=FIGSIZE)
            pl.clf        ()
            if log_scale:
                pl.imshow (np.log(np.maximum(1e-10, profiled_negative_log_likelihoods)), extent=(np.min(log_inv_rhos1), np.max(log_inv_rhos1), np.min(log_inv_rhos2), np.max(log_inv_rhos2)), origin='lower', cmap='viridis', interpolation='nearest')
            else:
                pl.imshow (profiled_negative_log_likelihoods, extent=(np.min(log_inv_rhos1), np.max(log_inv_rhos1), np.min(log_inv_rhos2), np.max(log_inv_rhos2)), origin='lower', cmap='viridis', interpolation='nearest')
            if self._optimization_method == "PMLE":
                pl.scatter([hyperparameters_0[0]], [hyperparameters_0[1]], color="blue", marker="d", s=200, label='Start' )
                pl.scatter([hyperparameters_f[0]], [hyperparameters_f[1]], color="red" , marker="d", s=200, label='Finish')
            pl.colorbar   ()
            pl.xticks     (fontsize=16)
            pl.yticks     (fontsize=16)
            pl.xlabel     ('-log(rho 1)')
            pl.ylabel     ('-log(rho 2)')
            pl.grid       ()
            if log_scale:
                pl.savefig(self._save_path + self._function_name + "/2.3.2 - Log profiled negative log likelihood.png")
            else:
                pl.savefig(self._save_path + self._function_name + "/2.3.2 - Profiled negative log likelihood.png")
            pl.show       ()
        if self._library == "plotly":
            if log_scale:
                figure = px.imshow  (np.log(np.maximum(1e-10, profiled_negative_log_likelihoods)), x=log_inv_rhos1[0], y=log_inv_rhos2.T[0], labels={'x': "-log(rho 1)", 'y': "-log(rho 2)", 'color': 'Log profiled negative log likelihood'})
                figure.update_layout(scene=dict(xaxis_title='-log(rho 1)', yaxis_title='-log(rho 2)', zaxis_title="Log profiled negative log likelihood"), width=800, height=800)
            else:
                figure = px.imshow  (profiled_negative_log_likelihoods, x=log_inv_rhos1[0], y=log_inv_rhos2.T[0], labels={'x': "-log(rho 1)", 'y': "-log(rho 2)", 'color': 'Profiled negative log likelihood'})
                figure.update_layout(scene=dict(xaxis_title='-log(rho 1)', yaxis_title='-log(rho 2)', zaxis_title="Profiled negative log likelihood"), width=800, height=800)
            if self._optimization_method == "PMLE":
                figure.add_trace    (go.Scatter(x=[hyperparameters_0[0]], y=[hyperparameters_0[1]], mode='markers', marker=dict(color='blue', symbol="diamond", size=10), name="Start" ))
                figure.add_trace    (go.Scatter(x=[hyperparameters_f[0]], y=[hyperparameters_f[1]], mode='markers', marker=dict(color='red' , symbol="diamond", size=10), name="Finish"))
            figure.update_yaxes     (autorange='reversed')
            figure.show             ()

    def plot_profiled_negative_log_likelihood_evolution_nD(self, log_scale):
        profiled_negative_log_likelihood_evolution = self._evaluation["Optimization functions evolution"]["Profiled negative log likelihood evolution"]
        hyperparameters_0                          = self._evaluation["Optimization infos"              ]["Hyperparameters history"                   ][0 ]
        hyperparameters_f                          = self._evaluation["Optimization infos"              ]["Hyperparameters history"                   ][-1]
        titles                                     = ["-log(rho{})".format(plot_index) for plot_index in range(len(profiled_negative_log_likelihood_evolution))]
        if self._library == "matplotlib":
            fig, axs = pl.subplots(int(np.ceil(len(profiled_negative_log_likelihood_evolution) / 2)), 2, figsize=FIGSIZE)
            for plot_index in range(len(profiled_negative_log_likelihood_evolution)):
                if log_scale:
                    axs    [int(np.floor(plot_index / 2))][plot_index % 2].plot(profiled_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Log inv rhos"], np.log(np.maximum(1e-10, profiled_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Profiled negative log likelihoods"])))
                    if self._optimization_method == "PMLE":
                        axs[int(np.floor(plot_index / 2))][plot_index % 2].scatter([hyperparameters_0[plot_index]], [np.log(profiled_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Profiled negative log likelihoods"][self._index_of_closest_element(hyperparameters_0[plot_index], profiled_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Log inv rhos"])])], color="blue", marker="d", s=200, label='Start' )
                        axs[int(np.floor(plot_index / 2))][plot_index % 2].scatter([hyperparameters_f[plot_index]], [np.log(profiled_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Profiled negative log likelihoods"][self._index_of_closest_element(hyperparameters_f[plot_index], profiled_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Log inv rhos"])])], color="red" , marker="d", s=200, label='Finish')
                    axs    [int(np.floor(plot_index / 2))][plot_index % 2].set_ylabel("Log profiled negative log likelihood")
                else:
                    axs    [int(np.floor(plot_index / 2))][plot_index % 2].plot(profiled_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Log inv rhos"], profiled_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Profiled negative log likelihoods"])
                    if self._optimization_method == "PMLE":
                        axs[int(np.floor(plot_index / 2))][plot_index % 2].scatter([hyperparameters_0[plot_index]], [profiled_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Profiled negative log likelihoods"][self._index_of_closest_element(hyperparameters_0[plot_index], profiled_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Log inv rhos"])]], color="blue", marker="d", s=200, label='Start' )
                        axs[int(np.floor(plot_index / 2))][plot_index % 2].scatter([hyperparameters_f[plot_index]], [profiled_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Profiled negative log likelihoods"][self._index_of_closest_element(hyperparameters_f[plot_index], profiled_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Log inv rhos"])]], color="red" , marker="d", s=200, label='Finish')
                    axs    [int(np.floor(plot_index / 2))][plot_index % 2].set_ylabel("Profiled negative log likelihood")
                axs        [int(np.floor(plot_index / 2))][plot_index % 2].set_title(titles[plot_index])
                axs        [int(np.floor(plot_index / 2))][plot_index % 2].set_xlabel("HP value")
                axs        [int(np.floor(plot_index / 2))][plot_index % 2].grid(True)
            pl.tight_layout()
            if log_scale:
                pl.savefig(self._save_path + self._function_name + "/2.3.2 - Log profiled negative log likelihood.png")
            else:
                pl.savefig(self._save_path + self._function_name + "/2.3.2 - Profiled negative log likelihood.png")
            pl.show()
        if self._library == "plotly":
            figure = make_subplots(rows=int(np.ceil(len(profiled_negative_log_likelihood_evolution) / 2)), cols=2, subplot_titles=titles)
            for plot_index in range(len(profiled_negative_log_likelihood_evolution)):
                if log_scale:
                    figure    .add_trace(go.Scatter(x=profiled_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Log inv rhos"], y=np.log(np.maximum(1e-10, profiled_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Profiled negative log likelihoods"]), mode='lines', name=titles[plot_index])), row=1 + int(np.floor(plot_index) / 2), col=1 + plot_index % 2)
                    if self._optimization_method == "PMLE":
                        figure.add_trace(go.Scatter(x=[hyperparameters_0[plot_index]], y=[np.log(profiled_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Profiled negative log likelihoods"][self._index_of_closest_element(hyperparameters_0[plot_index], profiled_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Log inv rhos"])])], mode='markers', marker=dict(color='blue', symbol="diamond", size=10), name="Start" ), row=1 + int(np.floor(plot_index / 2)), col=1 + plot_index % 2)
                        figure.add_trace(go.Scatter(x=[hyperparameters_f[plot_index]], y=[np.log(profiled_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Profiled negative log likelihoods"][self._index_of_closest_element(hyperparameters_f[plot_index], profiled_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Log inv rhos"])])], mode='markers', marker=dict(color='red' , symbol="diamond", size=10), name="Finish"), row=1 + int(np.floor(plot_index / 2)), col=1 + plot_index % 2)
                    figure.update_yaxes (title_text="Log profiled negative log likelihood", row=1 + int(np.floor(len(profiled_negative_log_likelihood_evolution) / 2)), col=1 + plot_index % 2, showgrid=True)
                else:
                    figure    .add_trace(go.Scatter(x=profiled_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Log inv rhos"], y=profiled_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Profiled negative log likelihoods"], mode='lines', name=titles[plot_index]), row=1 + int(np.floor(plot_index) / 2), col=1 + plot_index % 2)
                    if self._optimization_method == "PMLE":
                        figure.add_trace(go.Scatter(x=[hyperparameters_0[plot_index]], y=[profiled_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Profiled negative log likelihoods"][self._index_of_closest_element(hyperparameters_0[plot_index], profiled_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Log inv rhos"])]], mode='markers', marker=dict(color='blue', symbol="diamond", size=10), name="Start" ), row=1 + int(np.floor(plot_index / 2)), col=1 + plot_index % 2)
                        figure.add_trace(go.Scatter(x=[hyperparameters_f[plot_index]], y=[profiled_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Profiled negative log likelihoods"][self._index_of_closest_element(hyperparameters_f[plot_index], profiled_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Log inv rhos"])]], mode='markers', marker=dict(color='red' , symbol="diamond", size=10), name="Finish"), row=1 + int(np.floor(plot_index / 2)), col=1 + plot_index % 2)
                    figure.update_yaxes (title_text="Profiled negative log likelihood", row=1 + int(np.floor(len(profiled_negative_log_likelihood_evolution) / 2)), col=1 + plot_index % 2, showgrid=True)
            figure.update_xaxes (title_text="HP value", showgrid=True)
            figure.update_layout(height=800, width=1100)
            figure.show         ()

    def plot_profiled_pva_negative_log_likelihood_evolution_1D(self, log_scale):
        log_inv_rhos                          = self._evaluation["Optimization functions evolution"]["Profiled PVA negative log likelihood evolution"]["Log inv rhos"                         ]
        profiled_pva_negative_log_likelihoods = self._evaluation["Optimization functions evolution"]["Profiled PVA negative log likelihood evolution"]["Profiled PVA negative log likelihoods"]
        hyperparameters_0                     = self._evaluation["Optimization infos"              ]["Hyperparameters history"                       ][0 ]
        hyperparameters_f                     = self._evaluation["Optimization infos"              ]["Hyperparameters history"                       ][-1]
        if self._library == "matplotlib":
            pl.figure         (figsize=FIGSIZE)
            pl.clf            ()
            if log_scale:
                pl.plot       (log_inv_rhos, np.log(np.maximum(1e-10, profiled_pva_negative_log_likelihoods)), color="black", linewidth=1)
                if self._optimization_method == "PMLE - PVA":
                    pl.scatter([hyperparameters_0[0]], [np.log(profiled_pva_negative_log_likelihoods[self._index_of_closest_element(hyperparameters_0[0], log_inv_rhos)])], color="blue", marker="d", s=200, label='Start' )
                    pl.scatter([hyperparameters_f[0]], [np.log(profiled_pva_negative_log_likelihoods[self._index_of_closest_element(hyperparameters_f[0], log_inv_rhos)])], color="red" , marker="d", s=200, label='Finish')
                pl.ylabel     ("Log profiled PVA negative log likelihood", fontsize=20)
            else:
                pl.plot       (log_inv_rhos, profiled_pva_negative_log_likelihoods, color="black", linewidth=1)
                if self._optimization_method == "PMLE - PVA":
                    pl.scatter([hyperparameters_0[0]], [profiled_pva_negative_log_likelihoods[self._index_of_closest_element(hyperparameters_0[0], log_inv_rhos)]], color="blue", marker="d", s=200, label='Start' )
                    pl.scatter([hyperparameters_f[0]], [profiled_pva_negative_log_likelihoods[self._index_of_closest_element(hyperparameters_f[0], log_inv_rhos)]], color="red" , marker="d", s=200, label='Finish')
                pl.ylabel     ("Profiled PVA negative log likelihood", fontsize=20)
            pl.xticks         (fontsize=16)
            pl.yticks         (fontsize=16)
            pl.grid           ()
            pl.xlabel         ("-log(rho)", fontsize=20)
            if log_scale:
                pl.savefig    (self._save_path + self._function_name + "/2.3.3 - Log profiled PVA negative log likelihood.png")
            else:
                pl.savefig    (self._save_path + self._function_name + "/2.3.3 - Profiled PVA negative log likelihood.png")
            pl.show           ()
        if self._library == "plotly":
            figure = go.Figure()
            if log_scale:
                figure    .add_trace(go.Scatter(x=log_inv_rhos, y=np.log(np.maximum(1e-10, profiled_pva_negative_log_likelihoods)), mode='lines', line=dict(width=1, color="black" )))
                if self._optimization_method == "PMLE - PVA":
                    figure.add_trace(go.Scatter(x=[hyperparameters_0[0]], y=[np.log(profiled_pva_negative_log_likelihoods[self._index_of_closest_element(hyperparameters_0[0], log_inv_rhos)])], mode='markers', marker=dict(color='blue', symbol="diamond", size=10), name="Start" ))
                    figure.add_trace(go.Scatter(x=[hyperparameters_f[0]], y=[np.log(profiled_pva_negative_log_likelihoods[self._index_of_closest_element(hyperparameters_f[0], log_inv_rhos)])], mode='markers', marker=dict(color='red' , symbol="diamond", size=10), name="Finish"))
                figure.update_layout(xaxis_title="-log(rho)", yaxis_title="Log profiled PVA negative log likelihood", width=1100, height=700)
            else:
                figure    .add_trace(go.Scatter(x=log_inv_rhos, y=profiled_pva_negative_log_likelihoods, mode='lines', line=dict(width=1, color="black" )))
                if self._optimization_method == "PMLE - PVA":
                    figure.add_trace(go.Scatter(x=[hyperparameters_0[0]], y=[profiled_pva_negative_log_likelihoods[self._index_of_closest_element(hyperparameters_0[0], log_inv_rhos)]], mode='markers', marker=dict(color='blue', symbol="diamond", size=10), name="Start" ))
                    figure.add_trace(go.Scatter(x=[hyperparameters_f[0]], y=[profiled_pva_negative_log_likelihoods[self._index_of_closest_element(hyperparameters_f[0], log_inv_rhos)]], mode='markers', marker=dict(color='red' , symbol="diamond", size=10), name="Finish"))
                figure.update_layout(xaxis_title="-log(rho)", yaxis_title="Profiled PVA negative log likelihood", width=1100, height=700)
            figure.update_xaxes(tickprefix="<b>", ticksuffix="<b><br>")
            figure.update_yaxes(tickprefix="<b>", ticksuffix="<b><br>")
            figure.show        ()

    def plot_profiled_pva_negative_log_likelihood_evolution_2D_3D(self, log_scale):
        log_inv_rhos1                         = self._evaluation["Optimization functions evolution"]["Profiled PVA negative log likelihood evolution"]["Log inv rhos 1"                       ]
        log_inv_rhos2                         = self._evaluation["Optimization functions evolution"]["Profiled PVA negative log likelihood evolution"]["Log inv rhos 2"                       ]
        profiled_pva_negative_log_likelihoods = self._evaluation["Optimization functions evolution"]["Profiled PVA negative log likelihood evolution"]["Profiled PVA negative log likelihoods"]
        hyperparameters_0                     = self._evaluation["Optimization infos"              ]["Hyperparameters history"                       ][0 ]
        hyperparameters_f                     = self._evaluation["Optimization infos"              ]["Hyperparameters history"                       ][-1]
        if self._library == "matplotlib":
            figure = pl.figure         (figsize=FIGSIZE)
            axes   = figure.add_subplot(111, projection='3d')
            axes.set_xlabel            ("-log(rho 1)", fontsize=20)
            axes.set_ylabel            ("-log(rho 2)", fontsize=20)
            if log_scale:
                axes.plot_surface      (log_inv_rhos1, log_inv_rhos2, np.log(np.maximum(1e-10, profiled_pva_negative_log_likelihoods)), cmap='viridis')
                if self._optimization_method == "PMLE - PVA":
                    axes.scatter       ([hyperparameters_0[0]], [hyperparameters_0[1]], [0], color="blue", marker="d", s=200, label='Start' )
                    axes.scatter       ([hyperparameters_f[0]], [hyperparameters_f[1]], [0], color="red" , marker="d", s=200, label='Finish')
                axes.set_zlabel        ("Log profiled PVA negative log likelihood", fontsize=20)
                pl.savefig             (self._save_path + self._function_name + "/2.3.3 - Log profiled PVA negative log likelihood.png")
            else:
                axes.plot_surface      (log_inv_rhos1, log_inv_rhos2, profiled_pva_negative_log_likelihoods, cmap='viridis')
                if self._optimization_method == "PMLE - PVA":
                    axes.scatter       ([hyperparameters_0[0]], [hyperparameters_0[1]], [0], color="blue", marker="d", s=200, label='Start' )
                    axes.scatter       ([hyperparameters_f[0]], [hyperparameters_f[1]], [0], color="red" , marker="d", s=200, label='Finish')
                axes.set_zlabel        ("Profiled PVA negative log likelihood", fontsize=20)
                pl.savefig             (self._save_path + self._function_name + "/2.3.3 - Profiled PVA negative log likelihood.png")
            pl.show                    ()
        if self._library == "plotly":
            if log_scale:
                figure = go.Figure  (data=[go.Surface(z=np.log(np.maximum(1e-10, profiled_pva_negative_log_likelihoods)), x=log_inv_rhos1, y=log_inv_rhos2, colorscale='Viridis')])
                figure.update_layout(scene=dict(xaxis_title="-log(rho 1)", yaxis_title="-log(rho 2)", zaxis_title="Log profiled PVA negative log likelihood"), width=1100, height=700)
            else:
                figure = go.Figure  (data=[go.Surface(z=profiled_pva_negative_log_likelihoods, x=log_inv_rhos1, y=log_inv_rhos2, colorscale='Viridis')])
                figure.update_layout(scene=dict(xaxis_title="-log(rho 1)", yaxis_title="-log(rho 2)", zaxis_title="Profiled PVA negative log likelihood"), width=1100, height=700)
            if self._optimization_method == "PMLE - PVA":
                figure.add_trace    (go.Scatter3d(x=[hyperparameters_0[0]], y=[hyperparameters_0[1]], z=[0], mode='markers', marker=dict(size=10, color="blue", symbol="diamond"), name='Start' ))
                figure.add_trace    (go.Scatter3d(x=[hyperparameters_f[0]], y=[hyperparameters_f[1]], z=[0], mode='markers', marker=dict(size=10, color="red" , symbol="diamond"), name='Finish'))
            figure.show             ()

    def plot_profiled_pva_negative_log_likelihood_evolution_2D_heatmap(self, log_scale):
        log_inv_rhos1                         = self._evaluation["Optimization functions evolution"]["Profiled PVA negative log likelihood evolution"]["Log inv rhos 1"                       ]
        log_inv_rhos2                         = self._evaluation["Optimization functions evolution"]["Profiled PVA negative log likelihood evolution"]["Log inv rhos 2"                       ]
        profiled_pva_negative_log_likelihoods = self._evaluation["Optimization functions evolution"]["Profiled PVA negative log likelihood evolution"]["Profiled PVA negative log likelihoods"]
        hyperparameters_0                     = self._evaluation["Optimization infos"              ]["Hyperparameters history"                       ][0 ]
        hyperparameters_f                     = self._evaluation["Optimization infos"              ]["Hyperparameters history"                       ][-1]
        if self._library == "matplotlib":
            pl.figure     (figsize=FIGSIZE)
            pl.clf        ()
            if log_scale:
                pl.imshow (np.log(np.maximum(1e-10, profiled_pva_negative_log_likelihoods)), extent=(np.min(log_inv_rhos1), np.max(log_inv_rhos1), np.min(log_inv_rhos2), np.max(log_inv_rhos2)), origin='lower', cmap='viridis', interpolation='nearest')
            else:
                pl.imshow (profiled_pva_negative_log_likelihoods, extent=(np.min(log_inv_rhos1), np.max(log_inv_rhos1), np.min(log_inv_rhos2), np.max(log_inv_rhos2)), origin='lower', cmap='viridis', interpolation='nearest')
            if self._optimization_method == "PMLE - PVA":
                pl.scatter([hyperparameters_0[0]], [hyperparameters_0[1]], color="blue", marker="d", s=200, label='Start' )
                pl.scatter([hyperparameters_f[0]], [hyperparameters_f[1]], color="red" , marker="d", s=200, label='Finish')
            pl.colorbar   ()
            pl.xticks     (fontsize=16)
            pl.yticks     (fontsize=16)
            pl.xlabel     ('-log(rho 1)')
            pl.ylabel     ('-log(rho 2)')
            pl.grid       ()
            if log_scale:
                pl.savefig(self._save_path + self._function_name + "/2.3.3 - Log profiled PVA negative log likelihood.png")
            else:
                pl.savefig(self._save_path + self._function_name + "/2.3.3 - Profiled PVA negative log likelihood.png")
            pl.show       ()
        if self._library == "plotly":
            if log_scale:
                figure = px.imshow  (np.log(np.maximum(1e-10, profiled_pva_negative_log_likelihoods)), x=log_inv_rhos1[0], y=log_inv_rhos2.T[0], labels={'x': "-log(rho 1)", 'y': "-log(rho 2)", 'color': 'Log profiled PVA negative log likelihood'})
                figure.update_layout(scene=dict(xaxis_title='-log(rho 1)', yaxis_title='-log(rho 2)', zaxis_title='Log profiled PVA negative log likelihood'), width=800, height=800)
            else:
                figure = px.imshow  (profiled_pva_negative_log_likelihoods, x=log_inv_rhos1[0], y=log_inv_rhos2.T[0], labels={'x': "-log(rho 1)", 'y': "-log(rho 2)", 'color': 'Profiled PVA negative log likelihood'})
                figure.update_layout(scene=dict(xaxis_title='-log(rho 1)', yaxis_title='-log(rho 2)', zaxis_title='Profiled PVA negative log likelihood'), width=800, height=800)
            figure.update_yaxes     (autorange='reversed')
            if self._optimization_method == "PMLE - PVA":
                figure.add_trace(go.Scatter(x=[hyperparameters_0[0]], y=[hyperparameters_0[1]], mode='markers', marker=dict(color='blue', symbol="diamond", size=10), name="Start" ))
                figure.add_trace(go.Scatter(x=[hyperparameters_f[0]], y=[hyperparameters_f[1]], mode='markers', marker=dict(color='red' , symbol="diamond", size=10), name="Finish"))
            figure.show         ()

    def plot_profiled_pva_negative_log_likelihood_evolution_nD(self, log_scale):
        profiled_pva_negative_log_likelihood_evolution = self._evaluation["Optimization functions evolution"]["Profiled PVA negative log likelihood evolution"]
        hyperparameters_0                              = self._evaluation["Optimization infos"              ]["Hyperparameters history"                       ][0 ]
        hyperparameters_f                              = self._evaluation["Optimization infos"              ]["Hyperparameters history"                       ][-1]
        titles                                         = ["-log(rho{})".format(plot_index) for plot_index in range(len(profiled_pva_negative_log_likelihood_evolution))]
        if self._library == "matplotlib":
            fig, axs = pl.subplots(int(np.ceil(len(profiled_pva_negative_log_likelihood_evolution) / 2)), 2, figsize=FIGSIZE)
            for plot_index in range(len(profiled_pva_negative_log_likelihood_evolution)):
                if log_scale:
                    axs    [int(np.floor(plot_index / 2))][plot_index % 2].plot(profiled_pva_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Log inv rhos"], np.log(np.maximum(1e-10, profiled_pva_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Profiled PVA negative log likelihoods"])))
                    if self._optimization_method == "PMLE - PVA":
                        axs[int(np.floor(plot_index / 2))][plot_index % 2].scatter([hyperparameters_0[plot_index]], [np.log(profiled_pva_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Profiled PVA negative log likelihoods"][self._index_of_closest_element(hyperparameters_0[plot_index], profiled_pva_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Log inv rhos"])])], color="blue", marker="d", s=200, label='Start' )
                        axs[int(np.floor(plot_index / 2))][plot_index % 2].scatter([hyperparameters_f[plot_index]], [np.log(profiled_pva_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Profiled PVA negative log likelihoods"][self._index_of_closest_element(hyperparameters_f[plot_index], profiled_pva_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Log inv rhos"])])], color="red" , marker="d", s=200, label='Finish')
                    axs    [int(np.floor(plot_index / 2))][plot_index % 2].set_ylabel("Log profiled PVA negative log likelihood")
                else:
                    axs    [int(np.floor(plot_index / 2))][plot_index % 2].plot(profiled_pva_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Log inv rhos"], profiled_pva_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Profiled PVA negative log likelihoods"])
                    if self._optimization_method == "PMLE - PVA":
                        axs[int(np.floor(plot_index / 2))][plot_index % 2].scatter([hyperparameters_0[plot_index]], [profiled_pva_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Profiled PVA negative log likelihoods"][self._index_of_closest_element(hyperparameters_0[plot_index], profiled_pva_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Log inv rhos"])]], color="blue", marker="d", s=200, label='Start' )
                        axs[int(np.floor(plot_index / 2))][plot_index % 2].scatter([hyperparameters_f[plot_index]], [profiled_pva_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Profiled PVA negative log likelihoods"][self._index_of_closest_element(hyperparameters_f[plot_index], profiled_pva_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Log inv rhos"])]], color="red" , marker="d", s=200, label='Finish')
                    axs    [int(np.floor(plot_index / 2))][plot_index % 2].set_ylabel("Profiled PVA negative log likelihood")
                axs        [int(np.floor(plot_index / 2))][plot_index % 2].set_title(titles[plot_index])
                axs        [int(np.floor(plot_index / 2))][plot_index % 2].set_xlabel("HP value")
                axs        [int(np.floor(plot_index / 2))][plot_index % 2].grid(True)
            pl.tight_layout()
            if log_scale:
                pl.savefig(self._save_path + self._function_name + "/2.3.3 - Log profiled PVA negative log likelihood.png")
            else:
                pl.savefig(self._save_path + self._function_name + "/2.3.3 - Profiled PVA negative log likelihood.png")
            pl.show()
        if self._library == "plotly":
            figure = make_subplots(rows=int(np.ceil(len(profiled_pva_negative_log_likelihood_evolution) / 2)), cols=2, subplot_titles=titles)
            for plot_index in range(len(profiled_pva_negative_log_likelihood_evolution)):
                if log_scale:
                    figure    .add_trace(go.Scatter(x=profiled_pva_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Log inv rhos"], y=np.log(np.maximum(1e-10, profiled_pva_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Profiled PVA negative log likelihoods"]), mode='lines', name=titles[plot_index])), row=1 + int(np.floor(plot_index) / 2), col=1 + plot_index % 2)
                    if self._optimization_method == "PMLE - PVA":
                        figure.add_trace(go.Scatter(x=[hyperparameters_0[plot_index]], y=[np.log(profiled_pva_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Profiled PVA negative log likelihoods"][self._index_of_closest_element(hyperparameters_0[plot_index], profiled_pva_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Log inv rhos"])])], mode='markers', marker=dict(color='blue', symbol="diamond", size=10), name="Start" ), row=1 + int(np.floor(plot_index / 2)), col=1 + plot_index % 2)
                        figure.add_trace(go.Scatter(x=[hyperparameters_f[plot_index]], y=[np.log(profiled_pva_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Profiled PVA negative log likelihoods"][self._index_of_closest_element(hyperparameters_f[plot_index], profiled_pva_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Log inv rhos"])])], mode='markers', marker=dict(color='red' , symbol="diamond", size=10), name="Finish"), row=1 + int(np.floor(plot_index / 2)), col=1 + plot_index % 2)
                    figure.update_yaxes (title_text="Log profiled PVA negative log likelihood", row=1 + int(np.floor(len(profiled_pva_negative_log_likelihood_evolution) / 2)), col=1 + plot_index % 2, showgrid=True)
                else:
                    figure    .add_trace(go.Scatter(x=profiled_pva_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Log inv rhos"], y=profiled_pva_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Profiled PVA negative log likelihoods"], mode='lines', name=titles[plot_index]), row=1 + int(np.floor(plot_index) / 2), col=1 + plot_index % 2)
                    if self._optimization_method == "PMLE - PVA":
                        figure.add_trace(go.Scatter(x=[hyperparameters_0[plot_index]], y=[profiled_pva_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Profiled PVA negative log likelihoods"][self._index_of_closest_element(hyperparameters_0[plot_index], profiled_pva_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Log inv rhos"])]], mode='markers', marker=dict(color='blue', symbol="diamond", size=10), name="Start" ), row=1 + int(np.floor(plot_index / 2)), col=1 + plot_index % 2)
                        figure.add_trace(go.Scatter(x=[hyperparameters_f[plot_index]], y=[profiled_pva_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Profiled PVA negative log likelihoods"][self._index_of_closest_element(hyperparameters_f[plot_index], profiled_pva_negative_log_likelihood_evolution["Log inv rho " + str(plot_index)]["Log inv rhos"])]], mode='markers', marker=dict(color='red' , symbol="diamond", size=10), name="Finish"), row=1 + int(np.floor(plot_index / 2)), col=1 + plot_index % 2)
                    figure.update_yaxes (title_text="Profiled PVA negative log likelihood", row=1 + int(np.floor(len(profiled_pva_negative_log_likelihood_evolution) / 2)), col=1 + plot_index % 2, showgrid=True)
            figure.update_xaxes (title_text="HP value", showgrid=True)
            figure.update_layout(height=800, width=1100)
            figure.show         ()

    def plot_mean_square_error_evolution_1D(self, log_scale):
        log_inv_rhos       = self._evaluation["Optimization functions evolution"]["Mean square error evolution"]["Log inv rhos"      ]
        mean_square_errors = self._evaluation["Optimization functions evolution"]["Mean square error evolution"]["Mean square errors"]
        hyperparameters_0  = self._evaluation["Optimization infos"              ]["Hyperparameters history"    ][0 ]
        hyperparameters_f  = self._evaluation["Optimization infos"              ]["Hyperparameters history"    ][-1]
        if self._library == "matplotlib":
            pl.figure         (figsize=FIGSIZE)
            pl.clf            ()
            if log_scale:
                pl.plot       (log_inv_rhos, np.log(mean_square_errors), color="black", linewidth=1)
                if self._optimization_method == "MSE - PVA":
                    pl.scatter([hyperparameters_0[0]], [np.log(mean_square_errors[self._index_of_closest_element(hyperparameters_0[0], log_inv_rhos)])], color="blue", marker="d", s=200, label='Start' )
                    pl.scatter([hyperparameters_f[0]], [np.log(mean_square_errors[self._index_of_closest_element(hyperparameters_f[0], log_inv_rhos)])], color="red" , marker="d", s=200, label='Finish')
                pl.ylabel     ("Log mean square error", fontsize=20)
            else:
                pl.plot       (log_inv_rhos, mean_square_errors, color="black", linewidth=1)
                if self._optimization_method == "MSE - PVA":
                    pl.scatter([hyperparameters_0[0]], [mean_square_errors[self._index_of_closest_element(hyperparameters_0[0], log_inv_rhos)]], color="blue", marker="d", s=200, label='Start' )
                    pl.scatter([hyperparameters_f[0]], [mean_square_errors[self._index_of_closest_element(hyperparameters_f[0], log_inv_rhos)]], color="red" , marker="d", s=200, label='Finish')
                pl.ylabel     ("Mean square error", fontsize=20)
            pl.xticks         (fontsize=16)
            pl.yticks         (fontsize=16)
            pl.grid           ()
            pl.xlabel         ("-log(rho)", fontsize=20)
            if log_scale:
                pl.savefig    (self._save_path + self._function_name + "/2.3.4 - Log mean square error.png")
            else:
                pl.savefig    (self._save_path + self._function_name + "/2.3.4 - Mean square error.png")
            pl.show           ()
        if self._library == "plotly":
            figure = go.Figure()
            if log_scale:
                figure    .add_trace(go.Scatter(x=log_inv_rhos, y=np.log(mean_square_errors), mode='lines', line=dict(width=1, color="black" )))
                if self._optimization_method == "MSE - PVA":
                    figure.add_trace(go.Scatter(x=[hyperparameters_0[0]], y=[np.log(mean_square_errors[self._index_of_closest_element(hyperparameters_0[0], log_inv_rhos)])], mode='markers', marker=dict(color='blue', symbol="diamond", size=10), name="Start" ))
                    figure.add_trace(go.Scatter(x=[hyperparameters_f[0]], y=[np.log(mean_square_errors[self._index_of_closest_element(hyperparameters_f[0], log_inv_rhos)])], mode='markers', marker=dict(color='red' , symbol="diamond", size=10), name="Finish"))
                figure    .update_layout(xaxis_title="-log(rho)", yaxis_title="Log mean square error", width=1100, height=700)
            else:
                figure    .add_trace(go.Scatter(x=log_inv_rhos, y=mean_square_errors, mode='lines', line=dict(width=1, color="black" )))
                if self._optimization_method == "MSE - PVA":
                    figure.add_trace(go.Scatter(x=[hyperparameters_0[0]], y=[mean_square_errors[self._index_of_closest_element(hyperparameters_0[0], log_inv_rhos)]], mode='markers', marker=dict(color='blue', symbol="diamond", size=10), name="Start" ))
                    figure.add_trace(go.Scatter(x=[hyperparameters_f[0]], y=[mean_square_errors[self._index_of_closest_element(hyperparameters_f[0], log_inv_rhos)]], mode='markers', marker=dict(color='red' , symbol="diamond", size=10), name="Finish"))
                figure    .update_layout(xaxis_title="-log(rho)", yaxis_title="Mean square error", width=1100, height=700)
            figure.update_xaxes(tickprefix="<b>", ticksuffix="<b><br>")
            figure.update_yaxes(tickprefix="<b>", ticksuffix="<b><br>")
            figure.show        ()

    def plot_mean_square_error_evolution_2D_3D(self, log_scale):
        log_inv_rhos1      = self._evaluation["Optimization functions evolution"]["Mean square error evolution"]["Log inv rhos 1"    ]
        log_inv_rhos2      = self._evaluation["Optimization functions evolution"]["Mean square error evolution"]["Log inv rhos 2"    ]
        mean_square_errors = self._evaluation["Optimization functions evolution"]["Mean square error evolution"]["Mean square errors"]
        hyperparameters_0  = self._evaluation["Optimization infos"              ]["Hyperparameters history"    ][0 ]
        hyperparameters_f  = self._evaluation["Optimization infos"              ]["Hyperparameters history"    ][-1]
        if self._library == "matplotlib":
            figure = pl.figure         (figsize=FIGSIZE)
            axes   = figure.add_subplot(111, projection='3d')
            axes.set_xlabel            ("-log(rho 1)"            , fontsize=20)
            axes.set_ylabel            ("-log(rho 2)"            , fontsize=20)
            if log_scale:
                axes.plot_surface      (log_inv_rhos1, log_inv_rhos2, np.log(mean_square_errors), cmap='viridis')
                if self._optimization_method == "MSE - PVA":
                    axes.scatter       ([hyperparameters_0[0]], [hyperparameters_0[1]], [0], color="blue", marker="d", s=200, label='Start' )
                    axes.scatter       ([hyperparameters_f[0]], [hyperparameters_f[1]], [0], color="red" , marker="d", s=200, label='Finish')
                axes.set_zlabel        ("Log mean square error", fontsize=20)
                pl.savefig             (self._save_path + self._function_name + "/2.3.4 - Log mean square error.png")
            else:
                axes.plot_surface      (log_inv_rhos1, log_inv_rhos2, mean_square_errors, cmap='viridis')
                if self._optimization_method == "MSE - PVA":
                    axes.scatter       ([hyperparameters_0[0]], [hyperparameters_0[1]], [0], color="blue", marker="d", s=200, label='Start' )
                    axes.scatter       ([hyperparameters_f[0]], [hyperparameters_f[1]], [0], color="red" , marker="d", s=200, label='Finish')
                axes.set_zlabel        ("Mean square error", fontsize=20)
                pl.savefig             (self._save_path + self._function_name + "/2.3.4 - Mean square error.png")
            pl.show                    ()
        if self._library == "plotly":
            if log_scale:
                figure = go.Figure  (data=[go.Surface(z=np.log(mean_square_errors), x=log_inv_rhos1, y=log_inv_rhos2, colorscale='Viridis')])
                figure.update_layout(scene=dict(xaxis_title="-log(rho 1)", yaxis_title="-log(rho 2)", zaxis_title="Log mean square error"), width=1100, height=700)
            else:
                figure = go.Figure  (data=[go.Surface(z=mean_square_errors, x=log_inv_rhos1, y=log_inv_rhos2, colorscale='Viridis')])
                figure.update_layout(scene=dict(xaxis_title="-log(rho 1)", yaxis_title="-log(rho 2)", zaxis_title="Mean square error"), width=1100, height=700)
            if self._optimization_method == "MSE - PVA":
                figure.add_trace    (go.Scatter3d(x=[hyperparameters_0[0]], y=[hyperparameters_0[1]], z=[0], mode='markers', marker=dict(size=10, color="blue", symbol="diamond"), name='Start' ))
                figure.add_trace    (go.Scatter3d(x=[hyperparameters_f[0]], y=[hyperparameters_f[1]], z=[0], mode='markers', marker=dict(size=10, color="red" , symbol="diamond"), name='Finish'))
            figure.show             ()

    def plot_mean_square_error_evolution_2D_heatmap(self, log_scale):
        log_inv_rhos1      = self._evaluation["Optimization functions evolution"]["Mean square error evolution"]["Log inv rhos 1"    ]
        log_inv_rhos2      = self._evaluation["Optimization functions evolution"]["Mean square error evolution"]["Log inv rhos 2"    ]
        mean_square_errors = self._evaluation["Optimization functions evolution"]["Mean square error evolution"]["Mean square errors"]
        hyperparameters_0  = self._evaluation["Optimization infos"              ]["Hyperparameters history"    ][0 ]
        hyperparameters_f  = self._evaluation["Optimization infos"              ]["Hyperparameters history"    ][-1]
        if self._library == "matplotlib":
            pl.figure(figsize=FIGSIZE)
            pl.clf()
            if log_scale:
                pl.imshow(np.log(mean_square_errors), extent=(np.min(log_inv_rhos1), np.max(log_inv_rhos1), np.min(log_inv_rhos2), np.max(log_inv_rhos2)), origin='lower', cmap='viridis', interpolation='nearest')
            else:
                pl.imshow(mean_square_errors, extent=(np.min(log_inv_rhos1), np.max(log_inv_rhos1), np.min(log_inv_rhos2), np.max(log_inv_rhos2)), origin='lower', cmap='viridis', interpolation='nearest')
            if self._optimization_method == "MSE - PVA":
                pl.scatter([hyperparameters_0[0]], [hyperparameters_0[1]], color="blue", marker="d", s=200, label='Start' )
                pl.scatter([hyperparameters_f[0]], [hyperparameters_f[1]], color="red" , marker="d", s=200, label='Finish')
            pl.colorbar()
            pl.xticks(fontsize=16)
            pl.yticks(fontsize=16)
            pl.xlabel('-log(rho 1)')
            pl.ylabel('-log(rho 2)')
            pl.grid()
            if log_scale:
                pl.savefig(self._save_path + self._function_name + "/2.3.4 - Log mean square error.png")
            else:
                pl.savefig(self._save_path + self._function_name + "/2.3.4 - Mean square error.png")
            pl.show()
        if self._library == "plotly":
            if log_scale:
                figure = px.imshow(np.log(mean_square_errors), x=log_inv_rhos1[0], y=log_inv_rhos2.T[0], labels={'x': "-log(rho 1)", 'y': "-log(rho 2)", 'color': 'Log mean square error'})
                figure.update_layout(scene=dict(xaxis_title='-log(rho 1)', yaxis_title='-log(rho 2)', zaxis_title='Log mean square error'), width=800, height=800)
            else:
                figure = px.imshow(mean_square_errors, x=log_inv_rhos1[0], y=log_inv_rhos2.T[0], labels={'x': "-log(rho 1)", 'y': "-log(rho 2)", 'color': 'Mean square error'})
                figure.update_layout(scene=dict(xaxis_title='-log(rho 1)', yaxis_title='-log(rho 2)', zaxis_title='Mean square error'), width=800, height=800)
            if self._optimization_method == "MSE - PVA":
                figure.add_trace(go.Scatter(x=[hyperparameters_0[0]], y=[hyperparameters_0[1]], mode='markers', marker=dict(color='blue', symbol="diamond", size=10), name="Start" ))
                figure.add_trace(go.Scatter(x=[hyperparameters_f[0]], y=[hyperparameters_f[1]], mode='markers', marker=dict(color='red' , symbol="diamond", size=10), name="Finish"))
            figure.update_yaxes(autorange='reversed')
            figure.show()

    def plot_mean_square_error_evolution_nD(self, log_scale):
        mean_square_error_evolution = self._evaluation["Optimization functions evolution"]["Mean square error evolution"]
        hyperparameters_0           = self._evaluation["Optimization infos"              ]["Hyperparameters history"    ][0 ]
        hyperparameters_f           = self._evaluation["Optimization infos"              ]["Hyperparameters history"    ][-1]
        titles                      = ["-log(rho{})".format(plot_index) for plot_index in range(len(mean_square_error_evolution))]
        if self._library == "matplotlib":
            fig, axs = pl.subplots(int(np.ceil(len(mean_square_error_evolution) / 2)), 2, figsize=FIGSIZE)
            for plot_index in range(len(mean_square_error_evolution)):
                if log_scale:
                    axs    [int(np.floor(plot_index / 2))][plot_index % 2].plot(mean_square_error_evolution["Log inv rho " + str(plot_index)]["Log inv rhos"], np.log(mean_square_error_evolution["Log inv rho " + str(plot_index)]["Mean square errors"]))
                    if self._optimization_method == "MSE - PVA":
                        axs[int(np.floor(plot_index / 2))][plot_index % 2].scatter([hyperparameters_0[plot_index]], [np.log(mean_square_error_evolution["Log inv rho " + str(plot_index)]["Mean square errors"][self._index_of_closest_element(hyperparameters_0[plot_index], mean_square_error_evolution["Log inv rho " + str(plot_index)]["Log inv rhos"])])], color="blue", marker="d", s=200, label='Start' )
                        axs[int(np.floor(plot_index / 2))][plot_index % 2].scatter([hyperparameters_f[plot_index]], [np.log(mean_square_error_evolution["Log inv rho " + str(plot_index)]["Mean square errors"][self._index_of_closest_element(hyperparameters_f[plot_index], mean_square_error_evolution["Log inv rho " + str(plot_index)]["Log inv rhos"])])], color="red" , marker="d", s=200, label='Finish')
                    axs    [int(np.floor(plot_index / 2))][plot_index % 2].set_ylabel("Log mean square error")
                else:
                    axs    [int(np.floor(plot_index / 2))][plot_index % 2].plot(mean_square_error_evolution["Log inv rho " + str(plot_index)]["Log inv rhos"], mean_square_error_evolution["Log inv rho " + str(plot_index)]["Mean square errors"])
                    if self._optimization_method == "MSE - PVA":
                        axs[int(np.floor(plot_index / 2))][plot_index % 2].scatter([hyperparameters_0[plot_index]], [mean_square_error_evolution["Log inv rho " + str(plot_index)]["Mean square errors"][self._index_of_closest_element(hyperparameters_0[plot_index], mean_square_error_evolution["Log inv rho " + str(plot_index)]["Log inv rhos"])]], color="blue", marker="d", s=200, label='Start' )
                        axs[int(np.floor(plot_index / 2))][plot_index % 2].scatter([hyperparameters_f[plot_index]], [mean_square_error_evolution["Log inv rho " + str(plot_index)]["Mean square errors"][self._index_of_closest_element(hyperparameters_f[plot_index], mean_square_error_evolution["Log inv rho " + str(plot_index)]["Log inv rhos"])]], color="red" , marker="d", s=200, label='Finish')
                    axs    [int(np.floor(plot_index / 2))][plot_index % 2].set_ylabel("Mean square error")
                axs        [int(np.floor(plot_index / 2))][plot_index % 2].set_title(titles[plot_index])
                axs        [int(np.floor(plot_index / 2))][plot_index % 2].set_xlabel("HP value")
                axs        [int(np.floor(plot_index / 2))][plot_index % 2].grid(True)
            pl.tight_layout()
            if log_scale:
                pl.savefig(self._save_path + self._function_name + "/2.3.4 - Log mean square error.png")
            else:
                pl.savefig(self._save_path + self._function_name + "/2.3.4 - Mean square error.png")
            pl.show()
        if self._library == "plotly":
            figure = make_subplots(rows=int(np.ceil(len(mean_square_error_evolution) / 2)), cols=2, subplot_titles=titles)
            for plot_index in range(len(mean_square_error_evolution)):
                if log_scale:
                    figure    .add_trace(go.Scatter(x=mean_square_error_evolution["Log inv rho " + str(plot_index)]["Log inv rhos"], y=np.log(mean_square_error_evolution["Log inv rho " + str(plot_index)]["Mean square errors"]), mode='lines', name=titles[plot_index]), row=1 + int(np.floor(plot_index) / 2), col=1 + plot_index % 2)
                    if self._optimization_method == "MSE - PVA":
                        figure.add_trace(go.Scatter(x=[hyperparameters_0[plot_index]], y=[np.log(mean_square_error_evolution["Log inv rho " + str(plot_index)]["Mean square errors"][self._index_of_closest_element(hyperparameters_0[plot_index], mean_square_error_evolution["Log inv rho " + str(plot_index)]["Log inv rhos"])])], mode='markers', marker=dict(color='blue', symbol="diamond", size=10), name="Start" ), row=1 + int(np.floor(plot_index / 2)), col=1 + plot_index % 2)
                        figure.add_trace(go.Scatter(x=[hyperparameters_f[plot_index]], y=[np.log(mean_square_error_evolution["Log inv rho " + str(plot_index)]["Mean square errors"][self._index_of_closest_element(hyperparameters_f[plot_index], mean_square_error_evolution["Log inv rho " + str(plot_index)]["Log inv rhos"])])], mode='markers', marker=dict(color='red' , symbol="diamond", size=10), name="Finish"), row=1 + int(np.floor(plot_index / 2)), col=1 + plot_index % 2)
                    figure.update_yaxes (title_text="Log mean square error", row=1 + int(np.floor(len(mean_square_error_evolution) / 2)), col=1 + plot_index % 2, showgrid=True)
                else:
                    figure    .add_trace(go.Scatter(x=mean_square_error_evolution["Log inv rho " + str(plot_index)]["Log inv rhos"], y=mean_square_error_evolution["Log inv rho " + str(plot_index)]["Mean square errors"], mode='lines', name=titles[plot_index]), row=1 + int(np.floor(plot_index) / 2), col=1 + plot_index % 2)
                    if self._optimization_method == "MSE - PVA":
                        figure.add_trace(go.Scatter(x=[hyperparameters_0[plot_index]], y=[mean_square_error_evolution["Log inv rho " + str(plot_index)]["Mean square errors"][self._index_of_closest_element(hyperparameters_0[plot_index], mean_square_error_evolution["Log inv rho " + str(plot_index)]["Log inv rhos"])]], mode='markers', marker=dict(color='blue', symbol="diamond", size=10), name="Start" ), row=1 + int(np.floor(plot_index / 2)), col=1 + plot_index % 2)
                        figure.add_trace(go.Scatter(x=[hyperparameters_f[plot_index]], y=[mean_square_error_evolution["Log inv rho " + str(plot_index)]["Mean square errors"][self._index_of_closest_element(hyperparameters_f[plot_index], mean_square_error_evolution["Log inv rho " + str(plot_index)]["Log inv rhos"])]], mode='markers', marker=dict(color='red' , symbol="diamond", size=10), name="Finish"), row=1 + int(np.floor(plot_index / 2)), col=1 + plot_index % 2)
                    figure.update_yaxes (title_text="Mean square error", row=1 + int(np.floor(len(mean_square_error_evolution) / 2)), col=1 + plot_index % 2, showgrid=True)
            figure.update_xaxes (title_text="HP value", showgrid=True)
            figure.update_layout(height=800, width=1100)
            figure.show         ()

    def plot_empirical_coverage_probability_assuming_reliability(self, dataset):
        alphas =     self._evaluation[dataset]["Reliability"]["alphas"                      ]
        ECP    =     self._evaluation[dataset]["Reliability"]["ECP"                         ]
        n      = len(self._evaluation[dataset]["Reliability"]["Normalized prediction errors"])
        std    = np.sqrt(np.array([alpha*(1-alpha)/ n for alpha in alphas]))
        if self._library == "matplotlib":
            pl.figure      (figsize=FIGSIZE)
            pl.clf         ()
            pl.plot        (alphas, alphas, label="Theory", color="green", linewidth=1)
            pl.plot        (alphas, ECP   , label="ECP"   , color="blue" , linewidth=1)
            if dataset == "Test":
                pl.fill_between(alphas, alphas - std, alphas + std, alpha=0.4, color="lightskyblue", label="std")
            pl.xticks      (fontsize=16)
            pl.yticks      (fontsize=16)
            pl.grid        ()
            pl.legend      (fontsize=24)
            pl.xlabel      ('alpha', fontsize=20)
            pl.ylabel      ('ECP', fontsize=20)
            pl.savefig     (self._save_path + self._function_name + "/3.1 - ECP assuming reliability" + dataset + ".png")
            pl.show        ()
        if self._library == "plotly":
            figure = go.Figure  ()
            figure.add_trace    (go.Scatter(x=alphas, y=alphas, mode="lines"  , name="Theory", line=dict(width=1, color="green")))
            figure.add_trace    (go.Scatter(x=alphas, y=ECP   , mode='lines'  , name="ECP"   , line=dict(width=1, color="blue" )))
            figure.add_trace    (go.Scatter(x=np.concatenate([alphas, alphas[::-1]]), y=np.concatenate([alphas - std, (alphas + std)[::-1]]), fill='toself', fillcolor='rgba(0,0,0,0.2)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name='std'))
            figure.update_xaxes (tickprefix="<b>", ticksuffix="<b><br>")
            figure.update_yaxes (tickprefix="<b>", ticksuffix="<b><br>")
            figure.update_layout(xaxis_title="alpha", yaxis_title="ECP",width=1100, height=700)
            figure.show         ()

    def plot_predictive_variance_adequacy_distribution_assuming_reliability(self, dataset):
        PVA                =     self._evaluation[dataset]["Reliability"]["PVA"                         ]
        quantile_PVA       =     self._evaluation[dataset]["Reliability"]["Quantile PVA"                ]
        n                  = len(self._evaluation[dataset]["Reliability"]["Normalized prediction errors"])
        abscissa           = np.linspace(0, 3, 1000)
        gamma_pdf          = gamma.pdf(abscissa, n / 2, 0, 2 / n)
        lower_fill_between = [0 for _ in abscissa]
        upper_fill_between = [y * (x <= PVA) for (x, y) in zip(abscissa, gamma_pdf)]
        if self._library == "matplotlib":
            pl.figure      (figsize=FIGSIZE)
            pl.clf         ()
            pl.plot        (abscissa, gamma_pdf, color="black", label="Gamma(n/2, 2/n) distribution")
            pl.fill_between(abscissa, lower_fill_between, upper_fill_between, alpha=0.4, color="lightcoral")
            pl.axvline     (PVA                , color="red"  , label="Quantile : " + str(np.round(100 * quantile_PVA, 2)) + "%", linestyle='--')
            pl.xticks      (fontsize=16)
            pl.yticks      (fontsize=16)
            pl.grid        ()
            pl.legend      (fontsize=24)
            pl.xlabel      ('Value', fontsize=20)
            pl.ylabel      ('Density', fontsize=20)
            pl.savefig     (self._save_path + self._function_name + "/3.2 - PVA assuming reliability" + dataset + ".png")
            pl.show        ()
        if self._library == "plotly":
            figure = go.Figure  ()
            figure.add_trace    (go.Scatter(x=abscissa, y=gamma_pdf, mode='lines', name="Gamma(n/2, 2/n) distribution", line=dict(color='black')))
            figure.add_trace    (go.Scatter(x=[PVA, PVA], y=[0, max(gamma_pdf)], mode='lines', name="Quantile : " + str(np.round(100 * quantile_PVA, 2)) + "%", line=dict(color='red', dash='dash')))
            figure.add_trace    (go.Scatter(x=np.concatenate([abscissa, abscissa[::-1]]), y=np.concatenate([lower_fill_between, upper_fill_between[::-1]]), fill='toself', fillcolor='rgba(0,0,0,0.2)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name="Quantile"))
            figure.update_xaxes (tickprefix="<b>", ticksuffix="<b><br>")
            figure.update_yaxes (tickprefix="<b>", ticksuffix="<b><br>")
            figure.update_layout(xaxis_title="Value", yaxis_title="Density", width=1100, height=700)
            figure.show         ()

    def plot_normalized_prediction_errors(self, dataset):
        normalized_prediction_errors = self._evaluation[dataset]["Reliability"]["Normalized prediction errors"]
        abscissa                     = np.linspace(-4, 4, 1000)
        theoretical_pdf              = norm.pdf(abscissa, 0, 1)
        if self._library == "matplotlib":
            pl.figure(figsize=FIGSIZE)
            pl.clf    ()
            pl.hist   (normalized_prediction_errors, density=True, alpha=0.5, color="green", label="Normalized prediction errors")
            pl.plot   (abscissa, theoretical_pdf, color="blue", label='Theoretical Gaussian PDF' if dataset == "Test" else "")
            pl.xticks (fontsize=16)
            pl.yticks (fontsize=16)
            pl.grid   ()
            pl.legend (fontsize=24)
            pl.xlabel ('Value', fontsize=20)
            pl.ylabel ('Density', fontsize=20)
            pl.savefig(self._save_path + self._function_name + "/3.3 - Normalized prediction errors " + dataset + ".png")
            pl.show   ()
        if self._library == "plotly":
            hist_data = np.histogram(normalized_prediction_errors, density=True)
            figure = go.Figure      ()
            figure.add_trace        (go.Bar(x=hist_data[1][:-1], y=hist_data[0], width=(hist_data[1][1] - hist_data[1][0]), name="Correlated errors", opacity=0.5, marker_color="green"))
            figure.add_trace        (go.Scatter(x=abscissa, y=theoretical_pdf, mode='lines', name='Theoretical Gaussian PDF', line=dict(color="blue")))
            figure.update_xaxes     (tickprefix="<b>", ticksuffix="<b><br>")
            figure.update_yaxes     (tickprefix="<b>", ticksuffix="<b><br>")
            figure.update_layout    (xaxis_title="Value", yaxis_title="Density", width=1100, height=700)
            figure.show             ()

    def plot_whitened_data_assuming_gaussianity(self, dataset):
        Y_observed_whitened = self._evaluation[dataset]["Gaussianity"]["Y whitened"]
        abscissa            = np.linspace(-4, 4, 1000)
        theoretical_pdf     = norm.pdf(abscissa, 0, 1)
        if self._library == "matplotlib":
            pl.figure(figsize=FIGSIZE)
            pl.clf    ()
            pl.hist   (Y_observed_whitened, density=True, alpha=0.5, color="green", label="Whitened data")
            pl.plot   (abscissa, theoretical_pdf, color="blue", label='Theoretical Gaussian PDF')
            pl.xticks (fontsize=16)
            pl.yticks (fontsize=16)
            pl.grid   ()
            pl.legend (fontsize=24)
            pl.xlabel ('Value', fontsize=20)
            pl.ylabel ('Density', fontsize=20)
            pl.savefig(self._save_path + self._function_name + "/4.1 - Whitened data assuming Gaussianity" + dataset + ".png")
            pl.show   ()
        if self._library == "plotly":
            hist_data = np.histogram(Y_observed_whitened, density=True)
            figure    = go.Figure   ()
            figure.add_trace        (go.Bar(x=hist_data[1][:-1], y=hist_data[0], width=(hist_data[1][1] - hist_data[1][0]), name="Whitened data", opacity=0.5, marker_color="green"))
            figure.add_trace        (go.Scatter(x=abscissa, y=theoretical_pdf, mode='lines', name='Theoretical Gaussian PDF', line=dict(color="blue")))
            figure.update_xaxes     (tickprefix="<b>", ticksuffix="<b><br>")
            figure.update_yaxes     (tickprefix="<b>", ticksuffix="<b><br>")
            figure.update_layout    (xaxis_title="Value", yaxis_title="Density", width=1100, height=700)
            figure.show             ()

    def plot_mahalanobis_distance_distribution_assuming_gaussianity(self, dataset):
        mahalanobis_distance          =     self._evaluation[dataset]["Gaussianity"]["Mahalanobis distance"         ]
        quantile_mahalanobis_distance =     self._evaluation[dataset]["Gaussianity"]["Quantile Mahalanobis distance"]
        log_likelihood                =     self._evaluation[dataset]["Gaussianity"]["Log likelihood"               ]
        n                             = len(self._evaluation[dataset]["Gaussianity"]["Y whitened"                   ])
        abscissa                      = np.linspace(0, 3 * n, 1000)
        chi2_pdf                      = chi2.pdf(abscissa, n)
        lower_fill_between            = [0 for _ in abscissa]
        upper_fill_between            = [y * (x <= mahalanobis_distance) for (x, y) in zip(abscissa, chi2_pdf)]
        if self._library == "matplotlib":
            pl.figure      (figsize=FIGSIZE)
            pl.clf         ()
            pl.plot        (abscissa, chi2_pdf, color="black", label=f"Chi-squared({n})")
            pl.fill_between(abscissa, lower_fill_between, upper_fill_between, alpha=0.4, color="lightcoral")
            pl.axvline     (mahalanobis_distance, color="red", linestyle='--', label=f"Quantile : {np.round(100 * quantile_mahalanobis_distance, 2)}%")
            pl.xticks      (fontsize=16)
            pl.yticks      (fontsize=16)
            pl.grid        ()
            pl.legend      (fontsize=24)
            pl.xlabel      ('Value', fontsize=20)
            pl.ylabel      ('Density', fontsize=20)
            pl.title       ("Log-likelihood = " + str(log_likelihood))
            pl.savefig     (self._save_path + self._function_name + "/4.2 - Mahalanobis distance distribution assuming Gaussianity" + dataset + ".png")
            pl.show        ()
        if self._library == "plotly":
            figure = go.Figure  ()
            figure.add_trace    (go.Scatter(x=abscissa, y=chi2_pdf, mode='lines', name=f"Chi-squared({n})", line=dict(color='black')))
            figure.add_trace    (go.Scatter(x=[mahalanobis_distance, mahalanobis_distance], y=[0, max(chi2_pdf)], mode='lines', name=f"Quantile : {np.round(100 * quantile_mahalanobis_distance, 2)}%", line=dict(color='red', dash='dash')))
            figure.add_trace    (go.Scatter(x=np.concatenate([abscissa, abscissa[::-1]]), y=np.concatenate([lower_fill_between, upper_fill_between[::-1]]), fill='toself', fillcolor='rgba(0,0,0,0.2)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name="Quantile"))
            figure.update_xaxes (tickprefix="<b>", ticksuffix="<b><br>")
            figure.update_yaxes (tickprefix="<b>", ticksuffix="<b><br>")
            figure.update_layout(xaxis_title="Value", yaxis_title="Density", title="Log-likelihood = " + str(log_likelihood), width=1100, height=700)
            figure.show         ()

    def plot_equivalent_normal_observation_assuming_gaussianity(self, dataset):
        quantile_mahalanobis_distance = self._evaluation[dataset]["Gaussianity"]["Quantile Mahalanobis distance"]
        equivalent_normal_observation = self._evaluation[dataset]["Gaussianity"]["Equivalent normal observation"]
        abscissa_limit                = max(5, equivalent_normal_observation)
        abscissa                      = np.linspace(-abscissa_limit, abscissa_limit, 1000)
        normal_pdf                    = norm.pdf(abscissa, 0, 1)
        lower_fill_between            = [0 for _ in abscissa]
        upper_fill_between            = [y * (abs(x) <= equivalent_normal_observation) for (x, y) in zip(abscissa, normal_pdf)]
        if self._library == "matplotlib":
            pl.figure      (figsize=FIGSIZE)
            pl.clf         ()
            pl.plot        (abscissa, normal_pdf, color="black", label="N(0,1)")
            pl.fill_between(abscissa, lower_fill_between, upper_fill_between, alpha=0.4, color="lightcoral", label="Data")
            pl.axvline     (  equivalent_normal_observation, color="red", linestyle='--', label=f'Quantile: {np.round(100 * quantile_mahalanobis_distance, 2)}%')
            pl.axvline     (- equivalent_normal_observation, color="red", linestyle='--')
            pl.xticks      (fontsize=16)
            pl.yticks      (fontsize=16)
            pl.grid        ()
            pl.legend      (fontsize=24)
            pl.xlabel      ('Value', fontsize=20)
            pl.ylabel      ('Density', fontsize=20)
            pl.savefig     (self._save_path + self._function_name + "/4.3 - Equivalent normal observation assuming Gaussianity" + dataset + ".png")
            pl.show        ()
        if self._library == "plotly":
            figure = go.Figure  ()
            figure.add_trace    (go.Scatter(x=abscissa, y=normal_pdf, mode='lines', name="N(0,1)", line=dict(color='black')))
            figure.add_trace    (go.Scatter(x=np.concatenate([abscissa, abscissa[::-1]]), y=np.concatenate([lower_fill_between, upper_fill_between[::-1]]), fill='toself', fillcolor='rgba(0,0,0,0.2)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name="Data"))
            figure.add_trace    (go.Scatter(x=[  equivalent_normal_observation,   equivalent_normal_observation], y=[0, max(normal_pdf)], mode='lines', name=f'Quantile: {np.round(100 * quantile_mahalanobis_distance, 2)}%', line=dict(color='red', dash='dash')))
            figure.add_trace    (go.Scatter(x=[- equivalent_normal_observation, - equivalent_normal_observation], y=[0, max(normal_pdf)], mode='lines',                                                                        line=dict(color='red', dash='dash')))
            figure.update_xaxes (tickprefix="<b>", ticksuffix="<b><br>")
            figure.update_yaxes (tickprefix="<b>", ticksuffix="<b><br>")
            figure.update_layout(xaxis_title="Value", yaxis_title="Density", width=1100, height=700)
            figure.show         ()

    def plot_empirical_coverage_probability_assuming_gaussianity(self, dataset):
        alphas =     self._evaluation[dataset]["Gaussianity"]["alphas"]
        ECP    =     self._evaluation[dataset]["Gaussianity"]["ECP"   ]
        if self._library == "matplotlib":
            pl.figure      (figsize=FIGSIZE)
            pl.clf         ()
            pl.plot        (alphas, alphas, label="Theory", color="green", linewidth=1)
            pl.plot        (alphas, ECP   , label="ECP"   , color="blue" , linewidth=1)
            pl.xticks      (fontsize=16)
            pl.yticks      (fontsize=16)
            pl.grid        ()
            pl.legend      (fontsize=24)
            pl.xlabel      ('alpha', fontsize=20)
            pl.ylabel      ('ECP', fontsize=20)
            pl.savefig     (self._save_path + self._function_name + "/4.4 - ECP assuming Gaussianity" + dataset + ".png")
            pl.show        ()
        if self._library == "plotly":
            figure = go.Figure  ()
            figure.add_trace    (go.Scatter(x=alphas, y=alphas, mode="lines"  , name="Theory", line=dict(width=1, color="green")))
            figure.add_trace    (go.Scatter(x=alphas, y=ECP   , mode='lines'  , name="ECP"   , line=dict(width=1, color="blue" )))
            figure.update_xaxes (tickprefix="<b>", ticksuffix="<b><br>")
            figure.update_yaxes (tickprefix="<b>", ticksuffix="<b><br>")
            figure.update_layout(xaxis_title="alpha", yaxis_title="ECP",width=1100, height=700)
            figure.show         ()

    def plot_predictive_variance_adequacy_assuming_gaussianity(self, dataset):
        PVA                =     self._evaluation[dataset]["Gaussianity"]["PVA"         ]
        quantile_PVA       =     self._evaluation[dataset]["Gaussianity"]["Quantile PVA"]
        lambdas            =     self._evaluation[dataset]["Gaussianity"]["Lambdas"     ]
        n                  = len(lambdas)
        abscissa           = np.linspace(0, 3, 1000)
        generalized_chi2_pdf = gamma.pdf(abscissa, n / 2, 0, 2 / n) # wrong, we need the generalized chi2 package
        lower_fill_between = [0 for _ in abscissa]
        upper_fill_between = [y * (x <= PVA) for (x, y) in zip(abscissa, generalized_chi2_pdf)]
        if self._library == "matplotlib":
            pl.figure      (figsize=FIGSIZE)
            pl.clf         ()
            pl.plot        (abscissa, generalized_chi2_pdf, color="black", label="Generalized chi2 distribution")
            pl.fill_between(abscissa, lower_fill_between, upper_fill_between, alpha=0.4, color="lightcoral")
            pl.axvline     (PVA                , color="red"  , label="Quantile : " + str(np.round(100 * quantile_PVA, 2)) + "%", linestyle='--')
            pl.xticks      (fontsize=16)
            pl.yticks      (fontsize=16)
            pl.grid        ()
            pl.legend      (fontsize=24)
            pl.xlabel      ('Value', fontsize=20)
            pl.ylabel      ('Density', fontsize=20)
            pl.savefig     (self._save_path + self._function_name + "/4.5 - PVA assuming Gaussianity" + dataset + ".png")
            pl.show        ()
        if self._library == "plotly":
            figure = go.Figure  ()
            figure.add_trace    (go.Scatter(x=abscissa, y=generalized_chi2_pdf, mode='lines', name="Generalized chi2 distribution", line=dict(color='black')))
            figure.add_trace    (go.Scatter(x=[PVA, PVA], y=[0, max(generalized_chi2_pdf)], mode='lines', name="Quantile : " + str(np.round(100 * quantile_PVA, 2)) + "%", line=dict(color='red', dash='dash')))
            figure.add_trace    (go.Scatter(x=np.concatenate([abscissa, abscissa[::-1]]), y=np.concatenate([lower_fill_between, upper_fill_between[::-1]]), fill='toself', fillcolor='rgba(0,0,0,0.2)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name="Quantile"))
            figure.update_xaxes (tickprefix="<b>", ticksuffix="<b><br>")
            figure.update_yaxes (tickprefix="<b>", ticksuffix="<b><br>")
            figure.update_layout(xaxis_title="Value", yaxis_title="Density", width=1100, height=700)
            figure.show         ()

    def plot_normalized_prediction_errors_uncorrelated_assuming_gaussianity(self, dataset):
        uncorrelated_errors            = self._evaluation[dataset]["Gaussianity"]["Uncorrelated errors"]
        lambdas                        = self._evaluation[dataset]["Gaussianity"]["Lambdas"            ]
        uncorrelated_normalized_errors = uncorrelated_errors / np.sqrt(lambdas)
        abscissa                       = np.linspace(-4, 4, 1000)
        theoretical_pdf                = norm.pdf(abscissa, 0, 1)
        height, bins                   = np.histogram(uncorrelated_normalized_errors, density=True)
        x                              = [(bins[i + 1] + bins[i]) / 2 for i in range(len(height))]
        width                          = [ bins[i + 1] - bins[i]      for i in range(len(height))]
        all_sum_lambdas                = []
        for i in range(len(height)):
            sum_lambdas = 0
            for (uncorrelated_normalized_error, lambd) in zip(uncorrelated_normalized_errors, lambdas):
                if bins[i] <= uncorrelated_normalized_error < bins[i+1]:
                    sum_lambdas = sum_lambdas + lambd
            all_sum_lambdas.append(sum_lambdas)
        all_sum_lambdas = np.array(all_sum_lambdas)
        if self._library == "matplotlib":
            normalized_colors = mcolors.Normalize(vmin=all_sum_lambdas.min(), vmax=all_sum_lambdas.max())
            colormap          = cm.get_cmap('viridis')
            mapped_colors     = colormap(normalized_colors(all_sum_lambdas))
            figure, ax        = pl.subplots(figsize=FIGSIZE)
            ax.bar        (x=x, height=height, width=width, alpha=0.5, color=mapped_colors, label="Uncorrelated normalized errors")
            ax.plot       (abscissa, theoretical_pdf, color="blue", label='Theoretical Gaussian PDF')
            ax.grid       (True)
            ax.legend     (fontsize=24)
            ax.set_xlabel ("Value"  , fontsize=20)
            ax.set_ylabel ("Density", fontsize=20)
            cbar = figure.colorbar(cm.ScalarMappable(cmap=colormap, norm=normalized_colors), ax=ax)
            cbar.set_label("Sum of lambdas", fontsize=20)
            pl.savefig    (self._save_path + self._function_name + "/4.6 - Normalized prediction errors uncorrelated assuming Gaussianity" + dataset + ".png")
            pl.show       ()
        if self._library == "plotly":
            normalized_colors = (all_sum_lambdas - all_sum_lambdas.min()) / (all_sum_lambdas.max() - all_sum_lambdas.min())
            colormap          = px.colors.sequential.Viridis
            mapped_colors     = [colormap[int(c * (len(colormap) - 1))] for c in normalized_colors]
            figure = go.Figure  ()
            figure.add_trace    (go.Bar(x=x, y=height, width=width, name="Uncorrelated normalized errors", opacity=0.5, marker_color=mapped_colors, hoverinfo='x+y+text', text=['Sum of lambdas: {:.2f}'.format(c) for c in all_sum_lambdas]))
            figure.add_trace    (go.Scatter(x=abscissa, y=theoretical_pdf, mode='lines', name='Theoretical Gaussian PDF', line=dict(color="blue")))
            figure.update_xaxes (tickprefix="<b>", ticksuffix="<b><br>")
            figure.update_yaxes (tickprefix="<b>", ticksuffix="<b><br>")
            figure.update_layout(xaxis_title="Value", yaxis_title="Density", showlegend=True, width=1100, height=700)
            figure.show         ()

    def plot_lambdas_empirical_distribution(self, dataset):
        lambdas = self._evaluation[dataset]["Gaussianity"]["Lambdas"]
        if self._library == "matplotlib":
            pl.figure (figsize=FIGSIZE)
            pl.clf    ()
            pl.hist   (lambdas, density=False, alpha=0.5, color="green", label="Error weights")
            pl.xticks (fontsize=16)
            pl.yticks (fontsize=16)
            pl.grid   ()
            pl.legend (fontsize=24)
            pl.xlabel ('Value', fontsize=20)
            pl.ylabel ('Density', fontsize=20)
            pl.savefig(self._save_path + self._function_name + "/4.6 - Lambdas empirical distribution " + dataset + ".png")
            pl.show   ()
        if self._library == "plotly":
            hist_data = np.histogram(lambdas, density=False)
            figure    = go.Figure   ()
            figure.add_trace        (go.Bar(x=hist_data[1][:-1], y=hist_data[0], width=(hist_data[1][1] - hist_data[1][0]), name="Error weights", opacity=0.5, marker_color="green"))
            figure.update_xaxes     (tickprefix="<b>", ticksuffix="<b><br>")
            figure.update_yaxes     (tickprefix="<b>", ticksuffix="<b><br>")
            figure.update_layout    (xaxis_title="Value", yaxis_title="Density", width=1100, height=700)
            figure.show             ()

    def show_reliability_metrics(self, dataset):
        n    = self._evaluation[dataset]["Reliability metrics"]["N"   ]
        PVA  = self._evaluation[dataset]["Reliability metrics"]["PVA" ]
        ECP  = self._evaluation[dataset]["Reliability metrics"]["ECP" ][int(len(self._evaluation[dataset]["Reliability metrics"]["ECP"]) * self._alpha)]
        PMIW = self._evaluation[dataset]["Reliability metrics"]["PMIW"]
        IAE  = self._evaluation[dataset]["Reliability metrics"]["IAE" ]
        print(                                      "PVA       : " + str(PVA) + " - n = " + str(n))
        print(str(int(np.round(100*self._alpha))) + "% ECP : "   + str(ECP))
        print(                                      "IAE       : " + str(IAE))
        print(                                      "PMIW      : " + str(PMIW))

    def show_performance_metrics(self, dataset):
        MSE = self._evaluation[dataset]["Performance metrics"]["MSE"]
        Q2  = self._evaluation[dataset]["Performance metrics"]["Q2" ]
        print("MSE : " + str(MSE))
        print("Q2  : " + str(Q2 ))

    def show_hybrid_metrics(self, dataset):
        NLPD = self._evaluation[dataset]["Hybrid metrics"]["NLPD"]
        CRPS = self._evaluation[dataset]["Hybrid metrics"]["CRPS"]
        IS   = self._evaluation[dataset]["Hybrid metrics"]["IS"  ][int(len(self._evaluation[dataset]["Hybrid metrics"]["IS" ]) * self._alpha)]
        print(                                      "NLPD     : " + str(NLPD))
        print(                                      "CRPS     : " + str(CRPS))
        print(str(int(np.round(100*self._alpha))) + "% IS : "   + str(IS  ))
