import os
import numpy as np
from scipy.stats import norm, chi2, gamma
import matplotlib.pyplot as pl
import plotly.graph_objects as go
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import plotly.express as px
from plotly.subplots import make_subplots
from Code.data_generation.samplers import f


def index_of_closest_element(x, t):
    best_index, best_distance = 0, np.inf
    for index, y in enumerate(t):
        if np.abs(y - x) < best_distance:
            best_index, best_distance = index, np.abs(y - x)
    return best_index


def plot_hyperparameters_optimization_history(hyperparameters_history, titles, figsize, library, save_path):
    if library == "matplotlib":
        fig, axs = pl.subplots(int(np.ceil(len(hyperparameters_history) / 2)), 2, figsize=figsize)
        if int(np.ceil(len(hyperparameters_history) / 2)) == 1:
            axs = [axs]
        for plot_index in range(len(hyperparameters_history)):
            axs[int(np.floor(plot_index / 2))][plot_index % 2].plot([i + 1 for i in range(len(hyperparameters_history[0]))], hyperparameters_history[plot_index])
            axs[int(np.floor(plot_index / 2))][plot_index % 2].set_title(titles[plot_index])
            axs[int(np.floor(plot_index / 2))][plot_index % 2].set_xlabel("Iterations")
            axs[int(np.floor(plot_index / 2))][plot_index % 2].set_ylabel("HP value")
            axs[int(np.floor(plot_index / 2))][plot_index % 2].grid(True)
        pl.tight_layout()
        pl.savefig(save_path)
        pl.show()
    if library == "plotly":
        figure = make_subplots(rows=int(np.ceil(len(hyperparameters_history) / 2)), cols=2, subplot_titles=titles)
        for plot_index in range(len(hyperparameters_history)):
            figure.add_trace(go.Scatter(x=[i + 1 for i in range(len(hyperparameters_history[0]))], y=hyperparameters_history[plot_index], mode='lines', name=titles[plot_index]), row=1 + int(np.floor(plot_index) / 2), col=1 + plot_index % 2)
            figure.update_yaxes(title_text="HP value", row=1 + int(np.floor(plot_index / 2)), col=1 + plot_index % 2, showgrid=True)
        figure.update_xaxes(title_text="Iterations", showgrid=True)
        figure.update_layout(height=800, width=1100)
        figure.show()


def plot_criterion_history(criterion_history, criterion, figsize, library, save_path):
    if library == "matplotlib":
        pl.figure(figsize=figsize)
        pl.clf    ()
        pl.plot   ([i + 1 for i in range(len(criterion_history))], criterion_history, label="Criterion history", color="black", linewidth=1)
        pl.xticks (fontsize=16)
        pl.yticks (fontsize=16)
        pl.grid   ()
        pl.legend (fontsize=24)
        pl.xlabel ("Iterations", fontsize=20)
        pl.ylabel (criterion, fontsize=20)
        pl.savefig(save_path)
        pl.show   ()
    if library == "plotly":
        figure = go.Figure  ()
        figure.add_trace    (go.Scatter(x=[i + 1 for i in range(len(criterion_history))], y=criterion_history, mode='lines', name="Criterion history", line=dict(width=1, color="black" )))
        figure.update_xaxes (tickprefix="<b>", ticksuffix="<b><br>")
        figure.update_yaxes (tickprefix="<b>", ticksuffix="<b><br>")
        figure.update_layout(xaxis_title="Iterations", yaxis_title=criterion, width=1100, height=700)
        figure.show         ()


def plot_criterion_evolution_1D(X, Z, hyperparameters_0, hyperparameters_f, kernel_optimization_criterion_was_used, figsize, library, zlabel, save_path, log_scale):
    if library == "matplotlib":
        pl.figure         (figsize=figsize)
        pl.clf            ()
        if log_scale:
            pl.plot       (X, np.log(np.maximum(1e-10, Z)), color="black", linewidth=1)
            if kernel_optimization_criterion_was_used:
                pl.scatter([hyperparameters_0[0]], [np.log(Z[index_of_closest_element(hyperparameters_0[0], X)])], color="blue", marker="d", s=200, label='Start' )
                pl.scatter([hyperparameters_f[0]], [np.log(Z[index_of_closest_element(hyperparameters_f[0], X)])], color="red" , marker="d", s=200, label='Finish')
            pl.ylabel     ("Log " + zlabel, fontsize=20)
        else:
            pl.plot       (X, Z, color="black", linewidth=1)
            if kernel_optimization_criterion_was_used:
                pl.scatter([hyperparameters_0[0]], [       Z[index_of_closest_element(hyperparameters_0[0], X)] ], color="blue", marker="d", s=200, label='Start' )
                pl.scatter([hyperparameters_f[0]], [       Z[index_of_closest_element(hyperparameters_f[0], X)] ], color="red" , marker="d", s=200, label='Finish')
            pl.ylabel     (zlabel, fontsize=20)
        pl.xticks         (fontsize=16)
        pl.yticks         (fontsize=16)
        pl.grid           ()
        pl.xlabel         ("-log(rho)", fontsize=20)
        if log_scale:
            pl.savefig    (save_path)
        else:
            pl.savefig    (save_path)
        pl.show           ()
    if library == "plotly":
        figure = go.Figure()
        if log_scale:
            figure    .add_trace(go.Scatter(x=X, y=np.log(np.maximum(1e-10, Z)), mode='lines', line=dict(width=1, color="black" )))
            if kernel_optimization_criterion_was_used:
                figure.add_trace(go.Scatter(x=[hyperparameters_0[0]], y=[np.log(Z[index_of_closest_element(hyperparameters_0[0], X)])], mode='markers', marker=dict(color='blue', symbol="diamond", size=10), name="Start" ))
                figure.add_trace(go.Scatter(x=[hyperparameters_f[0]], y=[np.log(Z[index_of_closest_element(hyperparameters_f[0], X)])], mode='markers', marker=dict(color='red' , symbol="diamond", size=10), name="Finish"))
            figure.update_layout(xaxis_title="-log(rho)", yaxis_title="Log " + zlabel, width=1100, height=700)
        else:
            figure    .add_trace(go.Scatter(x=X, y=Z, mode='lines', line=dict(width=1, color="black" )))
            if kernel_optimization_criterion_was_used:
                figure.add_trace(go.Scatter(x=[hyperparameters_0[0]], y=[       Z[index_of_closest_element(hyperparameters_0[0], X)] ], mode='markers', marker=dict(color='blue', symbol="diamond", size=10), name="Start"))
                figure.add_trace(go.Scatter(x=[hyperparameters_f[0]], y=[       Z[index_of_closest_element(hyperparameters_f[0], X)] ], mode='markers', marker=dict(color='red', symbol="diamond", size=10), name="Finish"))
            figure.update_layout(xaxis_title="-log(rho)", yaxis_title=zlabel, width=1100, height=700)
        figure.update_xaxes(tickprefix="<b>", ticksuffix="<b><br>")
        figure.update_yaxes(tickprefix="<b>", ticksuffix="<b><br>")
        figure.show        ()


def plot_criterion_evolution_2D_3D(X, Y, Z, hyperparameters_0, hyperparameters_f, kernel_optimization_criterion_was_used, figsize, library, xlabel, ylabel, zlabel, save_path, log_scale):
    if library == "matplotlib":
        figure = pl.figure         (figsize=figsize)
        axes   = figure.add_subplot(111, projection='3d')
        axes.set_xlabel            (xlabel, fontsize=20)
        axes.set_ylabel            (ylabel, fontsize=20)
        if log_scale:
            axes.plot_surface      (X, Y, np.log(np.maximum(1e-10, Z)), cmap='viridis')
            axes.set_zlabel        ("Log " + zlabel, fontsize=20)
            if kernel_optimization_criterion_was_used:
                axes.scatter       ([hyperparameters_0[0]], [hyperparameters_0[1]], [0], color="blue", marker="d", s=200, label='Start' )
                axes.scatter       ([hyperparameters_f[0]], [hyperparameters_f[1]], [0], color="red" , marker="d", s=200, label='Finish')
            pl.savefig             (save_path)
        else:
            axes.plot_surface      (X, Y, Z                           , cmap='viridis')
            axes.set_zlabel        (zlabel, fontsize=20)
            if kernel_optimization_criterion_was_used:
                axes.scatter       ([hyperparameters_0[0]], [hyperparameters_0[1]], [0], color="blue", marker="d", s=200, label='Start' )
                axes.scatter       ([hyperparameters_f[0]], [hyperparameters_f[1]], [0], color="red" , marker="d", s=200, label='Finish')
            pl.savefig             (save_path)
        pl.show                    ()
    if library == "plotly":
        if log_scale:
            figure = go.Figure  (data=[go.Surface(z=np.log(np.maximum(1e-10, Z)), x=X, y=Y, colorscale='Viridis')])
            figure.update_layout(scene=dict(xaxis_title=xlabel, yaxis_title=ylabel, zaxis_title="Log " + zlabel), width=1100, height=700)
        else:
            figure = go.Figure  (data=[go.Surface(z=                         Z  , x=X, y=Y, colorscale='Viridis')])
            figure.update_layout(scene=dict(xaxis_title=xlabel, yaxis_title=ylabel, zaxis_title=zlabel), width=1100, height=700)
        if kernel_optimization_criterion_was_used:
            figure.add_trace    (go.Scatter3d(x=[hyperparameters_0[0]], y=[hyperparameters_0[1]], z=[0], mode='markers', marker=dict(size=10, color="blue", symbol="diamond"), name='Start' ))
            figure.add_trace    (go.Scatter3d(x=[hyperparameters_f[0]], y=[hyperparameters_f[1]], z=[0], mode='markers', marker=dict(size=10, color="red" , symbol="diamond"), name='Finish'))
        figure.show             ()


def plot_criterion_evolution_2D_heatmap(X, Y, Z, hyperparameters_0, hyperparameters_f, kernel_optimization_criterion_was_used, figsize, library, xlabel, ylabel, zlabel, save_path, log_scale):
    if library == "matplotlib":
        pl.figure     (figsize=figsize)
        pl.clf        ()
        if log_scale:
            pl.imshow (np.log(np.maximum(1e-10, Z)), extent=(np.min(X), np.max(X), np.min(Y), np.max(Y)), origin='lower', cmap='viridis', interpolation='nearest')
        else:
            pl.imshow (Z                           , extent=(np.min(X), np.max(X), np.min(Y), np.max(Y)), origin='lower', cmap='viridis', interpolation='nearest')
        if kernel_optimization_criterion_was_used:
            pl.scatter([hyperparameters_0[0]], [hyperparameters_0[1]], color="blue", marker="d", s=200, label='Start' )
            pl.scatter([hyperparameters_f[0]], [hyperparameters_f[1]], color="red" , marker="d", s=200, label='Finish')
        pl.colorbar   ()
        pl.xticks     (fontsize=16)
        pl.yticks     (fontsize=16)
        pl.xlabel     (xlabel)
        pl.ylabel     (ylabel)
        if log_scale:
            pl.savefig(save_path)
        else:
            pl.savefig(save_path)
        pl.show       ()
    if library == "plotly":
        if log_scale:
            figure = px.imshow  (np.log(np.maximum(1e-10, Z)), x=X[0], y=Y.T[0], labels={'x': xlabel, 'y': ylabel, 'color': "Log " + zlabel})
            figure.update_layout(scene=dict(xaxis_title=xlabel, yaxis_title=ylabel, zaxis_title='Log ' + zlabel), width=800, height=800)
        else:
            figure = px.imshow  (Z                           , x=X[0], y=Y.T[0], labels={'x': xlabel, 'y': ylabel, 'color':          zlabel})
            figure.update_layout(scene=dict(xaxis_title=xlabel, yaxis_title=ylabel, zaxis_title=zlabel), width=800, height=800)
        if kernel_optimization_criterion_was_used:
            figure.add_trace    (go.Scatter(x=[hyperparameters_0[0]], y=[hyperparameters_0[1]], mode='markers', marker=dict(color='blue', symbol="diamond", size=10), name="Start" ))
            figure.add_trace    (go.Scatter(x=[hyperparameters_f[0]], y=[hyperparameters_f[1]], mode='markers', marker=dict(color='red' , symbol="diamond", size=10), name="Finish"))
        figure.update_yaxes     (autorange='reversed')
        figure.show             ()


def plot_negative_log_likelihood_evolution_nD(Z, hyperparameters_0, hyperparameters_f, titles, kernel_optimization_criterion_was_used, figsize, library, zlabel, save_path, log_scale):
    if library == "matplotlib":
        fig, axs = pl.subplots(int(np.ceil(len(Z) / 2)), 2, figsize=figsize)
        for plot_index in range(len(Z)):
            key = list(Z.keys())[plot_index]
            if log_scale:
                axs[int(np.floor(plot_index / 2))][plot_index % 2].plot(Z[key]["Abscissas"], np.log(np.maximum(1e-10, Z[key]["Negative log likelihoods"])))
                if kernel_optimization_criterion_was_used:
                    axs[int(np.floor(plot_index / 2))][plot_index % 2].scatter   ([hyperparameters_0[plot_index]], [np.log(Z[key]["Negative log likelihoods"][index_of_closest_element(hyperparameters_0[plot_index], Z[key]["Abscissas"])])], color="blue", marker="d", s=200, label='Start' )
                    axs[int(np.floor(plot_index / 2))][plot_index % 2].scatter   ([hyperparameters_f[plot_index]], [np.log(Z[key]["Negative log likelihoods"][index_of_closest_element(hyperparameters_f[plot_index], Z[key]["Abscissas"])])], color="red" , marker="d", s=200, label='Finish')
                axs    [int(np.floor(plot_index / 2))][plot_index % 2].set_ylabel("Log " + zlabel)
            else:
                axs    [int(np.floor(plot_index / 2))][plot_index % 2].plot(Z[key]["Abscissas"], Z[key]["Negative log likelihoods"])
                if kernel_optimization_criterion_was_used:
                    axs[int(np.floor(plot_index / 2))][plot_index % 2].scatter   ([hyperparameters_0[plot_index]], [Z[key]["Negative log likelihoods"][index_of_closest_element(hyperparameters_0[plot_index], Z[key]["Abscissas"])]], color="blue", marker="d", s=200, label='Start' )
                    axs[int(np.floor(plot_index / 2))][plot_index % 2].scatter   ([hyperparameters_f[plot_index]], [Z[key]["Negative log likelihoods"][index_of_closest_element(hyperparameters_f[plot_index], Z[key]["Abscissas"])]], color="red" , marker="d", s=200, label='Finish')
                axs    [int(np.floor(plot_index / 2))][plot_index % 2].set_ylabel(zlabel)
            axs        [int(np.floor(plot_index / 2))][plot_index % 2].set_title (titles[plot_index])
            axs        [int(np.floor(plot_index / 2))][plot_index % 2].set_xlabel("HP value")
            axs        [int(np.floor(plot_index / 2))][plot_index % 2].grid      (True)
        pl.tight_layout()
        if log_scale:
            pl.savefig(save_path)
        else:
            pl.savefig(save_path)
        pl.show()
    if library == "plotly":
        figure = make_subplots(rows=int(np.ceil(len(Z) / 2)), cols=2, subplot_titles=titles)
        for plot_index in range(len(Z)):
            key = list(Z.keys())[plot_index]
            if log_scale:
                figure    .add_trace(go.Scatter(x=Z[key]["Abscissas"], y=np.log(np.maximum(1e-10, Z[key]["Negative log likelihoods"])), mode='lines', name=titles[plot_index]), row=1 + int(np.floor(plot_index) / 2), col=1 + plot_index % 2)
                if kernel_optimization_criterion_was_used:
                    figure.add_trace(go.Scatter(x=[hyperparameters_0[plot_index]], y=[np.log(Z[key]["Negative log likelihoods"][index_of_closest_element(hyperparameters_0[plot_index], Z[key]["Abscissas"])])], mode='markers', marker=dict(color='blue', symbol="diamond", size=10), name="Start" ), row=1 + int(np.floor(plot_index / 2)), col=1 + plot_index % 2)
                    figure.add_trace(go.Scatter(x=[hyperparameters_f[plot_index]], y=[np.log(Z[key]["Negative log likelihoods"][index_of_closest_element(hyperparameters_f[plot_index], Z[key]["Abscissas"])])], mode='markers', marker=dict(color='red' , symbol="diamond", size=10), name="Finish"), row=1 + int(np.floor(plot_index / 2)), col=1 + plot_index % 2)
                figure.update_yaxes (title_text="Log " + zlabel, row=1 + int(np.floor(plot_index / 2)), col=1 + plot_index % 2, showgrid=True)
            else:
                figure    .add_trace(go.Scatter(x=Z[key]["Abscissas"], y=Z[key]["Negative log likelihoods"], mode='lines', name=titles[plot_index]), row=1 + int(np.floor(plot_index) / 2), col=1 + plot_index % 2)
                if kernel_optimization_criterion_was_used:
                    figure.add_trace(go.Scatter(x=[hyperparameters_0[plot_index]], y=[Z[key]["Negative log likelihoods"][index_of_closest_element(hyperparameters_0[plot_index], Z[key]["Abscissas"])]], mode='markers', marker=dict(color='blue', symbol="diamond", size=10), name="Start" ), row=1 + int(np.floor(plot_index / 2)), col=1 + plot_index % 2)
                    figure.add_trace(go.Scatter(x=[hyperparameters_f[plot_index]], y=[Z[key]["Negative log likelihoods"][index_of_closest_element(hyperparameters_f[plot_index], Z[key]["Abscissas"])]], mode='markers', marker=dict(color='red' , symbol="diamond", size=10), name="Finish"), row=1 + int(np.floor(plot_index / 2)), col=1 + plot_index % 2)
                figure.update_yaxes (title_text=zlabel, row=1 + int(np.floor(plot_index / 2)), col=1 + plot_index % 2, showgrid=True)
        figure.update_xaxes (title_text="HP value", showgrid=True)
        figure.update_layout(height=800, width=1100)
        figure.show         ()


def plot_criterion_evolution_nD(Z, hyperparameters_0, hyperparameters_f, titles, kernel_optimization_criterion_was_used, figsize, library, zlabel, save_path, log_scale):
    if library == "matplotlib":
        fig, axs = pl.subplots(int(np.ceil(len(Z) / 2)), 2, figsize=figsize)
        for plot_index in range(len(Z)):
            if log_scale:
                axs    [int(np.floor(plot_index / 2))][plot_index % 2].plot(Z["Log inv rho " + str(plot_index)]["Log inv rhos"], np.log(np.maximum(1e-10, Z["Log inv rho " + str(plot_index)][zlabel + "s"])))
                if kernel_optimization_criterion_was_used:
                    axs[int(np.floor(plot_index / 2))][plot_index % 2].scatter([hyperparameters_0[plot_index]], [np.log(Z["Log inv rho " + str(plot_index)][zlabel + "s"][index_of_closest_element(hyperparameters_0[plot_index], Z["Log inv rho " + str(plot_index)]["Log inv rhos"])])], color="blue", marker="d", s=200, label='Start' )
                    axs[int(np.floor(plot_index / 2))][plot_index % 2].scatter([hyperparameters_f[plot_index]], [np.log(Z["Log inv rho " + str(plot_index)][zlabel + "s"][index_of_closest_element(hyperparameters_f[plot_index], Z["Log inv rho " + str(plot_index)]["Log inv rhos"])])], color="red" , marker="d", s=200, label='Finish')
                axs    [int(np.floor(plot_index / 2))][plot_index % 2].set_ylabel("Log " + zlabel)
            else:
                axs    [int(np.floor(plot_index / 2))][plot_index % 2].plot(Z["Log inv rho " + str(plot_index)]["Log inv rhos"],                          Z["Log inv rho " + str(plot_index)][zlabel + "s"]  )
                if kernel_optimization_criterion_was_used:
                    axs[int(np.floor(plot_index / 2))][plot_index % 2].scatter([hyperparameters_0[plot_index]], [       Z["Log inv rho " + str(plot_index)][zlabel + "s"][index_of_closest_element(hyperparameters_0[plot_index], Z["Log inv rho " + str(plot_index)]["Log inv rhos"])] ], color="blue", marker="d", s=200, label='Start' )
                    axs[int(np.floor(plot_index / 2))][plot_index % 2].scatter([hyperparameters_f[plot_index]], [       Z["Log inv rho " + str(plot_index)][zlabel + "s"][index_of_closest_element(hyperparameters_f[plot_index], Z["Log inv rho " + str(plot_index)]["Log inv rhos"])] ], color="red" , marker="d", s=200, label='Finish')
                axs    [int(np.floor(plot_index / 2))][plot_index % 2].set_ylabel(zlabel)
            axs        [int(np.floor(plot_index / 2))][plot_index % 2].set_title(titles[plot_index])
            axs        [int(np.floor(plot_index / 2))][plot_index % 2].set_xlabel("HP value")
            axs        [int(np.floor(plot_index / 2))][plot_index % 2].grid(True)
        pl.tight_layout()
        if log_scale:
            pl.savefig(save_path)
        else:
            pl.savefig(save_path)
        pl.show()
    if library == "plotly":
        figure = make_subplots(rows=int(np.ceil(len(Z) / 2)), cols=2, subplot_titles=titles)
        for plot_index in range(len(Z)):
            if log_scale:
                figure    .add_trace(go.Scatter(x=Z["Log inv rho " + str(plot_index)]["Log inv rhos"], y=np.log(np.maximum(1e-10, Z["Log inv rho " + str(plot_index)][zlabel + "s"])), mode='lines', name=titles[plot_index]), row=1 + int(np.floor(plot_index) / 2), col=1 + plot_index % 2)
                if kernel_optimization_criterion_was_used:
                    figure.add_trace(go.Scatter(x=[hyperparameters_0[plot_index]], y=[np.log(Z["Log inv rho " + str(plot_index)][zlabel + "s"][index_of_closest_element(hyperparameters_0[plot_index], Z["Log inv rho " + str(plot_index)]["Log inv rhos"])])], mode='markers', marker=dict(color='blue', symbol="diamond", size=10), name="Start" ), row=1 + int(np.floor(plot_index / 2)), col=1 + plot_index % 2)
                    figure.add_trace(go.Scatter(x=[hyperparameters_f[plot_index]], y=[np.log(Z["Log inv rho " + str(plot_index)][zlabel + "s"][index_of_closest_element(hyperparameters_f[plot_index], Z["Log inv rho " + str(plot_index)]["Log inv rhos"])])], mode='markers', marker=dict(color='red' , symbol="diamond", size=10), name="Finish"), row=1 + int(np.floor(plot_index / 2)), col=1 + plot_index % 2)
                figure.update_yaxes (title_text="Log " + zlabel, row=1 + int(np.floor(len(Z) / 2)), col=1 + plot_index % 2, showgrid=True)
            else:
                figure    .add_trace(go.Scatter(x=Z["Log inv rho " + str(plot_index)]["Log inv rhos"], y=                         Z["Log inv rho " + str(plot_index)][zlabel + "s"] , mode='lines', name=titles[plot_index] ), row=1 + int(np.floor(plot_index) / 2), col=1 + plot_index % 2)
                if kernel_optimization_criterion_was_used:
                    figure.add_trace(go.Scatter(x=[hyperparameters_0[plot_index]], y=[       Z["Log inv rho " + str(plot_index)][zlabel + "s"][index_of_closest_element(hyperparameters_0[plot_index], Z["Log inv rho " + str(plot_index)]["Log inv rhos"])] ], mode='markers', marker=dict(color='blue', symbol="diamond", size=10), name="Start" ), row=1 + int(np.floor(plot_index / 2)), col=1 + plot_index % 2)
                    figure.add_trace(go.Scatter(x=[hyperparameters_f[plot_index]], y=[       Z["Log inv rho " + str(plot_index)][zlabel + "s"][index_of_closest_element(hyperparameters_f[plot_index], Z["Log inv rho " + str(plot_index)]["Log inv rhos"])] ], mode='markers', marker=dict(color='red' , symbol="diamond", size=10), name="Finish"), row=1 + int(np.floor(plot_index / 2)), col=1 + plot_index % 2)
                figure.update_yaxes (title_text=zlabel, row=1 + int(np.floor(len(Z) / 2)), col=1 + plot_index % 2, showgrid=True)
        figure.update_xaxes (title_text="HP value", showgrid=True)
        figure.update_layout(height=800, width=1100)
        figure.show         ()


def plot_1D_function_prediction(train_X, train_Y, test_X, test_Y, function_X, function_Y, LOO_mean, LOO_variance, posterior_mean, posterior_std, q_alpha, figsize, library, save_path):
    if library == "matplotlib":
        pl.figure             (figsize=figsize)
        pl.clf                ()
        pl.scatter            (train_X   .ravel(), train_Y       , label="Samples" , color="red"            , marker="x", s=200)
        pl.scatter            (test_X    .ravel(), test_Y        , label="Test"    , color="blue"           , marker="o", s=200)
        LOO_plot = pl.scatter (train_X   .ravel(), LOO_mean      , label="LOO Mean", c=np.sqrt(LOO_variance), marker="s", s=200, cmap='viridis', alpha=0.25)
        cbar     = pl.colorbar(LOO_plot)
        cbar.set_label        ("LOO std", fontsize=20)
        pl.plot               (function_X.ravel(), function_Y    , label="f"      , color="green", linewidth=1, linestyle="--")
        pl.plot               (function_X.ravel(), posterior_mean, label="GP"     , color="blue" , linewidth=1)
        pl.fill_between       (function_X.ravel(), posterior_mean - q_alpha * posterior_std, posterior_mean + q_alpha * posterior_std, alpha=0.4, color="lightskyblue", label=str(int(np.round(100*q_alpha))) + "% CI")
        pl.xticks             (fontsize=16)
        pl.yticks             (fontsize=16)
        pl.grid               ()
        pl.legend             (fontsize=24)
        pl.xlabel             ('x', fontsize=20)
        pl.ylabel             ('f(x)', fontsize=20)
        pl.savefig            (save_path)
        pl.show               ()
    if library == "plotly":
        figure = go.Figure  ()
        figure.add_trace    (go.Scatter(x=train_X   .ravel(), y=train_Y       , mode='markers', name='Samples' , marker_color="red" , marker_size=10, marker_symbol="x"     ))
        figure.add_trace    (go.Scatter(x=test_X    .ravel(), y=test_Y        , mode='markers', name='Test'    , marker_color="blue", marker_size=10, marker_symbol="circle"))
        figure.add_trace    (go.Scatter(x=train_X   .ravel(), y=LOO_mean      , mode='markers', name='LOO Mean', marker=dict(size=10, color=LOO_variance, colorscale='Viridis', showscale=True, colorbar=dict(title='LOO Variance')), marker_symbol="square"))
        figure.add_trace    (go.Scatter(x=function_X.ravel(), y=function_Y    , mode="lines"  , name="f"       , line=dict(width=1, color="green", dash="dash")))
        figure.add_trace    (go.Scatter(x=function_X.ravel(), y=posterior_mean, mode='lines'  , name='GP'      , line=dict(width=1, color="blue" )))
        figure.add_trace    (go.Scatter(x=np.concatenate([function_X.ravel(), function_X.ravel()[::-1]]), y=np.concatenate([posterior_mean - q_alpha * posterior_std, (posterior_mean + q_alpha * posterior_std)[::-1]]), fill='toself', fillcolor='rgba(0,0,0,0.2)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name=str(int(np.round(100*q_alpha))) + "% CI"))
        figure.update_xaxes (tickprefix="<b>", ticksuffix="<b><br>")
        figure.update_yaxes (tickprefix="<b>", ticksuffix="<b><br>")
        figure.update_layout(legend=dict(x=0.8, y=0.95, xanchor='left', yanchor='top'), xaxis_title="x", yaxis_title="f(x)", width=1100, height=700)
        figure.show         ()


def plot_2D_function_prediction_3D(train_X, train_Y, test_X, test_Y, function_X, function_name, sampler_parameters, gaussian_process_model, figsize, library, save_path):
    X1_function             = np.unique  (function_X[:, 0])
    X2_function             = np.unique  (function_X[:, 1])
    X_mesh, Y_mesh          = np.meshgrid(X1_function, X2_function)
    Z_mesh                  = f(X_mesh, Y_mesh, function_name, sampler_parameters)
    Z_mesh_mean, Z_mesh_std = gaussian_process_model.predict_for_plot(X_mesh, Y_mesh)
    train_X1                = train_X[:, 0]
    train_X2                = train_X[:, 1]
    test_X1                 = test_X [:, 0]
    test_X2                 = test_X [:, 1]
    if library == "matplotlib":
        figure = pl.figure         (figsize=figsize)
        axes   = figure.add_subplot(221, projection='3d')
        axes.plot_surface          (X_mesh    , Y_mesh        , Z_mesh , cmap='viridis')
        axes.scatter               (train_X1  , train_X2   , train_Y, color="red" , marker="*", s=200, label='Samples')
        axes.scatter               (test_X1   , test_X2, test_Y,         color="blue", marker="o", s=200, label='Test'   )
        axes.set_xlabel            ('x1'      , fontsize=20)
        axes.set_ylabel            ('x2'      , fontsize=20)
        axes.set_zlabel            ('f(x1,x2)', fontsize=20)
        axes.set_title             ("Function", fontsize=24)
        axes   = figure.add_subplot(222, projection='3d')
        axes.plot_surface          (X_mesh    , Y_mesh, Z_mesh_mean, cmap='viridis')
        axes.scatter               (train_X1  , train_X2, train_Y, color="red", marker="*", s=200, label='Samples')
        axes.scatter               (test_X1   , test_X2, test_Y, color="blue", marker="o", s=200, label='Test')
        axes.set_xlabel            ('x1'      , fontsize=20)
        axes.set_ylabel            ('x2'      , fontsize=20)
        axes.set_zlabel            ('f(x1,x2)', fontsize=20)
        axes.set_title             ("Mean"    , fontsize=24)
        axes   = figure.add_subplot(224, projection='3d')
        axes.plot_surface          (X_mesh, Y_mesh, Z_mesh_std, cmap='viridis')
        axes.set_xlabel            ('x1'      , fontsize=20)
        axes.set_ylabel            ('x2'      , fontsize=20)
        axes.set_zlabel            ('f(x1,x2)', fontsize=20)
        axes.set_title             ("Std"     , fontsize=24)
        pl.savefig                 (save_path)
        pl.show                    ()
    if library == "plotly":
        figure = make_subplots(rows=2, cols=2, specs=[[{'type': 'surface'}, {'type': 'surface'}], [None, {'type': 'surface'}]], subplot_titles=("Function", "Mean", "Std"))
        figure.add_trace      (go.Surface  (z=Z_mesh     , x=X_mesh  , y=Y_mesh , colorscale='Viridis', showscale=False), row=1, col=1)
        figure.add_trace      (go.Scatter3d(x=train_X1   , y=train_X2, z=train_Y, mode      ='markers', marker=dict(color='red' , size=10, symbol='x'       ), name='Samples'), row=1, col=1)
        figure.add_trace      (go.Scatter3d(x=test_X1    , y=test_X2 , z=test_Y , mode      ='markers', marker=dict(color='blue', size=10, symbol='circle'  ), name='Test'   ), row=1, col=1)
        figure.add_trace      (go.Surface  (z=Z_mesh_mean, x=X_mesh  , y=Y_mesh , colorscale='Viridis', showscale=False), row=1, col=2)
        figure.add_trace      (go.Scatter3d(x=train_X1   , y=train_X2, z=train_Y, mode      ='markers', marker=dict(color='red' , size=10, symbol='x'       ), name='Samples'), row=1, col=2)
        figure.add_trace      (go.Scatter3d(x=test_X1    , y=test_X2 , z=test_Y , mode      ='markers', marker=dict(color='blue', size=10, symbol='circle'  ), name='Test'   ), row=1, col=2)
        figure.add_trace      (go.Surface  (z=Z_mesh_std , x=X_mesh  , y=Y_mesh , colorscale='Viridis', showscale=False), row=2, col=2)
        figure.update_layout  (scene =dict(xaxis_title='x1', yaxis_title='x2', zaxis_title='f(x1,x2)'), scene2=dict(xaxis_title='x1', yaxis_title='x2', zaxis_title='y'), scene3=dict(xaxis_title='x1', yaxis_title='x2', zaxis_title='y'), height=800, width=800)
        figure.show           ()


def plot_2D_function_prediction_heatmap(train_X, test_X, function_X, function_name, sampler_parameters, gaussian_process_model, figsize, library, save_path):
    function_X1             = np.unique  (function_X[:, 0])
    function_X2             = np.unique  (function_X[:, 1])
    X_mesh, Y_mesh          = np.meshgrid(function_X1, function_X2)
    Z_mesh                  = f(X_mesh, Y_mesh, function_name, sampler_parameters)
    Z_mesh_mean, Z_mesh_std = gaussian_process_model.predict_for_plot(X_mesh, Y_mesh)
    train_X1                = train_X[:, 0]
    train_X2                = train_X[:, 1]
    test_X1                 = test_X [:, 0]
    test_X2                 = test_X [:, 1]
    if library == "matplotlib":
        figure = pl.figure         (figsize=figsize)
        axes   = figure.add_subplot(221)
        c1     = axes.imshow       (Z_mesh    , extent=(min(function_X1), max(function_X1), min(function_X2), max(function_X2)), origin='lower', cmap='viridis', interpolation='nearest')
        axes.scatter               (train_X1  , train_X2, color="red", marker="x", s=200, label='Samples')
        axes.scatter               (test_X1   , test_X2 , color="blue", marker="o", s=200, label='Test')
        axes.set_xlabel            ('x1'      , fontsize=20)
        axes.set_ylabel            ('x2'      , fontsize=20)
        axes.set_title             ("Function", fontsize=24)
        figure.colorbar            (c1        , ax=axes)
        axes   = figure.add_subplot(222)
        c2     = axes.imshow       (Z_mesh_mean, extent=(min(function_X1), max(function_X1), min(function_X2), max(function_X2)), origin='lower', cmap='viridis', interpolation='nearest')
        axes.scatter               (train_X1   , train_X2, color="red", marker="x", s=200, label='Samples')
        axes.scatter               (test_X1    , test_X2 , color="blue", marker="o", s=200, label='Test')
        axes.set_xlabel            ('x1'       , fontsize=20)
        axes.set_ylabel            ('x2'       , fontsize=20)
        axes.set_title             ("Mean"     , fontsize=24)
        figure.colorbar            (c2         , ax=axes)
        axes   = figure.add_subplot(224)
        c3     = axes.imshow       (Z_mesh_std, extent=(min(function_X1), max(function_X1), min(function_X2), max(function_X2)), origin='lower', cmap='viridis', interpolation='nearest')
        axes.set_xlabel            ('x1'      , fontsize=20)
        axes.set_ylabel            ('x2'      , fontsize=20)
        axes.set_title             ("Std"     , fontsize=24)
        figure.colorbar            (c3        , ax=axes)
        pl.savefig                 (save_path)
        pl.show                    ()
    if library == "plotly":
        figure = make_subplots(rows=2, cols=2, subplot_titles=("Function", "Mean", "Std", "Std"), specs=[[{"type": "heatmap"}, {"type": "heatmap"}], [None, {"type": "heatmap"}]], column_widths=[0.5, 0.5], row_heights=[0.5, 0.5])
        figure.add_trace      (go.Heatmap(z=Z_mesh     , x=function_X1, y=function_X2 , colorscale='Viridis'    , colorbar=dict(title='Function', len=0.3, x=0.25)), row=1, col=1)
        figure.add_trace      (go.Scatter(x=train_X1   , y=train_X2   , mode='markers', marker=dict(color='red' , size=10, symbol='x'     ), name='Samples')       , row=1, col=1)
        figure.add_trace      (go.Scatter(x=test_X1    , y=test_X2    , mode='markers', marker=dict(color='blue', size=10, symbol='circle'), name='Test'   )       , row=1, col=1)
        figure.add_trace      (go.Heatmap(z=Z_mesh_mean, x=function_X1, y=function_X2 , colorscale='Viridis'    , showscale=False)                                 , row=1, col=2)
        figure.add_trace      (go.Scatter(x=train_X1   , y=train_X2   , mode='markers', marker=dict(color='red' , size=10, symbol='x'     ), name='Samples')       , row=1, col=2)
        figure.add_trace      (go.Scatter(x=test_X1    , y=test_X2    , mode='markers', marker=dict(color='blue', size=10, symbol='circle'), name='Test'   )       , row=1, col=2)
        figure.add_trace      (go.Heatmap(z=Z_mesh_std , x=function_X1, y=function_X2 , colorscale='Viridis', colorbar=dict(title='Std', len=0.3, x=1.03))         , row=2, col=2)
        figure.update_layout  (scene=dict(xaxis_title='x1', yaxis_title='x2'), scene2=dict(xaxis_title='x1', yaxis_title='x2', zaxis_title='f(x1,x2)'), scene3=dict(xaxis_title='x1', yaxis_title='x2'), height=800, width=800)
        figure.update_xaxes   (title_text="x1", row=1, col=1)
        figure.update_yaxes   (title_text="x2", row=1, col=1)
        figure.update_xaxes   (title_text="x1", row=1, col=2)
        figure.update_yaxes   (title_text="x2", row=1, col=2)
        figure.update_xaxes   (title_text="x1", row=2, col=2)
        figure.update_yaxes   (title_text="x2", row=2, col=2)
        figure.show           ()


def plot_truth_vs_predictions(true_values, predicted_values, std, RMSE, figsize, library, save_path):
    if library == "matplotlib":
        pl.figure             (figsize=figsize)
        pl.clf                ()
        pl.errorbar           (true_values, predicted_values, yerr=std, fmt='o', ecolor='gray', elinewidth=1, capsize=3, color='white', alpha=0.3)
        scatter = pl.scatter  (true_values, predicted_values, c   =std, cmap="viridis", label='Predictions')
        pl.plot               ([min(true_values), max(true_values)], [min(true_values), max(true_values)], color='green', linestyle='--', label="Reference")
        pl.xticks             (fontsize=16)
        pl.yticks             (fontsize=16)
        pl.grid               ()
        pl.legend             (fontsize=24)
        pl.xlabel             ('Truth', fontsize=20)
        pl.ylabel             ('Prediction', fontsize=20)
        colorbar = pl.colorbar(scatter)
        colorbar.set_label    ("Std", fontsize=24)
        pl.title              ("RMSE = " + str(np.round(RMSE, 2)), fontsize=24)
        pl.savefig            (save_path)
        pl.show               ()
    if library == "plotly":
        figure = go.Figure  ()
        figure.add_trace    (go.Scatter(x=true_values, y=predicted_values, mode='markers', name='Predicted values', marker=dict(color=std, colorscale='Viridis', size=10, colorbar=dict(title="Std")), error_y=dict(type='data', array=std, visible=True, color='gray')))
        figure.add_trace    (go.Scatter(x=[min(true_values), max(true_values)], y=[min(true_values), max(true_values)], mode='lines', name='Reference', line=dict(color='green', dash='dash')))
        figure.update_xaxes (tickprefix="<b>", ticksuffix="<b><br>")
        figure.update_yaxes (tickprefix="<b>", ticksuffix="<b><br>")
        figure.update_layout(legend=dict(x=0.8, y=0.95, xanchor='left', yanchor='top'), xaxis_title="Truth", yaxis_title="Prediction", title="RMSE = " + str(np.round(RMSE, 2)), width=1100, height=700)
        figure.show         ()


def plot_empirical_coverage_probability_assuming_reliability(alphas, ECP, n, figsize, library, save_path):
    std = np.sqrt(np.array([alpha*(1-alpha)/ n for alpha in alphas]))
    if library == "matplotlib":
        pl.figure      (figsize=figsize)
        pl.clf         ()
        pl.plot        (alphas, alphas, label="Theory", color="green", linewidth=1)
        pl.plot        (alphas, ECP   , label="ECP"   , color="blue" , linewidth=1)
        pl.fill_between(alphas, alphas - std, alphas + std, alpha=0.4, color="lightskyblue", label="std")
        pl.xticks      (fontsize=16)
        pl.yticks      (fontsize=16)
        pl.grid        ()
        pl.legend      (fontsize=24)
        pl.xlabel      ('alpha', fontsize=20)
        pl.ylabel      ('ECP', fontsize=20)
        pl.savefig     (save_path)
        pl.show        ()
    if library == "plotly":
        figure = go.Figure  ()
        figure.add_trace    (go.Scatter(x=alphas, y=alphas, mode="lines"  , name="Theory", line=dict(width=1, color="green")))
        figure.add_trace    (go.Scatter(x=alphas, y=ECP   , mode='lines'  , name="ECP"   , line=dict(width=1, color="blue" )))
        figure.add_trace    (go.Scatter(x=np.concatenate([alphas, alphas[::-1]]), y=np.concatenate([alphas - std, (alphas + std)[::-1]]), fill='toself', fillcolor='rgba(0,0,0,0.2)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name='std'))
        figure.update_xaxes (tickprefix="<b>", ticksuffix="<b><br>")
        figure.update_yaxes (tickprefix="<b>", ticksuffix="<b><br>")
        figure.update_layout(xaxis_title="alpha", yaxis_title="ECP",width=1100, height=700)
        figure.show         ()


def plot_predictive_variance_adequacy_distribution_assuming_reliability(PVA, p_value_PVA, n, figsize, library, save_path):
    abscissa           = np.linspace(0, 3, 1000)
    gamma_pdf          = gamma.pdf(abscissa, n / 2, 0, 2 / n)
    lower_fill_between = [0 for _ in abscissa]
    upper_fill_between = [y * (x <= PVA) for (x, y) in zip(abscissa, gamma_pdf)]
    if library == "matplotlib":
        pl.figure      (figsize=figsize)
        pl.clf         ()
        pl.plot        (abscissa, gamma_pdf, color="black", label="Gamma(n/2, 2/n) distribution")
        pl.fill_between(abscissa, lower_fill_between, upper_fill_between, alpha=0.4, color="lightcoral")
        pl.axvline     (PVA                , color="red"  , label="p-value : " + str(np.round(100 * p_value_PVA, 2)) + "%", linestyle='--')
        pl.xticks      (fontsize=16)
        pl.yticks      (fontsize=16)
        pl.grid        ()
        pl.legend      (fontsize=24)
        pl.xlabel      ('Value', fontsize=20)
        pl.ylabel      ('Density', fontsize=20)
        pl.savefig     (save_path)
        pl.show        ()
    if library == "plotly":
        figure = go.Figure  ()
        figure.add_trace    (go.Scatter(x=abscissa, y=gamma_pdf, mode='lines', name="Gamma(n/2, 2/n) distribution", line=dict(color='black')))
        figure.add_trace    (go.Scatter(x=[PVA, PVA], y=[0, max(gamma_pdf)], mode='lines', name="p-value : " + str(np.round(100 * p_value_PVA, 2)) + "%", line=dict(color='red', dash='dash')))
        figure.add_trace    (go.Scatter(x=np.concatenate([abscissa, abscissa[::-1]]), y=np.concatenate([lower_fill_between, upper_fill_between[::-1]]), fill='toself', fillcolor='rgba(0,0,0,0.2)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name="p-value"))
        figure.update_xaxes (tickprefix="<b>", ticksuffix="<b><br>")
        figure.update_yaxes (tickprefix="<b>", ticksuffix="<b><br>")
        figure.update_layout(xaxis_title="Value", yaxis_title="Density", width=1100, height=700)
        figure.show         ()


def plot_normalized_prediction_errors(normalized_prediction_errors, figsize, library, save_path):
    abscissa        = np.linspace(-4, 4, 1000)
    theoretical_pdf = norm.pdf(abscissa, 0, 1)
    if library == "matplotlib":
        pl.figure(figsize=figsize)
        pl.clf    ()
        pl.hist   (normalized_prediction_errors, density=True, alpha=0.5, color="green", label="Normalized prediction errors")
        pl.plot   (abscissa, theoretical_pdf, color="blue", label='Theoretical Gaussian PDF')
        pl.xticks (fontsize=16)
        pl.yticks (fontsize=16)
        pl.grid   ()
        pl.legend (fontsize=24)
        pl.xlabel ('Value', fontsize=20)
        pl.ylabel ('Density', fontsize=20)
        pl.savefig(save_path)
        pl.show   ()
    if library == "plotly":
        hist_data = np.histogram(normalized_prediction_errors, density=True)
        figure = go.Figure      ()
        figure.add_trace        (go.Bar(x=hist_data[1][:-1], y=hist_data[0], width=(hist_data[1][1] - hist_data[1][0]), name="Correlated errors", opacity=0.5, marker_color="green"))
        figure.add_trace        (go.Scatter(x=abscissa, y=theoretical_pdf, mode='lines', name='Theoretical Gaussian PDF', line=dict(color="blue")))
        figure.update_xaxes     (tickprefix="<b>", ticksuffix="<b><br>")
        figure.update_yaxes     (tickprefix="<b>", ticksuffix="<b><br>")
        figure.update_layout    (xaxis_title="Value", yaxis_title="Density", width=1100, height=700)
        figure.show             ()


def plot_whitened_data_assuming_gaussianity(train_Y_whitened, figsize, library, save_path):
    abscissa            = np.linspace(-4, 4, 1000)
    theoretical_pdf     = norm.pdf(abscissa, 0, 1)
    if library == "matplotlib":
        pl.figure(figsize=figsize)
        pl.clf    ()
        pl.hist   (train_Y_whitened, density=True, alpha=0.5, color="green", label="Whitened data")
        pl.plot   (abscissa, theoretical_pdf, color="blue", label='Theoretical Gaussian PDF')
        pl.xticks (fontsize=16)
        pl.yticks (fontsize=16)
        pl.grid   ()
        pl.legend (fontsize=24)
        pl.xlabel ('Value', fontsize=20)
        pl.ylabel ('Density', fontsize=20)
        pl.savefig(save_path)
        pl.show   ()
    if library == "plotly":
        hist_data = np.histogram(train_Y_whitened, density=True)
        figure    = go.Figure   ()
        figure.add_trace        (go.Bar(x=hist_data[1][:-1], y=hist_data[0], width=(hist_data[1][1] - hist_data[1][0]), name="Whitened data", opacity=0.5, marker_color="green"))
        figure.add_trace        (go.Scatter(x=abscissa, y=theoretical_pdf, mode='lines', name='Theoretical Gaussian PDF', line=dict(color="blue")))
        figure.update_xaxes     (tickprefix="<b>", ticksuffix="<b><br>")
        figure.update_yaxes     (tickprefix="<b>", ticksuffix="<b><br>")
        figure.update_layout    (xaxis_title="Value", yaxis_title="Density", width=1100, height=700)
        figure.show             ()


def plot_mahalanobis_distance_distribution_assuming_gaussianity(mahalanobis_distance, p_value_mahalanobis_distance, log_likelihood, n, figsize, library, save_path):
    abscissa                      = np.linspace(0, 3 * n, 1000)
    chi2_pdf                      = chi2.pdf(abscissa, n)
    lower_fill_between            = [0 for _ in abscissa]
    upper_fill_between            = [y * (x <= mahalanobis_distance) for (x, y) in zip(abscissa, chi2_pdf)]
    if library == "matplotlib":
        pl.figure      (figsize=figsize)
        pl.clf         ()
        pl.plot        (abscissa, chi2_pdf, color="black", label=f"Chi-squared({n})")
        pl.fill_between(abscissa, lower_fill_between, upper_fill_between, alpha=0.4, color="lightcoral")
        pl.axvline     (mahalanobis_distance, color="red", linestyle='--', label=f"p-value : {np.round(100 * p_value_mahalanobis_distance, 2)}%")
        pl.xticks      (fontsize=16)
        pl.yticks      (fontsize=16)
        pl.grid        ()
        pl.legend      (fontsize=24)
        pl.xlabel      ('Value', fontsize=20)
        pl.ylabel      ('Density', fontsize=20)
        pl.title       ("Log-likelihood = " + str(log_likelihood))
        pl.savefig     (save_path)
        pl.show        ()
    if library == "plotly":
        figure = go.Figure  ()
        figure.add_trace    (go.Scatter(x=abscissa, y=chi2_pdf, mode='lines', name=f"Chi-squared({n})", line=dict(color='black')))
        figure.add_trace    (go.Scatter(x=[mahalanobis_distance, mahalanobis_distance], y=[0, max(chi2_pdf)], mode='lines', name=f"p_value : {np.round(100 * p_value_mahalanobis_distance, 2)}%", line=dict(color='red', dash='dash')))
        figure.add_trace    (go.Scatter(x=np.concatenate([abscissa, abscissa[::-1]]), y=np.concatenate([lower_fill_between, upper_fill_between[::-1]]), fill='toself', fillcolor='rgba(0,0,0,0.2)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name="p-value"))
        figure.update_xaxes (tickprefix="<b>", ticksuffix="<b><br>")
        figure.update_yaxes (tickprefix="<b>", ticksuffix="<b><br>")
        figure.update_layout(xaxis_title="Value", yaxis_title="Density", title="Log-likelihood = " + str(log_likelihood), width=1100, height=700)
        figure.show         ()


def plot_equivalent_normal_observation_assuming_gaussianity(p_value_mahalanobis_distance, equivalent_normal_observation, figsize, library, save_path):
    abscissa_limit                = max(5, equivalent_normal_observation)
    abscissa                      = np.linspace(-abscissa_limit, abscissa_limit, 1000)
    normal_pdf                    = norm.pdf(abscissa, 0, 1)
    lower_fill_between            = [0 for _ in abscissa]
    upper_fill_between            = [y * (abs(x) <= equivalent_normal_observation) for (x, y) in zip(abscissa, normal_pdf)]
    if library == "matplotlib":
        pl.figure      (figsize=figsize)
        pl.clf         ()
        pl.plot        (abscissa, normal_pdf, color="black", label="N(0,1)")
        pl.fill_between(abscissa, lower_fill_between, upper_fill_between, alpha=0.4, color="lightcoral", label="Data")
        pl.axvline     (  equivalent_normal_observation, color="red", linestyle='--', label=f'p-value: {np.round(100 * p_value_mahalanobis_distance, 2)}%')
        pl.axvline     (- equivalent_normal_observation, color="red", linestyle='--')
        pl.xticks      (fontsize=16)
        pl.yticks      (fontsize=16)
        pl.grid        ()
        pl.legend      (fontsize=24)
        pl.xlabel      ('Value', fontsize=20)
        pl.ylabel      ('Density', fontsize=20)
        pl.savefig     (save_path)
        pl.show        ()
    if library == "plotly":
        figure = go.Figure  ()
        figure.add_trace    (go.Scatter(x=abscissa, y=normal_pdf, mode='lines', name="N(0,1)", line=dict(color='black')))
        figure.add_trace    (go.Scatter(x=np.concatenate([abscissa, abscissa[::-1]]), y=np.concatenate([lower_fill_between, upper_fill_between[::-1]]), fill='toself', fillcolor='rgba(0,0,0,0.2)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name="Data"))
        figure.add_trace    (go.Scatter(x=[  equivalent_normal_observation,   equivalent_normal_observation], y=[0, max(normal_pdf)], mode='lines', name=f'p-value: {np.round(100 * p_value_mahalanobis_distance, 2)}%', line=dict(color='red', dash='dash')))
        figure.add_trace    (go.Scatter(x=[- equivalent_normal_observation, - equivalent_normal_observation], y=[0, max(normal_pdf)], mode='lines',                                                                        line=dict(color='red', dash='dash')))
        figure.update_xaxes (tickprefix="<b>", ticksuffix="<b><br>")
        figure.update_yaxes (tickprefix="<b>", ticksuffix="<b><br>")
        figure.update_layout(xaxis_title="Value", yaxis_title="Density", width=1100, height=700)
        figure.show         ()


def plot_predictive_variance_adequacy_assuming_gaussianity(PVA, p_value_PVA, lambdas, figsize, library, save_path):
    n                    = len(lambdas)
    abscissa             = np.linspace(0, 3, 1000)
    generalized_chi2_pdf = gamma.pdf(abscissa, n / 2, 0, 2 / n) # wrong, we need the generalized chi2 package
    lower_fill_between   = [0 for _ in abscissa]
    upper_fill_between   = [y * (x <= PVA) for (x, y) in zip(abscissa, generalized_chi2_pdf)]
    if library == "matplotlib":
        pl.figure      (figsize=figsize)
        pl.clf         ()
        pl.plot        (abscissa, generalized_chi2_pdf, color="black", label="Generalized chi2 distribution")
        pl.fill_between(abscissa, lower_fill_between, upper_fill_between, alpha=0.4, color="lightcoral")
        pl.axvline     (PVA                , color="red"  , label="p-value : " + str(np.round(100 * p_value_PVA, 2)) + "%", linestyle='--')
        pl.xticks      (fontsize=16)
        pl.yticks      (fontsize=16)
        pl.grid        ()
        pl.legend      (fontsize=24)
        pl.xlabel      ('Value', fontsize=20)
        pl.ylabel      ('Density', fontsize=20)
        pl.savefig     (save_path)
        pl.show        ()
    if library == "plotly":
        figure = go.Figure  ()
        figure.add_trace    (go.Scatter(x=abscissa, y=generalized_chi2_pdf, mode='lines', name="Generalized chi2 distribution", line=dict(color='black')))
        figure.add_trace    (go.Scatter(x=[PVA, PVA], y=[0, max(generalized_chi2_pdf)], mode='lines', name="p-value : " + str(np.round(100 * p_value_PVA, 2)) + "%", line=dict(color='red', dash='dash')))
        figure.add_trace    (go.Scatter(x=np.concatenate([abscissa, abscissa[::-1]]), y=np.concatenate([lower_fill_between, upper_fill_between[::-1]]), fill='toself', fillcolor='rgba(0,0,0,0.2)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name="p-value"))
        figure.update_xaxes (tickprefix="<b>", ticksuffix="<b><br>")
        figure.update_yaxes (tickprefix="<b>", ticksuffix="<b><br>")
        figure.update_layout(xaxis_title="Value", yaxis_title="Density", width=1100, height=700)
        figure.show         ()


def plot_normalized_prediction_errors_uncorrelated_assuming_gaussianity(uncorrelated_errors, lambdas, figsize, library, save_path):
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
    if library == "matplotlib":
        normalized_colors = mcolors.Normalize(vmin=all_sum_lambdas.min(), vmax=all_sum_lambdas.max())
        colormap          = cm.get_cmap('viridis')
        mapped_colors     = colormap(normalized_colors(all_sum_lambdas))
        figure, ax        = pl.subplots(figsize=figsize)
        ax.bar        (x=x, height=height, width=width, alpha=0.5, color=mapped_colors, label="Uncorrelated normalized errors")
        ax.plot       (abscissa, theoretical_pdf, color="blue", label='Theoretical Gaussian PDF')
        ax.grid       (True)
        ax.legend     (fontsize=24)
        ax.set_xlabel ("Value"  , fontsize=20)
        ax.set_ylabel ("Density", fontsize=20)
        cbar = figure.colorbar(cm.ScalarMappable(cmap=colormap, norm=normalized_colors), ax=ax)
        cbar.set_label("Sum of lambdas", fontsize=20)
        pl.savefig    (save_path)
        pl.show       ()
    if library == "plotly":
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


def plot_lambdas_empirical_distribution(lambdas, figsize, library, save_path):
    if library == "matplotlib":
        pl.figure (figsize=figsize)
        pl.clf    ()
        pl.hist   (lambdas, density=False, alpha=0.5, color="green", label="Error weights")
        pl.xticks (fontsize=16)
        pl.yticks (fontsize=16)
        pl.grid   ()
        pl.legend (fontsize=24)
        pl.xlabel ('Value', fontsize=20)
        pl.ylabel ('Density', fontsize=20)
        pl.savefig(save_path)
        pl.show   ()
    if library == "plotly":
        hist_data = np.histogram(lambdas, density=False)
        figure    = go.Figure   ()
        figure.add_trace        (go.Bar(x=hist_data[1][:-1], y=hist_data[0], width=(hist_data[1][1] - hist_data[1][0]), name="Error weights", opacity=0.5, marker_color="green"))
        figure.update_xaxes     (tickprefix="<b>", ticksuffix="<b><br>")
        figure.update_yaxes     (tickprefix="<b>", ticksuffix="<b><br>")
        figure.update_layout    (xaxis_title="Value", yaxis_title="Density", width=1100, height=700)
        figure.show             ()


class ResultsVisualization:

    def __init__(self, function_name, sampler_parameters, gaussian_process_model, evaluation, visualization_parameters):
        self._function_name                 = function_name
        self._sampler_parameters            = sampler_parameters
        self._gaussian_process_model        = gaussian_process_model
        self._evaluation                    = evaluation
        self._figsize                       = visualization_parameters["Figsize"                   ]
        self._alpha                         = visualization_parameters["alpha"                     ]
        self._library                       = visualization_parameters["Library"                   ]
        self._plot_type_2D                  = visualization_parameters["Plot type 2D"              ]
        self._save_path                     = visualization_parameters["Save path"                 ]
        self._dimension                     = len(evaluation["Function prediction"]["Train X"][0])
        self._kernel_optimization_criterion = self._gaussian_process_model.get_kernel_optimization_criterion()

    def create_new_folder_for_usecase(self):
        if not os.path.exists(self._save_path + self._function_name):
            os.mkdir(self._save_path + self._function_name)

    def plot_hyperparameters_optimization_history(self):
        hyperparameters_history = np.array(self._evaluation["Optimization infos"]["Hyperparameters history"]).T
        titles                  = ["log(sigma2)"] + ["-log(rho{})".format(plot_index) for plot_index in range(len(hyperparameters_history) - 1)] if self._kernel_optimization_criterion in ["REML", "MLE", "REML - PVA", "MLE - PVA"] else ["-log(rho{})".format(plot_index) for plot_index in range(len(hyperparameters_history))]
        save_path               = self._save_path + self._function_name + "/1.1 - Hyperparameters history.png"
        plot_hyperparameters_optimization_history(hyperparameters_history, titles, self._figsize, self._library, save_path)

    def plot_criterion_history(self):
        criterion_history = self._evaluation["Optimization infos"]["Criterion history"]
        criterion         = "Log " + self._kernel_optimization_criterion.split(" - ")[0]
        save_path         = self._save_path + self._function_name + "/1.2 - Criterion history.png"
        plot_criterion_history(criterion_history, criterion, self._figsize, self._library, save_path)

    def plot_negative_log_likelihood_evolution(self, log_scale):
        log_sigma2s                            = self._evaluation["Criterions evolution"]["Negative log likelihood evolution"]["Log sigma2s"             ]
        hyperparameters_0                      = self._evaluation["Optimization infos"  ]["Hyperparameters history"          ][0 ]
        hyperparameters_f                      = self._evaluation["Optimization infos"  ]["Hyperparameters history"          ][-1]
        kernel_optimization_criterion_was_used = self._kernel_optimization_criterion in ["MLE", "MLE - PVA"]
        xlabel                                 = "log(sigma2)"
        ylabel                                 = "-log(rho)"
        zlabel                                 = "Negative log likelihood"
        save_path                              = self._save_path + self._function_name + "/1.3 - Negative log likelihood.png"
        if self._dimension == 1:
            log_inv_rhos             = self._evaluation["Criterions evolution"]["Negative log likelihood evolution"]["Log inv rhos"]
            negative_log_likelihoods = self._evaluation["Criterions evolution"]["Negative log likelihood evolution"]["Negative log likelihoods"]
            if self._plot_type_2D == "heatmap":
                plot_criterion_evolution_2D_heatmap  (log_sigma2s, log_inv_rhos, negative_log_likelihoods, hyperparameters_0, hyperparameters_f, kernel_optimization_criterion_was_used, self._figsize, self._library, xlabel, ylabel, zlabel, save_path, log_scale)
            if self._plot_type_2D == "3D":
                plot_criterion_evolution_2D_3D       (log_sigma2s, log_inv_rhos, negative_log_likelihoods, hyperparameters_0, hyperparameters_f, kernel_optimization_criterion_was_used, self._figsize, self._library, xlabel, ylabel, zlabel, save_path, log_scale)
        if self._dimension > 1:
            negative_log_likelihood_evolution = self._evaluation["Criterions evolution"]["Negative log likelihood evolution"]
            titles                            = ["log(sigma2)"] + ["-log(rho{})".format(plot_index) for plot_index in range(len(negative_log_likelihood_evolution) - 1)]
            plot_negative_log_likelihood_evolution_nD(negative_log_likelihood_evolution, hyperparameters_0, hyperparameters_f, titles          , kernel_optimization_criterion_was_used, self._figsize, self._library,                 zlabel, save_path, log_scale)

    def _plot_profiled_criterion_evolution(self, criterion, log_scale):
        criterion_number, criterion_name = None, None
        if criterion == "Profiled negative log likelihood":
            criterion_number, criterion_name = "4", "PMLE"
        if criterion == "Profiled PVA negative log likelihood":
            criterion_number, criterion_name = "5", "PMLE - PVA"
        if criterion == "Mean square error":
            criterion_number, criterion_name = "6", "MSE - PVA"
        hyperparameters_0                      = self._evaluation["Optimization infos"  ]["Hyperparameters history"][0]
        hyperparameters_f                      = self._evaluation["Optimization infos"  ]["Hyperparameters history"][-1]
        kernel_optimization_criterion_was_used = self._kernel_optimization_criterion == criterion_name
        xlabel                                 = "-log(rho 1)"
        ylabel                                 = "-log(rho 2)"
        zlabel                                 = criterion
        save_path                              = self._save_path + self._function_name + "/1." + criterion_number + " - Profiled negative log likelihood.png"
        if self._dimension == 1:
            log_inv_rhos  = self._evaluation["Criterions evolution"][criterion + " evolution"]["Log inv rhos" ]
            criterions    = self._evaluation["Criterions evolution"][criterion + " evolution"][criterion + "s"]
            plot_criterion_evolution_1D            (log_inv_rhos                , criterions, hyperparameters_0, hyperparameters_f, kernel_optimization_criterion_was_used, self._figsize, self._library                , zlabel, save_path, log_scale)
        if self._dimension == 2:
            log_inv_rhos1 = self._evaluation["Criterions evolution"][criterion + " evolution"]["Log inv rhos 1"]
            log_inv_rhos2 = self._evaluation["Criterions evolution"][criterion + " evolution"]["Log inv rhos 2"]
            criterions    = self._evaluation["Criterions evolution"][criterion + " evolution"][criterion + "s" ]
            if self._plot_type_2D == "heatmap":
                plot_criterion_evolution_2D_heatmap(log_inv_rhos1, log_inv_rhos2, criterions, hyperparameters_0, hyperparameters_f, kernel_optimization_criterion_was_used, self._figsize, self._library, xlabel, ylabel, zlabel, save_path, log_scale)
            if self._plot_type_2D == "3D":
                plot_criterion_evolution_2D_3D     (log_inv_rhos1, log_inv_rhos2, criterions, hyperparameters_0, hyperparameters_f, kernel_optimization_criterion_was_used, self._figsize, self._library, xlabel, ylabel, zlabel, save_path, log_scale)
        if self._dimension > 2:
            criterion_evolution = self._evaluation["Criterions evolution"][criterion + " evolution"]
            titles              = ["-log(rho{})".format(plot_index) for plot_index in range(len(criterion_evolution))]
            plot_criterion_evolution_nD            (criterion_evolution, hyperparameters_0, hyperparameters_f, titles             , kernel_optimization_criterion_was_used, self._figsize, self._library                , zlabel, save_path, log_scale)

    def plot_profiled_negative_log_likelihood_evolution(self, log_scale):
        self._plot_profiled_criterion_evolution("Profiled negative log likelihood"    , log_scale)

    def plot_profiled_pva_negative_log_likelihood_evolution(self, log_scale):
        self._plot_profiled_criterion_evolution("Profiled PVA negative log likelihood", log_scale)

    def plot_mean_square_error_evolution(self, log_scale):
        self._plot_profiled_criterion_evolution("Mean square error"                   , log_scale)

    def plot_function_predictions(self):
        train_X        =         self._evaluation["Function prediction"]["Train X"           ]
        train_Y        =         self._evaluation["Function prediction"]["Train Y"           ]
        LOO_mean       =         self._evaluation["Function prediction"]["LOO mean"          ]
        LOO_variance   =         self._evaluation["Function prediction"]["LOO variance"      ]
        test_X         =         self._evaluation["Function prediction"]["Test X"            ]
        test_Y         =         self._evaluation["Function prediction"]["Test Y"            ]
        function_X     =         self._evaluation["Function prediction"]["Function X"        ]
        function_Y     =         self._evaluation["Function prediction"]["Function Y"        ]
        posterior_mean =         self._evaluation["Function prediction"]["Posterior mean"    ]
        posterior_std  = np.sqrt(self._evaluation["Function prediction"]["Posterior variance"])
        q_alpha        =         self._evaluation["Train"]["Gaussianity"]["q alphas"][int(len(self._evaluation["Train"]["Gaussianity"]["q alphas"]) * self._alpha)]
        save_path      = self._save_path + self._function_name + "/2.1 - Function prediction.png"
        if self._dimension == 1:
            plot_1D_function_prediction        (train_X, train_Y, test_X, test_Y, function_X, function_Y, LOO_mean, LOO_variance, posterior_mean, posterior_std, q_alpha , self._figsize, self._library, save_path)
        if self._dimension == 2 and self._plot_type_2D == "heatmap":
            plot_2D_function_prediction_heatmap(train_X         , test_X        , function_X, self._function_name, self._sampler_parameters, self._gaussian_process_model, self._figsize, self._library, save_path)
        if self._dimension == 2 and self._plot_type_2D == "3D":
            plot_2D_function_prediction_3D     (train_X, train_Y, test_X, test_Y, function_X, self._function_name, self._sampler_parameters, self._gaussian_process_model, self._figsize, self._library, save_path)

    def plot_truth_vs_predictions(self, dataset):
        true_values      = self._evaluation[dataset]["Predictions"        ]["True values"     ]
        predicted_values = self._evaluation[dataset]["Predictions"        ]["Predicted values"]
        std              = self._evaluation[dataset]["Predictions"        ]["Std"             ]
        RMSE             = self._evaluation[dataset]["Performance metrics"]["RMSE"            ]
        save_path        = self._save_path + self._function_name + "/2.2 - Predictions " + dataset + ".png"
        plot_truth_vs_predictions(true_values, predicted_values, std, RMSE, self._figsize, self._library, save_path)

    def plot_empirical_coverage_probability_assuming_reliability(self, dataset):
        alphas    =     self._evaluation[dataset]["Reliability"]["alphas"                      ]
        ECP       =     self._evaluation[dataset]["Reliability"]["ECP"                         ]
        n         = len(self._evaluation[dataset]["Reliability"]["Normalized prediction errors"])
        save_path = self._save_path + self._function_name + "/3.1 - ECP assuming reliability" + dataset + ".png"
        plot_empirical_coverage_probability_assuming_reliability(alphas, ECP, n, self._figsize, self._library, save_path)

    def plot_predictive_variance_adequacy_distribution_assuming_reliability(self, dataset):
        PVA                =     self._evaluation[dataset]["Reliability"]["PVA"                         ]
        p_value_PVA        =     self._evaluation[dataset]["Reliability"]["p_value PVA"                 ]
        n                  = len(self._evaluation[dataset]["Reliability"]["Normalized prediction errors"])
        save_path          = self._save_path + self._function_name + "/3.2 - PVA assuming reliability" + dataset + ".png"
        plot_predictive_variance_adequacy_distribution_assuming_reliability(PVA, p_value_PVA, n, self._figsize, self._library, save_path)

    def plot_normalized_prediction_errors(self, dataset):
        normalized_prediction_errors = self._evaluation[dataset]["Reliability"]["Normalized prediction errors"]
        save_path                    = self._save_path + self._function_name + "/3.3 - Normalized prediction errors " + dataset + ".png"
        plot_normalized_prediction_errors(normalized_prediction_errors, self._figsize, self._library, save_path)

    def plot_whitened_data_assuming_gaussianity(self, dataset):
        train_Y_whitened = self._evaluation[dataset]["Gaussianity"]["Y whitened"]
        save_path        = self._save_path + self._function_name + "/4.1 - Whitened data assuming Gaussianity" + dataset + ".png"
        plot_whitened_data_assuming_gaussianity(train_Y_whitened, self._figsize, self._library, save_path)

    def plot_mahalanobis_distance_distribution_assuming_gaussianity(self, dataset):
        mahalanobis_distance         =     self._evaluation[dataset]["Gaussianity"]["Mahalanobis distance"        ]
        p_value_mahalanobis_distance =     self._evaluation[dataset]["Gaussianity"]["p_value Mahalanobis distance"]
        log_likelihood               =     self._evaluation[dataset]["Gaussianity"]["Log likelihood"              ]
        n                            = len(self._evaluation[dataset]["Gaussianity"]["Y whitened"                  ])
        save_path                    = self._save_path + self._function_name + "/4.2 - Mahalanobis distance distribution assuming Gaussianity" + dataset + ".png"
        plot_mahalanobis_distance_distribution_assuming_gaussianity(mahalanobis_distance, p_value_mahalanobis_distance, log_likelihood, n, self._figsize, self._library, save_path)

    def plot_equivalent_normal_observation_assuming_gaussianity(self, dataset):
        p_value_mahalanobis_distance  = self._evaluation[dataset]["Gaussianity"]["p_value Mahalanobis distance" ]
        equivalent_normal_observation = self._evaluation[dataset]["Gaussianity"]["Equivalent normal observation"]
        save_path                     = self._save_path + self._function_name + "/4.3 - Equivalent normal observation assuming Gaussianity" + dataset + ".png"
        plot_equivalent_normal_observation_assuming_gaussianity(p_value_mahalanobis_distance, equivalent_normal_observation, self._figsize, self._library, save_path)

    def plot_predictive_variance_adequacy_assuming_gaussianity(self, dataset):
        PVA                = self._evaluation[dataset]["Gaussianity"]["PVA"         ]
        p_value_PVA        = self._evaluation[dataset]["Gaussianity"]["p_value PVA" ]
        lambdas            = self._evaluation[dataset]["Gaussianity"]["Lambdas"     ]
        save_path          = self._save_path + self._function_name + "/4.4 - PVA assuming Gaussianity" + dataset + ".png"
        plot_predictive_variance_adequacy_assuming_gaussianity(PVA, p_value_PVA, lambdas, self._figsize, self._library, save_path)

    def plot_normalized_prediction_errors_uncorrelated_assuming_gaussianity(self, dataset):
        uncorrelated_errors = self._evaluation[dataset]["Gaussianity"]["Uncorrelated errors"]
        lambdas             = self._evaluation[dataset]["Gaussianity"]["Lambdas"            ]
        save_path           = self._save_path + self._function_name + "/4.6 - Normalized prediction errors uncorrelated assuming Gaussianity" + dataset + ".png"
        plot_normalized_prediction_errors_uncorrelated_assuming_gaussianity(uncorrelated_errors, lambdas, self._figsize, self._library, save_path)

    def plot_lambdas_empirical_distribution(self, dataset):
        lambdas   = self._evaluation[dataset]["Gaussianity"]["Lambdas"]
        save_path = self._save_path + self._function_name + "/4.6 - Lambdas empirical distribution " + dataset + ".png"
        plot_lambdas_empirical_distribution(lambdas, self._figsize, self._library, save_path)

    def show_reliability_metrics(self, dataset):
        n    = self._evaluation[dataset]["Reliability metrics"]["N"   ]
        PVA  = self._evaluation[dataset]["Reliability metrics"]["PVA" ]
        ECP  = self._evaluation[dataset]["Reliability metrics"]["ECP" ][int(len(self._evaluation[dataset]["Reliability metrics"]["ECP"]) * self._alpha)]
        PMIW = self._evaluation[dataset]["Reliability metrics"]["PMIW"]
        IAE  = self._evaluation[dataset]["Reliability metrics"]["IAE" ]
        print(                                      "PVA       : " + str(PVA) + " - n = " + str(n))
        print(str(int(np.round(100*self._alpha))) + "% ECP : "     + str(ECP))
        print(                                      "IAE       : " + str(IAE))
        print(                                      "PMIW      : " + str(PMIW))

    def show_performance_metrics(self, dataset):
        RMSE = self._evaluation[dataset]["Performance metrics"]["RMSE"]
        Q2   = self._evaluation[dataset]["Performance metrics"]["Q2"  ]
        print("RMSE : " + str(RMSE))
        print("Q2   : " + str(Q2  ))

    def show_hybrid_metrics(self, dataset):
        NLPD = self._evaluation[dataset]["Hybrid metrics"]["NLPD"]
        CRPS = self._evaluation[dataset]["Hybrid metrics"]["CRPS"]
        IS   = self._evaluation[dataset]["Hybrid metrics"]["IS"  ][int(len(self._evaluation[dataset]["Hybrid metrics"]["IS" ]) * self._alpha)]
        print(                                      "NLPD     : " + str(NLPD))
        print(                                      "CRPS     : " + str(CRPS))
        print(str(int(np.round(100*self._alpha))) + "% IS : "   + str(IS  ))
