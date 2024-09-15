import os
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as pl
from dataset_generation import f
import plotly.express as px


FIGSIZE = (20, 13)


def check_if_plot_is_possible(function_name):
    possible_1D_plot   = ["Power sinus 1D", "Heaviside 1D", "Multimodal sinus 1D", "Noisy sinus 1D", "Sinus times x 1D", "Sinus cardinal 1D", "Slide function", "Two bumps", "Gaussian process trajectory"]
    possible_2D_plot   = ["Power sinus 2D", "Heaviside 2D", "Multimodal sinus 2D", "Noisy sinus 2D", "Sinus times x 2D", "Sinus cardinal 2D", "Branin", "Branin hoo", "Goldstein", "Goldstein price", "Iooss1", "Iooss2", "Two input toy", "Highly non linear", "Mystery", "Six hump camelback"]
    impossible_plot    = ["Concrete compressive strength", "Energy efficiency", "Auto mpg", "Combined cycle power plant", "Airfoil self-noise", "Morokoff & caflisch 1", "Morokoff & caflisch 2", "Rosenbrock4", "Hartman4", "Hartman6", "Sphere6", "Wing weight", "Sobol", "Ishigami"]
    if function_name in possible_1D_plot:
        return "1D plot"
    if function_name in possible_2D_plot:
        return "2D plot"
    if function_name in impossible_plot:
        return "No plot"


def plot_1D_function(X_observed, Y_observed, X_test, Y_test, X_function, Y_function, library, save_path, function_name):
    X_observed, X_test, X_function = [x[0] for x in X_observed], [x[0] for x in X_test], [x[0] for x in X_function]
    if library == "matplotlib":
        pl.figure(figsize=FIGSIZE)
        pl.clf()
        pl.scatter(X_observed, Y_observed, label="Samples", color="red"  , marker="x", s=200)
        pl.scatter(X_test    , Y_test    , label="Test"   , color="blue" , marker="o", s=200)
        pl.plot   (X_function, Y_function, label="f"      , color="green", linewidth=1, linestyle="--")
        pl.xticks(fontsize=16)
        pl.yticks(fontsize=16)
        pl.grid()
        pl.legend(fontsize=24)
        pl.xlabel('x', fontsize=20)
        pl.ylabel('f(x)', fontsize=20)
        if not os.path.exists(save_path + function_name):
            os.mkdir(save_path + function_name)
        pl.savefig(save_path + function_name + "/Function.png")
        pl.show()
    if library == "plotly":
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=X_observed, y=Y_observed, mode='markers', name="Samples", marker_color="red" , marker_size=10, marker_symbol="x"     ))
        figure.add_trace(go.Scatter(x=X_test    , y=Y_test    , mode='markers', name="Test"   , marker_color="blue", marker_size=10, marker_symbol="circle"))
        figure.add_trace(go.Scatter(x=X_function, y=Y_function, mode="lines", name="f", line=dict(width=1, color="green", dash="dash")))
        figure.update_xaxes(tickprefix="<b>", ticksuffix="<b><br>")
        figure.update_yaxes(tickprefix="<b>", ticksuffix="<b><br>")
        figure.update_layout(xaxis_title="x", yaxis_title="f(x)", width=1100, height=700)
        figure.show()

def plot_2D_function_3D(X_observed, Y_observed, X_test, Y_test, X_function, function_name, function_parameters, library, save_path):
    X1_function    = np.unique(X_function[:, 0])
    X2_function    = np.unique(X_function[:, 1])
    X_mesh, Y_mesh = np.meshgrid(X1_function, X2_function)
    Z_mesh         = f(X_mesh, Y_mesh, function_name, function_parameters)
    X1_observed    = X_observed[:, 0]
    X2_observed    = X_observed[:, 1]
    X1_test        = X_test    [:, 0]
    X2_test        = X_test    [:, 1]
    if library == "matplotlib":
        figure     = pl.figure(figsize=FIGSIZE)
        axes       = figure.add_subplot(111, projection='3d')
        axes.plot_surface(X_mesh, Y_mesh, Z_mesh, cmap='viridis')
        axes.scatter(X1_observed, X2_observed, Y_observed, color="red" , marker="*", s=200, label='Samples')
        axes.scatter(X1_test    , X2_test    , Y_test    , color="blue", marker="o", s=200, label='Test'   )
        axes.set_xlabel('x1', fontsize=20)
        axes.set_ylabel('x2', fontsize=20)
        axes.set_zlabel('f(x1,x2)', fontsize=20)
        if not os.path.exists(save_path + function_name):
            os.mkdir(save_path + function_name)
        pl.savefig(save_path + function_name + "/Function.png")
        pl.show()
    if library == "plotly":
        figure = go.Figure(data=[go.Surface(z=Z_mesh, x=X_mesh, y=Y_mesh, colorscale='Viridis')])
        figure.add_trace(go.Scatter3d(x=X1_observed, y=X2_observed, z=Y_observed, mode='markers', marker=dict(size=10, color="red" , symbol="x"     ), name='Samples'))
        figure.add_trace(go.Scatter3d(x=X1_test    , y=X2_test    , z=Y_test    , mode='markers', marker=dict(size=10, color="blue", symbol="circle"), name='Test'   ))
        figure.update_layout(scene=dict(xaxis_title='x1', yaxis_title='x2', zaxis_title='f(x1,x2)'), width=800, height=800)
        figure.show()

def plot_2D_function_heatmap(X_observed, _, X_test, __, X_function, function_name, function_parameters, library, save_path):
    X1_function    = np.unique(X_function[:, 0])
    X2_function    = np.unique(X_function[:, 1])
    X_mesh, Y_mesh = np.meshgrid(X1_function, X2_function)
    Z_mesh         = f(X_mesh, Y_mesh, function_name, function_parameters)
    X1_observed    = X_observed[:, 0]
    X2_observed    = X_observed[:, 1]
    X1_test        = X_test    [:, 0]
    X2_test        = X_test    [:, 1]
    if library == "matplotlib":
        pl.figure(figsize=FIGSIZE)
        pl.clf()
        pl.scatter(X1_observed, X2_observed, color="red" , marker="x", s=200, label='Samples')
        pl.scatter(X1_test    , X2_test    , color="blue", marker="o", s=200, label='Test'   )
        pl.imshow(Z_mesh, extent=(min(X1_function), max(X1_function), min(X2_function), max(X2_function)), origin='lower', cmap='viridis', interpolation='nearest')
        pl.colorbar()
        pl.xticks(fontsize=16)
        pl.yticks(fontsize=16)
        pl.grid()
        pl.legend(fontsize=24)
        if not os.path.exists(save_path + function_name):
            os.mkdir(save_path + function_name)
        pl.savefig(save_path + function_name + "/Function.png")
        pl.show()
    if library == "plotly":
        figure   = px.imshow(Z_mesh, x=X1_function, y=X2_function, labels={'x': 'x1', 'y': 'x2', 'color': 'f(x1,x2)'})
        figure.add_trace(go.Scatter(x=X1_observed, y=X2_observed, mode='markers', marker=dict(color='red' , symbol="x"  , size=10), name="Samples"))
        figure.add_trace(go.Scatter(x=X1_test    , y=X2_test    , mode='markers', marker=dict(color='blue', symbol="circle", size=10), name="Test"   ))
        figure.update_yaxes(autorange='reversed')
        figure.update_layout(legend=dict(x=0.8, y=0.95, xanchor='left', yanchor='top'), scene=dict(xaxis_title='x1', yaxis_title='x2', zaxis_title='f(x1,x2)'), width=800, height=800)
        figure.show()
