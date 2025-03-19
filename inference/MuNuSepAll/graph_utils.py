from sklearn.metrics import auc
import plotly.graph_objects as go
import plotly.colors as pc

#=============Plots for binary classification============#

def plot_roc_auc(fpr, tpr, fig, name=f'ROC curve'):
    roc_auc = auc(fpr, tpr)
    # Add trace for ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode='lines', name=f'{name}(AUC = {roc_auc:.2f})',
        line=dict(width=2)
    ))
    # Display the plot
    return fig

def generate_plotly_colors(num_colors):
    # Use Plotly's built-in color scales to get distinct colors
    color_scale = pc.qualitative.Plotly  # This is a predefined qualitative color scale
    colors = color_scale * (num_colors // len(color_scale) + 1)  # Repeat the color scale if more colors are needed
    return colors[:num_colors]

def plot_metrics_vs_thresholds(fpr, tpr, thresholds, fig, name=f'', color='green'):
    # Plot TPR vs Threshold
    fig.add_trace(go.Scatter(
        x=thresholds, y=tpr, mode='lines', name=f"TPR / Recall / Sensitivity,<br>{name}",
        line=dict(color=color), showlegend=True
    ))
    # Plot FPR vs Threshold
    fig.add_trace(go.Scatter(
        x=thresholds, y=1-fpr, mode='lines', name=f"(1-FPR / Precision / Purity),<br>{name}",
        line=dict(dash='dot', color=color), showlegend=True
    ))
    return fig

def plot_metrics_vs_energy(energies, fpr, tpr, fig=None, name=f'', color='green'):
    if fig is None:
        fig = go.Figure()
    # Plot TPR vs Threshold
    fig.add_trace(go.Scatter(
        x=energies, y=tpr, mode='lines', name=f"TPR / Recall / Sensitivity,<br>{name}",
        line=dict(color=color), showlegend=True
    ))
    # Plot FPR vs Threshold
    fig.add_trace(go.Scatter(
        x=energies, y=1-fpr, mode='lines', name=f"(1-FPR / Precision / Purity),<br>{name}",
        line=dict(dash='dot', color=color), showlegend=True
    ))
    return fig

def plot_exp_muatm_preds_hist(arr_p_exp, arr_p_muatm, fig=None):
    if fig is None:
        fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=arr_p_exp,
        nbinsx=50,
        histnorm='percent',
        name='Exp data',
        marker_color='red',
        opacity=0.5 
    ))
    fig.add_trace(go.Histogram(
        x=arr_p_muatm,
        nbinsx=50,
        histnorm='percent',
        name='MC data',
        marker_color='blue',
        opacity=0.5
    ))
    fig.update_layout(
        title='Hists',
        xaxis_title='Predicted Value',
        yaxis_title='% (Log Scale)',
        yaxis_type='log',               # Logarithmic y-axis
        barmode='overlay',              # Overlay histograms (critical for overlaps)
        showlegend=True                 # Show legend to distinguish datasets
    )
    return fig