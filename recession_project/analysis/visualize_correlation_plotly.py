import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Render an interactive 2D correlation heatmap with a time slider using Plotly.")
    parser.add_argument("--npz", type=Path, required=True, help="Path to the correlation tensor .npz file.")
    parser.add_argument("--output", type=Path, required=True, help="Path to save the output HTML file.")
    parser.add_argument("--max-dates", type=int, default=500, help="Maximum number of timeline samples for the slider.")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load data from the .npz file
    data = np.load(args.npz, allow_pickle=True)
    dates = pd.to_datetime(data["dates"].astype(str))
    spreads = data["spreads"].astype(str)
    corr_scaled = data["corr_scaled"]

    n_dates, _, _ = corr_scaled.shape

    # Subsample dates to keep the visualization performant
    if n_dates > args.max_dates:
        print(f"[info] Reducing timeline from {n_dates} to {args.max_dates} samples for render.")
        idx = np.linspace(0, n_dates - 1, args.max_dates).round().astype(int)
        dates = dates[idx]
        corr_scaled = corr_scaled[idx]

    # Create the figure
    fig = go.Figure()

    # Add a heatmap trace for each date
    for i, date in enumerate(dates):
        fig.add_trace(
            go.Heatmap(
                z=corr_scaled[i],
                x=spreads,
                y=spreads,
                colorscale=[[0, 'red'], [1, 'blue']],
                zmin=0,
                zmax=1,
                visible=(i == len(dates) - 1), # Make only the last date visible initially
                colorbar=dict(title='Correlation')
            )
        )

    # Create frames for animation instead of multiple traces
    frames = []
    for i, date in enumerate(dates):
        frame = go.Frame(
            data=[go.Heatmap(
                z=corr_scaled[i],
                x=spreads,
                y=spreads,
                colorscale=[
                    [0, 'rgb(165,0,38)'],    # Dark red for low values
                    [0.25, 'rgb(215,48,39)'], # Light red
                    [0.5, 'rgb(244,109,67)'], # Orange
                    [0.75, 'rgb(69,117,180)'],# Light blue
                    [1, 'rgb(49,54,149)']     # Dark blue for high values
                ],
                zmin=0,
                zmax=1,
                colorbar=dict(
                    title=dict(
                        text='Correlation',
                        side='right',
                        font=dict(size=14)
                    ),
                    thickness=20,
                    thicknessmode='pixels',
                    len=0.75,
                    tickformat='.2f',
                    x=1.02
                ),
                xgap=1,  # Add gap between cells
                ygap=1,  # Add gap between cells
                hoverongaps=False,
                hovertemplate='Spread Y: %{y}<br>Spread X: %{x}<br>Correlation: %{z:.3f}<extra></extra>'
            )],
            name=date.strftime('%Y-%m-%d')
        )
        frames.append(frame)
    
    # Initialize with the last frame's data for the initial view
    fig = go.Figure(data=frames[-1].data)
    fig.frames = frames

    # Create slider steps that work with animation
    slider_steps = []
    for date in dates:
        step = dict(
            method='animate',
            args=[[date.strftime('%Y-%m-%d')],
                  {'frame': {'duration': 100, 'redraw': True},
                   'mode': 'immediate',
                   'transition': {'duration': 50}}],
            label=date.strftime('%Y-%m-%d')
        )
        slider_steps.append(step)

    sliders = [dict(
        active=len(dates) - 1,
        currentvalue={"prefix": "Date: "},
        pad={"t": 50},
        steps=slider_steps
    )]

    # Configure the layout
    fig.update_layout(
        title={
            'text': 'Interactive Correlation Heatmap',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        xaxis_title={
            'text': "Spread (i)",
            'font': dict(size=16)
        },
        yaxis_title={
            'text': "Spread (j)",
            'font': dict(size=16)
        },
        sliders=sliders,
        paper_bgcolor="#121212",
        plot_bgcolor="#121212",
        font_color="#e0e0e0",
        xaxis=dict(
            showgrid=False,
            tickangle=-45,
            tickfont=dict(size=12),
            side='bottom'
        ),
        yaxis=dict(
            showgrid=False,
            autorange='reversed',
            tickfont=dict(size=12),
            side='left'
        ),
        margin=dict(t=100, l=100, r=100, b=100),
        height=800  # Increase height for better visibility
    )

    # Create the custom HTML and JavaScript with proper slider control
    date_labels = [d.strftime('%Y-%m-%d') for d in dates]
    
    controls_html = """
    <div id="custom-controls" style="padding: 10px 20px; text-align: center; background-color: #121212;">
        <label for="date-search" style="margin-right: 10px; font-family: 'Roboto Mono', monospace; color: #e0e0e0;">
            Jump to Date:
        </label>
        <input type="date" id="date-search" style="background-color: #333; color: #eee; border: 1px solid #555; border-radius: 4px; padding: 5px;">
        <button id="jump-to-date-btn" style="background-color: #03dac6; color: #121212; border: none; border-radius: 4px; padding: 6px 12px; margin-left: 5px; cursor: pointer; font-weight: bold;">
            Go
        </button>
    </div>
    """

    js_code = f"""
    <script id="date-data" type="application/json">{json.dumps(date_labels)}</script>
    <script>
        const plotlyDiv = document.querySelector('.plotly-graph-div');
        
        function initControls() {{
            const gd = plotlyDiv;
            const dates = JSON.parse(document.getElementById('date-data').textContent);
            
            function jumpToDate(index) {{
                if (index >= 0 && index < dates.length) {{
                    // First update the frame
                    const frame = dates[index];
                    Plotly.animate(gd, [frame], {{
                        frame: {{ duration: 100, redraw: true }},
                        mode: 'immediate',
                        transition: {{ duration: 50 }}
                    }}).then(() => {{
                        // Then force update the slider and data
                        return Plotly.update(gd, 
                            // Update the data to match the frame
                            {{'z': [gd._transitionData._frames.find(f => f.name === frame).data[0].z]}},
                            // Update the slider position
                            {{'sliders[0].active': index}}
                        );
                    }});
                }}
            }}

            document.getElementById('jump-to-date-btn').addEventListener('click', () => {{
                const input = document.getElementById('date-search').value;
                if (!input) return;
                
                const targetDate = input.split('T')[0];
                const dateIndex = dates.findIndex(d => d === targetDate);
                
                if (dateIndex !== -1) {{
                    jumpToDate(dateIndex);
                }} else {{
                    const target = new Date(targetDate);
                    const closest = dates.reduce((prev, curr, idx) => {{
                        const prevDiff = Math.abs(new Date(dates[prev]) - target);
                        const currDiff = Math.abs(new Date(curr) - target);
                        return currDiff < prevDiff ? idx : prev;
                    }}, 0);
                    jumpToDate(closest);
                }}
            }});

            window.addEventListener('keydown', (e) => {{
                if (!['ArrowLeft', 'ArrowRight'].includes(e.key)) return;
                e.preventDefault();
                
                const curr = gd.layout.sliders[0].active;
                const next = e.key === 'ArrowLeft' ? 
                    Math.max(0, curr - 1) : 
                    Math.min(dates.length - 1, curr + 1);
                
                if (next !== curr) {{
                    jumpToDate(next);
                }}
            }});

            // Set up initial date input
            const dateInput = document.getElementById('date-search');
            dateInput.value = dates[dates.length - 1];
            dateInput.min = dates[0];
            dateInput.max = dates[dates.length - 1];

            // Also handle slider moves directly
            gd.on('slider', function(e) {{
                if (e && typeof e.step !== 'undefined') {{
                    jumpToDate(e.step);
                }}
            }});
        }}

        if (plotlyDiv) {{
            const observer = new MutationObserver((mutations, obs) => {{
                if (document.querySelector('.main-svg')) {{
                    initControls();
                    obs.disconnect();
                }}
            }});
            observer.observe(plotlyDiv, {{ childList: true, subtree: true }});
        }}
    </script>
    """

    # Write the complete HTML file
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(controls_html)
        fig.write_html(f, full_html=False, include_plotlyjs='cdn')
        f.write(js_code)

    print(f"[info] Interactive heatmap saved to {args.output}")

if __name__ == "__main__":
    main()