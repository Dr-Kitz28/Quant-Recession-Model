# recession_app/views.py
import subprocess
import os
from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponse

# This view now ONLY displays the page with the existing heatmap
def home(request):
    """
    Renders the home page, which displays the latest correlation heatmap image.
    This view is fast because it does not perform any data analysis.
    """
    context = {
        'interactive_plot_path': 'visuals/interactive_heatmap.html'
    }
    return render(request, 'home.html', context)

# This new view will run the heavy analysis when you navigate to its URL
def update_heatmap(request):
    """
    Triggers the data analysis scripts to generate a new correlation heatmap.
    This is a long-running process and should only be triggered when an update is needed.
    """
    # Define paths for output files using absolute paths from BASE_DIR
    output_dir = os.path.join(settings.BASE_DIR, 'outputs')
    # The interactive plot will be an HTML file in the static directory
    interactive_plot_path = os.path.join(settings.BASE_DIR, 'recession_app', 'static', 'visuals', 'interactive_heatmap.html')
    npz_path = os.path.join(output_dir, 'correlation_tensor_usa.npz')

    # Define paths to the analysis scripts
    generate_script_path = os.path.join(settings.BASE_DIR, 'analysis', 'generate_spreads_and_correlations.py')
    # Use the new Plotly visualization script
    visualize_script_path = os.path.join(settings.BASE_DIR, 'analysis', 'visualize_correlation_plotly.py')

    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(interactive_plot_path), exist_ok=True)

    try:
        # Run the analysis scripts and capture their output
        print("--- Running generate_spreads_and_correlations.py ---")
        result1 = subprocess.run(
            ['python', generate_script_path, '--output-dir', output_dir],
            check=True, capture_output=True, text=True
        )
        print(result1.stdout)
        if result1.stderr:
            print("Errors:\n", result1.stderr)

        print("\n--- Running visualize_correlation_plotly.py ---")
        result2 = subprocess.run(
            ['python', visualize_script_path, '--npz', npz_path, '--output', interactive_plot_path],
            check=True, capture_output=True, text=True
        )
        print(result2.stdout)
        if result2.stderr:
            print("Errors:\n", result2.stderr)

        context = {'success': True}
        return render(request, 'update_status.html', context)
    except subprocess.CalledProcessError as e:
        # If a script fails, render the status page with error details
        context = {
            'success': False,
            'error': {
                'cmd': e.cmd,
                'returncode': e.returncode,
                'stdout': e.stdout,
                'stderr': e.stderr,
            }
        }
        return render(request, 'update_status.html', context, status=500)