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
        'heatmap_image_path': 'images/correlation_heatmap.png'
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
    static_images_dir = os.path.join(settings.BASE_DIR, 'recession_app', 'static', 'images')
    heatmap_path = os.path.join(static_images_dir, 'correlation_heatmap.png')
    npz_path = os.path.join(output_dir, 'correlation_tensor_usa.npz')

    # Define paths to the analysis scripts
    generate_script_path = os.path.join(settings.BASE_DIR, 'analysis', 'generate_spreads_and_correlations.py')
    visualize_script_path = os.path.join(settings.BASE_DIR, 'analysis', 'visualize_correlation_matplotlib.py')

    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(static_images_dir, exist_ok=True)

    try:
        # Run the analysis scripts to generate the data and heatmap image
        # Using check=True will raise an exception if the scripts fail
        subprocess.run(
            ['python', generate_script_path, '--output-dir', output_dir],
            check=True, capture_output=True, text=True
        )
        subprocess.run(
            ['python', visualize_script_path, '--npz', npz_path, '--output', heatmap_path],
            check=True, capture_output=True, text=True
        )
        return HttpResponse("<h1>Success!</h1><p>The correlation heatmap has been updated.</p><a href='/'>Go back home</a>")
    except subprocess.CalledProcessError as e:
        # If a script fails, return an error message to the browser
        error_message = f"""
        <h1>Error updating heatmap</h1>
        <p>A script failed to execute.</p>
        <pre><strong>Command:</strong> {' '.join(e.cmd)}</pre>
        <pre><strong>Return Code:</strong> {e.returncode}</pre>
        <pre><strong>Output:</strong>\n{e.stdout}</pre>
        <pre><strong>Error Output:</strong>\n{e.stderr}</pre>
        """
        return HttpResponse(error_message, status=500)