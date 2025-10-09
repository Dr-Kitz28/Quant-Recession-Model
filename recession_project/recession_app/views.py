from django.shortcuts import render
import pandas as pd
import subprocess

def home(request):
    # Run the analysis scripts
    subprocess.run(['python', 'analysis/generate_spreads_and_correlations.py', '--output-dir', 'outputs'], check=True)
    subprocess.run(['python', 'analysis/visualize_correlation_matplotlib.py', '--npz', 'outputs/correlation_tensor_usa.npz', '--output', 'recession_project/recession_app/static/images/correlation_heatmap.png'], check=True)

    # Read the data from the CSV files
    bond_market_data = pd.read_csv('bond_market_data.csv')
    spreads_usa = pd.read_csv('outputs/spreads_usa.csv')

    context = {
        'bond_market_data': bond_market_data.to_html(),
        'spreads_usa': spreads_usa.to_html(),
    }
    return render(request, 'home.html', context)