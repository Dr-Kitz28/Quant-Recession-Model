import numpy as np

# Simulate the exact logic from frame_api.py
data = np.load('recession_project/outputs/correlation_tensor_usa.npz', allow_pickle=True, mmap_mode='r')

spreads = data["spreads"]
corr = data["corr"]
corr_scaled = data["corr_scaled"]

print(f'Original shapes:')
print(f'  spreads: {spreads.shape}')
print(f'  corr: {corr.shape}')
print(f'  corr_scaled: {corr_scaled.shape}')

# Parse spreads
def parse_maturity(s):
    if s.endswith('Y'):
        return int(s[:-1]) * 12
    elif s.endswith('M'):
        return int(s[:-1])
    return 0

def get_spread_diff(spread_name):
    parts = spread_name.split('-')
    if len(parts) == 2:
        return parse_maturity(parts[0]) - parse_maturity(parts[1])
    return 0

# Create sorting index
spread_diffs = [(i, get_spread_diff(str(s))) for i, s in enumerate(spreads)]
spread_diffs.sort(key=lambda x: x[1])
sort_indices = [i for i, _ in spread_diffs]

print(f'\nSort indices: {len(sort_indices)} elements')
print(f'First 10: {sort_indices[:10]}')

# Reorder spreads
sorted_spreads = spreads[sort_indices]
print(f'\nSorted spreads shape: {sorted_spreads.shape}')
print(f'First 5 spreads: {sorted_spreads[:5]}')

# Test EXACT reordering from frame_api.py line 87
sorted_corr = corr[:, sort_indices, :][:, :, sort_indices]
sorted_corr_scaled = corr_scaled[:, sort_indices, :][:, :, sort_indices]

print(f'\nAfter reordering:')
print(f'  sorted_corr shape: {sorted_corr.shape}')
print(f'  sorted_corr_scaled shape: {sorted_corr_scaled.shape}')

# Check a single frame
frame_0 = sorted_corr_scaled[0]
print(f'\nFrame 0 shape: {frame_0.shape}')
print(f'Frame 0 as float32 shape: {np.asarray(frame_0, dtype=np.float32).shape}')
