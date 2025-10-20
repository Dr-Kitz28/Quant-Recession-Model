import numpy as np

data = np.load('recession_project/outputs/correlation_tensor_usa.npz', allow_pickle=True)
print(f'Original corr shape: {data["corr"].shape}')
print(f'Spreads shape: {data["spreads"].shape}')

# Test the sorting logic
spreads = data["spreads"]
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

spread_diffs = [(i, get_spread_diff(str(s))) for i, s in enumerate(spreads)]
spread_diffs.sort(key=lambda x: x[1])
sort_indices = [i for i, _ in spread_diffs]

print(f'Sort indices length: {len(sort_indices)}')
print(f'First 10 indices: {sort_indices[:10]}')

# Test matrix reordering
corr = data["corr"]
print(f'\nOriginal corr shape: {corr.shape}')

# Wrong way (current code)
sorted_corr_wrong = corr[:, sort_indices, :][:, :, sort_indices]
print(f'Wrong reordering shape: {sorted_corr_wrong.shape}')

# Correct way using np.ix_
sorted_corr_correct = corr[:, sort_indices, :][:, :, sort_indices]
print(f'After first indexing: {corr[:, sort_indices, :].shape}')
print(f'After second indexing: {sorted_corr_wrong.shape}')
