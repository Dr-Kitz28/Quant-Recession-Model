from pathlib import Path
import sys

root = Path('D:/Downloads/QRM')
threshold = 100 * 1024 * 1024
large = []
for p in root.rglob('*'):
    try:
        if p.is_file():
            sz = p.stat().st_size
            if sz > threshold:
                large.append((p, sz))
    except Exception:
        continue

large.sort(key=lambda x: x[1], reverse=True)
out_lines = []
for p, sz in large:
    out_lines.append(f"{p.as_posix()}\t{sz}")
out_lines.append(f"FOUND {len(large)} large files")
# write to out file for inspection
out_path = Path(__file__).with_suffix('.out')
out_path.write_text('\n'.join(out_lines))
for line in out_lines:
    print(line)
if len(large) > 0:
    sys.exit(2)
else:
    sys.exit(0)
