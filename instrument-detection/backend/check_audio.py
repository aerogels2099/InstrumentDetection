import os
import soundfile as sf
from collections import defaultdict

audio_root = os.path.join(os.path.dirname(__file__), "data", "audio")

stats = defaultdict(lambda: {"count": 0, "corrupt": [], "total_dur": 0.0, "exts": set()})

for inst in os.listdir(audio_root):
    inst_dir = os.path.join(audio_root, inst)
    if not os.path.isdir(inst_dir):
        continue
    for fname in os.listdir(inst_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
            continue
        fpath = os.path.join(inst_dir, fname)
        stats[inst]["exts"].add(ext)
        try:
            info = sf.info(fpath)
            stats[inst]["count"] += 1
            stats[inst]["total_dur"] += info.duration
        except Exception as e:
            stats[inst]["corrupt"].append((fname, str(e)))

header = f"{'Instrument':<14} {'Files':>6} {'Corrupt':>7} {'Total (min)':>12} {'Avg (s)':>8}  Formats"
print(header)
print("-" * len(header))

grand_files = grand_corrupt = 0
grand_dur = 0.0

for inst in sorted(stats):
    s = stats[inst]
    n = s["count"]
    c = len(s["corrupt"])
    dur_min = s["total_dur"] / 60.0
    avg = s["total_dur"] / n if n else 0.0
    grand_files += n
    grand_corrupt += c
    grand_dur += s["total_dur"]
    fmts = ", ".join(sorted(s["exts"]))
    print(f"{inst:<14} {n:>6} {c:>7} {dur_min:>11.1f}m {avg:>7.1f}s  {fmts}")
    for fname, err in s["corrupt"][:5]:
        print(f"    !! {fname}: {err}")

print("-" * len(header))
avg_all = grand_dur / grand_files if grand_files else 0.0
print(f"{'TOTAL':<14} {grand_files:>6} {grand_corrupt:>7} {grand_dur/60:>11.1f}m {avg_all:>7.1f}s")
print()

# Sufficiency assessment
MIN_FILES = 200
print("=== Sufficiency Assessment ===")
for inst in sorted(stats):
    n = stats[inst]["count"]
    dur = stats[inst]["total_dur"] / 60.0
    flag = "OK" if n >= MIN_FILES else "LOW"
    print(f"  [{flag}] {inst:<12} {n} files  ({dur:.1f} min)")
