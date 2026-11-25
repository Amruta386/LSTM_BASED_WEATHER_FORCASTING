"""Train models ahead of time so the exhibition demo is instant.
Usage: python pretrain.py Delhi Mumbai Chennai
"""
import sys
from core import geocode, train_city

for name in sys.argv[1:] or ["Delhi"]:
    hits = geocode(name, count=1)
    if not hits:
        print(f"City not found: {name}")
        continue
    c = hits[0]
    print(f"Training {c['name']}, {c.get('country', '')} ...")
    m = train_city(c["latitude"], c["longitude"],
                   progress=lambda f, e, logs: print(f"  epoch {e} loss={logs['loss']:.4f}"))
    print(f"  done: LSTM MAE {m['lstm_mae']:.2f} C vs baseline {m['baseline_mae']:.2f} C")
