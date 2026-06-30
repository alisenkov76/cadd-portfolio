"""
ДИАГНОСТИКА ФАЙЛОВ

Показывает:
- Число sweeps в каждом файле
- Трассы всех sweeps
- Средний ток и std по каждому sweep
- Гистограмму амплитуд (чтобы видеть уровни)
"""

import pyabf
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# ====================== НАСТРОЙКИ ======================

ALL_FILES = [
    "data/raw/26629000.abf",
    "data/raw/26629001.abf",
    "data/raw/26629002.abf",
    "data/raw/26629003.abf",
    "data/raw/26629004.abf",
]

PREVIEW_SECONDS = 35   # сколько секунд показывать
HIST_BINS = 200        # бины гистограммы амплитуд

# =======================================================


def inspect_file(path: str) -> None:
    if not os.path.exists(path):
        print(f"⚠️  Не найден: {path}")
        return

    abf = pyabf.ABF(path)
    name = Path(path).name

    print(f"\n{'='*60}")
    print(f"ФАЙЛ: {name}")
    print(f"  Sweeps:    {abf.sweepCount}")
    print(f"  Длина:     {abf.sweepLengthSec:.1f} с")
    print(f"  Частота:   {abf.dataRate} Гц")
    print(f"  Ед. тока:  {abf.sweepUnitsY}")

    # --- Таблица средних токов ---
    print("\n  Sweep  |  mean (pA)  |  std (pA)")
    print("  " + "-"*36)
    for i in range(abf.sweepCount):
        abf.setSweep(i)
        m = np.mean(abf.sweepY)
        s = np.std(abf.sweepY)
        print(f"  {i:5d}  |  {m:10.2f}  |  {s:9.2f}")

    # --- Графики трасс ---
    n = abf.sweepCount
    cols = 2 if n > 2 else 1
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols,
                             figsize=(13, 3.0 * rows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for i in range(n):
        abf.setSweep(i)
        t = abf.sweepX
        y = abf.sweepY

        ax = axes_flat[i]
        ax.plot(t, y,
                color='steelblue',
                linewidth=0.6,
                alpha=0.9)

        ax.set_title(f"Sweep {i}", fontsize=10)
        ax.set_ylabel("pA", fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0, min(PREVIEW_SECONDS, abf.sweepLengthSec))

        m = np.mean(y)
        s = np.std(y)
        ax.text(0.98, 0.95,
                f"μ={m:.1f}  σ={s:.1f}",
                transform=ax.transAxes,
                fontsize=8,
                ha='right', va='top',
                bbox=dict(boxstyle='round',
                          facecolor='white',
                          alpha=0.85))

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.suptitle(name, fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = Path("results") / f"inspect_{Path(path).stem}.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"  → сохранено: {out}")

    # --- Гистограмма амплитуд по всем sweeps ---
    fig2, axes2 = plt.subplots(rows, cols,
                               figsize=(13, 3.0 * rows),
                               squeeze=False)
    axes2_flat = axes2.flatten()

    for i in range(n):
        abf.setSweep(i)
        y = abf.sweepY
        ax = axes2_flat[i]
        ax.hist(y, bins=HIST_BINS,
                color='steelblue',
                edgecolor='none',
                alpha=0.8,
                orientation='vertical')
        ax.set_title(f"Sweep {i} — histogram", fontsize=10)
        ax.set_xlabel("pA", fontsize=9)
        ax.set_ylabel("counts", fontsize=9)
        ax.grid(True, alpha=0.25)

    for j in range(n, len(axes2_flat)):
        axes2_flat[j].axis('off')

    plt.suptitle(f"{name} — амплитудные гистограммы",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out2 = Path("results") / f"hist_{Path(path).stem}.png"
    plt.savefig(out2, dpi=150)
    plt.show()
    print(f"  → сохранено: {out2}")


if __name__ == "__main__":
    print("ДИАГНОСТИКА ФАЙЛОВ ГРАМИЦИДИНА A")
    print("="*60)
    for f in ALL_FILES:
        inspect_file(f)
    print("\n" + "="*60)
    print("ДАЛЬШЕ:")
    print("1. Запиши sweep → напряжение для каждого файла")
    print("2. Выбери временные окна без сильного шума")
    print("3. Запускай gramicidin_iv_analysis.py")
    print("="*60)