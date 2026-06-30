#!/usr/bin/env python3
"""
Просмотр gap-free записей.
Показывает всю трассу с возможностью зума.
Помогает найти временные интервалы для каждого напряжения.
"""

import pyabf
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import numpy as np
from pathlib import Path

# ====================== НАСТРОЙКИ ======================

FILES = [
    "data/raw/26629000.abf",
    "data/raw/26629001.abf",
    "data/raw/26629002.abf",
    "data/raw/26629003.abf",
    "data/raw/26629004.abf",
]

DECIMATE = 50

# =======================================================


def browse_file(path: str) -> None:
    abf = pyabf.ABF(path)
    abf.setSweep(0)

    t = abf.sweepX
    y = abf.sweepY

    # Децимация
    t_dec = t[::DECIMATE]
    y_dec = y[::DECIMATE]

    name = Path(path).name
    duration = abf.sweepLengthSec

    print(f"\n{'='*60}")
    print(f"Файл: {name}")
    print(f"Длина: {duration:.0f} сек  "
          f"({duration/60:.1f} мин)")
    print(f"Отображаем каждую {DECIMATE}-ю точку")

    fig, ax = plt.subplots(figsize=(16, 5))

    ax.plot(t_dec, y_dec,
            color='steelblue',
            linewidth=0.5,
            alpha=0.85)

    ax.set_xlabel("Время (сек)", fontsize=12)
    ax.set_ylabel("Ток (pA)", fontsize=12)
    ax.set_title(
        f"{name}  |  {duration:.0f} сек  |  "
        f"{abf.dataRate} Гц  |  gap-free",
        fontsize=13
    )
    ax.grid(True, alpha=0.25)

    # Горизонтальные линии для ориентира
    ax.axhline(0, color='black', lw=0.8, linestyle='--')

    # Аннотация с подсказкой
    ax.text(
        0.01, 0.97,
        "Zoom → запиши t_start, t_end для каждого напряжения",
        transform=ax.transAxes,
        fontsize=9,
        va='top',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7)
    )

    plt.tight_layout()

    # Сохраняем обзорный график
    out = Path("results") / f"overview_{Path(path).stem}.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=100)
    print(f"  → сохранено: {out}")

    plt.show()


if __name__ == "__main__":
    print("ПРОСМОТР GAP-FREE ЗАПИСЕЙ")
    print("="*60)
    print("Для каждого файла:")
    print("  1. Смотри на трассу")
    print("  2. Находи участки стабильного тока")
    print("  3. Записывай время начала и конца")
    print("  4. Записывай напряжение (по знаку тока)")
    print("="*60)

    for f in FILES:
        browse_file(f)