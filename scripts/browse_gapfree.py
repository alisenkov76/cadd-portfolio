"""
Интерактивный просмотр gap-free записей.

Показывает трассу с тегами напряжения.
Помогает найти где начались каналы.

Запускется для каждого файла отдельно.
"""

import pyabf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ====================== НАСТРОЙКИ ======================

# Смотрим по одному файлу
FILE = "data/raw/26629000.abf"

# Децимация для отображения
# 5000 Гц × 3000 сек = 15М точек → берём каждую 25-ю
# итого ~600к точек → нормально
DECIMATE = 25

# Целевые напряжения
TARGET_VOLTAGES = {-150, -100, -50, 50, 100, 150}

# Цвета для разных напряжений
VOLTAGE_COLORS = {
     50:  '#2196F3',   # синий
    -50:  '#03A9F4',   # голубой
     100: '#FF9800',   # оранжевый
    -100: '#FF5722',   # красно-оранжевый
     150: '#F44336',   # красный
    -150: '#9C27B0',   # фиолетовый
}

# =======================================================


def parse_voltage(comment: str):
    import re
    m = re.search(r'=>\s*(-?\d+(?:\.\d+)?)\s*mV', comment)
    return float(m.group(1)) if m else None


def browse_file(path: str) -> None:
    abf = pyabf.ABF(path)
    abf.setSweep(0)

    t = abf.sweepX
    y = abf.sweepY

    # Децимация
    t_dec = t[::DECIMATE]
    y_dec = y[::DECIMATE]

    name     = Path(path).name
    duration = abf.sweepLengthSec

    print(f"\n{'='*60}")
    print(f"Файл:    {name}")
    print(f"Длина:   {duration:.0f} сек ({duration/60:.1f} мин)")
    print(f"Частота: {abf.dataRate} Гц")
    print(f"\nТеги напряжения:")

    # Парсим теги
    tag_times    = abf.tagTimesSec
    tag_comments = abf.tagComments

    valid_tags = []
    for t_tag, comment in zip(tag_times, tag_comments):
        v = parse_voltage(comment)
        if v is None:
            continue
        v_int = int(v)
        marker = "✅" if v_int in TARGET_VOLTAGES else "⏭ "
        print(f"  {marker} t={t_tag:8.1f}s  {v:+.0f} mV")
        if v_int in TARGET_VOLTAGES:
            valid_tags.append((t_tag, v_int))

    # --- Основной график ---
    fig, ax = plt.subplots(figsize=(18, 5))

    ax.plot(t_dec, y_dec,
            color='#455A64',
            linewidth=0.4,
            alpha=0.85,
            label='Ток (pA)')

    # Вертикальные линии тегов
    for t_tag, v_int in valid_tags:
        color = VOLTAGE_COLORS.get(v_int, 'gray')
        ax.axvline(
            t_tag,
            color=color,
            linewidth=1.5,
            alpha=0.8,
            linestyle='--'
        )
        ax.text(
            t_tag + duration * 0.003,
            ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 10,
            f"{v_int:+d}",
            fontsize=8,
            color=color,
            rotation=90,
            va='top'
        )

    ax.axhline(0, color='black', lw=0.8, linestyle=':')
    ax.set_xlabel("Время (сек)", fontsize=12)
    ax.set_ylabel("Ток (pA)", fontsize=12)
    ax.set_title(
        f"{name}  |  {duration:.0f} сек  |  "
        f"{abf.dataRate} Гц\n",
        fontsize=12
    )
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))


    plt.tight_layout()

    out = Path("results") / f"browse_{Path(path).stem}.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"\n→ сохранено: {out}")
    plt.show()


if __name__ == "__main__":
    browse_file(FILE)
    print("\nЧТО ДЕЛАТЬ:")
    print("1. Найди где начались ступеньки (каналы)")
    print("2. Запиши время начала каналов для каждого напряжения")
    print("3. Перенеси в MANUAL_EXCLUDE в gapfree_iv_analysis.py")