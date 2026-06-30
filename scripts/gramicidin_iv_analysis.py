#!/usr/bin/env python3
"""
АНАЛИЗ ОДИНОЧНОГО ФАЙЛА — ток одного канала через гистограмму амплитуд.

Логика:
1. Для каждого sweep выбираем временное окно
2. Маскируем шумовые участки
3. Строим гистограмму амплитуд
4. Находим два пика (закрыто / открыто)
5. delta_I = I_open - I_closed
"""

import pyabf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal as scipy_signal
from scipy.signal import find_peaks
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

# ====================== НАСТРОЙКИ ======================

MAIN_FILE    = "gramicidin_run1.abf"
CONTROL_FILE = "control.abf"

# sweep → напряжение (mV)
SWEEP_VOLTAGE_MAP = {
    0:  50,
    1: -50,
    2:  100,
    3: -100,
    4:  150,
    5: -150,
}

# Временные окна (сек): стабильные участки
MEASUREMENT_WINDOWS = {
     50:  (8.0, 25.0),
    -50:  (8.0, 25.0),
     100: (6.0, 20.0),
    -100: (6.0, 20.0),
     150: (4.0, 16.0),
    -150: (4.0, 16.0),
}

# Фильтрация
USE_FILTER  = True
LOWPASS_HZ  = 400

# Маскировка шума
NOISE_WINDOW_MS  = 50.0    # размер окна оценки шума (мс)
NOISE_THRESHOLD  = 4.0     # порог в единицах MAD

# Гистограмма
HIST_BINS        = 200
PEAK_MIN_HEIGHT  = 0.08    # минимальная высота пика (доля от max)
PEAK_MIN_DIST    = 15      # минимальное расстояние между пиками (бины)

# =======================================================


def lowpass_filter(data: np.ndarray,
                   fs: float,
                   cutoff: float) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = scipy_signal.butter(4, cutoff / nyq, btype='low')
    return scipy_signal.filtfilt(b, a, data)


def mask_noisy_segments(current: np.ndarray,
                         fs: float,
                         window_ms: float = 50.0,
                         threshold_sigma: float = 4.0) -> np.ndarray:
    """
    Возвращает булеву маску: True = хорошие точки.
    Участки где локальный std > threshold * MAD помечаются как шум.
    """
    window = int((window_ms / 1000.0) * fs)
    if window < 2:
        return np.ones(len(current), dtype=bool)

    global_mad = 1.4826 * np.median(
        np.abs(current - np.median(current))
    )

    mask = np.ones(len(current), dtype=bool)
    step = max(1, window // 2)

    for i in range(0, len(current) - window, step):
        chunk = current[i:i + window]
        if np.std(chunk) > threshold_sigma * global_mad:
            mask[i:i + window] = False

    frac_bad = 1.0 - mask.mean()
    if frac_bad > 0.0:
        print(f"    Шум: исключено {frac_bad*100:.1f}% точек")

    return mask


def get_single_channel_current(
        current: np.ndarray,
        bins: int = 200,
        peak_min_height: float = 0.08,
        peak_min_dist: int = 15) -> Dict:
    """
    Находит ток одного канала через два пика гистограммы амплитуд.

    Возвращает dict:
        I_closed, I_open, delta_I, sem_delta, n_peaks, histogram
    """
    hist, edges = np.histogram(current, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0

    peaks, _ = find_peaks(
        hist,
        height=hist.max() * peak_min_height,
        distance=peak_min_dist
    )

    result = {
        'I_closed':  np.nan,
        'I_open':    np.nan,
        'delta_I':   np.nan,
        'sem_delta': np.nan,
        'n_peaks':   len(peaks),
        'hist':      (hist, centers)
    }

    if len(peaks) < 2:
        # Один пик — берём как закрытое состояние
        result['I_closed'] = centers[np.argmax(hist)]
        return result

    # Берём два самых высоких пика
    top2 = peaks[np.argsort(hist[peaks])[::-1][:2]]
    top2 = top2[np.argsort(centers[top2])]  # сортируем по току

    result['I_closed'] = centers[top2[0]]
    result['I_open']   = centers[top2[1]]
    result['delta_I']  = abs(result['I_open'] - result['I_closed'])

    # Погрешность через FWHM каждого пика → σ = FWHM / 2.355
    def peak_sigma(peak_idx: int) -> float:
        h_half = hist[peak_idx] / 2.0
        left = peak_idx
        right = peak_idx
        while left > 0 and hist[left] > h_half:
            left -= 1
        while right < len(hist) - 1 and hist[right] > h_half:
            right += 1
        fwhm = centers[right] - centers[left]
        return fwhm / 2.355

    sig_c = peak_sigma(top2[0])
    sig_o = peak_sigma(top2[1])
    result['sem_delta'] = np.sqrt(sig_c**2 + sig_o**2)

    return result


def analyze_sweep(abf: pyabf.ABF,
                  sweep: int,
                  window: Tuple[float, float],
                  voltage_mV: int) -> Optional[Dict]:
    """
    Полный анализ одного sweep:
    фильтрация → маска шума → гистограмма → delta_I
    """
    abf.setSweep(sweep)
    t = abf.sweepX
    y = abf.sweepY.copy()

    # Временное окно
    mask_t = (t >= window[0]) & (t <= window[1])
    if not np.any(mask_t):
        print(f"    ⚠️  Окно {window} вне диапазона sweep {sweep}")
        return None

    current = y[mask_t]

    # Фильтрация
    if USE_FILTER:
        current = lowpass_filter(current, abf.dataRate, LOWPASS_HZ)

    # Маска шума
    noise_mask = mask_noisy_segments(
        current, abf.dataRate,
        window_ms=NOISE_WINDOW_MS,
        threshold_sigma=NOISE_THRESHOLD
    )
    current_clean = current[noise_mask]

    if len(current_clean) < 100:
        print(f"    ⚠️  Мало точек после маскировки ({len(current_clean)})")
        return None

    # Гистограмма → delta_I
    res = get_single_channel_current(
        current_clean,
        bins=HIST_BINS,
        peak_min_height=PEAK_MIN_HEIGHT,
        peak_min_dist=PEAK_MIN_DIST
    )

    print(f"    sweep {sweep} | {voltage_mV:+d} mV | "
          f"I_closed={res['I_closed']:.2f} pA | "
          f"I_open={res['I_open']:.2f} pA | "
          f"ΔI={res['delta_I']:.2f} ± {res['sem_delta']:.2f} pA | "
          f"пиков={res['n_peaks']}")

    return {
        'sweep':      sweep,
        'voltage_mV': voltage_mV,
        'I_closed':   res['I_closed'],
        'I_open':     res['I_open'],
        'delta_I':    res['delta_I'],
        'sem_delta':  res['sem_delta'],
        'n_peaks':    res['n_peaks'],
        'n_points':   len(current_clean),
        'hist':       res['hist'],
        'current':    current_clean
    }


def plot_diagnostics(results: list,
                     file_name: str) -> None:
    """
    Для каждого sweep — трасса + гистограмма с найденными уровнями.
    """
    n = len(results)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 2,
                             figsize=(14, 3.5 * n),
                             squeeze=False)

    for row, r in enumerate(results):
        v = r['voltage_mV']
        current = r['current']
        hist, centers = r['hist']

        # Трасса
        ax_t = axes[row, 0]
        ax_t.plot(current,
                  color='steelblue',
                  linewidth=0.5,
                  alpha=0.8)
        if not np.isnan(r['I_closed']):
            ax_t.axhline(r['I_closed'],
                         color='green',
                         linewidth=1.5,
                         linestyle='--',
                         label=f"closed={r['I_closed']:.1f}")
        if not np.isnan(r['I_open']):
            ax_t.axhline(r['I_open'],
                         color='red',
                         linewidth=1.5,
                         linestyle='--',
                         label=f"open={r['I_open']:.1f}")
        ax_t.set_title(f"{v:+d} mV — трасса (sweep {r['sweep']})")
        ax_t.set_ylabel("pA")
        ax_t.legend(fontsize=8)
        ax_t.grid(True, alpha=0.25)

        # Гистограмма
        ax_h = axes[row, 1]
        ax_h.bar(centers, hist,
                 width=(centers[1] - centers[0]),
                 color='steelblue',
                 alpha=0.7,
                 edgecolor='none')
        if not np.isnan(r['I_closed']):
            ax_h.axvline(r['I_closed'],
                         color='green',
                         linewidth=2,
                         label=f"closed={r['I_closed']:.1f}")
        if not np.isnan(r['I_open']):
            ax_h.axvline(r['I_open'],
                         color='red',
                         linewidth=2,
                         label=f"open={r['I_open']:.1f}")
        ax_h.set_title(f"{v:+d} mV — гистограмма")
        ax_h.set_xlabel("pA")
        ax_h.set_ylabel("counts")
        ax_h.legend(fontsize=8)
        ax_h.grid(True, alpha=0.25)

    plt.suptitle(f"Диагностика: {file_name}",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = Path("results") / f"diag_{Path(file_name).stem}.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"  → сохранено: {out}")


def analyze_file(path: str,
                 voltage_map: Dict[int, int],
                 windows: Dict[int, Tuple[float, float]],
                 is_control: bool = False) -> pd.DataFrame:

    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}")

    abf = pyabf.ABF(path)
    name = Path(path).name
    print(f"\n{'='*60}")
    print(f"{'КОНТРОЛЬ' if is_control else 'ЭКСПЕРИМЕНТ'}: {name}")
    print(f"  sweeps={abf.sweepCount}  "
          f"fs={abf.dataRate} Hz  "
          f"len={abf.sweepLengthSec:.1f}s")

    results = []

    for sweep, voltage in voltage_map.items():
        if sweep >= abf.sweepCount:
            print(f"  ⚠️  sweep {sweep} отсутствует")
            continue
        if voltage not in windows:
            print(f"  ⚠️  нет окна для {voltage} mV")
            continue

        res = analyze_sweep(abf, sweep, windows[voltage], voltage)
        if res is None:
            continue

        res['file'] = name
        res['is_control'] = is_control
        results.append(res)

    if results and not is_control:
        plot_diagnostics(results, name)

    df = pd.DataFrame([
        {k: v for k, v in r.items()
         if k not in ('hist', 'current')}
        for r in results
    ])

    return df


# ====================== ГЛАВНАЯ ПРОГРАММА ======================

if __name__ == "__main__":
    print("="*60)
    print("АНАЛИЗ ВАХ ГРАМИЦИДИНА A")
    print("="*60)

    exp_df = analyze_file(
        MAIN_FILE,
        SWEEP_VOLTAGE_MAP,
        MEASUREMENT_WINDOWS,
        is_control=False
    )

    ctrl_df = pd.DataFrame()
    if os.path.exists(CONTROL_FILE):
        ctrl_df = analyze_file(
            CONTROL_FILE,
            SWEEP_VOLTAGE_MAP,
            MEASUREMENT_WINDOWS,
            is_control=True
        )

    # Коррекция на контроль (вычитаем leak)
    if not ctrl_df.empty:
        ctrl_mean = (
            ctrl_df.groupby('voltage_mV')['delta_I']
            .mean()
            .reset_index()
            .rename(columns={'delta_I': 'leak_I'})
        )
        exp_df = exp_df.merge(ctrl_mean,
                              on='voltage_mV',
                              how='left')
        exp_df['corrected_delta_I'] = (
            exp_df['delta_I'] - exp_df['leak_I'].fillna(0)
        )
    else:
        exp_df['corrected_delta_I'] = exp_df['delta_I']
        print("\nКонтроль не найден — работаем без leak-коррекции")

    # Проводимость
    exp_df['conductance_pS'] = (
        exp_df['corrected_delta_I'] /
        exp_df['voltage_mV'].abs() * 1000
    )

    print("\n" + "="*60)
    print("ИТОГ:")
    cols = ['voltage_mV', 'delta_I', 'sem_delta',
            'corrected_delta_I', 'conductance_pS']
    print(exp_df[cols].to_string(index=False))

    Path("results").mkdir(exist_ok=True)
    exp_df.drop(columns=['hist', 'current'],
                errors='ignore').to_csv(
        "results/iv_single_channel.csv", index=False
    )

    # ВАХ
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        exp_df['voltage_mV'],
        exp_df['corrected_delta_I'],
        yerr=exp_df['sem_delta'],
        fmt='o-',
        color='steelblue',
        capsize=5,
        capthick=2,
        markersize=8,
        linewidth=2,
        label='Gramicidin A single channel ΔI'
    )
    ax.axhline(0, color='black', lw=0.8)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel("Voltage (mV)", fontsize=12)
    ax.set_ylabel("ΔI single channel (pA)", fontsize=12)
    ax.set_title("Gramicidin A — I–V curve\n"
                 "DOPhC membrane, 2M KCl", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("results/iv_curve.png", dpi=300)
    plt.show()
    print("  → results/iv_curve.png")