#!/usr/bin/env python3
"""
АНАЛИЗ ВАХ ДЛЯ GAP-FREE ЗАПИСЕЙ.

Пайплайн:
1. parse_tags → таблица интервалов
2. Для каждого интервала: фильтрация + маска шума
3. Гистограмма амплитуд → delta_I
4. Pooled SEM по напряжению
5. ВАХ с погрешностями
"""

import pyabf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from scipy.signal import find_peaks
from pathlib import Path
from typing import Dict, Tuple, Optional, List

from parse_tags import get_voltage_intervals

# ====================== НАСТРОЙКИ ======================

# Контрольный файл (мембрана без грамицидина)
CONTROL_FILE = "data/raw/26629000.abf"

# Экспериментальные файлы
EXP_FILES = [
    "data/raw/26629001.abf",
    "data/raw/26629002.abf",
    "data/raw/26629003.abf",
    "data/raw/26629004.abf",
]

# Минимальная длительность интервала (сек)
MIN_DURATION_SEC = 30.0

# Пропускать нулевое напряжение
SKIP_ZERO = True

# Фильтрация
USE_FILTER  = True
LOWPASS_HZ  = 400.0

# Маска шума
NOISE_WINDOW_MS = 50.0
NOISE_THRESHOLD = 4.0

# Гистограмма
HIST_BINS       = 300
PEAK_MIN_HEIGHT = 0.08
PEAK_MIN_DIST   = 20

# Напряжения для финальной ВАХ
# (оставляем только симметричные стандартные)
TARGET_VOLTAGES = [-150, -100, -75, -50,
                    50,   75,  100, 150]

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
                         threshold: float = 4.0) -> np.ndarray:
    window = max(2, int((window_ms / 1000.0) * fs))
    global_mad = 1.4826 * np.median(
        np.abs(current - np.median(current))
    )

    mask = np.ones(len(current), dtype=bool)
    step = max(1, window // 2)

    for i in range(0, len(current) - window, step):
        chunk = current[i:i + window]
        if np.std(chunk) > threshold * global_mad:
            mask[i:i + window] = False

    frac_bad = 1.0 - mask.mean()
    if frac_bad > 0.01:
        print(f"      шум: исключено {frac_bad*100:.1f}% точек")

    return mask


def get_single_channel_current(
        current: np.ndarray,
        bins: int = 300,
        peak_min_height: float = 0.08,
        peak_min_dist: int = 20) -> Dict:
    """
    Два пика гистограммы → I_closed, I_open, delta_I, sem_delta.
    """
    hist, edges = np.histogram(current, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0

    peaks, _ = find_peaks(
        hist,
        height=hist.max() * peak_min_height,
        distance=peak_min_dist
    )

    result = {
        'I_closed':  float(np.median(current)),
        'I_open':    np.nan,
        'delta_I':   np.nan,
        'sem_delta': np.nan,
        'n_peaks':   len(peaks),
        'hist':      (hist, centers)
    }

    if len(peaks) < 2:
        return result

    # Два самых высоких пика
    top2 = peaks[np.argsort(hist[peaks])[::-1][:2]]
    top2 = top2[np.argsort(centers[top2])]

    result['I_closed'] = float(centers[top2[0]])
    result['I_open']   = float(centers[top2[1]])
    result['delta_I']  = float(
        abs(result['I_open'] - result['I_closed'])
    )

    # Погрешность через FWHM → σ
    def peak_sigma(idx: int) -> float:
        h_half = hist[idx] / 2.0
        l, r = idx, idx
        while l > 0 and hist[l] > h_half:
            l -= 1
        while r < len(hist) - 1 and hist[r] > h_half:
            r += 1
        return (centers[r] - centers[l]) / 2.355

    result['sem_delta'] = float(np.sqrt(
        peak_sigma(top2[0])**2 + peak_sigma(top2[1])**2
    ))

    return result


def analyze_interval(abf: pyabf.ABF,
                     t_start: float,
                     t_end: float,
                     voltage_mV: int) -> Optional[Dict]:
    """
    Анализ одного временного интервала.
    """
    abf.setSweep(0)
    t = abf.sweepX
    y = abf.sweepY.copy()

    mask_t = (t >= t_start) & (t <= t_end)
    current = y[mask_t]

    if len(current) < 500:
        print(f"      ⚠️  мало точек: {len(current)}")
        return None

    # Фильтрация
    if USE_FILTER:
        current = lowpass_filter(
            current, abf.dataRate, LOWPASS_HZ
        )

    # Маска шума
    noise_mask = mask_noisy_segments(
        current, abf.dataRate,
        window_ms=NOISE_WINDOW_MS,
        threshold=NOISE_THRESHOLD
    )
    current_clean = current[noise_mask]

    if len(current_clean) < 200:
        print(f"      ⚠️  мало точек после маски: "
              f"{len(current_clean)}")
        return None

    # Гистограмма → delta_I
    res = get_single_channel_current(
        current_clean,
        bins=HIST_BINS,
        peak_min_height=PEAK_MIN_HEIGHT,
        peak_min_dist=PEAK_MIN_DIST
    )

    print(f"      {voltage_mV:+5d} mV  "
          f"[{t_start:.0f}–{t_end:.0f}s]  "
          f"I_cl={res['I_closed']:.2f}  "
          f"I_op={res['I_open']:.2f}  "
          f"ΔI={res['delta_I']:.2f} ± "
          f"{res['sem_delta']:.2f} pA  "
          f"peaks={res['n_peaks']}")

    return {
        'voltage_mV': voltage_mV,
        't_start':    t_start,
        't_end':      t_end,
        'duration':   t_end - t_start,
        'I_closed':   res['I_closed'],
        'I_open':     res['I_open'],
        'delta_I':    res['delta_I'],
        'sem_delta':  res['sem_delta'],
        'n_peaks':    res['n_peaks'],
        'n_points':   len(current_clean),
        'hist':       res['hist'],
        'current':    current_clean
    }


def analyze_file(path: str,
                 is_control: bool = False) -> pd.DataFrame:
    """
    Анализ одного gap-free файла.
    """
    name = Path(path).name
    print(f"\n{'='*60}")
    print(f"{'КОНТРОЛЬ' if is_control else 'ЭКСПЕРИМЕНТ'}: "
          f"{name}")

    # Автоматически парсим интервалы из тегов
    intervals = get_voltage_intervals(
        path,
        min_duration_sec=MIN_DURATION_SEC,
        skip_zero=SKIP_ZERO
    )

    if intervals.empty:
        print("  Нет подходящих интервалов")
        return pd.DataFrame()

    abf = pyabf.ABF(path)
    results = []
    hist_data = []

    for _, row in intervals.iterrows():
        res = analyze_interval(
            abf,
            row['t_start'],
            row['t_end'],
            int(row['voltage_mV'])
        )
        if res is None:
            continue

        res['file'] = name
        res['is_control'] = is_control
        hist_data.append(res)

        results.append({
            k: v for k, v in res.items()
            if k not in ('hist', 'current')
        })

    # Диагностические графики (только для эксперимента)
    if hist_data and not is_control:
        plot_diagnostics(hist_data, name)

    return pd.DataFrame(results)


def plot_diagnostics(results: List[Dict],
                     file_name: str) -> None:
    """
    Трасса + гистограмма для каждого интервала.
    """
    n = len(results)
    if n == 0:
        return

    fig, axes = plt.subplots(
        n, 2,
        figsize=(14, 3.5 * n),
        squeeze=False
    )

    for row_idx, r in enumerate(results):
        v = r['voltage_mV']
        current = r['current']
        hist, centers = r['hist']

        # Трасса
        ax_t = axes[row_idx, 0]
        ax_t.plot(current,
                  color='steelblue',
                  linewidth=0.4,
                  alpha=0.8)

        if not np.isnan(r['I_closed']):
            ax_t.axhline(
                r['I_closed'],
                color='green', lw=1.5,
                linestyle='--',
                label=f"closed={r['I_closed']:.1f} pA"
            )
        if not np.isnan(r['I_open']):
            ax_t.axhline(
                r['I_open'],
                color='red', lw=1.5,
                linestyle='--',
                label=f"open={r['I_open']:.1f} pA"
            )

        ax_t.set_title(
            f"{v:+d} mV  "
            f"[{r['t_start']:.0f}–{r['t_end']:.0f} s]"
        )
        ax_t.set_ylabel("pA")
        ax_t.legend(fontsize=8)
        ax_t.grid(True, alpha=0.25)

        # Гистограмма
        ax_h = axes[row_idx, 1]
        ax_h.bar(centers, hist,
                 width=(centers[1] - centers[0]),
                 color='steelblue',
                 alpha=0.75,
                 edgecolor='none')

        if not np.isnan(r['I_closed']):
            ax_h.axvline(r['I_closed'],
                         color='green', lw=2,
                         label=f"closed={r['I_closed']:.1f}")
        if not np.isnan(r['I_open']):
            ax_h.axvline(r['I_open'],
                         color='red', lw=2,
                         label=f"open={r['I_open']:.1f}")

        ax_h.set_title(
            f"ΔI = {r['delta_I']:.2f} ± "
            f"{r['sem_delta']:.2f} pA"
        )
        ax_h.set_xlabel("pA")
        ax_h.set_ylabel("counts")
        ax_h.legend(fontsize=8)
        ax_h.grid(True, alpha=0.25)

    plt.suptitle(
        f"Диагностика: {file_name}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()

    out = Path("results") / f"diag_{Path(file_name).stem}.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"  → {out}")


def pooled_sem(sems: np.ndarray,
               ns: np.ndarray) -> float:
    ns = ns.astype(float)
    pooled_var = np.sum(ns * sems**2) / np.sum(ns)
    return float(np.sqrt(pooled_var / np.sum(ns)))


def build_iv(exp_df: pd.DataFrame,
             ctrl_df: pd.DataFrame) -> pd.DataFrame:
    """
    Усредняет по напряжению, корректирует на контроль,
    считает проводимость.
    """
    rows = []
    for voltage, grp in exp_df.groupby('voltage_mV'):
        mean_dI = grp['delta_I'].mean()
        p_sem   = pooled_sem(
            grp['sem_delta'].values,
            grp['n_points'].values
        )
        rows.append({
            'voltage_mV':  voltage,
            'mean_delta_I': mean_dI,
            'pooled_sem':   p_sem,
            'n_intervals':  len(grp),
            'files':        ', '.join(grp['file'].unique())
        })

    summary = pd.DataFrame(rows).sort_values('voltage_mV')

    # Коррекция на контроль
    if not ctrl_df.empty:
        ctrl_mean = (
            ctrl_df.groupby('voltage_mV')
            .agg(
                leak_I=('delta_I', 'mean'),
                leak_sem=('sem_delta',
                          lambda x: float(
                              np.sqrt(np.mean(x**2))
                          ))
            )
            .reset_index()
        )
        summary = summary.merge(
            ctrl_mean, on='voltage_mV', how='left'
        )
        summary['corrected_dI'] = (
            summary['mean_delta_I'] -
            summary['leak_I'].fillna(0)
        )
        summary['corrected_sem'] = np.sqrt(
            summary['pooled_sem']**2 +
            summary['leak_sem'].fillna(0)**2
        )
    else:
        summary['corrected_dI']  = summary['mean_delta_I']
        summary['corrected_sem'] = summary['pooled_sem']
        summary['leak_I']        = np.nan

    # Проводимость
    mask_nonzero = summary['voltage_mV'] != 0
    summary.loc[mask_nonzero, 'conductance_pS'] = (
        summary.loc[mask_nonzero, 'corrected_dI'] /
        summary.loc[mask_nonzero, 'voltage_mV'].abs() * 1000
    )

    return summary


def plot_iv(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- ВАХ ---
    ax = axes[0]
    ax.errorbar(
        summary['voltage_mV'],
        summary['corrected_dI'],
        yerr=summary['corrected_sem'],
        fmt='o-',
        color='steelblue',
        capsize=5, capthick=2,
        markersize=9, linewidth=2.5,
        label='Gramicidin A ΔI'
    )

    if 'leak_I' in summary and summary['leak_I'].notna().any():
        ax.errorbar(
            summary['voltage_mV'],
            summary['leak_I'],
            fmt='s--',
            color='gray',
            capsize=3,
            markersize=7,
            alpha=0.7,
            label='Контроль (leak)'
        )

    ax.axhline(0, color='black', lw=0.8)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel("Voltage (mV)", fontsize=12)
    ax.set_ylabel("ΔI single channel (pA)", fontsize=12)
    ax.set_title(
        "I–V curve\nGramicidin A, DOPhC, 2M KCl",
        fontsize=13
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # --- Проводимость ---
    ax2 = axes[1]
    g_data = summary.dropna(subset=['conductance_pS'])
    ax2.plot(
        g_data['voltage_mV'],
        g_data['conductance_pS'],
        'o-',
        color='darkorange',
        markersize=9,
        linewidth=2.5
    )
    mean_g = g_data['conductance_pS'].mean()
    ax2.axhline(
        mean_g,
        color='gray',
        linestyle='--',
        label=f"mean = {mean_g:.1f} pS"
    )
    ax2.set_xlabel("Voltage (mV)", fontsize=12)
    ax2.set_ylabel("Conductance (pS)", fontsize=12)
    ax2.set_title(
        "Проводимость одного канала",
        fontsize=13
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)

    plt.suptitle(
        "Gramicidin A — полная ВАХ (gap-free, auto-tags)",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()

    out = Path("results") / "iv_curve_final.png"
    plt.savefig(out, dpi=300)
    plt.show()
    print(f"  → {out}")


# ====================== ГЛАВНАЯ ПРОГРАММА ======================

if __name__ == "__main__":
    print("="*60)
    print("АНАЛИЗ ВАХ ГРАМИЦИДИНА A (gap-free)")
    print("="*60)

    Path("results").mkdir(exist_ok=True)

    # Контроль
    ctrl_df = pd.DataFrame()
    if Path(CONTROL_FILE).exists():
        ctrl_df = analyze_file(CONTROL_FILE, is_control=True)
    else:
        print(f"⚠️  Контроль не найден: {CONTROL_FILE}")

    # Эксперимент
    exp_frames = []
    for f in EXP_FILES:
        if not Path(f).exists():
            print(f"⚠️  Не найден: {f}")
            continue
        df = analyze_file(f, is_control=False)
        if not df.empty:
            exp_frames.append(df)

    if not exp_frames:
        print("Нет данных!")
        raise SystemExit(1)

    exp_df = pd.concat(exp_frames, ignore_index=True)

    # Сохраняем сырые результаты
    exp_df.to_csv("results/iv_raw_intervals.csv", index=False)

    # Строим ВАХ
    summary = build_iv(exp_df, ctrl_df)

    print("\n" + "="*60)
    print("ИТОГОВАЯ ТАБЛИЦА:")
    cols = [
        'voltage_mV', 'corrected_dI',
        'corrected_sem', 'conductance_pS',
        'n_intervals'
    ]
    print(summary[cols].to_string(index=False))

    summary.to_csv("results/iv_summary.csv", index=False)
    print("\n  → results/iv_summary.csv")

    plot_iv(summary)

    # Итоговая статистика
    g_vals = summary['conductance_pS'].dropna()
    print("\n" + "="*60)
    print(f"Средняя проводимость: "
          f"{g_vals.mean():.1f} ± {g_vals.std():.1f} pS")
    print("Литература gA в DOPhC/KCl: ~14–17 pS")
    print("="*60)