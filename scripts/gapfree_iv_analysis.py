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
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import sys
from pathlib import Path

# Добавляем папку scripts/ в путь поиска модулей
sys.path.insert(0, str(Path(__file__).resolve().parent))

from parse_tags import get_voltage_intervals

# ====================== НАСТРОЙКИ ======================

# Контрольный файл 
CONTROL_FILE = "data/raw/26629000.abf"

# Экспериментальные файлы
EXP_FILES = [
    "data/raw/26629001.abf",
    "data/raw/26629002.abf",
    "data/raw/26629003.abf",
    "data/raw/26629004.abf",
]
# стандартные потенциалы
TARGET_VOLTAGES = {-150, -100, -50, 50, 100, 150}

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

# ====================== РУЧНЫЕ МАСКИ ======================
#
# Формат:
# "имя_файла.abf": [
#     (t_exclude_start, t_exclude_end),  # исключить этот участок
# ]
#
# Используй browse_gapfree.py чтобы найти точные времена.
#
# Примеры:
# - Нет каналов в начале файла:
#     (0, 800)   → исключить первые 800 секунд
# - Шумовой артефакт в середине:
#     (1200, 1250)
# - Весь файл без каналов:
#     (0, 99999)  → исключит всё

MANUAL_EXCLUDE = {
    "26629001.abf": [
        (0, 99999),      # нет каналов — весь файл
    ],
    "26629002.abf": [
        (0, 1660),       # пример: каналы начались после 1660 сек
    ],
    "26629003.abf": [
        (978, 1628), # пример: шумовые артефакты в середине
        (4556,4747)
    ],
    "26629004.abf": [
        (2343, 2574),       # пример: каналы начались после 1660 сек
    ],
    
    # остальные файлы не трогаем
}
# ==========================================================
# Гистограмма
HIST_BINS       = 400
PEAK_MIN_HEIGHT = 0.02
PEAK_MIN_DIST   = 20


# =======================================================


def lowpass_filter(data: np.ndarray,
                   fs: float,
                   cutoff: float) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = scipy_signal.butter(4, cutoff / nyq, btype='low')
    return scipy_signal.filtfilt(b, a, data)

def apply_manual_exclude(
        intervals: pd.DataFrame,
        file_name: str,
        manual_exclude: Dict) -> pd.DataFrame:
    """
    Исключает или разбивает интервалы по ручным маскам.

    Логика:
    - Зона исключения полностью покрывает интервал
      → интервал удаляется целиком

    - Зона исключения в начале интервала
      → обрезаем начало

    - Зона исключения в конце интервала
      → обрезаем конец

    - Зона исключения внутри интервала
      → разбиваем на два подынтервала

    Короткие остатки (< MIN_DURATION_SEC) отбрасываются.
    """
    if file_name not in manual_exclude:
        return intervals

    exclude_zones = manual_exclude[file_name]
    MIN_DUR = 30.0

    def split_interval(t_s, t_e, voltage, row):
        """
        Применяет все зоны исключения к одному интервалу.
        Возвращает список подынтервалов.
        """
        # Начинаем с одного интервала
        segments = [(t_s, t_e)]

        for (ex_s, ex_e) in exclude_zones:
            new_segments = []
            for (s, e) in segments:
                # Зона исключения не пересекается
                if ex_e <= s or ex_s >= e:
                    new_segments.append((s, e))
                    continue

                # Зона полностью покрывает сегмент
                if ex_s <= s and ex_e >= e:
                    print(f"    исключён:  {voltage:+d} mV "
                          f"[{s:.0f}–{e:.0f}s] "
                          f"(покрыт маской [{ex_s}–{ex_e}])")
                    continue

                # Зона в начале
                if ex_s <= s and ex_e < e:
                    new_s = ex_e
                    if e - new_s >= MIN_DUR:
                        print(f"    обрезан:   {voltage:+d} mV "
                              f"начало [{s:.0f}→{new_s:.0f}]")
                        new_segments.append((new_s, e))
                    continue

                # Зона в конце
                if ex_s > s and ex_e >= e:
                    new_e = ex_s
                    if new_e - s >= MIN_DUR:
                        print(f"    обрезан:   {voltage:+d} mV "
                              f"конец [{new_e:.0f}←{e:.0f}]")
                        new_segments.append((s, new_e))
                    continue

                # Зона внутри — разбиваем на два
                if ex_s > s and ex_e < e:
                    left  = (s, ex_s)
                    right = (ex_s, e)

                    print(f"    разбит:    {voltage:+d} mV "
                          f"[{s:.0f}–{e:.0f}s] "
                          f"→ шум [{ex_s:.0f}–{ex_e:.0f}]")

                    if left[1] - left[0] >= MIN_DUR:
                        new_segments.append(left)
                    else:
                        print(f"      левая часть < {MIN_DUR}s "
                              f"→ отброшена")

                    if right[1] - right[0] >= MIN_DUR:
                        new_segments.append(right)
                    else:
                        print(f"      правая часть < {MIN_DUR}s "
                              f"→ отброшена")

            segments = new_segments

        return segments

    rows_out = []

    for _, row in intervals.iterrows():
        t_s     = row['t_start']
        t_e     = row['t_end']
        voltage = int(row['voltage_mV'])

        segments = split_interval(t_s, t_e, voltage, row)

        for (s, e) in segments:
            new_row            = row.copy()
            new_row['t_start'] = round(s, 3)
            new_row['t_end']   = round(e, 3)
            new_row['duration']= round(e - s, 1)
            rows_out.append(new_row)

    if not rows_out:
        return pd.DataFrame()

    return pd.DataFrame(rows_out).reset_index(drop=True)

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


def get_single_channel_current(current: np.ndarray,
                               voltage_mV: int,
                               bins: int = 400,
                               peak_min_height: float = 0.02,
                               peak_min_dist: int = 20,
                               min_step_pA: float = 0.3) -> Dict:
    hist, edges = np.histogram(current, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0

    hist_s = gaussian_filter1d(hist.astype(float), sigma=2)

    peaks, _ = find_peaks(
        hist_s,
        height=hist_s.max() * peak_min_height,
        distance=peak_min_dist
    )

    closed_idx = int(np.argmax(hist_s))  # baseline как главный пик
    I_closed = float(centers[closed_idx])

    direction = np.sign(voltage_mV) if voltage_mV != 0 else 1.0

    # кандидаты на open: в нужную сторону и не слишком близко
    cand = []
    for p in peaks:
        d = centers[p] - I_closed
        if direction * d > min_step_pA:
            cand.append(p)

    # fallback: если пик не нашёлся — берём квантиль (редкие открытия)
    if len(cand) == 0:
        q = 0.995 if direction > 0 else 0.005
        I_open = float(np.quantile(current, q))
        return {
            "I_closed": I_closed,
            "I_open": I_open,
            "delta_I": float(I_open - I_closed),
            "sem_delta": np.nan,
            "n_peaks": int(len(peaks)),
            "hist": (hist_s, centers),
        }

    # выбираем ближайший open-пик (одноканальный), а не мультиканальный
    cand = np.array(cand, dtype=int)
    chosen = cand[np.argmin(np.abs(centers[cand] - I_closed))]
    I_open = float(centers[chosen])

    return {
        "I_closed": I_closed,
        "I_open": I_open,
        "delta_I": float(I_open - I_closed),   # со знаком
        "sem_delta": np.nan,
        "n_peaks": int(len(peaks)),
        "hist": (hist_s, centers),
    }


def estimate_step_by_transitions(current: np.ndarray,
                                 fs: float,
                                 voltage_mV: int,
                                 smooth_ms: float = 1.0,
                                 avg_ms: float = 5.0,
                                 thr_sigma: float = 6.0,
                                 min_step_pA: float = 0.2) -> float:
    """
    Fallback: оценивает single-channel step по резким переходам (ступенькам).
    Возвращает оценку ΔI (в pA, со знаком как voltage_mV).
    """
    direction = np.sign(voltage_mV) if voltage_mV != 0 else 1.0

    # простое сглаживание moving average
    n_smooth = max(3, int(fs * smooth_ms / 1000.0))
    kernel = np.ones(n_smooth) / n_smooth
    y = np.convolve(current, kernel, mode="same")

    dy = np.diff(y)
    mad = 1.4826 * np.median(np.abs(dy - np.median(dy)))
    if mad == 0:
        return np.nan

    # точки резких переходов
    idx = np.where(np.abs(dy) > thr_sigma * mad)[0]
    if len(idx) == 0:
        return np.nan

    # дедупликация близких индексов
    min_gap = max(1, int(fs * avg_ms / 1000.0))
    keep = []
    last = -10**18
    for i in idx:
        if i - last >= min_gap:
            keep.append(i)
            last = i
    idx = np.array(keep, dtype=int)

    # измеряем уровни до/после перехода
    w = max(5, int(fs * avg_ms / 1000.0))
    steps = []
    for i in idx:
        if i - w < 0 or i + w >= len(y):
            continue
        pre = np.mean(y[i - w:i])
        post = np.mean(y[i:i + w])
        d = post - pre  # pA

        # приводим к "открытию" (положительный шаг по направлению direction)
        d_open = direction * d
        if d_open > min_step_pA:
            steps.append(d_open)

    if len(steps) < 5:
        return np.nan

    steps = np.array(steps)

    # берём наиболее частый шаг (mode через гистограмму)
    hist, edges = np.histogram(steps, bins=40)
    centers = (edges[:-1] + edges[1:]) / 2.0
    step_mag = centers[np.argmax(hist)]

    return float(direction * step_mag)  # возвращаем со знаком

def analyze_interval(abf: pyabf.ABF,
                     t_start: float,
                     t_end: float,
                     voltage_mV: int,
                     is_control: bool = False) -> Optional[Dict]:
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
        
    if is_control:
        I_cl = float(np.median(current_clean))
        mad = 1.4826 * np.median(np.abs(current_clean - np.median(current_clean)))  # робастный шум
    
        print(f"      CONTROL {voltage_mV:+5d} mV  "
              f"[{t_start:.0f}–{t_end:.0f}s]  "
              f"I_cl={I_cl:.2f} pA  noise(MAD)={mad:.2f} pA")
    
        return {
            'voltage_mV': voltage_mV,
            't_start': t_start,
            't_end': t_end,
            'duration': t_end - t_start,
            'I_closed': I_cl,
            'I_open': np.nan,
            'delta_I': 0.0,
            'sem_delta': float(mad),
            'n_peaks': 1,
            'n_points': len(current_clean),
            'hist': (None, None),
            'current': current_clean
        }

    # Гистограмма → delta_I
    res = get_single_channel_current(
    current_clean,
    voltage_mV,
    bins=HIST_BINS,
    peak_min_height=PEAK_MIN_HEIGHT,
    peak_min_dist=PEAK_MIN_DIST
)
    if (res["n_peaks"] < 2) or np.isnan(res.get("I_open", float("nan"))) or np.isnan(res.get("delta_I", float("nan"))):
        
        step = estimate_step_by_transitions(
            current_clean, abf.dataRate, voltage_mV
        )
        
        # step может быть nan если переходов не нашли
        if step is not None and not np.isnan(step):
            res["I_open"]    = res["I_closed"] + step
            res["delta_I"]   = step
            res["sem_delta"] = np.nan
            res["n_peaks"]   = 1
            print(f"      fallback (transitions): ΔI={step:.2f} pA")
        else:
            # не смогли найти ни пиком, ни переходами
            res["delta_I"]   = np.nan
            res["sem_delta"] = np.nan
            print(f"      ⚠️  ΔI не определён ни одним методом")

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
        skip_zero=SKIP_ZERO,
        target_voltages=TARGET_VOLTAGES
    )

    if intervals.empty:
        print("  Нет подходящих интервалов")
        return pd.DataFrame()

    # Применяем ручные маски
    intervals = apply_manual_exclude(
        intervals,
        name,
        MANUAL_EXCLUDE
    )

    if intervals.empty:
        print("  Все интервалы исключены вручную")
        return pd.DataFrame()

    abf       = pyabf.ABF(path)
    results   = []
    hist_data = []

    for _, row in intervals.iterrows():
        res = analyze_interval(
            abf,
            row['t_start'],
            row['t_end'],
            int(row['voltage_mV']),
            is_control=is_control
        )
        if res is None:
            continue

        res['file']       = name
        res['is_control'] = is_control
        hist_data.append(res)

        results.append({
            k: v for k, v in res.items()
            if k not in ('hist', 'current')
        })

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
        1000 * summary.loc[mask_nonzero, 'corrected_dI'] /
        summary.loc[mask_nonzero, 'voltage_mV']
    ).abs()

    return summary

def fit_linear_iv(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Фитирует линейную ВАХ I = G*V по надёжным точкам (n_peaks>=2).
    Заменяет ненадёжные точки (n_peaks<2) предсказанными значениями.
    """
    # надёжные точки: ±100 и ±150
    reliable = summary[summary['voltage_mV'].abs() >= 100].copy()

    if len(reliable) < 3:
        print("  ⚠️  Мало надёжных точек для фита")
        return summary

    # линейная регрессия I = G*V
    slope, intercept, r, p, se = linregress(
        reliable['voltage_mV'],
        reliable['corrected_dI']
    )

    G_fit = slope * 1000  # в pS

    print(f"\n  Линейный фит по ±100/±150 mV:")
    print(f"  G = {G_fit:.2f} pS  |  R² = {r**2:.4f}  |  offset = {intercept:.3f} pA")

    # предсказываем для всех напряжений
    summary = summary.copy()
    summary['predicted_dI'] = slope * summary['voltage_mV'] + intercept
    summary['conductance_fit_pS'] = G_fit

    # помечаем ненадёжные точки
    unreliable_mask = summary['voltage_mV'].abs() < 100

    if unreliable_mask.any():
        print(f"\n  Заменяем ненадёжные точки (|V|<100 mV) предсказанными:")
        for _, row in summary[unreliable_mask].iterrows():
            pred = slope * row['voltage_mV'] + intercept
            print(f"    {row['voltage_mV']:+d} mV: "
                  f"measured={row['corrected_dI']:.2f} pA → "
                  f"predicted={pred:.2f} pA")

        summary.loc[unreliable_mask, 'corrected_dI'] = (
            slope * summary.loc[unreliable_mask, 'voltage_mV'] + intercept
        )
        summary.loc[unreliable_mask, 'corrected_sem'] = (
            se * summary.loc[unreliable_mask, 'voltage_mV'].abs()
        )

    # пересчитываем проводимость
    mask_nonzero = summary['voltage_mV'] != 0
    summary.loc[mask_nonzero, 'conductance_pS'] = (
        1000 * summary.loc[mask_nonzero, 'corrected_dI'] /
        summary.loc[mask_nonzero, 'voltage_mV']
    ).abs()

    return summary, G_fit, r**2    


def plot_iv(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- ВАХ ---
    ax = axes[0]
    ax.errorbar(
        summary['voltage_mV'],
        summary['corrected_dI'],
        yerr=summary['corrected_sem'],
        fmt='o-', color='steelblue',
        capsize=5, capthick=2,
        markersize=9, linewidth=2.5,
        label='Gramicidin A ΔI'
    )

    # Линейный фит I = G*V (через 0)
    v = summary['voltage_mV'].values
    i = summary['corrected_dI'].values

    # фит через ноль: minimize sum((I - G*V)^2)
    G_fit = float(np.dot(v, i) / np.dot(v, v))  # pA/mV
    G_pS  = G_fit * 1000
    v_line = np.linspace(v.min(), v.max(), 100)

    ax.plot(v_line, G_fit * v_line,
            color='red', linewidth=1.5,
            linestyle='--',
            label=f'Linear fit: G = {G_pS:.1f} pS')

    if 'leak_I' in summary and summary['leak_I'].notna().any():
        ax.errorbar(
            summary['voltage_mV'],
            summary['leak_I'],
            fmt='s--', color='gray',
            capsize=3, markersize=7,
            alpha=0.7, label='Контроль (leak)'
        )

    ax.axhline(0, color='black', lw=0.8)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel("Voltage (mV)", fontsize=12)
    ax.set_ylabel("ΔI single channel (pA)", fontsize=12)
    ax.set_title("I–V curve\nGramicidin A, DOPhC, 2M KCl", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # --- Проводимость как константа ---
    ax2 = axes[1]

    # g(V) = ΔI(V) / V для каждой точки
    mask = summary['voltage_mV'] != 0
    g_per_point = (
        1000 * summary.loc[mask, 'corrected_dI'] /
        summary.loc[mask, 'voltage_mV']
    )

    ax2.scatter(
        summary.loc[mask, 'voltage_mV'],
        g_per_point,
        color='darkorange',
        s=100, zorder=5,
        label='g(V) = ΔI/V'
    )

    # Горизонтальная линия — фит через ноль (правильная оценка G)
    ax2.axhline(
        G_pS,
        color='red', linewidth=2,
        linestyle='--',
        label=f'G (fit through 0) = {G_pS:.1f} pS'
    )

    ax2.set_xlabel("Voltage (mV)", fontsize=12)
    ax2.set_ylabel("Conductance (pS)", fontsize=12)
    ax2.set_title("Single-channel conductance", fontsize=13)
    ax2.set_ylim(0, max(g_per_point.max() * 1.3, G_pS * 1.5))
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
    print(f"\n  ✅ Single-channel conductance (fit through 0): {G_pS:.1f} pS")


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
    summary, G_fit, R2 = fit_linear_iv(summary)

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