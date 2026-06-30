"""
СБОРКА ВАХ ИЗ НЕСКОЛЬКИХ ФАЙЛОВ.

Вызывает analyze_file() для каждого файла,
объединяет результаты, усредняет по напряжению,
строит итоговую ВАХ с правильным pooled SEM.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# импортируем из основного скрипта
from gramicidin_iv_analysis import (
    analyze_file,
    MEASUREMENT_WINDOWS,
    LOWPASS_HZ,
    USE_FILTER
)

# ====================== НАСТРОЙКИ ======================

CONTROL_FILE = "control.abf"

CONTROL_SWEEP_VOLTAGES = {
    0:  50,
    1: -50,
    2:  100,
    3: -100,
    4:  150,
    5: -150,
}

# Каждый файл — свой словарь sweep → voltage
EXP_FILES = {
    "gramicidin_run1.abf": {
        0:  50,
        1: -50,
        3:  100,
    },
    "gramicidin_run2.abf": {
        0: -100,
        2:  150,
        4: -150,
    },
    "gramicidin_run3.abf": {
        1:  50,
        5: -50,
        6:  100,
    },
}

# =======================================================


def pooled_sem(sems: np.ndarray,
               ns: np.ndarray) -> float:
    """
    Правильный pooled SEM с учётом разного числа точек.
    """
    ns = ns.astype(float)
    pooled_var = np.sum(ns * sems**2) / np.sum(ns)
    return np.sqrt(pooled_var / np.sum(ns))


def build_iv(exp_frames: list,
             ctrl_df: pd.DataFrame) -> pd.DataFrame:
    """
    Объединяет все измерения, усредняет по напряжению,
    корректирует на контроль, считает проводимость.
    """
    all_exp = pd.concat(exp_frames, ignore_index=True)

    # Усреднение по напряжению
    groups = all_exp.groupby('voltage_mV')

    rows = []
    for voltage, grp in groups:
        mean_dI  = grp['delta_I'].mean()
        p_sem    = pooled_sem(
            grp['sem_delta'].values,
            grp['n_points'].values
        )
        n_sweeps = len(grp)
        files    = ', '.join(grp['file'].unique())

        rows.append({
            'voltage_mV': voltage,
            'mean_delta_I': mean_dI,
            'pooled_sem':   p_sem,
            'n_sweeps':     n_sweeps,
            'files':        files
        })

    summary = pd.DataFrame(rows).sort_values('voltage_mV')

    # Контроль — leak-коррекция
    if not ctrl_df.empty:
        ctrl_mean = (
            ctrl_df.groupby('voltage_mV')
            .agg(
                leak_I=('delta_I', 'mean'),
                leak_sem=('sem_delta',
                          lambda x: np.sqrt(np.mean(x**2)))
            )
            .reset_index()
        )
        summary = summary.merge(ctrl_mean,
                                on='voltage_mV',
                                how='left')
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

    # Проводимость (pS)
    summary['conductance_pS'] = (
        summary['corrected_dI'] /
        summary['voltage_mV'].abs() * 1000
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
    ax.set_title("I–V curve\nGramicidin A, DOPhC, 2M KCl",
                 fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # --- Проводимость ---
    ax2 = axes[1]
    ax2.plot(
        summary['voltage_mV'],
        summary['conductance_pS'],
        'o-',
        color='darkorange',
        markersize=9,
        linewidth=2.5
    )
    ax2.axhline(
        summary['conductance_pS'].mean(),
        color='gray',
        linestyle='--',
        label=f"mean = {summary['conductance_pS'].mean():.1f} pS"
    )
    ax2.set_xlabel("Voltage (mV)", fontsize=12)
    ax2.set_ylabel("Conductance (pS)", fontsize=12)
    ax2.set_title("Проводимость одного канала", fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)

    plt.suptitle("Gramicidin A — полная ВАХ\n"
                 "(несколько файлов, pooled SEM)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    Path("results").mkdir(exist_ok=True)
    plt.savefig("results/iv_curve_combined.png", dpi=300)
    plt.show()
    print("  → results/iv_curve_combined.png")


# ====================== ГЛАВНАЯ ПРОГРАММА ======================

if __name__ == "__main__":
    print("="*60)
    print("СБОРКА ВАХ ИЗ НЕСКОЛЬКИХ ФАЙЛОВ")
    print("="*60)

    # Контроль
    ctrl_df = pd.DataFrame()
    from pathlib import Path as _P
    if _P(CONTROL_FILE).exists():
        ctrl_df = analyze_file(
            CONTROL_FILE,
            CONTROL_SWEEP_VOLTAGES,
            MEASUREMENT_WINDOWS,
            is_control=True
        )

    # Эксперимент
    exp_frames = []
    for fname, sweep_map in EXP_FILES.items():
        if not _P(fname).exists():
            print(f"  ⚠️  Не найден: {fname}")
            continue
        df = analyze_file(
            fname,
            sweep_map,
            MEASUREMENT_WINDOWS,
            is_control=False
        )
        if not df.empty:
            exp_frames.append(df)

    if not exp_frames:
        print("Нет данных для анализа!")
        raise SystemExit(1)

    # Сборка ВАХ
    summary = build_iv(exp_frames, ctrl_df)

    print("\n" + "="*60)
    print("ИТОГОВАЯ ТАБЛИЦА:")
    cols = ['voltage_mV', 'corrected_dI',
            'corrected_sem', 'conductance_pS',
            'n_sweeps', 'files']
    print(summary[cols].to_string(index=False))

    summary.to_csv("results/iv_summary.csv", index=False)
    print("\n  → results/iv_summary.csv")

    plot_iv(summary)

    print("\n" + "="*60)
    mean_g = summary['conductance_pS'].mean()
    std_g  = summary['conductance_pS'].std()
    print(f"Средняя проводимость: {mean_g:.1f} ± {std_g:.1f} pS")
    print("Литература gA в DOPhC/KCl: ~14–17 pS")
    print("="*60)