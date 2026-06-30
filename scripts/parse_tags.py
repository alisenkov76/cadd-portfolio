"""
Парсинг тегов из gap-free ABF файлов.
Автоматически извлекает временные интервалы для каждого напряжения.

Запускается первым
"""

import pyabf
import numpy as np
import pandas as pd
import re
from pathlib import Path
from typing import List, Dict, Optional, Set


# ====================== НАСТРОЙКИ ======================

FILES = [
    "data/raw/26629000.abf",
    "data/raw/26629001.abf",
    "data/raw/26629002.abf",
    "data/raw/26629003.abf",
    "data/raw/26629004.abf",
]

# Только эти напряжения идут в ВАХ
TARGET_VOLTAGES: Set[int] = {-150, -100, -50, 50, 100, 150}

# Минимальная длительность интервала (сек)
# Короче — считаем переходным, пропускаем
MIN_DURATION_SEC: float = 30.0

# Пропускать нулевое напряжение (holding)
SKIP_ZERO: bool = True

# =======================================================


def parse_voltage_from_tag(comment: str) -> float:
    """
    Извлекает напряжение из строки тега Clampex.

    Пример:
        "Holding on 'Cmd 0' => -50 mV"  →  -50.0
        "Holding on 'Cmd 0' => 100 mV"  →  100.0
    """
    match = re.search(r'=>\s*(-?\d+(?:\.\d+)?)\s*mV', comment)
    if match:
        return float(match.group(1))
    raise ValueError(
        f"Не удалось распарсить напряжение из тега: '{comment}'"
    )


def get_voltage_intervals(
        path: str,
        min_duration_sec: float = 30.0,
        skip_zero: bool = True,
        target_voltages: Optional[Set[int]] = None
) -> pd.DataFrame:
    """
    Строит таблицу интервалов из тегов gap-free файла.

    Логика:
    - Теги задают моменты переключения напряжения
    - Интервал [t_tag_i, t_tag_{i+1}] → напряжение из тега i
    - Последний тег → интервал до конца файла
    - Фильтры:
        1. Слишком короткие интервалы → пропуск (переходные)
        2. Нулевое напряжение → пропуск (holding)
        3. Напряжение не в target_voltages → пропуск

    Параметры:
        path              — путь к .abf файлу
        min_duration_sec  — минимальная длина интервала (сек)
        skip_zero         — пропускать 0 mV
        target_voltages   — множество нужных напряжений (int)
                            если None — берём все

    Возвращает:
        pd.DataFrame с колонками:
            file, voltage_mV, t_start, t_end, duration
    """
    if not Path(path).exists():
        print(f"  ⚠️  Файл не найден: {path}")
        return pd.DataFrame()

    abf = pyabf.ABF(path)
    name = Path(path).name

    comments = abf.tagComments
    times    = abf.tagTimesSec

    if not comments:
        print(f"  ⚠️  {name}: теги отсутствуют")
        return pd.DataFrame()

    # Парсим напряжения из тегов
    voltages    = []
    valid_times = []

    for comment, t in zip(comments, times):
        try:
            v = parse_voltage_from_tag(comment)
            voltages.append(v)
            valid_times.append(t)
        except ValueError as e:
            print(f"    ⚠️  {e}")

    if not voltages:
        print(f"  ⚠️  {name}: не удалось распарсить ни одного тега")
        return pd.DataFrame()

    total_duration = abf.sweepLengthSec
    rows           = []
    skipped        = []

    for i, (v, t_start) in enumerate(zip(voltages, valid_times)):
        # Конец интервала
        t_end    = valid_times[i + 1] if i + 1 < len(valid_times) \
                   else total_duration
        duration = t_end - t_start
        v_int    = int(v)

        # --- Фильтры ---

        # 1. Слишком короткий → переходное напряжение
        if duration < min_duration_sec:
            skipped.append(
                f"    пропуск {v:+.0f} mV "
                f"[{t_start:.1f}–{t_end:.1f}s]  "
                f"длина={duration:.1f}s < {min_duration_sec}s"
            )
            continue

        # 2. Нулевое напряжение (holding)
        if skip_zero and v == 0.0:
            skipped.append(
                f"    пропуск  0 mV "
                f"[{t_start:.1f}–{t_end:.1f}s]  (holding)"
            )
            continue

        # 3. Не в целевом списке напряжений
        if target_voltages is not None and v_int not in target_voltages:
            skipped.append(
                f"    пропуск {v:+.0f} mV "
                f"[{t_start:.1f}–{t_end:.1f}s]  "
                f"(переходное, не в TARGET_VOLTAGES)"
            )
            continue

        rows.append({
            'file':       name,
            'voltage_mV': v_int,
            't_start':    round(t_start, 3),
            't_end':      round(t_end, 3),
            'duration':   round(duration, 1),
        })

    # Вывод результатов
    print(f"\n  {'─'*54}")
    print(f"  {name}")
    print(f"  {'─'*54}")

    if skipped:
        print("  Пропущено:")
        for s in skipped:
            print(s)

    df = pd.DataFrame(rows)

    if df.empty:
        print("  ❌ Нет подходящих интервалов")
        return df

    print(f"\n  Принято ({len(df)} интервалов):")
    print(
        df[['voltage_mV', 't_start', 't_end', 'duration']]
        .to_string(index=False)
    )

    # Проверяем покрытие напряжений
    if target_voltages is not None:
        found    = set(df['voltage_mV'].unique())
        missing  = set(target_voltages) - found
        if missing:
            print(f"\n  ⚠️  Отсутствуют напряжения: "
                  f"{sorted(missing)} mV")
        else:
            print(f"\n  ✅ Все целевые напряжения найдены")

    return df


# ====================== MAIN ======================

if __name__ == "__main__":
    print("ПАРСИНГ ТЕГОВ GAP-FREE ЗАПИСЕЙ")
    print("="*60)
    print(f"Целевые напряжения: {sorted(TARGET_VOLTAGES)} mV")
    print(f"Мин. длительность:  {MIN_DURATION_SEC} сек")
    print("="*60)

    all_intervals = []

    for f in FILES:
        df = get_voltage_intervals(
            f,
            min_duration_sec=MIN_DURATION_SEC,
            skip_zero=SKIP_ZERO,
            target_voltages=TARGET_VOLTAGES
        )
        if not df.empty:
            all_intervals.append(df)

    if not all_intervals:
        print("\n❌ Нет данных ни в одном файле")
        raise SystemExit(1)

    combined = pd.concat(all_intervals, ignore_index=True)

    # Итоговая сводка
    print("\n" + "="*60)
    print("ИТОГОВАЯ СВОДКА ПО ВСЕМ ФАЙЛАМ:")
    print("="*60)

    pivot = combined.groupby(
        ['file', 'voltage_mV']
    )['duration'].sum().unstack(fill_value=0)
    print(pivot.to_string())

    print(f"\nВсего интервалов: {len(combined)}")
    print(
        f"Напряжения:       "
        f"{sorted(combined['voltage_mV'].unique())} mV"
    )
    print(
        f"Суммарное время:  "
        f"{combined['duration'].sum():.0f} сек  "
        f"({combined['duration'].sum()/60:.1f} мин)"
    )

    # Сохраняем
    out = Path("results") / "voltage_intervals.csv"
    out.parent.mkdir(exist_ok=True)
    combined.to_csv(out, index=False)
    print(f"\n→ сохранено: {out}")
    print("\nДАЛЬШЕ: запускай gapfree_iv_analysis.py")
    print("="*60)