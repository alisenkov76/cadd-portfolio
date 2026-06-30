#!/usr/bin/env python3
"""
Парсинг тегов из gap-free ABF файлов.
Автоматически извлекает временные интервалы для каждого напряжения.
"""

import pyabf
import numpy as np
import pandas as pd
import re
from pathlib import Path
from typing import List, Dict


def parse_voltage_from_tag(comment: str) -> float:
    """
    Извлекает напряжение из строки тега.
    "Holding on 'Cmd 0' => -50 mV" → -50.0
    """
    match = re.search(r'=>\s*(-?\d+(?:\.\d+)?)\s*mV', comment)
    if match:
        return float(match.group(1))
    raise ValueError(f"Не удалось распарсить напряжение из: {comment}")


def get_voltage_intervals(
        path: str,
        min_duration_sec: float = 30.0,
        skip_zero: bool = True) -> pd.DataFrame:
    """
    Строит таблицу интервалов: voltage, t_start, t_end, duration.

    Логика:
    - Теги задают моменты переключения напряжения
    - Интервал [t_tag_i, t_tag_{i+1}] → напряжение из тега i
    - Последний интервал → до конца файла
    - Фильтруем слишком короткие (переходы, артефакты)
    - Опционально пропускаем 0 mV (holding)
    """
    abf = pyabf.ABF(path)
    name = Path(path).name

    comments = abf.tagComments
    times    = abf.tagTimesSec

    if not comments:
        print(f"  ⚠️  {name}: нет тегов")
        return pd.DataFrame()

    # Парсим напряжения
    voltages = []
    valid_times = []
    for comment, t in zip(comments, times):
        try:
            v = parse_voltage_from_tag(comment)
            voltages.append(v)
            valid_times.append(t)
        except ValueError as e:
            print(f"  ⚠️  {e}")

    if not voltages:
        return pd.DataFrame()

    total_duration = abf.sweepLengthSec

    rows = []
    for i, (v, t_start) in enumerate(zip(voltages, valid_times)):
        # Конец интервала = начало следующего тега или конец файла
        if i + 1 < len(valid_times):
            t_end = valid_times[i + 1]
        else:
            t_end = total_duration

        duration = t_end - t_start

        # Пропускаем слишком короткие
        if duration < min_duration_sec:
            continue

        # Пропускаем holding (0 mV)
        if skip_zero and v == 0.0:
            continue

        rows.append({
            'file':       name,
            'voltage_mV': int(v),
            't_start':    round(t_start, 3),
            't_end':      round(t_end, 3),
            'duration':   round(duration, 1)
        })

    df = pd.DataFrame(rows)

    if not df.empty:
        print(f"\n  {name}:")
        print(df[['voltage_mV', 't_start',
                   't_end', 'duration']].to_string(index=False))

    return df


if __name__ == "__main__":
    FILES = [
        "data/raw/26629000.abf",
        "data/raw/26629001.abf",
        "data/raw/26629002.abf",
        "data/raw/26629003.abf",
        "data/raw/26629004.abf",
    ]

    print("АВТОМАТИЧЕСКИЙ ПАРСИНГ ТЕГОВ")
    print("="*60)

    all_intervals = []
    for f in FILES:
        df = get_voltage_intervals(
            f,
            min_duration_sec=30.0,
            skip_zero=True
        )
        if not df.empty:
            all_intervals.append(df)

    if all_intervals:
        combined = pd.concat(all_intervals, ignore_index=True)
        out = Path("results") / "voltage_intervals.csv"
        out.parent.mkdir(exist_ok=True)
        combined.to_csv(out, index=False)
        print(f"\n→ сохранено: {out}")
        print(f"\nВсего интервалов: {len(combined)}")
        print(f"Уникальные напряжения: "
              f"{sorted(combined['voltage_mV'].unique())}")