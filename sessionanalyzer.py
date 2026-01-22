#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sessionanalyzer.py
==================
Анализатор выгрузок PokerCraft (GG / PokerOK)
Назначение:
  - Анализ кеш- или MTT-сессий из CSV/JSON PokerCraft
  - Расчет bb/100, EV bb/100, рейка, динамики, разрезов по позициям/лимитам
  - Вывод отчета в консоль и Markdown, опционально графики
Запуск:
  python sessionanalyzer.py --input pokercraft.csv --game cash --bb-normalize --plot
"""

import argparse
import logging
import json
import re
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dateutil import parser as dateparser
from dateutil import tz


# ----------------------------
# Настройка логгирования
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("sessionanalyzer")


# ----------------------------
# Парсер аргументов CLI
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Анализатор PokerCraft CSV/JSON (GG/PokerOK)"
    )
    parser.add_argument("--input", required=True, help="Путь к CSV или JSON PokerCraft выгрузке")
    parser.add_argument("--game", choices=["cash", "mtt"], default="cash", help="Тип игры (cash|mtt)")
    parser.add_argument("--currency", default=None, help="Валюта данных (например, USD, CNY)")
    parser.add_argument("--stakes-filter", default=None, help="Фильтр по лимитам, через запятую")
    parser.add_argument("--positions", default=None, help="Список позиций для отчета (SB,BB,EP,MP,CO,BTN)")
    parser.add_argument("--start", default=None, help="Начальная дата (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="Конечная дата (YYYY-MM-DD)")
    parser.add_argument("--tz", default="UTC", help="Таймзона (например, Europe/Madrid)")
    parser.add_argument("--bb-normalize", action="store_true", help="Пересчитать результаты в BB")
    parser.add_argument("--report", default="summary,positions,stakes,timeline",
                        help="Секции отчета (summary,positions,stakes,stacks,timeline)")
    parser.add_argument("--plot", action="store_true", help="Построить графики PNG")
    parser.add_argument("--out", default=None, help="Сохранить Markdown-отчет")
    parser.add_argument("--img-out", default="charts", help="Папка для PNG-графиков")
    parser.add_argument("--sep", default=",", help="Разделитель CSV (по умолчанию ,)")
    parser.add_argument("--limit", type=int, default=0, help="Ограничить число строк (0 = все)")
    parser.add_argument("--verbose", action="store_true", help="Подробный лог")
    parser.add_argument("--dry-run", action="store_true", help="Только парсинг, без расчетов")
    return parser.parse_args()


# ----------------------------
# Загрузка данных
# ----------------------------
def load_data(path: Path, sep: str = ",", limit: int = 0) -> pd.DataFrame:
    """Загружает CSV или JSON из PokerCraft"""
    if not path.exists():
        logger.error(f"Файл не найден: {path}")
        sys.exit(1)

    if path.suffix.lower() == ".csv":
        logger.info("Загрузка CSV...")
        df = pd.read_csv(path, sep=sep, nrows=limit if limit > 0 else None)
    elif path.suffix.lower() == ".json":
        logger.info("Загрузка JSON...")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.json_normalize(data)
        if limit > 0:
            df = df.head(limit)
    else:
        logger.error("Поддерживаются только CSV и JSON.")
        sys.exit(1)

    logger.info(f"Загружено строк: {len(df)}")
    return df


# ----------------------------
# Маппинг и нормализация колонок
# ----------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Приводим имена полей к внутренним стандартам"""
    rename_map = {
        'Game ID': 'hand_id',
        'Net Win': 'net_win',
        'Rake': 'rake',
        'All-in EV Diff': 'allin_ev_diff',
        'Position': 'position',
        'Start Time': 'start_time',
        'Stakes': 'stakes',
        'Table Name': 'table_name'
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k: v}, inplace=True)

    # Привести названия колонок к нижнему регистру
    df.columns = [c.lower() for c in df.columns]
    return df


# ----------------------------
# Парсинг лимитов (stakes)
# ----------------------------
def parse_stakes(stakes_str):
    """Парсит строку лимита вида '0.5/1' -> (0.5, 1.0)"""
    if not isinstance(stakes_str, str):
        return (np.nan, np.nan)
    m = re.match(r"([\d\.]+)/([\d\.]+)", stakes_str)
    if m:
        return float(m.group(1)), float(m.group(2))
    return (np.nan, np.nan)


# ----------------------------
# Приведение времени
# ----------------------------
def parse_datetime(dt_str, target_tz="UTC"):
    """Парсинг даты с приведением к нужной таймзоне"""
    if pd.isna(dt_str):
        return None
    try:
        dt = dateparser.parse(str(dt_str))
        if not dt.tzinfo:
            dt = dt.replace(tzinfo=tz.gettz("UTC"))
        return dt.astimezone(tz.gettz(target_tz))
    except Exception:
        return None


# ----------------------------
# Фильтрация набора данных
# ----------------------------
def apply_filters(df: pd.DataFrame, args):
    """Фильтрует по дате, лимитам, позициям"""
    if args.start:
        start = dateparser.parse(args.start)
        df = df[df["start_time"] >= start]
    if args.end:
        end = dateparser.parse(args.end)
        df = df[df["start_time"] <= end]
    if args.stakes_filter:
        filt = [s.strip() for s in args.stakes_filter.split(",")]
        df = df[df["stakes"].isin(filt)]
    if args.positions:
        pos = [p.strip().upper() for p in args.positions.split(",")]
        df = df[df["position"].str.upper().isin(pos)]
    return df


# ----------------------------
# Пересчет результатов в BB
# ----------------------------
def normalize_to_bb(df: pd.DataFrame):
    """Пересчитывает net_win и EV в BB"""
    df["sb"], df["bb"] = zip(*df["stakes"].map(parse_stakes))
    df["bb_result"] = df["net_win"] / df["bb"]
    if "allin_ev_diff" in df.columns:
        df["bb_ev"] = df["allin_ev_diff"] / df["bb"]
    else:
        df["bb_ev"] = np.nan
    return df


# ----------------------------
# Расчет сводной статистики
# ----------------------------
def aggregate_summary(df: pd.DataFrame, normalize_bb: bool = True):
    """Возвращает словарь с ключевыми метриками"""
    hands = len(df)
    total_net = df["net_win"].sum()
    total_ev = df["allin_ev_diff"].sum() if "allin_ev_diff" in df.columns else np.nan
    total_rake = df["rake"].sum() if "rake" in df.columns else np.nan

    if normalize_bb and "bb_result" in df.columns:
        bb_per_100 = df["bb_result"].mean() * 100
        ev_bb_per_100 = df["bb_ev"].mean() * 100 if "bb_ev" in df.columns else np.nan
    else:
        bb_per_100 = ev_bb_per_100 = np.nan

    return {
        "hands": hands,
        "net": total_net,
        "ev": total_ev,
        "rake": total_rake,
        "bb/100": bb_per_100,
        "ev_bb/100": ev_bb_per_100
    }


# ----------------------------
# Отчеты по позициям / лимитам
# ----------------------------
def aggregate_positions(df: pd.DataFrame):
    """Агрегаты по позициям"""
    if "position" not in df.columns:
        return pd.DataFrame()
    pos_stats = df.groupby("position").agg(
        hands=("net_win", "count"),
        net=("net_win", "sum"),
        bb100=("bb_result", "mean"),
        evbb100=("bb_ev", "mean")
    ).reset_index()
    pos_stats["bb100"] *= 100
    pos_stats["evbb100"] *= 100
    return pos_stats


def aggregate_stakes(df: pd.DataFrame):
    """Агрегаты по лимитам"""
    stake_stats = df.groupby("stakes").agg(
        hands=("net_win", "count"),
        net=("net_win", "sum"),
        bb100=("bb_result", "mean"),
        evbb100=("bb_ev", "mean")
    ).reset_index()
    stake_stats["bb100"] *= 100
    stake_stats["evbb100"] *= 100
    return stake_stats


# ----------------------------
# Графики
# ----------------------------
def make_plots(df: pd.DataFrame, out_dir: Path):
    """Строит PNG-графики (кумулятивная прибыль и bb/100 по времени)"""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.sort_values("start_time")

    if "bb_result" in df.columns:
        df["cum_bb"] = df["bb_result"].cumsum()
        plt.figure()
        plt.plot(df["start_time"], df["cum_bb"])
        plt.title("Кумулятивная прибыль (BB)")
        plt.xlabel("Время")
        plt.ylabel("Прибыль (BB)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / "profit_timeline.png")
        plt.close()
        logger.info("Сохранен график: profit_timeline.png")


# ----------------------------
# Вывод в консоль
# ----------------------------
def render_console(summary: dict, pos_stats=None, stake_stats=None):
    """Форматирует и печатает данные в консоль"""
    print("\n=== SESSION ANALYZER REPORT ===\n")
    print(f"Hands: {summary['hands']}")
    print(f"Net: {summary['net']:.2f}")
    print(f"Rake: {summary['rake']:.2f}" if not pd.isna(summary['rake']) else "Rake: N/A")
    if not pd.isna(summary["bb/100"]):
        print(f"bb/100: {summary['bb/100']:.2f}")
    if not pd.isna(summary["ev_bb/100"]):
        print(f"EV bb/100: {summary['ev_bb/100']:.2f}")

    if pos_stats is not None and not pos_stats.empty:
        print("\n-- By Position --")
        print(pos_stats.to_string(index=False, justify="left", col_space=10))

    if stake_stats is not None and not stake_stats.empty:
        print("\n-- By Stakes --")
        print(stake_stats.to_string(index=False, justify="left", col_space=10))


# ----------------------------
# Главная функция
# ----------------------------
def main():
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    df = load_data(Path(args.input), sep=args.sep, limit=args.limit)
    df = normalize_columns(df)

    # Преобразуем время
    if "start_time" in df.columns:
        df["start_time"] = df["start_time"].apply(lambda x: parse_datetime(x, args.tz))
        df = df.dropna(subset=["start_time"])

    if args.dry_run:
        logger.info(f"Колонки: {list(df.columns)}")
        logger.info("Завершено (режим dry-run).")
        sys.exit(0)

    # Фильтры
    df = apply_filters(df, args)

    # Нормализация в BB
    if args.bb_normalize:
        df = normalize_to_bb(df)

    # Метрики
    summary = aggregate_summary(df, normalize_bb=args.bb_normalize)
    pos_stats = aggregate_positions(df)
    stake_stats = aggregate_stakes(df)

    # Вывод отчета
    render_console(summary, pos_stats, stake_stats)

    # Графики
    if args.plot:
        make_plots(df, Path(args.img_out))


if __name__ == "__main__":
    main()