#!/usr/bin/env python3
"""
Детектор свечных паттернов по Стиву Нисону («Японские свечи») для Московской биржи.

Данные берутся из открытого API MOEX ISS (без ключа и регистрации).

Примеры:
    python nison_detector.py                        15 голубых фишек MOEX, сводная таблица
    python nison_detector.py --demo                 проверка без интернета (синтетика)
    python nison_detector.py SBER --stats           акция, дневки + статистика на истории
    python nison_detector.py Si --futures --tf 1h   фьючерс (ближайший контракт), часовики
    python nison_detector.py SiZ6 --futures         конкретный контракт
    python nison_detector.py GAZP --tf 4h --last 30 --confirm --csv gazp.csv
    python nison_detector.py SBER GAZP LKOH         свой список, сводная таблица
    python nison_detector.py --recent 14 --full     голубые фишки, сигналы за 14 дней + отчёты
    python nison_detector.py --bars 3               сигналы на 3 последних свечах вместо дней

Нисон подчёркивает: разворотная модель имеет смысл только при наличии тренда,
который можно развернуть, поэтому все разворотные паттерны проверяются
в контексте предшествующего тренда (см. --trend-len).
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ISS = "https://iss.moex.com/iss"

# таймфрейм -> (interval ISS, глубина истории по умолчанию в днях)
TIMEFRAMES = {
    "1m": (1, 7),
    "10m": (10, 90),
    "1h": (60, 365),
    "4h": (60, 730),  # собирается из часовиков
    "1d": (24, 3650),
    "1w": (7, 7300),
    "1M": (31, 9000),
}

# Запасной состав индекса голубых фишек MOEXBC (если не удалось получить его с биржи)
BLUE_CHIPS = ["SBER", "GAZP", "LKOH", "YDEX", "T", "ROSN", "NVTK", "GMKN",
              "TATN", "PLZL", "CHMF", "NLMK", "MTSS", "VTBR", "X5"]

# Пороговые коэффициенты (в долях среднего тела / диапазона свечи)
LONG_BODY = 1.2    # «длинное» тело: >= 1.2 * среднего тела
SMALL_BODY = 0.5   # «маленькое» тело: <= 0.5 * среднего тела
DOJI_BODY = 0.1    # доджи: тело <= 10% диапазона
AVG_WINDOW = 14    # окно для средних тела и диапазона


# ---------------------------------------------------------------------------
# Загрузка данных MOEX ISS
# ---------------------------------------------------------------------------

def _iss_get(path: str, params: dict) -> dict:
    import requests

    params = {"iss.meta": "off", **params}
    for attempt in range(4):
        try:
            r = requests.get(f"{ISS}/{path}", params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)  # обрыв связи: повторяем через 1, 2, 4 с


def _table(js: dict, name: str) -> pd.DataFrame:
    block = js.get(name) or {}
    return pd.DataFrame(block.get("data", []), columns=block.get("columns", []))


def blue_chips() -> tuple[list[str], str]:
    """Текущий состав индекса голубых фишек MOEXBC с биржи, иначе встроенный список."""
    try:
        js = _iss_get("statistics/engines/stock/markets/index/analytics/MOEXBC.json",
                      {"limit": 100})
        tickers = _table(js, "analytics")["ticker"].dropna().astype(str).unique().tolist()
        if len(tickers) >= 10:
            return tickers, "состав индекса MOEXBC с биржи"
    except Exception:
        pass
    return BLUE_CHIPS, "встроенный список (состав индекса получить не удалось)"


def resolve_futures(code: str) -> str:
    """'Si' -> самый ликвидный (по открытому интересу) торгуемый контракт, напр. 'SiZ6'.
    Если передан уже конкретный SECID, возвращает его."""
    js = _iss_get(
        "engines/futures/markets/forts/securities.json",
        {"iss.only": "securities"},
    )
    sec = _table(js, "securities")
    if sec.empty:
        raise RuntimeError("MOEX ISS не вернул список фьючерсов")

    if code in set(sec["SECID"]):
        return code

    cand = sec[sec["ASSETCODE"].str.upper() == code.upper()].copy()
    if cand.empty:
        raise RuntimeError(f"Фьючерсы с базовым активом '{code}' не найдены")

    cand["LASTTRADEDATE"] = pd.to_datetime(cand["LASTTRADEDATE"], errors="coerce")
    cand = cand[cand["LASTTRADEDATE"] >= pd.Timestamp(date.today())]
    if cand.empty:
        raise RuntimeError(f"Нет торгуемых контрактов для '{code}'")

    if "PREVOPENPOSITION" in cand and cand["PREVOPENPOSITION"].fillna(0).gt(0).any():
        best = cand.sort_values("PREVOPENPOSITION", ascending=False).iloc[0]
    else:
        best = cand.sort_values("LASTTRADEDATE").iloc[0]
    return str(best["SECID"])


def load_moex(ticker: str, futures: bool, tf: str, days: int | None,
              board: str = "TQBR") -> tuple[pd.DataFrame, str]:
    interval, default_days = TIMEFRAMES[tf]
    days = days or default_days
    start = (date.today() - timedelta(days=days)).isoformat()

    if futures:
        secid = resolve_futures(ticker)
        path = f"engines/futures/markets/forts/securities/{secid}/candles.json"
        title = f"{secid} (FORTS)"
    else:
        secid = ticker.upper()
        path = f"engines/stock/markets/shares/boards/{board}/securities/{secid}/candles.json"
        title = f"{secid} ({board})"

    frames, offset = [], 0
    while True:
        js = _iss_get(path, {"interval": interval, "from": start, "start": offset})
        part = _table(js, "candles")
        if part.empty:
            break
        frames.append(part)
        offset += len(part)
        if len(part) < 500:  # ISS отдаёт не более 500 свечей за запрос
            break

    if not frames:
        raise RuntimeError(f"Нет свечей для {title}: проверьте тикер/режим торгов")

    df = pd.concat(frames, ignore_index=True)
    df["begin"] = pd.to_datetime(df["begin"])
    df = (df.set_index("begin")[["open", "high", "low", "close", "volume"]]
            .astype(float).sort_index())
    df = df[~df.index.duplicated()]

    if tf == "4h":
        df = df.resample("4h", origin="start_day").agg(
            {"open": "first", "high": "max", "low": "min",
             "close": "last", "volume": "sum"}).dropna()

    return df, f"{title}, {tf}"


# ---------------------------------------------------------------------------
# Синтетические данные для --demo
# ---------------------------------------------------------------------------

def demo_data(seed: int = 7) -> pd.DataFrame:
    """Случайное блуждание + вставки «учебных» паттернов в нужном тренде."""
    rng = np.random.default_rng(seed)
    rows: list[tuple[float, float, float, float]] = []
    price = 100.0

    def bar(o, c, up_sh, dn_sh):
        nonlocal price
        h = max(o, c) * (1 + up_sh)
        lo = min(o, c) * (1 - dn_sh)
        rows.append((o, h, lo, c))
        price = c

    def noise(n, drift=0.0, vol=0.01):
        for _ in range(n):
            o = price * (1 + rng.normal(0, vol * 0.2))
            c = o * (1 + drift + rng.normal(0, vol))
            bar(o, c, abs(rng.normal(0, vol * 0.4)), abs(rng.normal(0, vol * 0.4)))

    def trend(n, step):
        for _ in range(n):
            o = price * (1 + rng.normal(0, 0.001))
            c = o * (1 + step + rng.normal(0, 0.002))
            bar(o, c, 0.002, 0.002)

    def script(*candles):
        """Свечи в долях цены на начало вставки: (open, high, low, close)."""
        nonlocal price
        p = price
        for o, h, lo, c in candles:
            rows.append((p * o, p * h, p * lo, p * c))
        price = p * candles[-1][3]

    patterns = [
        # (направление тренда перед паттерном, свечи)
        (-1, [(0.990, 0.996, 0.968, 0.995)]),                                  # молот
        (+1, [(1.000, 1.012, 0.998, 1.010), (1.013, 1.015, 0.993, 0.995)]),    # медв. поглощение
        (-1, [(1.000, 1.002, 0.968, 0.970), (0.964, 0.968, 0.958, 0.963),
              (0.967, 0.992, 0.965, 0.990)]),                                  # утренняя звезда
        (+1, [(1.000, 1.032, 0.998, 1.030), (1.036, 1.042, 1.033, 1.037),
              (1.033, 1.035, 1.005, 1.007)]),                                  # вечерняя звезда
        (-1, [(1.000, 1.002, 0.968, 0.970), (0.965, 0.991, 0.963, 0.990)]),    # просвет
        (+1, [(1.000, 1.032, 0.998, 1.030), (1.036, 1.038, 1.008, 1.010)]),    # завеса
        (+1, [(1.005, 1.035, 1.002, 1.012)]),                                  # падающая звезда
        (-1, [(1.000, 1.002, 0.968, 0.970), (0.980, 0.986, 0.977, 0.983)]),    # бычий харами
        (+1, [(1.000, 1.002, 0.978, 0.980), (0.983, 0.985, 0.958, 0.960),
              (0.963, 0.965, 0.938, 0.940)]),                                  # три вороны
    ]

    noise(40)
    for _ in range(4):
        for direction, candles in patterns:
            trend(10, 0.008 * direction)
            script(*candles)
            noise(int(rng.integers(8, 20)), drift=-0.004 * direction)
        noise(30)

    idx = pd.bdate_range(end=pd.Timestamp(date.today()), periods=len(rows))
    df = pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close"])
    df["volume"] = rng.integers(1_000, 10_000, len(df)).astype(float)
    return df


# ---------------------------------------------------------------------------
# Паттерны
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Pattern:
    name: str
    bars: int          # сколько свечей в модели
    direction: int     # +1 бычий, -1 медвежий
    kind: str          # "разворот" / "продолжение"
    check: Callable[[int], bool]


class Candles:
    def __init__(self, df: pd.DataFrame, trend_len: int):
        self.o = df["open"].to_numpy(float)
        self.h = df["high"].to_numpy(float)
        self.l = df["low"].to_numpy(float)
        self.c = df["close"].to_numpy(float)
        self.n = len(df)

        self.body = np.abs(self.c - self.o)
        self.rng = self.h - self.l
        self.top = np.maximum(self.o, self.c)
        self.bot = np.minimum(self.o, self.c)
        self.upper = self.h - self.top
        self.lower = self.bot - self.l
        self.mid = (self.o + self.c) / 2

        # Средние считаются по свечам ДО текущей, чтобы не заглядывать вперёд.
        prior = lambda x: pd.Series(x).rolling(AVG_WINDOW, min_periods=5).mean().shift(1).to_numpy()
        self.avg_body = prior(self.body)
        self.avg_rng = prior(self.rng)

        # Тренд на баре j: цена выше/ниже SMA и выше/ниже, чем trend_len баров назад.
        close = pd.Series(self.c)
        sma = close.rolling(trend_len).mean()
        past = close.shift(trend_len - 1)
        self.up = ((close > sma) & (close > past)).to_numpy()
        self.dn = ((close < sma) & (close < past)).to_numpy()

    # --- элементарные свойства свечи ---
    def white(self, i): return self.c[i] > self.o[i]
    def black(self, i): return self.c[i] < self.o[i]
    def long(self, i): return self.body[i] >= LONG_BODY * self.avg_body[i]
    def small(self, i): return self.body[i] <= SMALL_BODY * self.avg_body[i]
    def doji(self, i): return self.rng[i] > 0 and self.body[i] <= DOJI_BODY * self.rng[i]
    def sized(self, i): return self.rng[i] >= self.avg_body[i]  # не «микросвеча»
    def tol(self, i): return 0.05 * self.avg_rng[i]

    # --- формы одиночных свечей ---
    def hammer_shape(self, i):
        return (self.sized(i) and not self.doji(i)
                and self.lower[i] >= 2 * self.body[i]
                and self.upper[i] <= 0.1 * self.rng[i])

    def inv_hammer_shape(self, i):
        return (self.sized(i) and not self.doji(i)
                and self.upper[i] >= 2 * self.body[i]
                and self.lower[i] <= 0.1 * self.rng[i])

    def gravestone(self, i):
        return (self.doji(i) and self.sized(i)
                and self.lower[i] <= 0.1 * self.rng[i] and self.upper[i] >= 0.7 * self.rng[i])

    def dragonfly(self, i):
        return (self.doji(i) and self.sized(i)
                and self.upper[i] <= 0.1 * self.rng[i] and self.lower[i] >= 0.7 * self.rng[i])


def build_patterns(k: Candles) -> list[Pattern]:
    up, dn = k.up, k.dn
    o, h, l, c = k.o, k.h, k.l, k.c
    P = Pattern

    def harami(i, bull, cross):
        p = i - 1
        trend = dn[i - 2] if bull else up[i - 2]
        color = k.black(p) if bull else k.white(p)
        return (trend and color and k.long(p)
                and k.top[i] <= k.top[p] and k.bot[i] >= k.bot[p]
                and k.body[i] <= 0.5 * k.body[p]
                and k.doji(i) == cross)

    def star(i, bull, doji_star):
        a, b = i - 2, i - 1
        if bull:
            ok = (dn[i - 3] and k.black(a) and k.long(a)
                  and k.mid[b] < c[a] and k.white(i) and c[i] > k.mid[a])
        else:
            ok = (up[i - 3] and k.white(a) and k.long(a)
                  and k.mid[b] > c[a] and k.black(i) and c[i] < k.mid[a])
        return ok and (k.small(b) or k.doji(b)) and k.doji(b) == doji_star

    def soldiers(i, bull):
        ids = (i - 2, i - 1, i)
        trend = dn[i - 3] if bull else up[i - 3]
        if not trend:
            return False
        for j in ids:
            if not (k.white(j) if bull else k.black(j)):
                return False
            if k.body[j] < 0.7 * k.avg_body[j]:
                return False
            # закрытие у экстремума
            shadow = k.upper[j] if bull else k.lower[j]
            if shadow > 0.3 * k.body[j]:
                return False
        for p, j in zip(ids, ids[1:]):
            if bull and not (c[j] > c[p] and o[p] < o[j] <= c[p]):
                return False
            if not bull and not (c[j] < c[p] and c[p] <= o[j] < o[p]):
                return False
        return True

    def three_methods(i, bull):
        first, inner = i - 4, (i - 3, i - 2, i - 1)
        trend = up[i - 5] if bull else dn[i - 5]
        if not (trend and k.long(first) and (k.white(first) if bull else k.black(first))):
            return False
        for j in inner:
            if k.body[j] > 0.6 * k.body[first] or h[j] > h[first] or l[j] < l[first]:
                return False
        if bull:
            return c[i - 1] < c[i - 3] and k.white(i) and c[i] > c[first] and k.long(i)
        return c[i - 1] > c[i - 3] and k.black(i) and c[i] < c[first] and k.long(i)

    return [
        # ---------- одиночные свечи ----------
        P("Молот", 1, +1, "разворот", lambda i: dn[i - 1] and k.hammer_shape(i)),
        P("Повешенный", 1, -1, "разворот", lambda i: up[i - 1] and k.hammer_shape(i)),
        P("Перевёрнутый молот", 1, +1, "разворот", lambda i: dn[i - 1] and k.inv_hammer_shape(i)),
        P("Падающая звезда", 1, -1, "разворот", lambda i: up[i - 1] and k.inv_hammer_shape(i)),
        P("Доджи-стрекоза", 1, +1, "разворот", lambda i: dn[i - 1] and k.dragonfly(i)),
        P("Доджи-надгробие", 1, -1, "разворот", lambda i: up[i - 1] and k.gravestone(i)),
        P("Доджи на вершине", 1, -1, "разворот",
          lambda i: up[i - 1] and k.doji(i) and not k.gravestone(i)
          and k.white(i - 1) and k.long(i - 1)),
        P("Бычья свеча-пояс", 1, +1, "разворот",
          lambda i: dn[i - 1] and k.white(i) and k.long(i) and k.lower[i] <= 0.05 * k.rng[i]),
        P("Медвежья свеча-пояс", 1, -1, "разворот",
          lambda i: up[i - 1] and k.black(i) and k.long(i) and k.upper[i] <= 0.05 * k.rng[i]),

        # ---------- двухсвечные ----------
        P("Бычье поглощение", 2, +1, "разворот",
          lambda i: dn[i - 2] and k.black(i - 1) and k.white(i)
          and o[i] <= c[i - 1] and c[i] >= o[i - 1] and k.body[i] > k.body[i - 1]),
        P("Медвежье поглощение", 2, -1, "разворот",
          lambda i: up[i - 2] and k.white(i - 1) and k.black(i)
          and o[i] >= c[i - 1] and c[i] <= o[i - 1] and k.body[i] > k.body[i - 1]),
        P("Бычий харами", 2, +1, "разворот", lambda i: harami(i, True, False)),
        P("Медвежий харами", 2, -1, "разворот", lambda i: harami(i, False, False)),
        P("Бычий крест харами", 2, +1, "разворот", lambda i: harami(i, True, True)),
        P("Медвежий крест харами", 2, -1, "разворот", lambda i: harami(i, False, True)),
        P("Просвет в облаках", 2, +1, "разворот",
          lambda i: dn[i - 2] and k.black(i - 1) and k.long(i - 1) and k.white(i)
          and o[i] < c[i - 1] and k.mid[i - 1] < c[i] < o[i - 1]),
        P("Завеса из тёмных облаков", 2, -1, "разворот",
          lambda i: up[i - 2] and k.white(i - 1) and k.long(i - 1) and k.black(i)
          and o[i] > c[i - 1] and o[i - 1] < c[i] < k.mid[i - 1]),
        P("Пинцет (основание)", 2, +1, "разворот",
          lambda i: dn[i - 2] and k.black(i - 1) and k.white(i)
          and abs(l[i] - l[i - 1]) <= k.tol(i)),
        P("Пинцет (вершина)", 2, -1, "разворот",
          lambda i: up[i - 2] and k.white(i - 1) and k.black(i)
          and abs(h[i] - h[i - 1]) <= k.tol(i)),
        P("Бычьи встречные свечи", 2, +1, "разворот",
          lambda i: dn[i - 2] and k.black(i - 1) and k.long(i - 1) and k.white(i) and k.long(i)
          and o[i] < c[i - 1] and abs(c[i] - c[i - 1]) <= 2 * k.tol(i)),
        P("Медвежьи встречные свечи", 2, -1, "разворот",
          lambda i: up[i - 2] and k.white(i - 1) and k.long(i - 1) and k.black(i) and k.long(i)
          and o[i] > c[i - 1] and abs(c[i] - c[i - 1]) <= 2 * k.tol(i)),
        P("Окно вверх", 2, +1, "продолжение", lambda i: up[i - 1] and l[i] > h[i - 1]),
        P("Окно вниз", 2, -1, "продолжение", lambda i: dn[i - 1] and h[i] < l[i - 1]),

        # ---------- трёхсвечные и длиннее ----------
        P("Утренняя звезда", 3, +1, "разворот", lambda i: star(i, True, False)),
        P("Вечерняя звезда", 3, -1, "разворот", lambda i: star(i, False, False)),
        P("Утренняя доджи-звезда", 3, +1, "разворот", lambda i: star(i, True, True)),
        P("Вечерняя доджи-звезда", 3, -1, "разворот", lambda i: star(i, False, True)),
        P("Три белых солдата", 3, +1, "разворот", lambda i: soldiers(i, True)),
        P("Три чёрные вороны", 3, -1, "разворот", lambda i: soldiers(i, False)),
        P("Две вороны в окне", 3, -1, "разворот",
          lambda i: up[i - 3] and k.white(i - 2) and k.long(i - 2)
          and k.black(i - 1) and c[i - 1] > c[i - 2]
          and k.black(i) and o[i] > o[i - 1] and c[i - 2] < c[i] < c[i - 1]),
        P("Нарастающие три метода", 5, +1, "продолжение", lambda i: three_methods(i, True)),
        P("Падающие три метода", 5, -1, "продолжение", lambda i: three_methods(i, False)),
    ]


def detect(df: pd.DataFrame, trend_len: int = 10) -> pd.DataFrame:
    k = Candles(df, trend_len)
    patterns = build_patterns(k)
    start = max(trend_len, AVG_WINDOW) + 6
    hits = []
    for i in range(start, k.n):
        for p in patterns:
            if p.check(i):
                hits.append((i, df.index[i], p.name, p.direction, p.kind, p.bars, k.c[i]))
    return pd.DataFrame(hits, columns=["i", "time", "pattern", "dir", "kind", "bars", "close"])


# ---------------------------------------------------------------------------
# Подтверждение и статистика
# ---------------------------------------------------------------------------

def add_confirmation(sig: pd.DataFrame, close: np.ndarray) -> pd.DataFrame:
    """Нисон советует ждать подтверждения: следующая свеча закрывается
    в сторону сигнала. None — подтверждающей свечи ещё нет."""
    def status(row):
        j = row.i + 1
        if j >= len(close):
            return None
        return bool((close[j] - close[row.i]) * row.dir > 0)
    sig = sig.copy()
    sig["confirmed"] = [status(r) for r in sig.itertuples()] if len(sig) else []
    return sig


def stats(df: pd.DataFrame, sig: pd.DataFrame, horizons: list[int],
          confirm: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, float]]:
    close = df["close"].to_numpy(float)
    n = len(close)

    rows = []
    for name, grp in sig.groupby("pattern"):
        d = int(grp["dir"].iloc[0])
        entries = []
        for r in grp.itertuples():
            if confirm:
                if r.confirmed is not True:
                    continue
                entries.append(r.i + 1)
            else:
                entries.append(r.i)
        row = {"Паттерн": name, "": "▲" if d > 0 else "▼", "N": len(entries)}
        for hz in horizons:
            rets = np.array([(close[e + hz] / close[e] - 1) * d
                             for e in entries if e + hz < n])
            row[f"win% {hz}"] = 100 * (rets > 0).mean() if len(rets) else np.nan
            row[f"ср.% {hz}"] = 100 * rets.mean() if len(rets) else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values("N", ascending=False).reset_index(drop=True)

    # Бейзлайн: то же самое для «случайного входа» на любом баре.
    base_rows = []
    for d, label in ((+1, "Любой бар, лонг"), (-1, "Любой бар, шорт")):
        row = {"Паттерн": label, "": "▲" if d > 0 else "▼", "N": n}
        for hz in horizons:
            rets = (close[hz:] / close[:-hz] - 1) * d if hz < n else np.array([])
            row[f"win% {hz}"] = 100 * (rets > 0).mean() if len(rets) else np.nan
            row[f"ср.% {hz}"] = 100 * rets.mean() if len(rets) else np.nan
        base_rows.append(row)
    std = {hz: 100 * float(np.std(close[hz:] / close[:-hz] - 1)) if hz < n else np.nan
           for hz in horizons}
    return table, pd.DataFrame(base_rows), std


# ---------------------------------------------------------------------------
# Вывод
# ---------------------------------------------------------------------------

def fmt_time(t: pd.Timestamp) -> str:
    return t.strftime("%Y-%m-%d") if (t.hour, t.minute) == (0, 0) else t.strftime("%Y-%m-%d %H:%M")


def key_horizon(horizons: list[int]) -> int:
    """Горизонт, по которому делается краткий вывод."""
    return 5 if 5 in horizons else horizons[len(horizons) // 2]


def judge(row, base: pd.DataFrame, std: dict[int, float], hz: int, min_n: int) -> tuple[str, float]:
    """Сравнивает паттерн со случайным входом в ту же сторону.
    Возвращает (вердикт, перевес в п.п.)."""
    b = base[base[""] == row[""]].iloc[0]
    edge = row[f"ср.% {hz}"] - b[f"ср.% {hz}"]
    if row["N"] < min_n or pd.isna(edge):
        return "мало случаев, выводов не делать", edge
    # перевес относительно стандартной ошибки среднего: |t| >= 2 — вряд ли случайность
    t = edge / (std[hz] / np.sqrt(row["N"])) if std[hz] > 0 else 0.0
    if t >= 2:
        return "работал лучше случайного входа", edge
    if t <= -2:
        return "работал ХУЖЕ случайного входа", edge
    return "не лучше случайного входа", edge


SHORT_VERDICT = {
    "работал лучше случайного входа": "лучше случайного",
    "работал ХУЖЕ случайного входа": "ХУЖЕ случайного",
    "не лучше случайного входа": "не лучше случайного",
    "мало случаев, выводов не делать": "мало случаев",
}


def signal_info(r, table, base, std, hz: int, min_n: int, n: int, confirm: bool) -> dict:
    """Всё о сигнале для сводки: когда, подтверждение и как паттерн отрабатывал раньше."""
    ago = n - 1 - r.i
    info = {"i": int(r.i), "time": fmt_time(r.time), "dir": int(r.dir), "pattern": r.pattern,
            "kind": r.kind, "ago": ago,
            "when": "последняя" if ago == 0 else f"{ago} св. назад",
            "confirmed": r.confirmed if confirm else "", "hist": None}
    row = table[table["Паттерн"] == r.pattern] if table is not None and not table.empty else []
    if len(row):
        row = row.iloc[0]
        b = base[base[""] == row[""]].iloc[0]
        verdict, edge = judge(row, base, std, hz, min_n)
        info["hist"] = {"N": int(row["N"]), "win": row[f"win% {hz}"], "bwin": b[f"win% {hz}"],
                        "avg": row[f"ср.% {hz}"], "bavg": b[f"ср.% {hz}"],
                        "verdict": SHORT_VERDICT[verdict]}
    return info


def hist_text(h: dict | None) -> str:
    if not h:
        return ""
    if h["win"] != h["win"]:  # NaN
        return f"N={h['N']:<4} → {h['verdict']}"
    return (f"N={h['N']:<4} win {h['win']:3.0f}% (случ. {h['bwin']:.0f}%), "
            f"ср. {h['avg']:+.2f}% → {h['verdict']}")


def print_signals(sig: pd.DataFrame, last: int, confirm: bool, n_bars: int, explain: bool) -> None:
    if sig.empty:
        print("Паттерны не найдены.")
        return
    tail = sig.tail(last)
    print(f"Последние сигналы ({len(tail)} из {len(sig)} найденных за всю историю):")
    for r in tail.itertuples():
        arrow = "▲ бычий   " if r.dir > 0 else "▼ медвежий"
        line = f"  {fmt_time(r.time):<16}  {arrow}  {r.pattern:<26} ({r.kind}) close={r.close:g}"
        if confirm:
            line += {True: "  ✓ подтверждён", False: "  ✗ не подтверждён",
                     None: "  … ждёт подтверждения"}[r.confirmed]
        print(line)
    if explain:
        print("""
  Как читать:
    дата        — свеча, на которой паттерн сформировался (его последняя свеча);
    ▲ бычий     — паттерн предвещает рост, ▼ медвежий — падение;
    разворот    — смена текущего тренда, продолжение — тренд, скорее всего, продолжится;
    close       — цена закрытия этой свечи.
  Несколько паттернов в одну дату — это одна и та же свеча, подходящая под разные модели.
  По Нисону, свечной сигнал — предупреждение, а не приказ: дождитесь подтверждения
  следующей свечой (ключ --confirm) и учитывайте уровни поддержки/сопротивления.""")


def print_last_candle(sig: pd.DataFrame, n_bars: int, table: pd.DataFrame | None,
                      base: pd.DataFrame | None, std: dict[int, float] | None,
                      hz: int, min_n: int) -> None:
    on_last = sig[sig["i"] == n_bars - 1]
    print()
    if on_last.empty:
        print("На последней свече паттернов нет.")
        return
    names = ", ".join(f"{p} ({'▲' if d > 0 else '▼'})"
                      for p, d in zip(on_last["pattern"], on_last["dir"]))
    print(f"На последней свече: {names}")
    if table is None or table.empty:
        return
    for name in on_last["pattern"]:
        row = table[table["Паттерн"] == name]
        if row.empty:
            continue
        row = row.iloc[0]
        b = base[base[""] == row[""]].iloc[0]
        verdict, _ = judge(row, base, std, hz, min_n)
        print(f"  {name}: в истории {row['N']} раз; через {hz} баров цена шла в сторону сигнала "
              f"в {row[f'win% {hz}']:.0f}% случаев (случайный вход — {b[f'win% {hz}']:.0f}%), "
              f"в среднем {row[f'ср.% {hz}']:+.2f}% (случайный — {b[f'ср.% {hz}']:+.2f}%) "
              f"→ {verdict}.")


def print_stats(table: pd.DataFrame, base: pd.DataFrame, std: dict[int, float],
                horizons: list[int], confirm: bool, min_n: int, explain: bool) -> None:
    print()
    print("Статистика на истории (доходность в сторону сигнала, закрытие → закрытие через h баров"
          + (", вход после подтверждения" if confirm else "") + "):")
    if table.empty:
        print("  нет сигналов")
        return
    shown = table.copy()
    rare = shown["N"] < min_n
    shown.loc[rare, "Паттерн"] = shown.loc[rare, "Паттерн"] + " *"
    full = pd.concat([shown, base], ignore_index=True)
    fmts = {col: (lambda v: "" if pd.isna(v) else f"{v:6.1f}") for col in full.columns
            if col.startswith("win%")}
    fmts.update({col: (lambda v: "" if pd.isna(v) else f"{v:+6.2f}") for col in full.columns
                 if col.startswith("ср.%")})
    text = full.to_string(index=False, formatters=fmts, na_rep="")
    lines = text.splitlines()
    sep = "-" * len(lines[0])
    print(lines[0])
    print(sep)
    for line in lines[1:len(shown) + 1]:
        print(line)
    print(sep)
    for line in lines[len(shown) + 1:]:
        print(line)
    if rare.any():
        print(f"* меньше {min_n} случаев — статистика ненадёжна")

    hz = key_horizon(horizons)
    if explain:
        h_list = ", ".join(map(str, horizons))
        print(f"""
  Как читать таблицу:
    N         — сколько раз паттерн встретился на истории;
    win% h    — в скольких % случаев через h баров (h = {h_list}) цена ушла в сторону
                сигнала: для ▲ выросла, для ▼ упала;
    ср.% h    — средний результат сделки в сторону сигнала через h баров, в %
                (для ▼ — как шорт: плюс означает, что цена упала);
    Любой бар — бейзлайн: те же цифры для входа на КАЖДОМ баре без всякого паттерна.
                Строка «лонг» — с чем сравнивать ▲, «шорт» — с чем сравнивать ▼.
  Паттерн полезен, только если его win% и ср.% заметно лучше бейзлайна той же стороны.
  Цифры — без учёта комиссий и проскальзывания; прошлое не гарантирует будущего.""")

    print(f"\nИтог (по горизонту {hz} баров, паттерны с N ≥ {min_n}):")
    verdicts = []
    for _, row in table.iterrows():
        if row["N"] < min_n:
            continue
        verdict, edge = judge(row, base, std, hz, min_n)
        verdicts.append((edge, row, verdict))
    if not verdicts:
        print("  ни у одного паттерна недостаточно случаев — увеличьте историю (--days)")
        return
    for edge, row, verdict in sorted(verdicts, key=lambda x: -x[0]):
        print(f"  {row['Паттерн'] + ' ' + row['']:<28} перевес над случайным входом "
              f"{edge:+6.2f} п.п. → {verdict}")
    if explain:
        print("  «Лучше/хуже» — перевес больше двух стандартных ошибок (вряд ли случайность);"
              " «не лучше» — разница в пределах шума.")


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# HTML-отчёт (один самодостаточный файл: открывается в любом браузере без интернета)
# ---------------------------------------------------------------------------

def esc(x) -> str:
    return html.escape(str(x), quote=True)


def fmt_price(x: float) -> str:
    return f"{x:,.6g}".replace(",", " ") if abs(x) >= 1000 else f"{x:.6g}"


def save_html(page: str, path: str, open_browser: bool) -> None:
    import webbrowser

    out = Path(path).resolve()
    out.write_text(page, encoding="utf-8")
    print(f"\nHTML-отчёт: {out}")
    if open_browser:
        try:
            webbrowser.open(out.as_uri())
        except Exception:
            pass


def svg_chart(df: pd.DataFrame, signals: list[dict], bars: int, width: int = 640,
              height: int = 200) -> str:
    """Свечной график последних `bars` свечей с отметками сигналов."""
    tail = df.iloc[-bars:]
    off = len(df) - len(tail)
    pad_l, pad_r, pad_t, pad_b = 4, 58, 16, 22
    w, h = width - pad_l - pad_r, height - pad_t - pad_b
    lo, hi = float(tail["low"].min()), float(tail["high"].max())
    span = (hi - lo) or abs(hi) * 0.01 or 1.0
    lo, hi = lo - span * 0.08, hi + span * 0.08

    def y(v):
        return pad_t + (hi - v) / (hi - lo) * h

    step = w / len(tail)
    bw = max(1.0, min(9.0, step * 0.62))
    by_bar: dict[int, list[dict]] = {}
    for sg in signals:
        by_bar.setdefault(sg["i"], []).append(sg)

    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="Свечной график, последние {len(tail)} свечей">']
    for k in range(4):  # сетка и ценовые метки справа
        v = lo + (hi - lo) * (k + 0.5) / 4
        parts.append(f'<line class="grid" x1="{pad_l}" x2="{pad_l + w}" y1="{y(v):.1f}" '
                     f'y2="{y(v):.1f}"/><text class="axis" x="{pad_l + w + 6}" '
                     f'y="{y(v) + 4:.1f}">{esc(f"{v:.5g}")}</text>')
    for k in (0, len(tail) // 2, len(tail) - 1):  # даты снизу
        x = pad_l + step * (k + 0.5)
        anchor = "start" if k == 0 else "end" if k == len(tail) - 1 else "middle"
        parts.append(f'<text class="axis" x="{x:.1f}" y="{height - 6}" '
                     f'text-anchor="{anchor}">{esc(fmt_time(tail.index[k]))}</text>')

    for k, (ts, r) in enumerate(tail.iterrows()):
        i = off + k
        x = pad_l + step * (k + 0.5)
        o, hh, ll, c = r["open"], r["high"], r["low"], r["close"]
        cls = "up" if c >= o else "dn"
        top, bot = y(max(o, c)), y(min(o, c))
        sigs = by_bar.get(i, [])
        tip = {"t": fmt_time(ts), "o": fmt_price(o), "h": fmt_price(hh), "l": fmt_price(ll),
               "c": fmt_price(c),
               "s": [("▲ " if sg["dir"] > 0 else "▼ ") + sg["pattern"] for sg in sigs]}
        parts.append(
            f'<g class="cnd {cls}" data-tip="{esc(json.dumps(tip, ensure_ascii=False))}">'
            f'<rect class="hit" x="{x - step / 2:.1f}" y="{pad_t}" width="{step:.1f}" height="{h}"/>'
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y(hh):.1f}" y2="{y(ll):.1f}"/>'
            f'<rect class="body" x="{x - bw / 2:.1f}" y="{top:.1f}" width="{bw:.1f}" '
            f'height="{max(1.0, bot - top):.1f}" rx="1"/></g>')
        for n_sig, sg in enumerate(sigs[:1]):  # одна отметка на свечу
            if sg["dir"] > 0:
                ty = y(ll) + 6
                pts = f"{x:.1f},{ty:.1f} {x - 5:.1f},{ty + 8:.1f} {x + 5:.1f},{ty + 8:.1f}"
            else:
                ty = y(hh) - 6
                pts = f"{x:.1f},{ty:.1f} {x - 5:.1f},{ty - 8:.1f} {x + 5:.1f},{ty - 8:.1f}"
            parts.append(f'<polygon class="mark {"up" if sg["dir"] > 0 else "dn"}" points="{pts}"/>')
    parts.append("</svg>")
    return "".join(parts)


VERDICT_BADGE = {
    "лучше случайного": ("good", "✓"),
    "ХУЖЕ случайного": ("bad", "✗"),
    "не лучше случайного": ("neutral", "≈"),
    "мало случаев": ("muted", "?"),
}


def signal_rows_html(signals: list[dict], hz: int, confirm: bool) -> str:
    out = []
    for sg in reversed(signals):  # свежие сверху
        d = "up" if sg["dir"] > 0 else "dn"
        h = sg["hist"]
        verdict = h["verdict"] if h else ""
        vcls, vicon = VERDICT_BADGE.get(verdict, ("muted", ""))
        if h and h["win"] == h["win"]:
            hist = (f'<span class="num">{h["win"]:.0f}%</span> против '
                    f'<span class="num">{h["bwin"]:.0f}%</span> у случайного входа · '
                    f'ср. <span class="num">{h["avg"]:+.2f}%</span> против '
                    f'<span class="num">{h["bavg"]:+.2f}%</span> · N={h["N"]}')
        elif h:
            hist = f"N={h['N']}"
        else:
            hist = ""
        conf = ""
        if confirm:
            conf = {True: '<span class="conf ok">✓ подтверждён</span>',
                    False: '<span class="conf no">✗ не подтверждён</span>',
                    None: '<span class="conf wait">… ждёт подтверждения</span>'}.get(sg["confirmed"], "")
        badge = (f'<span class="badge {vcls}">{vicon} {esc(verdict)}</span>' if verdict else "")
        out.append(
            f'<div class="sig" data-dir="{d}" data-verdict="{vcls}">'
            f'<div class="sig-main"><span class="dir {d}">{"▲" if d == "up" else "▼"}</span>'
            f'<span class="pname">{esc(sg["pattern"])}</span>'
            f'<span class="kind">{esc(sg["kind"])}</span>{conf}</div>'
            f'<div class="sig-meta"><span>{esc(sg["time"])} · {esc(sg["when"])}</span>{badge}</div>'
            + (f'<div class="sig-hist">История на этом тикере через {hz} баров: {hist}</div>'
               if hist else "")
            + "</div>")
    return "".join(out)


CSS = """
:root{color-scheme:light;--bg:#f6f6f4;--surface:#fcfcfb;--border:#e4e3df;--text:#0b0b0b;
--text2:#52514e;--muted:#7d7c77;--grid:#ecebe7;--up:#0ca30c;--dn:#d03b3b;--accent:#2a78d6;
--good-bg:#e3f4e3;--good:#086b08;--bad-bg:#fbe5e5;--bad:#a42727;--neu-bg:#efeeea;--neu:#52514e;
--chip:#efeeea}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#121211;
--surface:#1a1a19;--border:#2e2e2b;--text:#fff;--text2:#c3c2b7;--muted:#8f8e86;--grid:#262624;
--up:#2fbf2f;--dn:#e66767;--accent:#3987e5;--good-bg:#173317;--good:#7fdc7f;--bad-bg:#3a1c1c;
--bad:#f29a9a;--neu-bg:#262624;--neu:#c3c2b7;--chip:#262624}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#121211;--surface:#1a1a19;--border:#2e2e2b;
--text:#fff;--text2:#c3c2b7;--muted:#8f8e86;--grid:#262624;--up:#2fbf2f;--dn:#e66767;
--accent:#3987e5;--good-bg:#173317;--good:#7fdc7f;--bad-bg:#3a1c1c;--bad:#f29a9a;
--neu-bg:#262624;--neu:#c3c2b7;--chip:#262624}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
font:15px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:24px;margin:0 0 4px}
.sub{color:var(--text2);margin:0 0 20px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:16px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:12px 16px}
.tile .v{font-size:28px;font-weight:600;font-variant-numeric:tabular-nums}
.tile .l{color:var(--text2);font-size:13px}
.filters{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:0 0 16px;
position:sticky;top:0;background:var(--bg);padding:8px 0;z-index:5}
.filters button{border:1px solid var(--border);background:var(--surface);color:var(--text);
border-radius:999px;padding:6px 14px;font:inherit;cursor:pointer}
.filters button[aria-pressed="true"]{background:var(--text);color:var(--surface);border-color:var(--text)}
.filters label{color:var(--text2);display:flex;gap:6px;align-items:center;margin-left:8px;cursor:pointer}
.filters input[type=search]{border:1px solid var(--border);background:var(--surface);color:var(--text);
border-radius:999px;padding:6px 14px;font:inherit;min-width:140px}
.grid-cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,520px),1fr));gap:16px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:16px;min-width:0}
.card-h{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap}
.tk{font-size:20px;font-weight:700}
.ttl{color:var(--muted);font-size:13px}
.price{font-variant-numeric:tabular-nums;font-size:18px;font-weight:600}
.chg{font-size:13px;margin-left:6px;font-variant-numeric:tabular-nums}
.chg.up{color:var(--up)}.chg.dn{color:var(--dn)}
.chart{width:100%;height:auto;display:block;margin:8px 0 4px}
.chart .grid{stroke:var(--grid);stroke-width:1}
.chart .axis{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
.chart .cnd line{stroke-width:1}
.chart .cnd.up line{stroke:var(--up)}.chart .cnd.dn line{stroke:var(--dn)}
.chart .cnd.up .body{fill:var(--up)}.chart .cnd.dn .body{fill:var(--dn)}
.chart .hit{fill:transparent}
.chart .cnd:hover .hit{fill:var(--grid)}
.chart .mark{stroke:var(--surface);stroke-width:1.5}
.chart .mark.up{fill:var(--up)}.chart .mark.dn{fill:var(--dn)}
.sig{border-top:1px solid var(--border);padding:10px 0}
.sig-main{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.dir{font-weight:700}.dir.up{color:var(--up)}.dir.dn{color:var(--dn)}
.pname{font-weight:600}
.kind{color:var(--muted);font-size:13px}
.sig-meta{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;color:var(--text2);
font-size:13px;margin-top:2px}
.sig-hist{color:var(--text2);font-size:13px;margin-top:2px}
.num{font-variant-numeric:tabular-nums;color:var(--text)}
.badge{border-radius:999px;padding:1px 10px;font-size:12px;font-weight:600;white-space:nowrap}
.badge.good{background:var(--good-bg);color:var(--good)}
.badge.bad{background:var(--bad-bg);color:var(--bad)}
.badge.neutral,.badge.muted{background:var(--neu-bg);color:var(--neu)}
.conf{font-size:12px;color:var(--text2)}
.conf.ok{color:var(--good)}.conf.no{color:var(--bad)}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chips span{background:var(--chip);border-radius:999px;padding:2px 10px;font-size:13px;color:var(--text2)}
section{margin-top:28px}
h2{font-size:17px;margin:0 0 10px}
details{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:12px 16px;
margin-top:28px;color:var(--text2)}
details summary{cursor:pointer;color:var(--text);font-weight:600}
details li{margin:4px 0}
.err{color:var(--bad);font-size:14px}
.empty{color:var(--text2);padding:24px;text-align:center;background:var(--surface);
border:1px dashed var(--border);border-radius:12px}
.tbl-wrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;text-align:right;border-bottom:1px solid var(--border);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--text2);font-weight:600;position:sticky;top:0;background:var(--surface)}
tr.base td{background:var(--chip);font-weight:600}
tr.rare td{color:var(--muted)}
td.pos{color:var(--good)}td.neg{color:var(--bad)}
#tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--text);
border:1px solid var(--border);border-radius:8px;padding:8px 10px;font-size:12px;
box-shadow:0 4px 16px rgba(0,0,0,.15);display:none;z-index:10;font-variant-numeric:tabular-nums}
#tip b{display:block;margin-bottom:2px}
.foot{color:var(--muted);font-size:12px;margin-top:28px}
"""

JS = """
const tip=document.getElementById('tip');
document.addEventListener('mousemove',e=>{
  const g=e.target.closest&&e.target.closest('.cnd');
  if(!g){tip.style.display='none';return}
  const d=JSON.parse(g.dataset.tip);
  tip.innerHTML='<b>'+d.t+'</b>O '+d.o+' · H '+d.h+'<br>L '+d.l+' · C '+d.c+
    (d.s.length?'<br>'+d.s.join('<br>'):'');
  tip.style.display='block';
  const x=Math.min(e.clientX+14,innerWidth-tip.offsetWidth-8);
  const y=Math.min(e.clientY+14,innerHeight-tip.offsetHeight-8);
  tip.style.left=x+'px';tip.style.top=y+'px';
});
const state={dir:'all',hideRare:false,onlyGood:false,q:''};
function apply(){
  let shown=0;
  document.querySelectorAll('.card[data-ticker]').forEach(card=>{
    let any=0;
    card.querySelectorAll('.sig').forEach(s=>{
      const ok=(state.dir==='all'||s.dataset.dir===state.dir)&&
        !(state.hideRare&&s.dataset.verdict==='muted')&&
        !(state.onlyGood&&s.dataset.verdict!=='good');
      s.hidden=!ok;if(ok)any++;
    });
    const qok=!state.q||card.dataset.ticker.includes(state.q);
    card.hidden=!(any&&qok);if(!card.hidden)shown++;
  });
  const em=document.getElementById('none');if(em)em.hidden=shown>0;
}
document.querySelectorAll('[data-f]').forEach(b=>b.addEventListener('click',()=>{
  state.dir=b.dataset.f;
  document.querySelectorAll('[data-f]').forEach(x=>x.setAttribute('aria-pressed',x===b));
  apply();
}));
const r=document.getElementById('hideRare');if(r)r.addEventListener('change',()=>{state.hideRare=r.checked;apply()});
const g=document.getElementById('onlyGood');if(g)g.addEventListener('change',()=>{state.onlyGood=g.checked;apply()});
const q=document.getElementById('q');if(q)q.addEventListener('input',()=>{state.q=q.value.trim().toUpperCase();apply()});
"""


def help_html(hz: int, min_n: int) -> str:
    return f"""<details><summary>Как читать отчёт</summary><ul>
<li><b>▲ бычий</b> паттерн предвещает рост, <b>▼ медвежий</b> — падение. На графике сигнал
отмечен треугольником под/над свечой. Наведите мышь на свечу, чтобы увидеть цены.</li>
<li><b>Разворот</b> — возможная смена тренда, <b>продолжение</b> — тренд, скорее всего, продолжится.</li>
<li><b>История</b> — как этот же паттерн отрабатывал раньше на этом же тикере: в скольких % случаев
через {hz} баров цена шла в сторону сигнала и средний результат — в сравнении со входом на
случайной свече.</li>
<li><b>✓ лучше случайного</b> — перевес больше статистического шума; <b>≈ не лучше</b> — разница
в пределах шума; <b>✗ хуже</b> — после паттерна цена чаще шла против него;
<b>? мало случаев</b> — меньше {min_n}, выводов не делать.</li>
<li>Свеча «последняя» может быть ещё не закрыта — тогда паттерн может исчезнуть до конца дня.</li>
<li>По Нисону свечной сигнал — предупреждение, а не приказ: ждите подтверждения следующей свечой,
смотрите на уровни поддержки/сопротивления и ставьте стоп. Цифры без комиссий;
прошлое не гарантирует будущего.</li></ul></details>"""


def page_html(title: str, body: str) -> str:
    return (f'<!doctype html><html lang="ru"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{esc(title)}</title><style>{CSS}</style></head><body>'
            f'<div class="wrap">{body}<p class="foot">Создано nison_detector.py · '
            f'{datetime.now():%d.%m.%Y %H:%M} · данные MOEX ISS</p></div>'
            f'<div id="tip"></div><script>{JS}</script></body></html>')


def card_html(ticker: str, title: str, df: pd.DataFrame, signals: list[dict], hz: int,
              confirm: bool, bars: int, width: int = 640, height: int = 200) -> str:
    last = df["close"].iloc[-1]
    ref = df["close"].iloc[-min(len(df), bars)]
    chg = (last / ref - 1) * 100 if ref else 0.0
    return (f'<article class="card" data-ticker="{esc(ticker.upper())}">'
            f'<div class="card-h"><div><span class="tk">{esc(ticker)}</span> '
            f'<span class="ttl">{esc(title)}</span></div>'
            f'<div><span class="price">{esc(fmt_price(last))}</span>'
            f'<span class="chg {"up" if chg >= 0 else "dn"}" title="изменение за период графика">'
            f'{"▲" if chg >= 0 else "▼"} {chg:+.1f}%</span></div></div>'
            f'{svg_chart(df, signals, bars, width, height)}'
            f'{signal_rows_html(signals, hz, confirm)}</article>')


def filters_html(with_search: bool) -> str:
    return ('<div class="filters" role="toolbar" aria-label="Фильтры">'
            '<button data-f="all" aria-pressed="true">Все</button>'
            '<button data-f="up" aria-pressed="false">▲ Бычьи</button>'
            '<button data-f="dn" aria-pressed="false">▼ Медвежьи</button>'
            '<label><input type="checkbox" id="onlyGood"> только «лучше случайного»</label>'
            '<label><input type="checkbox" id="hideRare"> скрыть «мало случаев»</label>'
            + ('<input type="search" id="q" placeholder="Тикер…" aria-label="Поиск по тикеру">'
               if with_search else "") + "</div>")


def html_scan(cards: list[dict], quiet: list[str], failed: list, meta: dict) -> str:
    n_sig = sum(len(c["signals"]) for c in cards)
    n_up = sum(sg["dir"] > 0 for c in cards for sg in c["signals"])
    n_good = sum(1 for c in cards for sg in c["signals"]
                 if sg["hist"] and sg["hist"]["verdict"] == "лучше случайного")
    tiles = [(meta["count"], "инструментов проверено"), (len(cards), "с сигналами"),
             (n_up, "▲ бычьих сигналов"), (n_sig - n_up, "▼ медвежьих сигналов"),
             (n_good, "✓ исторически лучше случайного")]
    body = [f'<h1>Свечные паттерны Нисона</h1><p class="sub">{esc(meta["source"][:1].upper() + meta["source"][1:])} · '
            f'таймфрейм {esc(meta["tf"])} · {esc(meta["period"])}</p>',
            '<div class="tiles">' + "".join(
                f'<div class="tile"><div class="v">{v}</div><div class="l">{esc(l)}</div></div>'
                for v, l in tiles) + "</div>"]
    if cards:
        body.append(filters_html(True))
        body.append('<div class="grid-cards">' + "".join(
            card_html(c["ticker"], c["title"].split(",")[0], c["df"], c["signals"], meta["hz"],
                      meta["confirm"], 60) for c in cards) + "</div>")
        body.append('<p class="empty" id="none" hidden>Под фильтр ничего не попало.</p>')
    else:
        body.append('<p class="empty">Сигналов нет ни у одного инструмента за этот период.</p>')
    if quiet:
        body.append('<section><h2>Без сигналов</h2><div class="chips">'
                    + "".join(f"<span>{esc(t)}</span>" for t in quiet) + "</div></section>")
    if failed:
        body.append('<section><h2>Не удалось загрузить</h2>' + "".join(
            f'<p class="err">{esc(t)}: {esc(e)}</p>' for t, e in failed) + "</section>")
    body.append(help_html(meta["hz"], meta["min_n"]))
    return page_html(f"Паттерны Нисона — {datetime.now():%d.%m.%Y}", "".join(body))


def stats_table_html(table: pd.DataFrame, base: pd.DataFrame, horizons: list[int],
                     min_n: int) -> str:
    head = "".join(f"<th>win% {h}</th><th>ср.% {h}</th>" for h in horizons)
    rows = []
    for cls, frame in (("", table), ("base", base)):
        for _, r in frame.iterrows():
            rare = cls == "" and r["N"] < min_n
            cells = []
            for h in horizons:
                w, a = r[f"win% {h}"], r[f"ср.% {h}"]
                cells.append("<td>–</td><td>–</td>" if w != w else
                             f'<td>{w:.1f}</td><td class="{"pos" if a > 0 else "neg"}">{a:+.2f}</td>')
            rows.append(f'<tr class="{cls}{" rare" if rare else ""}"><td>{r[""]} {esc(r["Паттерн"])}'
                        f'{" *" if rare else ""}</td><td>{r["N"]}</td>{"".join(cells)}</tr>')
    return (f'<div class="tbl-wrap"><table><thead><tr><th>Паттерн</th><th>N</th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>'
            f'<p class="ttl">* меньше {min_n} случаев — статистика ненадёжна. Строки «Любой бар» — '
            f'вход на каждой свече без паттерна: сравнивайте ▲ с «лонг», ▼ с «шорт».</p>')


def html_single(df: pd.DataFrame, title: str, signals: list[dict], table, base,
                horizons: list[int], meta: dict) -> str:
    ticker = title.split(" ")[0]
    body = [f'<h1>{esc(ticker)}</h1><p class="sub">{esc(title)} · {len(df)} свечей, '
            f'{esc(fmt_time(df.index[0]))} — {esc(fmt_time(df.index[-1]))}</p>']
    if signals:
        body.append(filters_html(False))
    body.append(card_html(ticker, f"последние {len(signals)} сигналов", df, signals,
                          meta["hz"], meta["confirm"], 120, 1040, 340))
    if table is not None and not table.empty:
        body.append(f'<section class="card"><h2>Статистика на истории</h2>'
                    f'{stats_table_html(table, base, horizons, meta["min_n"])}</section>')
    body.append(help_html(meta["hz"], meta["min_n"]))
    return page_html(f"{ticker} — паттерны Нисона", "".join(body))


def report(df: pd.DataFrame, title: str, args, horizons: list[int]) -> None:
    """Подробный отчёт по одному инструменту."""
    sig = detect(df, args.trend_len)
    if args.confirm or args.stats:
        sig = add_confirmation(sig, df["close"].to_numpy(float))
    sig_print = sig.drop(columns="confirmed") if not args.confirm and "confirmed" in sig else sig

    print(f"{title}: {len(df)} свечей, {fmt_time(df.index[0])} — {fmt_time(df.index[-1])}, "
          f"последнее закрытие {df['close'].iloc[-1]:g}")
    print()
    explain = not args.brief
    table = base = std = None
    if args.stats:
        table, base, std = stats(df, sig, horizons, args.confirm)

    print_signals(sig_print, args.last, args.confirm, len(df), explain)
    if not sig.empty:
        print_last_candle(sig, len(df), table, base, std, key_horizon(horizons), args.min_n)
    if args.stats:
        print_stats(table, base, std, horizons, args.confirm, args.min_n, explain)

    if args.csv:
        out = sig.copy()
        out["direction"] = np.where(out["dir"] > 0, "bull", "bear")
        out.drop(columns=["i", "dir"]).to_csv(args.csv, index=False)
        print(f"\nСигналы сохранены: {args.csv}")

    if not args.no_html and not getattr(args, "_in_scan", False):
        hz = key_horizon(horizons)
        infos = [signal_info(r, table, base, std, hz, args.min_n, len(df), args.confirm)
                 for r in sig.tail(args.last).itertuples()]
        page = html_single(df, title, infos, table, base, horizons, {
            "hz": hz, "min_n": args.min_n, "confirm": args.confirm})
        save_html(page, args.html, not args.no_open)


def scan(tickers: list[str], source: str, args, horizons: list[int]) -> int:
    """Сводная таблица по списку инструментов: сигналы за последние --recent дней
    (или на последних --bars свечах)
    и как этот паттерн отрабатывал раньше на этом же инструменте."""
    hz = key_horizon(horizons)
    since = pd.Timestamp(date.today()) - pd.Timedelta(days=args.recent - 1)
    rows, quiet, failed, all_sig, cards = [], [], [], [], []
    for num, t in enumerate(tickers, 1):
        print(f"Загрузка {t} ({num}/{len(tickers)})...".ljust(40), end="\r",
              file=sys.stderr, flush=True)
        try:
            df, title = load_moex(t, args.futures, args.tf, args.days, args.board)
        except Exception as e:
            failed.append((t, str(e).splitlines()[0][:80]))
            continue
        if len(df) < 30:
            failed.append((t, f"мало свечей ({len(df)})"))
            continue

        sig = add_confirmation(detect(df, args.trend_len), df["close"].to_numpy(float))
        table, base, std = stats(df, sig, horizons, args.confirm)
        if args.full:
            print(" " * 40, end="\r", file=sys.stderr)
            print("=" * 100)
            report(df, title, argparse.Namespace(**{**vars(args), "stats": True, "csv": None,
                                                    "_in_scan": True}),
                   horizons)
            print()
        if args.csv:
            all_sig.append(sig.assign(ticker=t))

        n = len(df)
        if args.bars:
            recent = sig[sig["i"] >= n - args.bars]
        else:
            recent = sig[sig["time"] >= since]
        if recent.empty:
            quiet.append(t)
            continue
        infos = [signal_info(r, table, base, std, hz, args.min_n, n, args.confirm)
                 for r in recent.itertuples()]
        cards.append({"ticker": t, "title": title, "df": df, "signals": infos})
        for info in infos:
            conf = {True: "✓", False: "✗", None: "…", "": ""}[info["confirmed"]]
            rows.append((t, df["close"].iloc[-1], info["time"], info["when"],
                         ("▲ " if info["dir"] > 0 else "▼ ") + info["pattern"], conf,
                         hist_text(info["hist"])))
    print(" " * 40, end="\r", file=sys.stderr)

    if args.full:
        print("=" * 100)
    print(f"Сводка ({source}), инструментов: {len(tickers)}, таймфрейм {args.tf}, "
          + (f"сигналы на последних {args.bars} свечах" if args.bars
             else f"сигналы за последние {args.recent} дн. (с {since:%Y-%m-%d})"))
    print()
    if rows:
        conf_col = "Подтв. " if args.confirm else ""
        print(f"{'Тикер':<7}{'Цена':>10}  {'Свеча':<17}{'Когда':<13}{'Сигнал':<30}{conf_col}"
              f"История на этом тикере (через {hz} баров)")
        print("-" * 130)
        prev = None
        for t, price, ts, when, name, conf, hist in rows:
            head = f"{t:<7}{price:>10g}" if t != prev else " " * 17
            conf_txt = f"{conf:<7}" if args.confirm else ""
            print(f"{head}  {ts:<17}{when:<13}{name:<30}{conf_txt}{hist}")
            prev = t
    else:
        print("Сигналов нет ни у одного инструмента.")
    if quiet:
        print(f"\nБез сигналов: {', '.join(quiet)}")
    for t, err in failed:
        print(f"Не удалось загрузить {t}: {err}")

    if not args.brief and rows:
        print(f"""
  Как читать:
    Цена      — последнее закрытие; Свеча — дата свечи, на которой сформировался паттерн;
    Когда     — «последняя» = самая свежая свеча (если торги ещё идут, она не закрыта
                и паттерн может исчезнуть), «N св. назад» — сколько свечей прошло;
    История   — как этот паттерн отрабатывал раньше на ЭТОМ ЖЕ тикере: N — сколько раз встречался,
                в скольких % случаев через {hz} баров цена ушла в сторону сигнала, сравнение
                со входом на случайной свече («случ.») и средний результат в %.
    Вывод «лучше случайного» — перевес больше статистического шума; «мало случаев» — меньше
    {args.min_n} раз, выводов не делать. Сигнал — повод посмотреть график, а не приказ на вход.
  Подробный отчёт по одному тикеру: python nison_detector.py SBER --stats""")

    if args.csv and all_sig:
        out = pd.concat(all_sig, ignore_index=True)
        out["direction"] = np.where(out["dir"] > 0, "bull", "bear")
        out.drop(columns=["i", "dir"]).to_csv(args.csv, index=False)
        print(f"\nСигналы сохранены: {args.csv}")

    if not args.no_html:
        period = (f"сигналы на последних {args.bars} свечах" if args.bars
                  else f"сигналы за последние {args.recent} дн. (с {since:%d.%m.%Y})")
        page = html_scan(cards, quiet, failed, {
            "source": source, "count": len(tickers), "tf": args.tf, "period": period,
            "hz": hz, "min_n": args.min_n, "confirm": args.confirm})
        save_html(page, args.html, not args.no_open)
    return 0 if len(failed) < len(tickers) else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Свечные паттерны Нисона на данных Московской биржи (MOEX ISS).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Примеры:")[1].split("Нисон")[0] if __doc__ else None,
    )
    ap.add_argument("tickers", nargs="*",
                    help="тикер(ы) акций (SBER) или фьючерсов (Si, RTS, SiZ6); "
                         "без тикера — 15 голубых фишек")
    ap.add_argument("--demo", action="store_true", help="синтетические данные, без интернета")
    ap.add_argument("--futures", action="store_true", help="срочный рынок FORTS")
    ap.add_argument("--board", default="TQBR", help="режим торгов для акций (по умолчанию TQBR)")
    ap.add_argument("--tf", default="1d", choices=list(TIMEFRAMES), help="таймфрейм (по умолчанию 1d)")
    ap.add_argument("--days", type=int, help="глубина истории в днях")
    ap.add_argument("--stats", action="store_true", help="статистика отработки паттернов на истории")
    ap.add_argument("--horizons", default="1,3,5,10", help="горизонты статистики в барах")
    ap.add_argument("--confirm", action="store_true",
                    help="учитывать подтверждение следующей свечой (по Нисону)")
    ap.add_argument("--trend-len", type=int, default=10, help="баров для определения тренда")
    ap.add_argument("--last", type=int, default=15, help="сколько последних сигналов показать")
    ap.add_argument("--recent", type=int, default=7,
                    help="сводка: сигналы за сколько последних календарных дней (по умолчанию 7)")
    ap.add_argument("--bars", type=int,
                    help="сводка: вместо дней — сигналы на N последних свечах")
    ap.add_argument("--full", action="store_true",
                    help="сводка: дополнительно подробный отчёт по каждому тикеру")
    ap.add_argument("--min-n", type=int, default=20, help="порог «мало данных» в статистике")
    ap.add_argument("--csv", help="сохранить все сигналы в CSV")
    ap.add_argument("--brief", action="store_true", help="без пояснений к выводу")
    ap.add_argument("--html", default="nison_report.html",
                    help="куда сохранить HTML-отчёт (по умолчанию nison_report.html)")
    ap.add_argument("--no-html", action="store_true", help="не создавать HTML-отчёт")
    ap.add_argument("--no-open", action="store_true", help="не открывать отчёт в браузере")
    args = ap.parse_args(argv)

    try:
        horizons = sorted({int(x) for x in args.horizons.split(",") if x.strip()})
        if not horizons or min(horizons) < 1:
            raise ValueError
    except ValueError:
        ap.error("--horizons: список положительных целых, напр. 1,3,5,10")
    if args.recent < 1 or (args.bars is not None and args.bars < 1):
        ap.error("--recent и --bars должны быть >= 1")

    if args.demo:
        args.stats = True
        report(demo_data(), "DEMO (синтетика), 1d", args, horizons)
        return 0

    if len(args.tickers) == 1:
        try:
            df, title = load_moex(args.tickers[0], args.futures, args.tf, args.days, args.board)
        except Exception as e:  # сеть, неверный тикер и т.п.
            print(f"Ошибка загрузки данных MOEX: {e}", file=sys.stderr)
            print("Проверьте тикер и доступ к iss.moex.com, или запустите --demo.", file=sys.stderr)
            return 1
        if len(df) < 30:
            print(f"Слишком мало свечей ({len(df)}) для анализа.", file=sys.stderr)
            return 1
        report(df, title, args, horizons)
        return 0

    if args.tickers:
        return scan([t.upper() if not args.futures else t for t in args.tickers],
                    "заданный список", args, horizons)
    if args.futures:
        ap.error("для фьючерсов укажите тикер(ы), напр.: Si RTS BR --futures")
    tickers, source = blue_chips()
    return scan(tickers, source, args, horizons)


if __name__ == "__main__":
    sys.exit(main())
