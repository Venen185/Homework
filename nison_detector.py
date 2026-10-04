#!/usr/bin/env python3
"""
Детектор свечных паттернов по Стиву Нисону («Японские свечи») для Московской биржи.

Данные берутся из открытого API MOEX ISS (без ключа и регистрации).

Примеры:
    python nison_detector.py --demo                 проверка без интернета (синтетика)
    python nison_detector.py SBER --stats           акция, дневки + статистика на истории
    python nison_detector.py Si --futures --tf 1h   фьючерс (ближайший контракт), часовики
    python nison_detector.py SiZ6 --futures         конкретный контракт
    python nison_detector.py GAZP --tf 4h --last 30 --confirm --csv gazp.csv

Нисон подчёркивает: разворотная модель имеет смысл только при наличии тренда,
который можно развернуть, поэтому все разворотные паттерны проверяются
в контексте предшествующего тренда (см. --trend-len).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
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
    r = requests.get(f"{ISS}/{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _table(js: dict, name: str) -> pd.DataFrame:
    block = js.get(name) or {}
    return pd.DataFrame(block.get("data", []), columns=block.get("columns", []))


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
          confirm: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return table, pd.DataFrame(base_rows)


# ---------------------------------------------------------------------------
# Вывод
# ---------------------------------------------------------------------------

def fmt_time(t: pd.Timestamp) -> str:
    return t.strftime("%Y-%m-%d") if (t.hour, t.minute) == (0, 0) else t.strftime("%Y-%m-%d %H:%M")


def print_signals(sig: pd.DataFrame, last: int, confirm: bool, n_bars: int) -> None:
    if sig.empty:
        print("Паттерны не найдены.")
        return
    tail = sig.tail(last)
    print(f"Последние сигналы ({len(tail)} из {len(sig)}):")
    for r in tail.itertuples():
        arrow = "▲ бычий   " if r.dir > 0 else "▼ медвежий"
        line = f"  {fmt_time(r.time):<16}  {arrow}  {r.pattern:<26} ({r.kind}) close={r.close:g}"
        if confirm:
            line += {True: "  ✓ подтверждён", False: "  ✗ не подтверждён",
                     None: "  … ждёт подтверждения"}[r.confirmed]
        print(line)

    on_last = sig[sig["i"] == n_bars - 1]
    print()
    if on_last.empty:
        print("На последней свече паттернов нет.")
    else:
        names = ", ".join(f"{p} ({'▲' if d > 0 else '▼'})"
                          for p, d in zip(on_last["pattern"], on_last["dir"]))
        print(f"На последней свече: {names}")


def print_stats(table: pd.DataFrame, base: pd.DataFrame, horizons: list[int],
                confirm: bool, min_n: int) -> None:
    print()
    print("Статистика на истории (доходность в сторону сигнала, закрытие → закрытие через h баров"
          + (", вход после подтверждения" if confirm else "") + "):")
    if table.empty:
        print("  нет сигналов")
        return
    table = table.copy()
    rare = table["N"] < min_n
    table.loc[rare, "Паттерн"] = table.loc[rare, "Паттерн"] + " *"
    full = pd.concat([table, base], ignore_index=True)
    fmts = {col: (lambda v: "" if pd.isna(v) else f"{v:6.1f}") for col in full.columns
            if col.startswith("win%")}
    fmts.update({col: (lambda v: "" if pd.isna(v) else f"{v:+6.2f}") for col in full.columns
                 if col.startswith("ср.%")})
    text = full.to_string(index=False, formatters=fmts)
    lines = text.splitlines()
    sep = "-" * len(lines[0])
    print(lines[0])
    print(sep)
    for line in lines[1:len(table) + 1]:
        print(line)
    print(sep)
    for line in lines[len(table) + 1:]:
        print(line)
    if rare.any():
        print(f"\n* меньше {min_n} случаев — статистика ненадёжна")
    print("Сравнивайте win%/ср.% паттерна с бейзлайном «Любой бар» той же стороны:"
          " преимущество есть, только если паттерн заметно лучше.")


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Свечные паттерны Нисона на данных Московской биржи (MOEX ISS).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Примеры:")[1].split("Нисон")[0] if __doc__ else None,
    )
    ap.add_argument("ticker", nargs="?", help="тикер акции (SBER) или фьючерса (Si, RTS, SiZ6)")
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
    ap.add_argument("--min-n", type=int, default=20, help="порог «мало данных» в статистике")
    ap.add_argument("--csv", help="сохранить все сигналы в CSV")
    args = ap.parse_args(argv)

    try:
        horizons = sorted({int(x) for x in args.horizons.split(",") if x.strip()})
        if not horizons or min(horizons) < 1:
            raise ValueError
    except ValueError:
        ap.error("--horizons: список положительных целых, напр. 1,3,5,10")

    if args.demo:
        df, title = demo_data(), "DEMO (синтетика), 1d"
        args.stats = True
    elif args.ticker:
        try:
            df, title = load_moex(args.ticker, args.futures, args.tf, args.days, args.board)
        except Exception as e:  # сеть, неверный тикер и т.п.
            print(f"Ошибка загрузки данных MOEX: {e}", file=sys.stderr)
            print("Проверьте тикер и доступ к iss.moex.com, или запустите --demo.", file=sys.stderr)
            return 1
    else:
        ap.error("укажите тикер или --demo")

    if len(df) < 30:
        print(f"Слишком мало свечей ({len(df)}) для анализа.", file=sys.stderr)
        return 1

    sig = detect(df, args.trend_len)
    if args.confirm or args.stats:
        sig = add_confirmation(sig, df["close"].to_numpy(float))
    if not args.confirm and "confirmed" in sig:
        sig_print = sig.drop(columns="confirmed")
    else:
        sig_print = sig

    print(f"{title}: {len(df)} свечей, {fmt_time(df.index[0])} — {fmt_time(df.index[-1])}, "
          f"последнее закрытие {df['close'].iloc[-1]:g}")
    print()
    print_signals(sig_print, args.last, args.confirm, len(df))

    if args.stats:
        table, base = stats(df, sig, horizons, args.confirm)
        print_stats(table, base, horizons, args.confirm, args.min_n)

    if args.csv:
        out = sig.copy()
        out["direction"] = np.where(out["dir"] > 0, "bull", "bear")
        out.drop(columns=["i", "dir"]).to_csv(args.csv, index=False)
        print(f"\nСигналы сохранены: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
