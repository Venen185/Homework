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
    python nison_detector.py --bars 5 --full        голубые фишки, сигналы за 5 свечей + отчёты

Нисон подчёркивает: разворотная модель имеет смысл только при наличии тренда,
который можно развернуть, поэтому все разворотные паттерны проверяются
в контексте предшествующего тренда (см. --trend-len).
"""

from __future__ import annotations

import argparse
import sys
import time
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


def scan(tickers: list[str], source: str, args, horizons: list[int]) -> int:
    """Сводная таблица по списку инструментов: сигналы на последних --bars свечах
    и как этот паттерн отрабатывал раньше на этом же инструменте."""
    hz = key_horizon(horizons)
    rows, quiet, failed, all_sig = [], [], [], []
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
            report(df, title, argparse.Namespace(**{**vars(args), "stats": True, "csv": None}),
                   horizons)
            print()
        if args.csv:
            all_sig.append(sig.assign(ticker=t))

        n = len(df)
        recent = sig[sig["i"] >= n - args.bars]
        if recent.empty:
            quiet.append(t)
            continue
        for r in recent.itertuples():
            row = table[table["Паттерн"] == r.pattern]
            hist = ""
            if not row.empty:
                row = row.iloc[0]
                b = base[base[""] == row[""]].iloc[0]
                verdict, _ = judge(row, base, std, hz, args.min_n)
                if row[f"win% {hz}"] == row[f"win% {hz}"]:  # не NaN
                    hist = (f"N={row['N']:<4} win {row[f'win% {hz}']:3.0f}% "
                            f"(случ. {b[f'win% {hz}']:.0f}%), ср. {row[f'ср.% {hz}']:+.2f}% "
                            f"→ {SHORT_VERDICT[verdict]}")
                else:
                    hist = f"N={row['N']:<4} → {SHORT_VERDICT[verdict]}"
            ago = n - 1 - r.i
            when = "последняя" if ago == 0 else f"{ago} св. назад"
            conf = {True: "✓", False: "✗", None: "…"}[r.confirmed] if args.confirm else ""
            rows.append((t, df["close"].iloc[-1], fmt_time(r.time), when,
                         ("▲ " if r.dir > 0 else "▼ ") + r.pattern, conf, hist))
    print(" " * 40, end="\r", file=sys.stderr)

    if args.full:
        print("=" * 100)
    print(f"Сводка ({source}), инструментов: {len(tickers)}, таймфрейм {args.tf}, "
          f"сигналы на последних {args.bars} свечах")
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
    ap.add_argument("--bars", type=int, default=3,
                    help="сводка: сигналы на скольких последних свечах показывать (по умолчанию 3)")
    ap.add_argument("--full", action="store_true",
                    help="сводка: дополнительно подробный отчёт по каждому тикеру")
    ap.add_argument("--min-n", type=int, default=20, help="порог «мало данных» в статистике")
    ap.add_argument("--csv", help="сохранить все сигналы в CSV")
    ap.add_argument("--brief", action="store_true", help="без пояснений к выводу")
    args = ap.parse_args(argv)

    try:
        horizons = sorted({int(x) for x in args.horizons.split(",") if x.strip()})
        if not horizons or min(horizons) < 1:
            raise ValueError
    except ValueError:
        ap.error("--horizons: список положительных целых, напр. 1,3,5,10")
    if args.bars < 1:
        ap.error("--bars должен быть >= 1")

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
