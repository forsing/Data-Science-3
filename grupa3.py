"""Grupa 3 — sekvence / redosled (Loto 7/39)."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SEED = 39
FRONT_N = 39
FRONT_SELECT = 7
CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "loto7_4648_k55.csv"

np.random.seed(SEED)


def load_draws(csv_path: Path = CSV_PATH) -> np.ndarray:
    draws = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) < FRONT_SELECT:
                continue
            try:
                draw = sorted(int(x.strip()) for x in row[:FRONT_SELECT])
            except ValueError:
                continue
            if len(draw) == FRONT_SELECT and all(1 <= x <= FRONT_N for x in draw):
                if len(set(draw)) == FRONT_SELECT:
                    draws.append(draw)
    if not draws:
        raise ValueError(f"Nema validnih kola u {csv_path}")
    return np.array(draws, dtype=int)


def transition_matrix(draws: np.ndarray) -> dict:
    """P(broj u t+1 | broj u t) — brojačka transition matrica 39×39."""
    trans = np.zeros((FRONT_N + 1, FRONT_N + 1), dtype=float)
    for i in range(len(draws) - 1):
        a = set(draws[i].tolist())
        b = set(draws[i + 1].tolist())
        for x in a:
            for y in b:
                trans[x, y] += 1.0
    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    probs = trans / row_sums
    return {"counts": trans, "probs": probs}


def markov_top_successors(draws: np.ndarray, top_k: int = 5) -> dict:
    """Za svaki broj: top naslednici u sledećem kolu."""
    tm = transition_matrix(draws)["probs"]
    out = {}
    for n in range(1, FRONT_N + 1):
        row = tm[n, 1:]
        idx = np.argsort(-row)[:top_k]
        out[n] = [(int(j + 1), float(row[j])) for j in idx]
    return out


def sequential_carry_stats(draws: np.ndarray) -> dict:
    """Koliko brojeva „pređe“ iz kola t u t+1 (carry / overlap sekvence)."""
    carries = [len(set(draws[i]) & set(draws[i + 1])) for i in range(len(draws) - 1)]
    return {
        "mean_carry": float(np.mean(carries)),
        "hist": dict(sorted(Counter(carries).items())),
    }


def ngram_number_sequences(draws: np.ndarray, top_k: int = 15) -> dict:
    """
    Sekvence pojavljivanja broja kroz vreme (prisutan=1):
    bigram/trigram obrazaca 1→1 (ponavljanje u uzastopnim kolima).
    """
    presence = np.zeros((len(draws), FRONT_N + 1), dtype=int)
    for i, draw in enumerate(draws):
        for n in draw.tolist():
            presence[i, n] = 1

    bigrams = Counter()
    trigrams = Counter()
    for n in range(1, FRONT_N + 1):
        seq = presence[:, n]
        for i in range(len(seq) - 1):
            bigrams[(n, int(seq[i]), int(seq[i + 1]))] += 1
        for i in range(len(seq) - 2):
            trigrams[(n, int(seq[i]), int(seq[i + 1]), int(seq[i + 2]))] += 1

    # najčešći 1→1 bigrami (broj se ponovi sledeće kolo)
    repeat = [((n, 1, 1), c) for (n, a, b), c in bigrams.items() if a == 1 and b == 1]
    repeat.sort(key=lambda x: (-x[1], x[0][0]))
    return {"top_repeat_1to1": repeat[:top_k]}


def learn_next_rule(draws: np.ndarray) -> dict:
    """
    Pravilo next iz grupe 3:
    skor(y) = sum_x in last P(y|x)  (Markov naslednici poslednjeg kola).
    """
    last = set(draws[-1].tolist())
    probs = transition_matrix(draws)["probs"]
    number_score = {}
    for y in range(1, FRONT_N + 1):
        number_score[y] = float(sum(probs[x, y] for x in last))
    # blagi prior frekvencije da ne bude prazno
    freq = Counter(draws.reshape(-1).tolist())
    max_f = max(freq.values()) if freq else 1
    for y in range(1, FRONT_N + 1):
        number_score[y] += 0.15 * (freq.get(y, 0) / max_f)

    return {
        "number_score": number_score,
        "last_draw": sorted(last),
        "mean_carry": sequential_carry_stats(draws)["mean_carry"],
        "target_sum": float(draws.sum(axis=1).mean()),
    }


def _combo_fit(combo: list[int], rule: dict) -> float:
    score = sum(rule["number_score"][x] for x in combo)
    carry = len(set(combo) & set(rule["last_draw"]))
    score -= 0.35 * abs(carry - rule["mean_carry"])
    score -= 0.015 * abs(sum(combo) - rule["target_sum"])
    return score


def predict_next_from_rule(draws: np.ndarray, rule: dict | None = None) -> list[int]:
    if rule is None:
        rule = learn_next_rule(draws)
    ranked = sorted(rule["number_score"], key=lambda n: (-rule["number_score"][n], n))
    best = None
    best_fit = -1e18
    for start in range(0, min(20, FRONT_N - FRONT_SELECT + 1)):
        base = sorted(ranked[start : start + FRONT_SELECT])
        for repl in ranked[:28]:
            cand = sorted(set(base[1:] + [repl]))
            if len(cand) != FRONT_SELECT:
                continue
            fit = _combo_fit(cand, rule)
            if fit > best_fit:
                best_fit = fit
                best = cand
    return best if best is not None else sorted(ranked[:FRONT_SELECT])


def run_grupa3(csv_path: Path = CSV_PATH) -> None:
    draws = load_draws(csv_path)
    print(f"CSV: {csv_path.name}")
    print(f"Kola: {len(draws)} | seed={SEED} | 7/39 | grupa3")
    print()

    print("=== sequential carry (t→t+1 overlap) ===")
    print(sequential_carry_stats(draws))
    print()

    print("=== Markov top successors (sample numbers) ===")
    tops = markov_top_successors(draws)
    for n in [8, 23, 34, draws[-1][0], draws[-1][-1]]:
        print(f"  {n} → {tops[int(n)]}")
    print()

    print("=== n-gram repeat 1→1 (top) ===")
    print(ngram_number_sequences(draws)["top_repeat_1to1"][:10])
    print()

    print("=== pravilo → next (grupa 3) ===")
    rule = learn_next_rule(draws)
    combo = predict_next_from_rule(draws, rule)
    print(
        "rule:",
        {
            "last_draw": rule["last_draw"],
            "mean_carry": round(rule["mean_carry"], 3),
            "target_sum": round(rule["target_sum"], 2),
        },
    )
    print("next:", combo)


if __name__ == "__main__":
    run_grupa3()


"""
3. Sekvence / redosled
GSP, PrefixSpan, Spade, CM-SPADE, CloSpan, BIDE, sequential rules,
episode mining, motif discovery, n-gram, skip-gram na nizovima,
transition matrix, Markov 1..k, higher-order Markov, variable-order Markov,
suffix tree/array obrasci
"""



"""
CSV: loto7_4648_k55.csv
Kola: 4648 | seed=39 | 7/39 | grupa3

=== sequential carry (t→t+1 overlap) ===
{'mean_carry': 1.2336991607488703, 'hist': {0: 1018, 1: 1979, 2:1266, 3: 321, 4: 56, 5: 7}}

=== Markov top successors (sample numbers) ===
  8 → [(9, 0.029039812646370025), (39, 0.02841530054644809), (38, 0.02841530054644809), (34, 0.027946916471506635), (16, 0.027790788446526153)]
  23 → [(37, 0.02973568281938326), (13, 0.0289490245437382), (23, 0.028791692888609187), (29, 0.028791692888609187), (5, 0.028634361233480177)]
  34 → [(10, 0.02902804957599478), (39, 0.028538812785388126), (7, 0.02837573385518591), (25, 0.02756033920417482), (31, 0.02756033920417482)]
  3 → [(10, 0.02974041602200447), (26, 0.029568506102802133), (3, 0.02888086642599278), (23, 0.02870895650679044), (8, 0.028365136668385766)]
  29 → [(23, 0.03051643192488263), (22, 0.029175050301810865), (13, 0.029175050301810865), (9, 0.028504359490274984), (35, 0.028169014084507043)]

=== n-gram repeat 1→1 (top) ===
[((23, 1, 1), 183), ((8, 1, 1), 169), ((3, 1, 1), 168), ((25, 1,1), 168), ((34, 1, 1), 168), ((5, 1, 1), 164), ((35, 1, 1), 163), ((37, 1, 1), 162), ((22, 1, 1), 160), ((11, 1, 1), 159)]

=== pravilo → next (grupa 3) ===
rule: {'last_draw': [3, 7, 12, 13, 18, 24, 29], 'mean_carry': 1.234, 'target_sum': 140.43}
next: [2, 7, 9, 21, 25, 38, 39]
"""
