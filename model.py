import random
import time

from card_utils import Card
from game_state import GameStateView
from player_action import PlayerAction

# Cards are handled internally as ints: (value - 2) * 4 + suit index.
_SUITS = "CDHS"
_STRAIGHTS = [(0b11111 << (h - 4), h) for h in range(14, 5, -1)]
_STRAIGHTS.append(((1 << 14) | 0b111100, 5))  # wheel A-2-3-4-5 (bits by value)


def _to_int(card):
    return (card.value - 2) * 4 + _SUITS.index(card.suit)


def _straight_high(mask):
    for pattern, high in _STRAIGHTS:
        if mask & pattern == pattern:
            return high
    return 0


def _score(cards):
    counts = [0] * 15
    suit_masks = [0, 0, 0, 0]
    suit_counts = [0, 0, 0, 0]
    mask = 0
    for c in cards:
        v = (c >> 2) + 2
        s = c & 3
        counts[v] += 1
        suit_masks[s] |= 1 << v
        suit_counts[s] += 1
        mask |= 1 << v
    for s in range(4):
        if suit_counts[s] >= 5:
            sm = suit_masks[s]
            sh = _straight_high(sm)
            if sh:
                return (8 << 20) | sh
            top = [v for v in range(14, 1, -1) if sm >> v & 1][:5]
            return (5 << 20) | _pack(top)
    quads = trips = 0
    pairs = []
    singles = []
    for v in range(14, 1, -1):
        n = counts[v]
        if n == 4:
            quads = v
        elif n == 3:
            if trips:
                pairs.append(v)
            else:
                trips = v
        elif n == 2:
            pairs.append(v)
        elif n == 1:
            singles.append(v)
    if quads:
        kick = max([v for v in range(14, 1, -1) if counts[v] and v != quads])
        return (7 << 20) | _pack([quads, kick])
    if trips and pairs:
        return (6 << 20) | _pack([trips, pairs[0]])
    sh = _straight_high(mask)
    if sh:
        return (4 << 20) | sh
    if trips:
        return (3 << 20) | _pack([trips] + singles[:2])
    if len(pairs) >= 2:
        rest = sorted(pairs[2:] + singles, reverse=True)
        return (2 << 20) | _pack(pairs[:2] + rest[:1])
    if pairs:
        return (1 << 20) | _pack([pairs[0]] + singles[:3])
    return _pack(singles[:5])


def _pack(values):
    out = 0
    for i in range(5):
        out = (out << 4) | (values[i] if i < len(values) else 0)
    return out


def _preflop_strength(a, b):
    va, vb = (a >> 2) + 2, (b >> 2) + 2
    hi, lo = max(va, vb), min(va, vb)
    score = hi * 2 + lo
    if hi == lo:
        score += 14 + hi
    if (a & 3) == (b & 3):
        score += 4
    gap = hi - lo
    if gap == 1:
        score += 3
    elif gap == 2:
        score += 1
    return min(1.0, (score - 7) / 50.0)


class Model:
    """Monte Carlo equity bot with simple per-team opponent modelling."""

    def __init__(self):
        self.hand = []
        self.player_id = None
        self.opp = {}

    def _stats(self, team):
        if team not in self.opp:
            self.opp[team] = {"raise": 0, "call": 0, "fold": 0,
                              "faced": 0, "folded_to": 0}
        return self.opp[team]

    def _observe(self, actions, game_state):
        try:
            teams = game_state.teams
            my_bet = False
            for a in actions:
                if a.kind not in ("fold", "call", "raise"):
                    continue
                if a.player_id == self.player_id:
                    my_bet = a.kind == "raise"
                    continue
                st = self._stats(teams[a.player_id])
                st[a.kind] += 1
                if my_bet:
                    st["faced"] += 1
                    if a.kind == "fold":
                        st["folded_to"] += 1
                    my_bet = False
        except Exception:
            pass

    def _aggression(self, team):
        st = self._stats(team)
        n = st["raise"] + st["call"] + st["fold"]
        return (st["raise"] + 2.0) / (n + 6.0)

    def _fold_to_bet(self, team):
        st = self._stats(team)
        return (st["folded_to"] + 1.0) / (st["faced"] + 3.0)

    def _equity(self, board, opponents, min_strength, budget):
        mine = [_to_int(c) for c in self.hand]
        brd = [_to_int(c) for c in board]
        used = set(mine + brd)
        unseen = [c for c in range(52) if c not in used]
        missing = 5 - len(brd)
        need = missing + 2 * opponents
        won = 0.0
        n = 0
        deadline = time.time() + budget
        sample = random.sample
        while n < 3000:
            if n % 100 == 0 and n >= 200 and time.time() > deadline:
                break
            cards = sample(unseen, need)
            holes = cards[missing:]
            if min_strength > 0:
                ok = True
                for i in range(opponents):
                    if _preflop_strength(holes[2 * i], holes[2 * i + 1]) < min_strength:
                        ok = False
                        break
                if not ok and random.random() < 0.85:
                    continue
            full = brd + cards[:missing]
            me = _score(mine + full)
            best = max(_score(holes[2 * i:2 * i + 2] + full)
                       for i in range(opponents))
            if me > best:
                won += 1
            elif me == best:
                won += 0.5
            n += 1
        return won / n if n else 0.5

    def _raise(self, gs, total):
        me = self.player_id
        if not gs.can_raise or gs.raise_range is None:
            return PlayerAction.call(me)
        low, high = gs.raise_range
        total = int(min(high, max(low, total)))
        return PlayerAction.raise_to(me, total)

    def _bet_frac(self, gs, frac):
        return gs.current_bet + frac * (gs.pot + gs.call_amount)

    def seed(self, hand: list[Card], player_id: int):
        self.hand = hand
        self.player_id = player_id

    def run(self, actions: list[PlayerAction], game_state: GameStateView) -> PlayerAction:
        self._observe(actions, game_state)
        try:
            return self._decide(game_state)
        except Exception:
            if game_state.call_amount == 0:
                return PlayerAction.call(self.player_id)
            return PlayerAction.fold(self.player_id)

    def _decide(self, gs):
        me = self.player_id
        opps = [p for p in gs.players_in_hand if p != me]
        n_opp = max(1, len(opps))
        teams = gs.teams
        aggr = max(self._aggression(teams[p]) for p in opps) if opps else 0.33
        fold_to = min(self._fold_to_bet(teams[p]) for p in opps) if opps else 0.33
        call_amt = gs.call_amount
        pot = gs.pot
        bb = gs.big_blind
        facing = call_amt > 0
        preflop = gs.street == "preflop"
        bet_ratio = call_amt / max(1, pot - call_amt)

        min_strength = 0.0
        tightness = max(0.0, min(1.0, (0.55 - aggr) / 0.35))
        if facing and (not preflop or gs.current_bet > bb):
            min_strength = 0.25 * tightness * min(1.5, 0.5 + bet_ratio)

        eq = self._equity(list(gs.board), n_opp, min_strength, 0.6)
        if facing and not preflop:
            eq -= 0.10 * tightness * min(1.5, bet_ratio)

        r = random.random()

        if preflop:
            if not facing or call_amt <= bb // 2:
                if eq > 0.60:
                    return self._raise(gs, 3 * bb)
                if eq > 0.40 or (eq > 0.34 and fold_to > 0.45):
                    return self._raise(gs, int(2.5 * bb))
                if call_amt == 0:
                    return PlayerAction.call(me)
                if eq > 0.36:
                    return PlayerAction.call(me)
                return PlayerAction.fold(me)
            pot_odds = gs.pot_odds
            if eq > 0.70 and r < 0.8:
                return self._raise(gs, 3 * gs.current_bet)
            if eq > 0.62 and gs.current_bet < 6 * bb:
                return self._raise(gs, 3 * gs.current_bet)
            if (fold_to > 0.5 and gs.current_bet <= 4 * bb and eq > 0.45
                    and r < 0.4):
                return self._raise(gs, 3 * gs.current_bet)
            if eq > pot_odds + 0.04:
                return PlayerAction.call(me)
            return PlayerAction.fold(me)

        pot_odds = gs.pot_odds
        station = fold_to < 0.25
        value_size = 0.9 if station else 0.65
        bluff_rate = 0.0 if station else max(0.15, min(0.7, 1.3 * fold_to))
        if gs.street == "river":
            bluff_rate = bluff_rate if fold_to > 0.5 else 0.0
        if eq > 0.82:
            if gs.spr(me) < 1.5:
                return self._raise(gs, gs.max_raise_to or 0)
            return self._raise(gs, self._bet_frac(gs, 0.85))
        if eq > 0.66:
            if not facing or r < 0.35:
                return self._raise(gs, self._bet_frac(gs, value_size))
            return PlayerAction.call(me)
        if not facing:
            if eq > 0.55 and r < (0.75 if station else 0.5):
                return self._raise(gs, self._bet_frac(gs, 0.5))
            if r < bluff_rate:
                return self._raise(gs, self._bet_frac(gs, 0.5))
            return PlayerAction.call(me)
        if (gs.street == "flop" and fold_to > 0.5 and eq > 0.35
                and bet_ratio <= 0.75 and r < 0.25):
            return self._raise(gs, 3 * gs.current_bet)
        if eq > pot_odds + 0.03:
            return PlayerAction.call(me)
        return PlayerAction.fold(me)

    def end_game(self, actions: list[PlayerAction], game_state: GameStateView):
        self._observe(actions, game_state)
