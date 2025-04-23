# trader_follow_camilla.py
#
# Place this file in your round‑7 folder (or anywhere on the PYTHONPATH) and
# run:  prosperity-backtester trader_follow_camilla.py --round 5 --days 2 3 4

from datamodel import Order, TradingState
from typing import Dict, List, Any
import json, jsonpickle

# ─────────────────────────── Logger (full, unchanged) ─────────────────────────
logger = None
class Logger:
    def __init__(self) -> None:
        self.logs, self.max_log_length = "", 3750

    def print(self, *objs: Any, sep=" ", end="\n") -> None:
        self.logs += sep.join(map(str, objs)) + end

    def flush(self, state: TradingState, orders, conversions, tdata):
        base = len(self.to_json([self._state(state,""), self._orders(orders),
                                 conversions, "", ""]))
        lim  = (self.max_log_length - base)//3
        print(self.to_json([
            self._state(state, self._trunc(state.traderData, lim)),
            self._orders(orders), conversions,
            self._trunc(tdata, lim), self._trunc(self.logs, lim)]))
        self.logs = ""

    # helpers (identical to Prosperity’s template) ---------------------------
    def _state(self, s: TradingState, td: str):
        return [
            s.timestamp, td,
            [[l.symbol,l.product,l.denomination] for l in s.listings.values()],
            {sym:[od.buy_orders,od.sell_orders] for sym,od in s.order_depths.items()},
            [[t.symbol,t.price,t.quantity,t.buyer,t.seller,t.timestamp]
             for ts in s.own_trades.values() for t in ts],
            [[t.symbol,t.price,t.quantity,t.buyer,t.seller,t.timestamp]
             for ts in s.market_trades.values() for t in ts],
            s.position, []  # we ignore observations to save log space
        ]
    def _orders(self, orders): return [[o.symbol,o.price,o.quantity]
                                       for ol in orders.values() for o in ol]
    def to_json(self,v): return json.dumps(v,separators=(",",":"))
    def _trunc(self,v,l): return v if len(v)<=l else v[:l-3]+"..."
logger = Logger()
# ──────────────────────────────────────────────────────────────────────────────


class Trader:
    """Mirror Camilla’s voucher trades one tick later."""
    LIMIT = 400                     # max long OR short per symbol

    # memory of what to execute next timestamp
    def __init__(self):
        self.queue_next: Dict[str, int] = {}

    # ---------- utility ------------------------------------------------------
    def _cross(self, sym: str, qty: int, state: TradingState,
               book: Dict[str, List[Order]]):
        """Send aggressive order guaranteeing immediate fill."""
        od = state.order_depths.get(sym)
        if qty > 0:   # BUY
            price = min(od.sell_orders) if od and od.sell_orders else 10**9
        else:         # SELL
            price = max(od.buy_orders)  if od and od.buy_orders  else 0
        book.setdefault(sym, []).append(Order(sym, price, qty))

    # ---------- step 1: replicate queued orders ------------------------------
    def _execute_queue(self, state: TradingState, orders: Dict[str, List[Order]]):
        for sym, queued_qty in list(self.queue_next.items()):
            if queued_qty == 0:
                continue
            pos = state.position.get(sym, 0)
            # clip to not break inventory limit
            max_buy  = self.LIMIT - pos
            max_sell = -self.LIMIT - pos
            qty = max(min(queued_qty, max_buy), max_sell)
            if qty != 0:
                self._cross(sym, qty, state, orders)
                logger.print(f"[{state.timestamp}] EXEC {sym} {qty}")
            # reset queue for this symbol
            self.queue_next[sym] = 0

    # ---------- step 2: observe Camilla this tick ---------------------------
    def _camilla_trades_this_tick(self, state: TradingState) -> Dict[str, int]:
        delta: Dict[str, int] = {}
        for sym, trades in state.market_trades.items():
            if "VOUCHER" not in sym:        # *** remove this line if you want
                continue                    #     to follow her in everything
            net = 0
            for t in trades:
                if t.buyer == "Camilla":
                    net += t.quantity
                elif t.seller == "Camilla":
                    net -= t.quantity
            if net:
                delta[sym] = delta.get(sym, 0) + net
        return delta

    # ---------- main entry ---------------------------------------------------
    def run(self, state: TradingState):
        orders: Dict[str, List[Order]] = {}
        conversions = 0

        # 1. fire orders stored from previous tick
        self._execute_queue(state, orders)

        # 2. watch Camilla now and queue for next tick
        cam_delta = self._camilla_trades_this_tick(state)
        for sym, qty in cam_delta.items():
            self.queue_next[sym] = self.queue_next.get(sym, 0) + qty
            side = "BUY" if qty > 0 else "SELL"
            logger.print(f"[{state.timestamp}] Will {side} {abs(qty)} {sym} next tick")

        # 3. flush & finish
        tdata = jsonpickle.encode({})
        logger.flush(state, orders, conversions, tdata)
        return orders, conversions, tdata
