"""
chess_app.py - a complete chess game with a Streamlit front-end.

Run with:   streamlit run chess_app.py
Requires:   streamlit >= 1.39  (everything else is the Python standard library)

The file has two clearly separated halves:

  1. BACKEND   Piece, Move, Board, Game, ChessAI - pure Python, no Streamlit.
               The position is an 8x8 matrix of strings: "wp" = white pawn,
               "bk" = black king, "" = empty square.  Row 0 is rank 8 and
               column 0 is file a, so grid[6][4] is e2.
  2. FRONTEND  Streamlit rendering and interaction (clickable board, sidebar
               with settings, captured pieces and move history).

To split it into two files, move everything in section 1 into `chess_engine.py`
and put `from chess_engine import *` at the top of `app.py`.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Dict, List, NamedTuple, Optional, Tuple

import streamlit as st

# =============================================================================
# 1. BACKEND
# =============================================================================

Square = Tuple[int, int]  # (row, col) - row 0 is rank 8, col 0 is file "a"
WHITE, BLACK = "w", "b"
FILES = "abcdefgh"

KNIGHT_STEPS = ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1))
KING_STEPS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
ROOK_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
BISHOP_DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
QUEEN_DIRS = ROOK_DIRS + BISHOP_DIRS

# Moving from, or capturing on, one of these squares removes castling rights.
RIGHTS_LOST_ON = {
    (7, 4): "KQ", (7, 0): "Q", (7, 7): "K",
    (0, 4): "kq", (0, 0): "q", (0, 7): "k",
}


def opponent(color: str) -> str:
    return BLACK if color == WHITE else WHITE


def square_name(r: int, c: int) -> str:
    """(6, 4) -> 'e2'"""
    return f"{FILES[c]}{8 - r}"


class Piece:
    """Small value object wrapping a piece code such as 'wp' or 'bk'.

    The board matrix stores plain strings; this class adds names, symbols and
    values on top of them (used by the UI and by the AI).
    """

    __slots__ = ("code",)

    NAMES = {"p": "Pawn", "n": "Knight", "b": "Bishop", "r": "Rook", "q": "Queen", "k": "King"}
    # Filled glyphs for both colours (colour is applied with CSS). The trailing
    # U+FE0E forces text presentation so phones do not draw the pawn as an emoji.
    SYMBOLS = {"k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟"}
    VALUES = {"p": 100, "n": 320, "b": 330, "r": 500, "q": 900, "k": 0}  # centipawns

    def __init__(self, code: str):
        if len(code) != 2 or code[0] not in "wb" or code[1] not in "pnbrqk":
            raise ValueError(f"Invalid piece code: {code!r}")
        self.code = code

    @property
    def color(self) -> str:
        return self.code[0]

    @property
    def kind(self) -> str:
        return self.code[1]

    @property
    def is_white(self) -> bool:
        return self.code[0] == WHITE

    @property
    def name(self) -> str:
        return self.NAMES[self.kind]

    @property
    def symbol(self) -> str:
        return self.SYMBOLS[self.kind] + "\ufe0e"

    @property
    def value(self) -> int:
        return self.VALUES[self.kind]

    @property
    def points(self) -> int:
        """Classic 1/3/3/5/9 piece values."""
        return self.VALUES[self.kind] // 100

    def __repr__(self) -> str:
        return f"Piece({self.code!r})"


class Move(NamedTuple):
    start: Square
    end: Square
    piece: str
    captured: str = ""
    promotion: str = ""  # "q", "r", "b" or "n"
    castle: str = ""  # "K" (king side) or "Q" (queen side)
    en_passant: bool = False

    @property
    def uci(self) -> str:
        return square_name(*self.start) + square_name(*self.end) + self.promotion


class Board:
    """An 8x8 matrix of piece codes plus the extra state needed for move rules
    (castling rights, en-passant target square, king locations)."""

    def __init__(self) -> None:
        self.grid: List[List[str]] = [[""] * 8 for _ in range(8)]
        self.castling: str = ""  # subset of "KQkq"
        self.ep_target: Optional[Square] = None  # square a pawn skipped over
        self.kings: Dict[str, Optional[Square]] = {WHITE: None, BLACK: None}

    # ---- construction -------------------------------------------------------
    @classmethod
    def initial(cls) -> "Board":
        b = cls()
        back = ["r", "n", "b", "q", "k", "b", "n", "r"]
        b.grid[0] = [BLACK + k for k in back]
        b.grid[1] = [BLACK + "p"] * 8
        b.grid[6] = [WHITE + "p"] * 8
        b.grid[7] = [WHITE + k for k in back]
        b.castling = "KQkq"
        b.kings = {WHITE: (7, 4), BLACK: (0, 4)}
        return b

    @classmethod
    def from_fen(cls, fen: str) -> Tuple["Board", str, int, int]:
        """Parse a FEN string -> (board, side_to_move, halfmove_clock, fullmove_number)."""
        parts = fen.split()
        b = cls()
        for r, row in enumerate(parts[0].split("/")):
            c = 0
            for ch in row:
                if ch.isdigit():
                    c += int(ch)
                else:
                    color = WHITE if ch.isupper() else BLACK
                    b.grid[r][c] = color + ch.lower()
                    if ch.lower() == "k":
                        b.kings[color] = (r, c)
                    c += 1
        turn = parts[1] if len(parts) > 1 else WHITE
        b.castling = "" if len(parts) < 3 or parts[2] == "-" else parts[2]
        ep = parts[3] if len(parts) > 3 else "-"
        b.ep_target = None if ep == "-" else (8 - int(ep[1]), FILES.index(ep[0]))
        halfmove = int(parts[4]) if len(parts) > 4 else 0
        fullmove = int(parts[5]) if len(parts) > 5 else 1
        return b, turn, halfmove, fullmove

    def copy(self) -> "Board":
        b = Board.__new__(Board)
        b.grid = [row[:] for row in self.grid]
        b.castling = self.castling
        b.ep_target = self.ep_target
        b.kings = dict(self.kings)
        return b

    # ---- attack detection ---------------------------------------------------
    def is_attacked(self, r: int, c: int, by: str) -> bool:
        """Is square (r, c) attacked by any piece of colour `by`?"""
        g = self.grid
        # pawns (white pawns attack towards row 0, black pawns towards row 7)
        pr = r + 1 if by == WHITE else r - 1
        if 0 <= pr < 8:
            pawn = by + "p"
            if c > 0 and g[pr][c - 1] == pawn:
                return True
            if c < 7 and g[pr][c + 1] == pawn:
                return True
        knight, king = by + "n", by + "k"
        for dr, dc in KNIGHT_STEPS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < 8 and 0 <= cc < 8 and g[rr][cc] == knight:
                return True
        for dr, dc in KING_STEPS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < 8 and 0 <= cc < 8 and g[rr][cc] == king:
                return True
        for dirs, sliders in ((ROOK_DIRS, (by + "r", by + "q")), (BISHOP_DIRS, (by + "b", by + "q"))):
            for dr, dc in dirs:
                rr, cc = r + dr, c + dc
                while 0 <= rr < 8 and 0 <= cc < 8:
                    p = g[rr][cc]
                    if p:
                        if p in sliders:
                            return True
                        break
                    rr += dr
                    cc += dc
        return False

    def in_check(self, color: str) -> bool:
        k = self.kings[color]
        return k is not None and self.is_attacked(k[0], k[1], opponent(color))

    # ---- move generation ----------------------------------------------------
    def pseudo_legal_moves(self, color: str) -> List[Move]:
        """Moves that obey piece movement rules but may leave the king in check."""
        out: List[Move] = []
        for r in range(8):
            row = self.grid[r]
            for c in range(8):
                p = row[c]
                if p and p[0] == color:
                    self._piece_moves(r, c, p, out)
        return out

    def legal_moves(self, color: str, captures_only: bool = False) -> List[Move]:
        """Fully legal moves: pseudo-legal moves that don't leave `color` in check."""
        legal = []
        for m in self.pseudo_legal_moves(color):
            if captures_only and not (m.captured or m.promotion):
                continue
            nb = self.copy()
            nb.apply_move(m)
            if not nb.in_check(color):
                legal.append(m)
        return legal

    def _piece_moves(self, r: int, c: int, piece: str, out: List[Move]) -> None:
        g = self.grid
        color, kind = piece[0], piece[1]
        if kind == "p":
            self._pawn_moves(r, c, piece, out)
        elif kind in "nk":
            for dr, dc in (KNIGHT_STEPS if kind == "n" else KING_STEPS):
                rr, cc = r + dr, c + dc
                if 0 <= rr < 8 and 0 <= cc < 8:
                    t = g[rr][cc]
                    if not t or t[0] != color:
                        out.append(Move((r, c), (rr, cc), piece, t))
            if kind == "k":
                self._castling_moves(r, c, color, out)
        else:
            dirs = ROOK_DIRS if kind == "r" else BISHOP_DIRS if kind == "b" else QUEEN_DIRS
            for dr, dc in dirs:
                rr, cc = r + dr, c + dc
                while 0 <= rr < 8 and 0 <= cc < 8:
                    t = g[rr][cc]
                    if not t:
                        out.append(Move((r, c), (rr, cc), piece))
                    else:
                        if t[0] != color:
                            out.append(Move((r, c), (rr, cc), piece, t))
                        break
                    rr += dr
                    cc += dc

    def _pawn_moves(self, r: int, c: int, piece: str, out: List[Move]) -> None:
        g = self.grid
        color = piece[0]
        d = -1 if color == WHITE else 1
        start_row = 6 if color == WHITE else 1
        last_row = 0 if color == WHITE else 7
        nr = r + d
        if not 0 <= nr < 8:
            return

        def add(end: Square, captured: str = "", ep: bool = False) -> None:
            if end[0] == last_row:  # promotion: one move per choice
                for promo in "qrbn":
                    out.append(Move((r, c), end, piece, captured, promo))
            else:
                out.append(Move((r, c), end, piece, captured, en_passant=ep))

        if not g[nr][c]:
            add((nr, c))
            if r == start_row and not g[r + 2 * d][c]:
                out.append(Move((r, c), (r + 2 * d, c), piece))
        for dc in (-1, 1):
            nc = c + dc
            if 0 <= nc < 8:
                t = g[nr][nc]
                if t and t[0] != color:
                    add((nr, nc), t)
                elif not t and self.ep_target == (nr, nc):
                    add((nr, nc), opponent(color) + "p", ep=True)

    def _castling_moves(self, r: int, c: int, color: str, out: List[Move]) -> None:
        home = 7 if color == WHITE else 0
        if (r, c) != (home, 4):
            return
        g, foe = self.grid, opponent(color)
        k_right, q_right = ("K", "Q") if color == WHITE else ("k", "q")
        king, rook = color + "k", color + "r"
        if (
            k_right in self.castling and g[home][7] == rook
            and not g[home][5] and not g[home][6]
            and not any(self.is_attacked(home, col, foe) for col in (4, 5, 6))
        ):
            out.append(Move((home, 4), (home, 6), king, castle="K"))
        if (
            q_right in self.castling and g[home][0] == rook
            and not g[home][1] and not g[home][2] and not g[home][3]
            and not any(self.is_attacked(home, col, foe) for col in (4, 3, 2))
        ):
            out.append(Move((home, 4), (home, 2), king, castle="Q"))

    # ---- applying a move ----------------------------------------------------
    def apply_move(self, m: Move) -> None:
        """Mutate this board by playing `m` (assumed to be pseudo-legal)."""
        g = self.grid
        (sr, sc), (er, ec) = m.start, m.end
        color = m.piece[0]
        g[sr][sc] = ""
        if m.en_passant:
            g[sr][ec] = ""  # the captured pawn sits beside the moving pawn
        g[er][ec] = color + m.promotion if m.promotion else m.piece
        if m.castle == "K":
            g[er][5], g[er][7] = color + "r", ""
        elif m.castle == "Q":
            g[er][3], g[er][0] = color + "r", ""
        if m.piece[1] == "k":
            self.kings[color] = (er, ec)
        for sq in (m.start, m.end):
            lost = RIGHTS_LOST_ON.get(sq)
            if lost:
                self.castling = "".join(ch for ch in self.castling if ch not in lost)
        if m.piece[1] == "p" and abs(er - sr) == 2:
            self.ep_target = ((sr + er) // 2, sc)
        else:
            self.ep_target = None

    # ---- misc ---------------------------------------------------------------
    def insufficient_material(self) -> bool:
        """K v K, K+minor v K, or only same-coloured bishops: nobody can mate."""
        minors = []
        for r in range(8):
            for c in range(8):
                p = self.grid[r][c]
                if not p or p[1] == "k":
                    continue
                if p[1] in "pqr":
                    return False
                minors.append((p[1], (r + c) % 2))
        if len(minors) <= 1:
            return True
        return all(k == "b" for k, _ in minors) and len({shade for _, shade in minors}) == 1

    def matrix_text(self) -> str:
        """Pretty-print the underlying 2D matrix (empty strings shown as '..')."""
        lines = [f"{8 - r}  [" + " ".join(p or ".." for p in row) + "]" for r, row in enumerate(self.grid)]
        lines.append("     " + "  ".join(FILES))
        return "\n".join(lines)


class HistoryEntry(NamedTuple):
    number: int  # full-move number
    color: str
    move: Move
    san: str  # algebraic notation, e.g. "Nf3", "exd5", "O-O", "e8=Q#"


class Game:
    """Game state on top of a Board: whose turn it is, captured pieces, move
    history (with undo), and detection of check / checkmate / stalemate / draws."""

    TERMINAL = {"checkmate", "stalemate", "insufficient", "fifty_move", "repetition"}

    def __init__(self, fen: Optional[str] = None) -> None:
        self.reset(fen)

    def reset(self, fen: Optional[str] = None) -> None:
        if fen:
            self.board, self.turn, self.halfmove_clock, self.fullmove_number = Board.from_fen(fen)
        else:
            self.board, self.turn = Board.initial(), WHITE
            self.halfmove_clock, self.fullmove_number = 0, 1
        self.captured: Dict[str, List[str]] = {WHITE: [], BLACK: []}  # pieces each side has taken
        self.history: List[HistoryEntry] = []
        self.last_move: Optional[Move] = None
        self.status = "ongoing"
        self.result = "*"
        self._snapshots: list = []
        self._legal_cache: Optional[List[Move]] = None
        self.position_counts: Counter = Counter()
        self.position_counts[self._position_key()] += 1
        self._refresh_status()

    # ---- queries ------------------------------------------------------------
    @property
    def is_over(self) -> bool:
        return self.status in self.TERMINAL

    @property
    def winner(self) -> Optional[str]:
        return opponent(self.turn) if self.status == "checkmate" else None

    @property
    def in_check(self) -> bool:
        return self.board.in_check(self.turn)

    def _all_legal(self) -> List[Move]:
        if self._legal_cache is None:
            self._legal_cache = self.board.legal_moves(self.turn)
        return self._legal_cache

    def legal_moves(self) -> List[Move]:
        return [] if self.is_over else self._all_legal()

    def legal_moves_from(self, square: Square) -> List[Move]:
        return [m for m in self.legal_moves() if m.start == square]

    def move_from_uci(self, uci: str) -> Optional[Move]:
        """'e2e4' / 'e7e8q' -> the matching legal move, or None."""
        for m in self.legal_moves():
            if m.uci == uci:
                return m
        return None

    def material_balance(self) -> int:
        """White's material minus Black's, in pawns (from the pieces on the board)."""
        total = 0
        for row in self.board.grid:
            for p in row:
                if p and p[1] != "k":
                    pts = Piece.VALUES[p[1]] // 100
                    total += pts if p[0] == WHITE else -pts
        return total

    # ---- playing moves ------------------------------------------------------
    def make_move(self, move: Move) -> str:
        """Play a legal move; returns its algebraic notation."""
        legal = self.legal_moves()
        if move not in legal:
            raise ValueError(f"Illegal move: {move}")
        san = self._san(move, legal)
        self._snapshots.append((
            self.board.copy(), self.turn, self.halfmove_clock, self.fullmove_number,
            {k: list(v) for k, v in self.captured.items()}, Counter(self.position_counts), self.last_move,
        ))
        mover, number = self.turn, self.fullmove_number
        if move.captured:
            self.captured[mover].append(move.captured)
        if move.piece[1] == "p" or move.captured:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1
        self.board.apply_move(move)
        if mover == BLACK:
            self.fullmove_number += 1
        self.turn = opponent(mover)
        self.last_move = move
        self._legal_cache = None
        self.position_counts[self._position_key()] += 1
        self._refresh_status()
        if self.status == "checkmate":
            san += "#"
        elif self.in_check:
            san += "+"
        self.history.append(HistoryEntry(number, mover, move, san))
        return san

    def undo(self) -> bool:
        """Take back the last move. Returns False if there is nothing to undo."""
        if not self.history:
            return False
        (self.board, self.turn, self.halfmove_clock, self.fullmove_number,
         self.captured, self.position_counts, self.last_move) = self._snapshots.pop()
        self.history.pop()
        self._legal_cache = None
        self._refresh_status()
        return True

    # ---- status, notation ---------------------------------------------------
    def _position_key(self):
        ep = self.board.ep_target
        # an en-passant square only matters if an en-passant capture is possible
        ep_ok = ep is not None and any(m.en_passant for m in self._all_legal())
        return (tuple(tuple(row) for row in self.board.grid), self.turn, self.board.castling, ep if ep_ok else None)

    def _refresh_status(self) -> None:
        legal = self._all_legal()
        in_check = self.board.in_check(self.turn)
        self.result = "*"
        if not legal:
            if in_check:
                self.status = "checkmate"
                self.result = "0-1" if self.turn == WHITE else "1-0"
            else:
                self.status, self.result = "stalemate", "1/2-1/2"
        elif self.board.insufficient_material():
            self.status, self.result = "insufficient", "1/2-1/2"
        elif self.halfmove_clock >= 100:
            self.status, self.result = "fifty_move", "1/2-1/2"
        elif self.position_counts[self._position_key()] >= 3:
            self.status, self.result = "repetition", "1/2-1/2"
        else:
            self.status = "check" if in_check else "ongoing"

    @staticmethod
    def _san(move: Move, legal: List[Move]) -> str:
        if move.castle:
            return "O-O" if move.castle == "K" else "O-O-O"
        kind, dest = move.piece[1], square_name(*move.end)
        if kind == "p":
            text = (FILES[move.start[1]] + "x" if move.captured else "") + dest
            return text + ("=" + move.promotion.upper() if move.promotion else "")
        text = kind.upper()
        rivals = [m for m in legal if m.piece == move.piece and m.end == move.end and m.start != move.start]
        if rivals:  # disambiguate: file first, then rank, then both
            if all(m.start[1] != move.start[1] for m in rivals):
                text += FILES[move.start[1]]
            elif all(m.start[0] != move.start[0] for m in rivals):
                text += str(8 - move.start[0])
            else:
                text += square_name(*move.start)
        return text + ("x" if move.captured else "") + dest

    def pgn(self) -> str:
        parts = []
        for i, e in enumerate(self.history):
            if e.color == WHITE:
                parts.append(f"{e.number}. {e.san}")
            elif i == 0 or self.history[i - 1].color != WHITE:
                parts.append(f"{e.number}... {e.san}")
            else:
                parts.append(e.san)
        parts.append(self.result)
        return " ".join(parts)


# ---- a small AI opponent ----------------------------------------------------
# Piece-square tables (White's point of view, row 0 = rank 8).
_PST = {
    "p": [[0, 0, 0, 0, 0, 0, 0, 0], [50, 50, 50, 50, 50, 50, 50, 50], [10, 10, 20, 30, 30, 20, 10, 10],
          [5, 5, 10, 25, 25, 10, 5, 5], [0, 0, 0, 20, 20, 0, 0, 0], [5, -5, -10, 0, 0, -10, -5, 5],
          [5, 10, 10, -20, -20, 10, 10, 5], [0, 0, 0, 0, 0, 0, 0, 0]],
    "n": [[-50, -40, -30, -30, -30, -30, -40, -50], [-40, -20, 0, 0, 0, 0, -20, -40],
          [-30, 0, 10, 15, 15, 10, 0, -30], [-30, 5, 15, 20, 20, 15, 5, -30],
          [-30, 0, 15, 20, 20, 15, 0, -30], [-30, 5, 10, 15, 15, 10, 5, -30],
          [-40, -20, 0, 5, 5, 0, -20, -40], [-50, -40, -30, -30, -30, -30, -40, -50]],
    "b": [[-20, -10, -10, -10, -10, -10, -10, -20], [-10, 0, 0, 0, 0, 0, 0, -10],
          [-10, 0, 5, 10, 10, 5, 0, -10], [-10, 5, 5, 10, 10, 5, 5, -10],
          [-10, 0, 10, 10, 10, 10, 0, -10], [-10, 10, 10, 10, 10, 10, 10, -10],
          [-10, 5, 0, 0, 0, 0, 5, -10], [-20, -10, -10, -10, -10, -10, -10, -20]],
    "r": [[0, 0, 0, 0, 0, 0, 0, 0], [5, 10, 10, 10, 10, 10, 10, 5], [-5, 0, 0, 0, 0, 0, 0, -5],
          [-5, 0, 0, 0, 0, 0, 0, -5], [-5, 0, 0, 0, 0, 0, 0, -5], [-5, 0, 0, 0, 0, 0, 0, -5],
          [-5, 0, 0, 0, 0, 0, 0, -5], [0, 0, 0, 5, 5, 0, 0, 0]],
    "q": [[-20, -10, -10, -5, -5, -10, -10, -20], [-10, 0, 0, 0, 0, 0, 0, -10],
          [-10, 0, 5, 5, 5, 5, 0, -10], [-5, 0, 5, 5, 5, 5, 0, -5], [0, 0, 5, 5, 5, 5, 0, -5],
          [-10, 5, 5, 5, 5, 5, 0, -10], [-10, 0, 5, 0, 0, 0, 0, -10],
          [-20, -10, -10, -5, -5, -10, -10, -20]],
}
_KING_MIDDLE = [[-30, -40, -40, -50, -50, -40, -40, -30]] * 4 + [
    [-20, -30, -30, -40, -40, -30, -30, -20], [-10, -20, -20, -20, -20, -20, -20, -10],
    [20, 20, 0, 0, 0, 0, 20, 20], [20, 30, 10, 0, 0, 10, 30, 20]]
_KING_END = [[-50, -40, -30, -20, -20, -30, -40, -50], [-30, -20, -10, 0, 0, -10, -20, -30],
             [-30, -10, 20, 30, 30, 20, -10, -30], [-30, -10, 30, 40, 40, 30, -10, -30],
             [-30, -10, 30, 40, 40, 30, -10, -30], [-30, -10, 20, 30, 30, 20, -10, -30],
             [-30, -30, 0, 0, 0, 0, -30, -30], [-50, -30, -30, -30, -30, -30, -30, -50]]


class ChessAI:
    """Negamax with alpha-beta pruning, a capture-only quiescence search and a
    material + piece-square evaluation. Level 0 plays random legal moves."""

    # level = how many half-moves (plies) the engine searches
    LEVELS = {"Beginner (random moves)": 0, "Easy": 1, "Medium": 2, "Hard": 3,
              "Expert (can take a few seconds)": 4}
    MATE, INF, QDEPTH = 100_000, 10 ** 9, 4

    @staticmethod
    def evaluate(board: Board) -> int:
        """Static score in centipawns from White's point of view."""
        pieces, material = [], 0
        for r in range(8):
            for c in range(8):
                p = board.grid[r][c]
                if p:
                    pieces.append((r, c, p))
                    if p[1] not in "pk":
                        material += Piece.VALUES[p[1]]
        endgame = material <= 2600
        score = 0
        for r, c, p in pieces:
            kind = p[1]
            table = _PST[kind] if kind != "k" else (_KING_END if endgame else _KING_MIDDLE)
            if p[0] == WHITE:
                score += Piece.VALUES[kind] + table[r][c]
            else:
                score -= Piece.VALUES[kind] + table[7 - r][c]
        return score

    def _score(self, board: Board, color: str) -> int:
        s = self.evaluate(board)
        return s if color == WHITE else -s

    @staticmethod
    def _order_key(m: Move) -> int:
        """Try captures (most valuable victim, cheapest attacker) and promotions first."""
        s = 0
        if m.captured:
            s += 10 * Piece.VALUES[m.captured[1]] - Piece.VALUES[m.piece[1]]
        if m.promotion:
            s += Piece.VALUES[m.promotion]
        return s

    def _quiesce(self, board: Board, color: str, alpha: int, beta: int, depth: int) -> int:
        stand_pat = self._score(board, color)
        if depth == 0 or stand_pat >= beta:
            return stand_pat
        alpha = max(alpha, stand_pat)
        for m in sorted(board.legal_moves(color, captures_only=True), key=self._order_key, reverse=True):
            nb = board.copy()
            nb.apply_move(m)
            score = -self._quiesce(nb, opponent(color), -beta, -alpha, depth - 1)
            if score >= beta:
                return score
            alpha = max(alpha, score)
        return alpha

    def _search(self, board: Board, color: str, depth: int, alpha: int, beta: int, ply: int) -> int:
        if depth == 0:
            return self._quiesce(board, color, alpha, beta, self.QDEPTH)
        moves = board.legal_moves(color)
        if not moves:
            return -self.MATE + ply if board.in_check(color) else 0
        best = -self.INF
        for m in sorted(moves, key=self._order_key, reverse=True):
            nb = board.copy()
            nb.apply_move(m)
            score = -self._search(nb, opponent(color), depth - 1, -beta, -alpha, ply + 1)
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best

    def choose_move(self, game: Game, level: int) -> Optional[Move]:
        moves = game.legal_moves()
        if not moves:
            return None
        if level <= 0:
            return random.choice(moves)
        color = game.turn
        best, best_moves = -self.INF, []
        for m in sorted(moves, key=self._order_key, reverse=True):
            nb = game.board.copy()
            nb.apply_move(m)
            # window (best-1, inf) keeps scores *equal* to the best exact, so ties
            # can be broken at random and the AI doesn't repeat itself.
            alpha = -self.INF if best == -self.INF else best - 1
            score = -self._search(nb, opponent(color), level - 1, -self.INF, -alpha, 1)
            if score > best:
                best, best_moves = score, [m]
            elif score == best:
                best_moves.append(m)
        return random.choice(best_moves)


# =============================================================================
# 2. FRONTEND (Streamlit)
# =============================================================================

MODE_2P = "Two players (same screen)"
MODE_AI = "Play against the computer"
ORIENTATIONS = ["Auto", "White at bottom", "Black at bottom", "Rotate every turn"]
COLOR_NAME = {WHITE: "White", BLACK: "Black"}
BLANK = "\u00a0"  # label for empty squares (st.button needs a non-empty label)

LIGHT_SQ, DARK_SQ = "#f0d9b5", "#b58863"
LAST_LIGHT, LAST_DARK = "#cdd26a", "#aaa23a"
SELECTED_SQ = "#f7ec5d"

BASE_CSS = """
.st-key-css_holder{display:none !important;}
.st-key-game_area{max-width:560px;margin:0 auto;}
.st-key-board{gap:0 !important;container-type:inline-size;}
.st-key-board [data-testid="stHorizontalBlock"]{gap:0 !important;flex-wrap:nowrap !important;}
.st-key-board [data-testid="stColumn"],.st-key-board [data-testid="column"]{flex:1 1 0 !important;min-width:0 !important;width:auto !important;}
.st-key-board [data-testid="stElementContainer"],.st-key-board .element-container,.st-key-board [data-testid="stButton"]{width:100% !important;}
.st-key-board button{position:relative;width:100% !important;height:auto !important;min-height:0 !important;aspect-ratio:1/1;padding:0 !important;border:none !important;border-radius:0 !important;box-shadow:none !important;outline:none !important;display:flex;align-items:center;justify-content:center;overflow:hidden;cursor:pointer;user-select:none;}
.st-key-board button:hover{filter:brightness(1.08);}
.st-key-board button *{color:inherit !important;text-shadow:inherit !important;margin:0 !important;padding:0 !important;line-height:1 !important;font-family:"Noto Sans Symbols 2","Segoe UI Symbol","Apple Symbols","DejaVu Sans",sans-serif !important;font-size:clamp(1.4rem,6vw,3rem) !important;font-size:8.4cqw !important;}
.st-key-board button::before,.st-key-board button::after{position:absolute;font-size:.68rem;font-weight:700;line-height:1;text-shadow:none !important;pointer-events:none;font-family:sans-serif;}
.st-key-board button::before{top:3px;left:4px;}
.st-key-board button::after{bottom:3px;right:4px;}
"""


def css_rule(selectors: List[str], declarations: str) -> str:
    return f"{','.join(selectors)}{{{declarations}}}" if selectors else ""


def sq_sel(r: int, c: int) -> str:
    return f".st-key-sq_{r}_{c} button"


def build_board_css(game: Game, selected: Optional[Square], targets: Dict[Square, bool], flipped: bool) -> str:
    """Per-square CSS: checkerboard colours, piece colours, highlights, coordinates."""
    grid = game.board.grid
    light, dark, white_pieces, black_pieces = [], [], [], []
    for r in range(8):
        for c in range(8):
            s = sq_sel(r, c)
            (light if (r + c) % 2 == 0 else dark).append(s)
            p = grid[r][c]
            if p:
                (white_pieces if p[0] == WHITE else black_pieces).append(s)
    rules = [
        css_rule(light, f"background-color:{LIGHT_SQ} !important"),
        css_rule(dark, f"background-color:{DARK_SQ} !important"),
        css_rule(white_pieces, "color:#fff !important;text-shadow:0 0 2px #000,0 0 3px #000,0 1px 4px #000 !important"),
        css_rule(black_pieces, "color:#151515 !important;text-shadow:0 0 1px rgba(255,255,255,.5) !important"),
    ]
    if game.last_move:
        for r, c in (game.last_move.start, game.last_move.end):
            colour = LAST_LIGHT if (r + c) % 2 == 0 else LAST_DARK
            rules.append(css_rule([sq_sel(r, c)], f"background-color:{colour} !important"))
    if selected:
        rules.append(css_rule([sq_sel(*selected)], f"background-color:{SELECTED_SQ} !important"))
    if game.in_check and game.board.kings[game.turn]:
        kr, kc = game.board.kings[game.turn]
        rules.append(css_rule([sq_sel(kr, kc)], "background-image:radial-gradient(circle,rgba(255,0,0,.95) 0%,"
                                                 "rgba(231,0,0,.8) 25%,rgba(169,0,0,0) 89%) !important"))
    for (r, c), is_capture in targets.items():
        if is_capture:
            img = "radial-gradient(circle,rgba(0,0,0,0) 56%,rgba(214,48,49,.75) 58%)"
        else:
            img = "radial-gradient(circle,rgba(20,90,40,.55) 0%,rgba(20,90,40,.55) 20%,rgba(0,0,0,0) 23%)"
        rules.append(css_rule([sq_sel(r, c)], f"background-image:{img} !important"))
    # rank numbers on the left edge, file letters along the bottom edge
    left_col, bottom_row = (7, 0) if flipped else (0, 7)
    for r in range(8):
        for c in range(8):
            label = DARK_SQ if (r + c) % 2 == 0 else LIGHT_SQ
            if c == left_col:
                rules.append(css_rule([sq_sel(r, c) + "::before"], f'content:"{8 - r}";color:{label}'))
            if r == bottom_row:
                rules.append(css_rule([sq_sel(r, c) + "::after"], f'content:"{FILES[c]}";color:{label}'))
    return BASE_CSS + "".join(rules)


# ---- session state & callbacks ----------------------------------------------
def init_state() -> None:
    ss = st.session_state
    if "game" not in ss:
        ss.game = Game()
        ss.ai = ChessAI()
        ss.selected = None  # (row, col) of the selected piece
        ss.pending_promotion = None  # (from_square, to_square) waiting for a piece choice


def ai_color() -> Optional[str]:
    ss = st.session_state
    if ss.get("mode") != MODE_AI:
        return None
    return BLACK if ss.get("human_color", "White") == "White" else WHITE


def is_ai_turn() -> bool:
    ac = ai_color()
    game = st.session_state.game
    return ac is not None and game.turn == ac and not game.is_over


def board_flipped() -> bool:
    ss = st.session_state
    choice = ss.get("orientation", "Auto")
    if choice == "White at bottom":
        return False
    if choice == "Black at bottom":
        return True
    if choice == "Rotate every turn" and ss.get("mode") == MODE_2P:
        return ss.game.turn == BLACK
    return ss.get("mode") == MODE_AI and ss.get("human_color") == "Black"  # "Auto"


def on_square_click(r: int, c: int) -> None:
    ss = st.session_state
    game = ss.game
    ss.pending_promotion = None
    if game.is_over or is_ai_turn():
        return
    sel = ss.selected
    if sel is not None:
        candidates = [m for m in game.legal_moves_from(sel) if m.end == (r, c)]
        if candidates:
            if len(candidates) > 1:  # several promotion choices: ask which piece
                ss.pending_promotion = (sel, (r, c))
            else:
                game.make_move(candidates[0])
                ss.selected = None
            return
    piece = game.board.grid[r][c]
    ss.selected = (r, c) if piece and piece[0] == game.turn and sel != (r, c) else None


def choose_promotion(kind: str) -> None:
    ss = st.session_state
    start, end = ss.pending_promotion
    move = next(m for m in ss.game.legal_moves_from(start) if m.end == end and m.promotion == kind)
    ss.game.make_move(move)
    ss.selected = ss.pending_promotion = None


def cancel_promotion() -> None:
    st.session_state.selected = st.session_state.pending_promotion = None


def new_game() -> None:
    ss = st.session_state
    ss.game.reset()
    ss.selected = ss.pending_promotion = None


def can_undo() -> bool:
    ss = st.session_state
    minimum = 1 if ai_color() == WHITE else 0  # keep the computer's opening move
    return len(ss.game.history) > minimum


def undo_move() -> None:
    ss = st.session_state
    game = ss.game
    game.undo()
    ac = ai_color()
    if ac is not None and game.turn == ac and game.history:
        game.undo()  # also take back the computer's reply so it's the human's turn
    ss.selected = ss.pending_promotion = None


# ---- rendering ---------------------------------------------------------------
def piece_span(code: str, size: str = "1.6rem") -> str:
    p = Piece(code)
    style = ("color:#fff;text-shadow:0 0 2px #000,0 0 3px #000;" if p.is_white
             else "color:#151515;text-shadow:0 0 2px #fff,0 0 3px #fff;")
    return f'<span style="font-size:{size};line-height:1.1;{style}">{p.symbol}</span>'


def render_settings() -> None:
    sb = st.sidebar
    sb.title("♟️ Chess")
    sb.radio("Game mode", [MODE_2P, MODE_AI], key="mode")
    vs_ai = st.session_state.mode == MODE_AI
    sb.radio("You play as", ["White", "Black"], key="human_color", horizontal=True, disabled=not vs_ai)
    sb.selectbox("Computer level", list(ChessAI.LEVELS), index=2, key="level", disabled=not vs_ai)
    sb.selectbox("Board orientation", ORIENTATIONS, key="orientation")
    sb.button("🔄 New game", key="new_game_sidebar", on_click=new_game)


def render_status(game: Game) -> None:
    ac = ai_color()
    turn = COLOR_NAME[game.turn]
    if game.status == "checkmate":
        winner = opponent(game.turn)
        message = f"🏁 Checkmate. {COLOR_NAME[winner]} wins."
        (st.error if ac is not None and game.turn != ac else st.success)(message)  # red only if the human lost
    elif game.status == "stalemate":
        st.info("🤝 Stalemate. The game is a draw.")
    elif game.status == "insufficient":
        st.info("🤝 Draw: neither side has enough pieces to checkmate.")
    elif game.status == "fifty_move":
        st.info("🤝 Draw by the 50-move rule.")
    elif game.status == "repetition":
        st.info("🤝 Draw by threefold repetition.")
    else:
        who = "" if ac is None else (" (computer)" if game.turn == ac else " (you)")
        prefix = "⚠️ Check! " if game.status == "check" else ""
        (st.warning if game.status == "check" else st.info)(f"{prefix}{turn} to move{who}")


def render_promotion_picker(game: Game) -> None:
    if not st.session_state.pending_promotion:
        return
    st.markdown("**Promote your pawn to:**")
    cols = st.columns(5)
    for col, (kind, name) in zip(cols, [("q", "Queen"), ("r", "Rook"), ("b", "Bishop"), ("n", "Knight")]):
        with col:
            st.button(f"{Piece(game.turn + kind).symbol} {name}", key=f"promo_{kind}",
                      on_click=choose_promotion, args=(kind,))
    with cols[4]:
        st.button("Cancel", key="promo_cancel", on_click=cancel_promotion)


def render_board(game: Game, flipped: bool) -> None:
    order = range(7, -1, -1) if flipped else range(8)
    with st.container(key="board"):
        for r in order:
            cols = st.columns(8)
            for col, c in zip(cols, order):
                piece = game.board.grid[r][c]
                label = Piece(piece).symbol if piece else BLANK
                with col:
                    st.button(label, key=f"sq_{r}_{c}", on_click=on_square_click, args=(r, c))


def render_captured(game: Game) -> None:
    balance = game.material_balance()
    for color in (WHITE, BLACK):
        taken = sorted(game.captured[color], key=lambda code: -Piece(code).value)
        lead = balance if color == WHITE else -balance
        pieces = "".join(piece_span(code, "1.4rem") for code in taken) or '<span style="opacity:.6">none yet</span>'
        lead_txt = f' <b style="font-size:.9rem;margin-left:6px">+{lead}</b>' if lead > 0 else ""
        st.sidebar.markdown(
            f'<div style="margin-bottom:6px"><b>{COLOR_NAME[color]} has captured</b>{lead_txt}<br>{pieces}</div>',
            unsafe_allow_html=True,
        )


def history_html(game: Game) -> str:
    if not game.history:
        return '<div style="opacity:.65">No moves yet.</div>'
    rows: Dict[int, List[str]] = {}
    for e in game.history:
        rows.setdefault(e.number, ["", ""])[0 if e.color == WHITE else 1] = e.san
    last = game.history[-1]

    def cell(text: str, active: bool) -> str:
        style = "font-weight:700;background:rgba(247,236,93,.35);" if active else ""
        return f'<td style="padding:2px 8px;{style}">{text}</td>'

    body = "".join(
        f'<tr><td style="opacity:.6;padding:2px 6px">{n}.</td>'
        f'{cell(w, last.number == n and last.color == WHITE)}{cell(b, last.number == n and last.color == BLACK)}</tr>'
        for n, (w, b) in sorted(rows.items())
    )
    # column-reverse keeps the scroll position pinned to the newest move
    return ('<div style="max-height:280px;overflow-y:auto;display:flex;flex-direction:column-reverse">'
            '<table style="width:100%;border-collapse:collapse;font-family:monospace;font-size:.95rem">'
            f"{body}</table></div>")


def render_sidebar_info(game: Game) -> None:
    sb = st.sidebar
    sb.divider()
    render_captured(game)
    sb.divider()
    sb.subheader("Move history")
    sb.markdown(history_html(game), unsafe_allow_html=True)
    with sb.expander("PGN"):
        st.code(game.pgn(), language=None)
        st.download_button("Download PGN", data=game.pgn(), file_name="game.pgn", key="download_pgn")
    with sb.expander("Board matrix (backend state)"):
        st.code(game.board.matrix_text(), language=None)
        st.caption('Each cell holds a piece code such as "wp" or "bk"; empty squares are "" (shown as "..").')


def check_streamlit_version() -> None:
    try:
        major, minor = (int(x) for x in st.__version__.split(".")[:2])
        if (major, minor) < (1, 39):
            st.warning("This app styles the board using features from Streamlit 1.39+. "
                       "Run `pip install -U streamlit` for the full look.")
    except ValueError:
        pass


def main() -> None:
    st.set_page_config(page_title="Chess", page_icon="♟️", layout="centered", initial_sidebar_state="auto")
    check_streamlit_version()
    init_state()
    ss = st.session_state
    render_settings()

    game: Game = ss.game
    # drop a stale selection (e.g. after switching sides or modes)
    if ss.selected is not None:
        piece = game.board.grid[ss.selected[0]][ss.selected[1]]
        if game.is_over or not piece or piece[0] != game.turn:
            ss.selected = None
    moves = game.legal_moves_from(ss.selected) if ss.selected else []
    targets = {m.end: bool(m.captured) for m in moves}
    flipped = board_flipped()

    with st.container(key="css_holder"):
        st.markdown(f"<style>{build_board_css(game, ss.selected, targets, flipped)}</style>", unsafe_allow_html=True)

    with st.container(key="game_area"):
        render_status(game)
        render_promotion_picker(game)
        render_board(game, flipped)

        if ss.selected and not ss.pending_promotion:
            st.caption(f"Selected {square_name(*ss.selected)}. Green dots show where it can move; "
                       f"click it again to deselect.")
        elif not game.is_over and not is_ai_turn():
            st.caption("Click one of your pieces, then click where it should go.")

        left, right = st.columns(2)
        with left:
            st.button("🔄 New game", key="new_game", on_click=new_game)
        with right:
            st.button("↩️ Undo move", key="undo", on_click=undo_move, disabled=not can_undo())

    render_sidebar_info(game)

    # Computer's turn: runs after the board is drawn so the human's move is already visible.
    if is_ai_turn():
        with st.spinner("Computer is thinking…"):
            move = ss.ai.choose_move(game, ChessAI.LEVELS[ss.level])
            if move:
                game.make_move(move)
        st.rerun()


if __name__ == "__main__":
    main()