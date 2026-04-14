"""Fast board simulation via C extension."""

import ctypes
from pathlib import Path

import numpy as np

_lib = None
_so_path = Path(__file__).parent / "board_sim_c.so"
if _so_path.exists():
    try:
        _lib = ctypes.CDLL(str(_so_path))
        _lib.simulate_placement.restype = ctypes.c_int
        _lib.simulate_placement.argtypes = [
            ctypes.POINTER(ctypes.c_int8),   # board_in
            ctypes.POINTER(ctypes.c_int8),   # board_out
            ctypes.POINTER(ctypes.c_uint8),  # piece
            ctypes.c_int, ctypes.c_int,      # h, w
            ctypes.c_int, ctypes.c_int,      # px, py
        ]
        _lib.check_collision.restype = ctypes.c_int
        _lib.check_collision.argtypes = [
            ctypes.POINTER(ctypes.c_int8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
        ]
        _lib.find_drop_y.restype = ctypes.c_int
        _lib.find_drop_y.argtypes = [
            ctypes.POINTER(ctypes.c_int8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int,
        ]
        _lib.enumerate_all.restype = ctypes.c_int
        _lib.enumerate_all.argtypes = [
            ctypes.POINTER(ctypes.c_int8),   # board
            ctypes.POINTER(ctypes.c_uint8),  # piece
            ctypes.c_int, ctypes.c_int,      # h, w
            ctypes.POINTER(ctypes.c_int8),   # boards_out
            ctypes.POINTER(ctypes.c_int),    # meta_out
            ctypes.c_int,                    # max_out
        ]
    except OSError:
        _lib = None


def is_available() -> bool:
    return _lib is not None


def simulate(board: np.ndarray, piece_matrix: np.ndarray, px: int, py: int) -> tuple[np.ndarray, int]:
    """Simulate placement. Returns (resulting_board, lines_cleared).

    board: (20, 10) int8
    piece_matrix: (h, w) uint8
    px, py: piece position
    """
    board_flat = np.ascontiguousarray(board.flatten(), dtype=np.int8)
    piece_flat = np.ascontiguousarray(piece_matrix.flatten(), dtype=np.uint8)
    result = np.zeros(200, dtype=np.int8)

    h, w = piece_matrix.shape
    lines = _lib.simulate_placement(
        board_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        result.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        piece_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        h, w, px, py,
    )
    return result.reshape(20, 10), lines


def collision(board: np.ndarray, piece_matrix: np.ndarray, px: int, py: int) -> bool:
    """Check if piece collides at position."""
    board_flat = np.ascontiguousarray(board.flatten(), dtype=np.int8)
    piece_flat = np.ascontiguousarray(piece_matrix.flatten(), dtype=np.uint8)
    h, w = piece_matrix.shape
    return bool(_lib.check_collision(
        board_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        piece_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        h, w, px, py,
    ))


def drop_y(board: np.ndarray, piece_matrix: np.ndarray, px: int) -> int:
    """Find lowest valid y for piece at column px."""
    board_flat = np.ascontiguousarray(board.flatten(), dtype=np.int8)
    piece_flat = np.ascontiguousarray(piece_matrix.flatten(), dtype=np.uint8)
    h, w = piece_matrix.shape
    return _lib.find_drop_y(
        board_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        piece_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        h, w, px,
    )


MAX_PLACEMENTS = 80  # more than enough for any piece


def enumerate_all(board: np.ndarray, piece_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Enumerate all valid placements across 4 rotations.

    board: (20, 10) int8 playfield
    piece_matrix: (h, w) uint8 rotation-0 piece

    Returns (boards, meta, count):
        boards: (count, 20, 10) int8 resulting boards
        meta: (count, 4) int32 — [rotation, x, y, lines_cleared]
        count: number of valid placements
    """
    board_flat = np.ascontiguousarray(board.flatten(), dtype=np.int8)
    piece_flat = np.ascontiguousarray(piece_matrix.flatten(), dtype=np.uint8)
    h, w = piece_matrix.shape

    boards_buf = np.zeros(MAX_PLACEMENTS * 200, dtype=np.int8)
    meta_buf = np.zeros(MAX_PLACEMENTS * 4, dtype=np.int32)

    count = _lib.enumerate_all(
        board_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        piece_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        h, w,
        boards_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        meta_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        MAX_PLACEMENTS,
    )

    boards = boards_buf[:count * 200].reshape(count, 20, 10)
    meta = meta_buf[:count * 4].reshape(count, 4)
    return boards, meta, count
