/*
 * Fast board simulation for Tetris placement enumeration.
 * Called from Python via ctypes.
 *
 * Board is a flat int8 array of 20*10, row-major.
 * Piece matrix is a flat uint8 array of h*w.
 */

#include <string.h>
#include <stdint.h>

#define ROWS 20
#define COLS 10

/*
 * Simulate placing a piece on the board, clear lines, return result.
 *
 * board_in:  20*10 int8 array (0=empty, >0=filled)
 * board_out: 20*10 int8 array (result after placement + line clears)
 * piece:     h*w uint8 array (0=empty, >0=filled)
 * h, w:      piece dimensions
 * px, py:    piece position on board
 *
 * Returns number of lines cleared.
 */
int simulate_placement(
    const int8_t *board_in,
    int8_t *board_out,
    const uint8_t *piece,
    int h, int w,
    int px, int py
) {
    /* Copy board */
    memcpy(board_out, board_in, ROWS * COLS * sizeof(int8_t));

    /* Place piece */
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            if (piece[r * w + c] != 0) {
                int br = py + r;
                int bc = px + c;
                if (br >= 0 && br < ROWS && bc >= 0 && bc < COLS) {
                    board_out[br * COLS + bc] = 1;
                }
            }
        }
    }

    /* Clear lines */
    int lines = 0;
    int write_row = ROWS - 1;

    for (int r = ROWS - 1; r >= 0; r--) {
        int full = 1;
        for (int c = 0; c < COLS; c++) {
            if (board_out[r * COLS + c] == 0) {
                full = 0;
                break;
            }
        }
        if (full) {
            lines++;
        } else {
            if (write_row != r) {
                memcpy(&board_out[write_row * COLS], &board_out[r * COLS], COLS * sizeof(int8_t));
            }
            write_row--;
        }
    }

    /* Clear top rows */
    for (int r = 0; r <= write_row; r++) {
        memset(&board_out[r * COLS], 0, COLS * sizeof(int8_t));
    }

    return lines;
}

/*
 * Check collision: returns 1 if piece collides at (px, py), 0 otherwise.
 */
int check_collision(
    const int8_t *board,
    const uint8_t *piece,
    int h, int w,
    int px, int py
) {
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            if (piece[r * w + c] == 0) continue;
            int br = py + r;
            int bc = px + c;
            if (br < 0 || br >= ROWS || bc < 0 || bc >= COLS) return 1;
            if (board[br * COLS + bc] != 0) return 1;
        }
    }
    return 0;
}

/*
 * Find drop position: returns the lowest y where piece doesn't collide.
 */
int find_drop_y(
    const int8_t *board,
    const uint8_t *piece,
    int h, int w,
    int px
) {
    int y = 0;
    while (!check_collision(board, piece, h, w, px, y + 1)) {
        y++;
    }
    return y;
}
