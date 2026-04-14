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

/*
 * Rotate a piece matrix 90 degrees clockwise.
 * in: h*w, out: w*h (dimensions swap)
 */
void rotate_cw(const uint8_t *in, uint8_t *out, int h, int w) {
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            out[c * h + (h - 1 - r)] = in[r * w + c];
        }
    }
}

/*
 * Enumerate ALL valid placements across all 4 rotations.
 *
 * board: 20*10 int8 playfield
 * piece: h*w uint8 piece matrix (rotation 0)
 * h, w: piece dimensions
 * boards_out: buffer for resulting boards (max_out * 200 bytes)
 * meta_out: buffer for (rotation, x, y, lines_cleared) per placement (max_out * 4 ints)
 * max_out: max placements
 *
 * Returns number of placements found.
 */
int enumerate_all(
    const int8_t *board,
    const uint8_t *piece_r0,
    int h0, int w0,
    int8_t *boards_out,
    int *meta_out,
    int max_out
) {
    int count = 0;
    uint8_t pieces[4][16];  /* max 4x4 piece */
    int hs[4], ws[4];

    /* Generate all 4 rotations */
    memcpy(pieces[0], piece_r0, h0 * w0);
    hs[0] = h0; ws[0] = w0;

    for (int rot = 1; rot < 4; rot++) {
        rotate_cw(pieces[rot - 1], pieces[rot], hs[rot - 1], ws[rot - 1]);
        hs[rot] = ws[rot - 1];
        ws[rot] = hs[rot - 1];
    }

    for (int rot = 0; rot < 4; rot++) {
        int h = hs[rot], w = ws[rot];
        uint8_t *piece = pieces[rot];

        for (int x = -(w - 1); x < COLS; x++) {
            if (check_collision(board, piece, h, w, x, 0)) continue;

            int y = find_drop_y(board, piece, h, w, x);

            int8_t *out = &boards_out[count * ROWS * COLS];
            int lines = simulate_placement(board, out, piece, h, w, x, y);

            meta_out[count * 4 + 0] = rot;
            meta_out[count * 4 + 1] = x;
            meta_out[count * 4 + 2] = y;
            meta_out[count * 4 + 3] = lines;
            count++;

            if (count >= max_out) return count;
        }
    }

    return count;
}
