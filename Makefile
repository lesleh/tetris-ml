CC = gcc
CFLAGS = -O3 -shared

SO_DIR = tetris
SOURCES = $(SO_DIR)/board_sim_c.c
TARGETS = $(SOURCES:.c=.so)

all: $(TARGETS)

$(SO_DIR)/%.so: $(SO_DIR)/%.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean
