# Makefile

CPPFLAGS = -MMD
CC = gcc -fsanitize=address
CFLAGS = -Wall -Wextra -Wformat=2 -pedantic -std=c99 -Og -g3 $(shell pkg-config --cflags ANN)
LDFLAGS = $(shell pkg-config --libs ANN)
LDLIBS = -lm -lANN

SRC_DIR=src/
SRC = $(shell find ${SRC_DIR} -type f -name "*.c")
OBJ = ${SRC:.c=.o}
DEP = ${SRC:.c=.d}

all: main

main: ${OBJ} 
	${CC} ${CFLAGS} -o main ${OBJ} ${LDFLAGS} ${LDLIBS}

.PHONY: clean

clean:
	${RM} ${OBJ}   # remove object files
	${RM} ${DEP}   # remove dependency files
	${RM} main     # remove main program

-include ${DEP}

# END

