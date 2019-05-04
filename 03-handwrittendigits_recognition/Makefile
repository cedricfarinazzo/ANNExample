# Makefile

CPPFLAGS = -MMD
CC = gcc -fsanitize=address
CFLAGS = -Wall -Wextra -Wformat=2 -pedantic -std=c99 -Og -g3 $(shell pkg-config --cflags ANN) $(shell pkg-config --cflags SDL_image) 
LDFLAGS = 
LDLIBS = -lm -lpthread $(shell pkg-config --libs ANN) $(shell pkg-config --libs SDL_image)

SRC_DIR=src/
SRC = $(shell find ${SRC_DIR} -type f -name "*.c")
OBJ = ${SRC:.c=.o}
DEP = ${SRC:.c=.d}

all: main

train: main
	@./main train ./dataset/training ./dataset/testing

check: main
	@./main check ./dataset/testing

main: ${OBJ} 
	${CC} ${CFLAGS} -o main ${OBJ} ${LDFLAGS} ${LDLIBS}

.PHONY: clean run

clean:
	${RM} ${OBJ}   # remove object files
	${RM} ${DEP}   # remove dependency files
	${RM} main     # remove main program

-include ${DEP}

# END

