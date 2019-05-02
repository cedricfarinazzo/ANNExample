#ifndef _SRC_DATASET_H
#define _SRC_DATASET_H

#include <stdlib.h>
#include <SDL/SDL.h>
#include <SDL/SDL_image.h>
#include <dirent.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include "sdl.h"

struct DATASET {
    size_t size, target_size;
    double **data, **target;
};

struct DATASET *LoadDataset(char *path, size_t target_size);

void FreeDataset(struct DATASET *d);

#endif /* _SRC_DATASET_H */
