#include <stdlib.h>
#include <stdio.h>
#include <SDL/SDL.h>
#include <SDL/SDL_image.h>
#include <dirent.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include "sdl.h"

#include "dataset.h"

int getDigitFromFileName(char *file)
{
    size_t i = 0;
    while (i < strlen(file) && file[i] != '_') ++i;
    if (i >= strlen(file)) return -1;
    char ascii[i];
    for (size_t j = 0; j < i; ++j) ascii[j] = file[j];
    int digit;
    sscanf(ascii, "%d", &digit);
    return digit;
}

double *SDLSurfToMat(SDL_Surface *img)
{
    double *mat = malloc(sizeof(double) * (img->w * img->h));
    if (mat == NULL) return NULL;
    size_t i = 0;
    Uint8 r, g, b;
    for(int y = 0; y < img->h; ++y)
        for(int x = 0; x < img->w; ++x, ++i)
        {
            Uint32 pixel = get_pixel(img, x, y);
            SDL_GetRGB(pixel, img->format, &r, &g, &b);
            if (r < 127 && g < 127 && b < 127)
                mat[i] = 0.0;
            else
                mat[i] = 1.0;
        }
    return mat;
}

double *genExpectedOutput(size_t size, int digit)
{
    double *expected = calloc(size, sizeof(double));
    if (expected == NULL) return NULL;
    expected[digit] = 1.0;
    return expected;
}

struct DATASET *LoadDataset(char *path, size_t target_size)
{
    struct DATASET *dataset = malloc(sizeof(struct DATASET));
    if (dataset == NULL) return NULL;
    dataset->size = 0;
    dataset->target_size = target_size;
    dataset->data = dataset->target = NULL;
    size_t len = 0;
    DIR *d;
    struct dirent *handler;
    d = opendir(path);
    if (d == NULL) return dataset;

    double **data = malloc(sizeof(double*) * len);
    double **target = malloc(sizeof(double*) * len);

    while ((handler = readdir(d)) != NULL)
    {
        char *name = handler->d_name;
        if (strcmp(name, ".") == 0 ||
            strcmp(name, "..") == 0)
            continue;

        char filename[PATH_MAX];
        strcpy(filename, path);
        strcat(filename, name);
        int digit = getDigitFromFileName(name);
        if (digit == -1) continue;
        SDL_Surface *img = load_image(filename);
        if (img == NULL) continue;
        double *expout = genExpectedOutput(target_size, digit);
        if (expout == NULL) { SDL_FreeSurface(img); continue; }
        double *imgmat = SDLSurfToMat(img);
        SDL_FreeSurface(img);
        if (imgmat == NULL) { free(expout); continue; }

        ++len;
        data = realloc(data, sizeof(double*) * len);
        if (data == NULL) { free(expout); free(imgmat); free(target); free(dataset); return NULL; }
        target = realloc(target, sizeof(double*) *len);
        if (target == NULL) { free(expout); free(imgmat); free(data); free(dataset); return NULL; }
        data[len - 1] = imgmat;
        target[len - 1] = expout;
    }
    closedir(d);
    dataset->size = len;
    dataset->data = data;
    dataset->target = target;
    return dataset;
}

void FreeDataset(struct DATASET *d)
{
    for (size_t i = 0; i < d->size; ++i)
    {
        free(d->data[i]);  
        free(d->target[i]); 
    }
    free(d->data);
    free(d->target);
    free(d);
}
