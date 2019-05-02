#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "dataset.h"
#include "sdl.h"
#include "nn.h"

int main(int argc, char *argv[])
{
    if (argc == 2)
    {
        char *path = argv[1];
        printf("Dataset path: %s\n", path);
        struct DATASET *dataset = LoadDataset(path, 10);
        printf("Dataset: Loaded %ld in memory\n", dataset->size);

        FreeDataset(dataset);
    }

    return EXIT_SUCCESS;
}
