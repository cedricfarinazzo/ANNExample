#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "dataset.h"
#include "sdl.h"
#include "nn.h"

int main(int argc, char *argv[])
{
    srand(time(NULL));
    if (argc == 3)
    {
        char *path1 = argv[1];
        char *path2 = argv[2];
        
        struct DATASET *dtr = LoadDataset(path1, 10);
        printf("Training dataset (%s): Loaded %ld digit images in memory\n", path1, dtr->size);
       
        struct DATASET *dte = LoadDataset(path2, 10);
        printf("Testing dataset (%s): Loaded %ld digit images in memory\n", path2, dte->size);

        struct PCFNN_NETWORK *net = init_net();
        printf("Init neural network\n");
        printf("Training...\n");
        train(net, dtr);
        printf("Training done\n");
        FreeDataset(dtr);

        printf("Testing...\n");
        size_t nberror = check(net, dte);
        printf("Testing done\n");
        printf("Error: %ld (%f%%)\n", nberror, (double)nberror / (double)dte->size * 100);

        FreeDataset(dte);
        PCFNN_NETWORK_free(net);
    }

    return EXIT_SUCCESS;
}
