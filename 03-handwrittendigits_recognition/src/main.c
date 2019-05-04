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

        printf("Init neural network\n");
        struct PCFNN_NETWORK *net = init_load_net();
        
        printf("Training...\n");
        train_status(net, dtr);
        printf("Training done\n");
        FreeDataset(dtr);

        printf("Saving network config\n");
        savenn(net);

        check(net, dte);

        FreeDataset(dte);
        PCFNN_NETWORK_free(net);
    }

    return EXIT_SUCCESS;
}
