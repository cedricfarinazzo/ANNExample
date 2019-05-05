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
        if (strcmp("check", argv[1]) == 0)
        {
            char *path1 = argv[2];

            struct DATASET *dte = LoadDataset(path1, 10);
            if (dte == NULL)
            {
                printf("Failed to load dataset!\n");
                return EXIT_FAILURE;
            }
            printf("Testing dataset (%s): Loaded %ld digit images in memory\n", path1, dte->size);

            printf("Init neural network\n");
            struct PCFNN_NETWORK *net = init_load_net();
            if (net == NULL)
            {
                FreeDataset(dte);
                printf("Failed to init network!\n");
                return EXIT_FAILURE;
            }
            print_ram_usage(net);

            check(net, dte);

            FreeDataset(dte);
            PCFNN_NETWORK_free(net);
        }

        if (strcmp("get", argv[1]) == 0)
        {
            char *path1 = argv[2];

            struct DATASET *dte = LoadDataset(path1, 10);
            if (dte == NULL)
            {
                printf("Failed to load dataset!\n");
                return EXIT_FAILURE;
            }
            printf("Testing dataset (%s): Loaded %ld digit images in memory\n", path1, dte->size);

            printf("Init neural network\n");
            struct PCFNN_NETWORK *net = init_load_net();
            if (net == NULL)
            {
                FreeDataset(dte);
                printf("Failed to init network!\n");
                return EXIT_FAILURE;
            }
            print_ram_usage(net);

            get(net, dte);

            FreeDataset(dte);
            PCFNN_NETWORK_free(net);
        }

    }

    if (argc == 4)
    {
        if (strcmp("train", argv[1]) == 0)
        {
            char *path1 = argv[2];
            char *path2 = argv[3];

            struct DATASET *dtr = LoadDataset(path1, 10);
            if (dtr == NULL)
            {
                printf("Failed to load training dataset!\n");
                return EXIT_FAILURE;
            }
            printf("Training dataset (%s): Loaded %ld digit images in memory\n", path1, dtr->size);

            struct DATASET *dte = LoadDataset(path2, 10);
            if (dte == NULL)
            {
                FreeDataset(dtr);
                printf("Failed to load testing dataset!\n");
                return EXIT_FAILURE;
            }
            printf("Testing dataset (%s): Loaded %ld digit images in memory\n", path2, dte->size);

            printf("Init neural network\n");
            struct PCFNN_NETWORK *net = init_load_net();
            if (net == NULL)
            {
                FreeDataset(dte);
                FreeDataset(dtr);
                printf("Failed to init network!\n");
                return EXIT_FAILURE;
            }
            print_ram_usage(net);

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
    }

    return EXIT_SUCCESS;
}
