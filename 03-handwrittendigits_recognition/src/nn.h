#ifndef _SRC_NN_H
#define _SRC_NN_H

#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <ANN/models/PCFNN/network.h>
#include <ANN/models/PCFNN/train.h>
#include <ANN/models/PCFNN/feedforward.h>
#include <ANN/models/PCFNN/batch.h>
#include <ANN/models/PCFNN/file.h>
#include <ANN/tools.h>


struct PCFNN_NETWORK *init_net();

struct PCFNN_NETWORK *init_load_net();

void savenn(struct PCFNN_NETWORK *net);

void train(struct PCFNN_NETWORK *net, struct DATASET *d);

void train_status(struct PCFNN_NETWORK *net, struct DATASET *d);

size_t check(struct PCFNN_NETWORK *net, struct DATASET *d);

void get(struct PCFNN_NETWORK *net, struct DATASET *d);

#endif /* _SRC_NN_H */
