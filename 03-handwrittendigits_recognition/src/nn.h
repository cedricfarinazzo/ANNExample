#ifndef _SRC_NN_H
#define _SRC_NN_H

#include <stdlib.h>
#include <ANN/models/PCFNN/network.h>
#include <ANN/models/PCFNN/train.h>
#include <ANN/models/PCFNN/feedforward.h>
#include <ANN/models/PCFNN/batch.h>
#include <ANN/models/PCFNN/file.h>
#include <ANN/tools.h>


struct PCFNN_NETWORK *init_net();

void train(struct PCFNN_NETWORK *net, struct DATASET *d);

size_t check(struct PCFNN_NETWORK *net, struct DATASET *d);

#endif /* _SRC_NN_H */
