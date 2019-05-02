#include <stdlib.h>
#include <ANN/models/PCFNN/network.h>
#include <ANN/models/PCFNN/train.h>
#include <ANN/models/PCFNN/feedforward.h>
#include <ANN/models/PCFNN/batch.h>
#include <ANN/models/PCFNN/file.h>
#include <ANN/tools.h>
#include "dataset.h"

#include "nn.h"

struct PCFNN_NETWORK *init_net()
{
    struct PCFNN_NETWORK *net = PCFNN_NETWORK_new();
    struct PCFNN_LAYER *l1 = PCFNN_LAYER_new_input(784, f_act_input, f_act_input_de);
    struct PCFNN_LAYER *l2 = PCFNN_LAYER_new(NULL, NULL, NULL);
    struct PCFNN_LAYER *l3 = PCFNN_LAYER_new(NULL, NULL, NULL);
    struct PCFNN_LAYER *l4 = PCFNN_LAYER_new(NULL, NULL, NULL);
    PCFNN_NETWORK_addl(net, l1); PCFNN_NETWORK_addl(net, l2); 
    PCFNN_NETWORK_addl(net, l3); PCFNN_NETWORK_addl(net, l4);

    PCFNN_LAYER_connect(l1, l2, 784, 60, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);
    PCFNN_LAYER_connect(l2, l3, 60, 32, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);
    PCFNN_LAYER_connect(l3, l4, 32, 10, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);

    PCFNN_NETWORK_build(net);

    return net;
}

void train(struct PCFNN_NETWORK *net, struct DATASET *d)
{
     PCFNN_NETWORK_train(net, d->data, d->target,
        d->size, 0.0, NULL, 1, 4, 10, 0.5, f_cost_quadratic_loss_de);
}

int isOk(double *out, double *target, size_t size)
{
    int ok = 1;
    for (size_t i = 0; i < size && ok; ++i)
    {
        if (target[i] > 0.6)
            ok = out[i] > 0.6;
        if (target[i] < 0.4)
            ok = out[i] < 0.4;
    }
    return ok;
}

size_t check(struct PCFNN_NETWORK *net, struct DATASET *d)
{
    size_t nberror = 0;
    for(size_t i = 0; i < d->size; ++i) 
    {
        PCFNN_NETWORK_feedforward(net, d->data[i]);
        double *out = PCFNN_NETWORK_get_output(net);
        nberror += !isOk(out, d->target[i], d->target_size);
        free(out);
    }
    return nberror;
}
