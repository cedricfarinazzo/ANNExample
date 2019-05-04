#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
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

    PCFNN_LAYER_connect(l1, l2, 784, 128, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);
    PCFNN_LAYER_connect(l2, l3, 128, 64, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);
    PCFNN_LAYER_connect(l3, l4, 64, 10, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);

    PCFNN_NETWORK_build(net);

    return net;
}

struct PCFNN_NETWORK *init_load_net()
{
    struct PCFNN_NETWORK *net = init_net();
    if (net == NULL) return NULL;
    if( access("conf.nn", F_OK) != -1 ) {
        FILE *f = fopen("conf.nn", "r");
        PCFNN_NETWORK_load_conf(net, f);
        printf("Load network config from %s\n", "conf.nn");
    }
    return net;
}

void savenn(struct PCFNN_NETWORK *net)
{
    FILE *f = fopen("conf.nn", "w+");
    PCFNN_NETWORK_save_conf(net, f);
}

void train(struct PCFNN_NETWORK *net, struct DATASET *d)
{
     PCFNN_NETWORK_train(net, d->data, d->target,
        d->size, 0.0, NULL, 1, 7, 40, 0.03, 0, f_cost_quadratic_loss_de, NULL);
}

struct trainthread {
    struct PCFNN_NETWORK *net;
    struct DATASET *d;
    double *status;
};

void *train_worker(void *a)
{
    struct trainthread *t = (struct trainthread*)a;
    PCFNN_NETWORK_train(t->net, t->d->data, t->d->target,
        t->d->size, 0.0, NULL, 1, 7, 10, 0.1, 0.9, f_cost_quadratic_loss_de, t->status);
    return NULL;
}

void train_status(struct PCFNN_NETWORK *net, struct DATASET *d)
{
    struct trainthread tt;
    tt.d = d; tt.net = net;
    double status; tt.status = &status;

    clock_t start = clock();
    clock_t end;
    double time_taken; 

    pthread_attr_t tattr;
    struct sched_param param;
    pthread_attr_init(&tattr);
    pthread_attr_getschedparam(&tattr, &param);
    (param.sched_priority)++;
    pthread_attr_setschedparam(&tattr, &param);
    
    pthread_t t;
    if (pthread_create(&t, &tattr, train_worker, &tt) != 0)
        return;
    while (pthread_tryjoin_np(t, NULL) != 0)
    {
        end = clock();
        time_taken = ((double)(end - start))/CLOCKS_PER_SEC; // in seconds 
        printf("\rTraining status : %d%% (%f seconds)", (int)status, time_taken);
        fflush(stdout);
        sleep(1);
    }
    end = clock();
    time_taken = ((double)end - start)/CLOCKS_PER_SEC; // in seconds 
    printf("\rTraining status : %d%% (%f seconds)\n", (int)status, time_taken);
    return;
}

int isOk(double *out, double *target, size_t size)
{
    size_t outm = 0; size_t tm = 0;
    for (size_t i = 0; i < size; ++i)
    {
        if (out[i] > out[outm])
            outm = i;
        if (target[i] > target[tm])
            tm = i;
    }
    return outm == tm && abs(target[tm] - out[outm]) < 0.3;
}

size_t check(struct PCFNN_NETWORK *net, struct DATASET *d)
{
    printf("Testing...\n");
    size_t nbdigit[10]; size_t errdigit[10];
    for (size_t i = 0; i < 10; ++i)
        errdigit[i] = nbdigit[i] = 0;
    size_t nberror = 0;
    for(size_t i = 0; i < d->size; ++i) 
    {
        int digit;
        for (digit = 0; d->target[i][digit] != 1 && digit < 10; ++digit);
        ++(nbdigit[digit]);

        PCFNN_NETWORK_feedforward(net, d->data[i]);
        double *out = PCFNN_NETWORK_get_output(net);
        
        int iserr = !isOk(out, d->target[i], d->target_size);
        nberror += iserr;
        errdigit[digit] += iserr;

        free(out);
    }
    printf("Testing done\n");
    for (size_t i = 0; i < 10; ++i)
        printf("[Digit %ld]: error : %ld / %ld (%f%%)\n", i, errdigit[i], nbdigit[i], 
               nbdigit[i] != 0 ? (double)errdigit[i] / (double)nbdigit[i] * 100 : 0);
    printf("Error: %ld (%f%%)\n", nberror, (double)nberror / (double)d->size * 100);
    return nberror;
}

void get(struct PCFNN_NETWORK *net, struct DATASET *d)
{
    if (d->size != 1) return;
    
    PCFNN_NETWORK_feedforward(net, d->data[0]);
    double *out = PCFNN_NETWORK_get_output(net);

    size_t outm = 0; size_t tm = 0;
    for (size_t i = 0; i < d->target_size; ++i)
    {
        if (out[i] > out[outm])
            outm = i;
        if (d->target[0][i] > d->target[0][tm])
            tm = i;
        printf("[Digit %ld]: propa %f%%)\n", i, out[i] * 100);
    }
    printf("Result: it's a %ld! (propa: %f%%)    | expected: %ld\n", outm, out[outm] * 100, tm);
    free(out);
}
