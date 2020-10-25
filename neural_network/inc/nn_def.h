#ifndef __NN_DEF_H__
#define __NN_DEF_H__
#include <stdbool.h>

//#define NN_TYPE				NN_REGRESS
//#define NN_TYPE			NN_PREDICT

#define MAX_LAYERS			3
//#define WEIGHT_UPPER_LIMIT	1
//#define WEIGHT_LOWER_LIMIT	0

typedef float (*f_actv_fn) (float);
typedef void* (*f_train_fn) (void*);
typedef void* (*f_predict_fn) (void*);

typedef enum {
    eINPUT,
    eHIDDEN,
    eOUTPUT,
    eUNDEF = -1,
}e_ltype;

typedef struct {
	bool by_pass;
    float *weight;
    f_actv_fn activate;
    f_actv_fn deactivate;
}t_neuron;

typedef struct{
    int n_layers;
    int layer_size[MAX_LAYERS];
}t_layer_info;

typedef struct nn_layer{
    e_ltype layer_type;
    int n_output;
    int n_input;

    float *i_back_err;
    float *o_back_err;
    float *input;
    float *output;
    float *error;

    t_neuron *neuron;
}t_layer;

typedef struct {
    int n_hlayer;
    t_layer *i_layer;
    t_layer **h_layer;
    t_layer *o_layer;
    float eta;

    f_train_fn train;
    f_predict_fn predict;
}t_neural_network;


t_neural_network* create_neural_network(int n_in, t_layer_info hidden, int n_out);
void feed_forward(t_neural_network *nwk, float *input);
void back_propogate(t_neural_network *nwk, float *output);
void destroy_neural_network(t_neural_network *nwk);

#endif
