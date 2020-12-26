#pragma once

#include <stdbool.h>


#define MAX_LAYERS					1
#define WEIGHT_UPPER_LIMIT	1
#define WEIGHT_LOWER_LIMIT	0

typedef float (*f_actv_fn) (float);
typedef void* (*f_train_fn) (void*);
typedef void* (*f_predict_fn) (void*);

typedef enum {
	eREGRESS,
	eCLASSIFY,
}e_nntype;

typedef enum {
	eINPUT,
	eHIDDEN,
	eOUTPUT,
	eUNDEF = -1,
}e_ltype;

typedef enum {
	eLINEAR,
	eRELU,
	eSIGMOID,
}e_atvfn;

typedef struct {
	bool by_pass;
	float *weight;
	f_actv_fn activate;
	f_actv_fn deactivate;
}t_neuron;

typedef struct{
	int size;
	e_ltype layer_type;
	e_atvfn actv;
}t_lyrinfo;

typedef struct nn_layer{
	e_ltype layer_type;
	e_atvfn actv;
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
	e_nntype type;
	int n_hlayer;
	t_layer *i_layer;
	t_layer **h_layer;
	t_layer *o_layer;
	float eta;

	f_train_fn train;
	f_predict_fn predict;
}t_neural_network;

void feed_forward(t_neural_network *nwk, float *input);
void back_propogate(t_neural_network *nwk, float *output);

t_layer* create_layer(t_lyrinfo l);
void join_layers(t_layer *A, t_layer *B);
void destroy_layer(t_layer *l);
