#pragma once
#include "nn_def.h"

typedef struct {
	float *in;
	float *out;
	float *error;
} t_sample;

typedef struct {
	e_nntype type;
	float eta;
	int n_in;
	int n_out;
	e_atvfn oactv;
	int n_hdn;
	t_lyrinfo *hinfo;
}t_nn_cfg;


void* create_neural_network(t_nn_cfg config);
void destroy_neural_network(void *nwk);
void train_network  (void *obj, t_sample train);
void predict_network(void *obj, t_sample test);
