#pragma once
#include "nn_def.h"

typedef struct {
	e_nntype type;
	float eta;
	int n_in;
	int n_out;
	e_atvfn oactv;
	int n_hdn;
	t_lyrinfo *hinfo;
}t_nn_cfg;

typedef struct {
  int val;
}t_kon;

t_kon * konichiwa(int k);
t_nn_cfg* get_nn_config(int nhdn);
t_neural_network* create_neural_network(t_nn_cfg *config);
void destroy_neural_network(t_neural_network *nwk);
void train_network(t_neural_network *nwk,float *in, float *out);
void predict_network(t_neural_network *nwk, float *in, float *out, int n_out);
