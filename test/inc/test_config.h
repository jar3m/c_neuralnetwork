#pragma once

//#include "nn_def.h" 
#include "../../neural_network/inc/nn_def.h" 

typedef struct {
	char  file_name[100];
	FILE  *fp;
	float **in;
	float **out;
}t_nn_set;

typedef struct test_nn_config{
	int n_in;
	int n_out;
	t_layer_info h;
	int n_train_set;
	int n_test_set;
	float *mean_in;
	float *mean_out;
	float *std_dev_in;
	float *std_dev_out;
	t_nn_set train;
	t_nn_set test;
}t_nn_config;

