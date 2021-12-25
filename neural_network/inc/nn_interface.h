/*! @file nn_interface.h
    @author jar3m
    @brief 
    Defines strutures exposed to the user for interfacing with neural network (called from python)
*/
#pragma once

#include "nn_def.h"

/*!
 *  @struct t_sample
 *  Defines a sample: pair of input-output-error of a given instance
*/
typedef struct {
	float *in;		///< Pointer to input at an instance
	float *out;		///< Pointer to outputs at an instance
	float *error;		///< Pointer to errors at an instance
} t_sample;

/*!
 *  @struct t_nn_cfg
 *  define configurations required to create a neural network
*/
typedef struct {
	e_nntype type;		///< Type of neural network @see e_nntype
	float eta;		///< Nearning rate
	int n_in;		///< No of inputs
	int n_out;		///< No of outputs
	e_atvfn oactv;		///< Output activation to be defined in case of Regression layer
	int n_hdn;		///< No of hidden layers
	t_lyrinfo *hinfo;	///< Information of each hidden layer @see t_lyrinfo
}t_nn_cfg;


void* create_neural_network(t_nn_cfg config);
void destroy_neural_network(void *nwk);
void train_network  (void *obj, t_sample train);
void predict_network(void *obj, t_sample test);
