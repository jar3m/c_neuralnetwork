/*! @file nn_def.h
    @author jar3m
    @brief 
    Contains declations of neural network structures
*/
#pragma once

#include <stdbool.h>


#define WEIGHT_UPPER_LIMIT	1
#define WEIGHT_LOWER_LIMIT	0

typedef float (*f_actv_fn) (float);	///< fn pointer(prototype) for activation function
typedef void* (*f_train_fn) (void*);	///< fn pointer(prototype) for training function
typedef void* (*f_predict_fn) (void*);	///< fn pointer(prototype) for predicion function

/*!
 *  @enum e_nntype
 *  Defines the type of neural network
*/
typedef enum {
	eREGRESS,		///< Regression Based Neural Network
	eCLASSIFY,		///< Classification Based Neural Network
}e_nntype;

/*!
 *  @enum e_ltype
 *  Defines the type of neural network layer
*/
typedef enum {
	eINPUT,			///< Input layer
	eHIDDEN,		///< Hidden layer
	eOUTPUT,		///< Output layer
	eUNDEF = -1,		///< Undefined layer
}e_ltype;

/*!
 *  @enum e_atvfn
 *  Defines the type of activation function to be used for the layer
*/
typedef enum {
	eLINEAR,		///< Linear Activation function
	eRELU,			///< ReLU Activation function
	eSIGMOID,		///< Sigmoid Activation function
}e_atvfn;

/*!
 *  @struct t_neuron
 *  Defines a neuron params
*/
typedef struct {
	bool by_pass;		///< Flag to bypass neuron activation, i.e,out of neuron = selcted input
	float *weight;		///< Weights associated to inputs to neuron
	f_actv_fn activate;	///< Activation to be performed on input in feed forward
	f_actv_fn deactivate;	///< Activation to be performed on output in back propogation
}t_neuron;

/*!
 *  @struct t_lyrinfo
 *  Defines layer information
*/
typedef struct{
	int size;		///< Size of the given layer
	e_ltype layer_type;	///< Type of later @see e_ltype
	e_atvfn actv;		///< Type of activation funtion to be used @see e_atvfn
}t_lyrinfo;

/*!
 *  @struct t_lyrinfo
 *  Defines a neural network layer
*/
typedef struct nn_layer{
	e_ltype layer_type;	///< Type of layer @see e_ltype
	e_atvfn actv;		///< Type of activation funtion for current layer @see e_atvfn
	int n_output;		///< No of oututs to layer
	int n_input;		///< No of inputs to layer

	float *i_back_err;	///< Pointer to layer's input errors used for backpropogation
	float *o_back_err;	///< Pointer to layer's output errors used for backpropogation
	float *input;		///< Pointer to layer's inputs 
	float *output;		///< Pointer to layer's outputs
	float *error;		///< Pointer to layer's errors (used only in Output layer)

	t_neuron *neuron;	///< Pointer to layer's neurons @see t_neuron
}t_layer;

/*!
 *  @struct t_neural_network
 *  Defines neural network layer
*/
typedef struct {
	e_nntype type;		///< Type of neural network @see e_nntype
	int n_hlayer;		///< Num of hidden layers
	t_layer *i_layer;	///< Pointer to input layer of the neural network
	t_layer **h_layer;	///< Pointer to hidden layers of the neural network
	t_layer *o_layer;	///< Pointer to output layer of the neural network
	float eta;		///< Learning rate of the neural network

	f_train_fn train;	///< Pointer to train function @see f_train_fn
	f_predict_fn predict;	///< Pointer to predict function @see f_predict_fn
}t_neural_network;

void feed_forward(t_neural_network *nwk, float *input);
void back_propogate(t_neural_network *nwk, float *output);

t_layer* create_layer(t_lyrinfo l);
void join_layers(t_layer *A, t_layer *B);
void destroy_layer(t_layer *l);
