#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h> 
#include <time.h> 
#include "nn_def.h" 

float relu (float x)
{
	float temp = (x >= 0)? x : 0;
	return temp;
}

float d_relu (float x)
{
	float temp = (x > 0)? 1 : 0;
	return temp;
}
float sigmoid (float x)
{
	float temp = 1 / (1 + exp(-x));
	return temp;
}

float d_sigmoid (float x)
{
	float temp = sigmoid(x);

	temp = temp * (1 - temp);
	return temp;
}

float linear_actv(float x)
{
	return x;
}

float d_linear_actv(float x)
{
	return 1;
}

void set_bypass_neuron(t_neural_network *nwk, int layer_num, int in_num,int neuron_num)
{
	nwk->h_layer[layer_num]->neuron[neuron_num].by_pass = true;
	nwk->h_layer[layer_num]->neuron[neuron_num].weight[0] = in_num;
}

void set_bypass_neuron_op(t_neural_network *nwk, int layer_num, int in_num,int neuron_num)
{
	nwk->o_layer->neuron[neuron_num].by_pass = true;
	nwk->o_layer->neuron[neuron_num].weight[0] = in_num;
}

t_layer* create_layer(e_ltype type, int size)
{
	t_layer *layer = malloc(sizeof(t_layer));
	
	layer->n_output = layer->n_input = 0;
	layer->input = layer->output = layer->error = NULL;
	layer->o_back_err = layer->i_back_err  = NULL;
	layer->neuron = NULL;
	layer->layer_type = type;

	switch(type)
	{
		case eINPUT:
			layer->input = calloc(size,sizeof(float));
			layer->output = layer->input;
			layer->n_output = layer->n_input = size;
			layer->o_back_err = calloc(size,sizeof(float));
			break;
		case eOUTPUT:
			layer->error = calloc(size,sizeof(float));

		case eHIDDEN:
			layer->neuron = calloc(size,sizeof(t_neuron));

			layer->output = calloc(size,sizeof(float));

			layer->o_back_err = calloc(size,sizeof(float));
			layer->n_output = size;
			break;
		default:
			//assert();
			break;
	}

	return layer;
}

void join_layers(t_layer *A, t_layer *B)
{
	int i,j;
	//assert(A && B);
	
	if((A->layer_type == eINPUT && B->layer_type == eHIDDEN) ||
	   (A->layer_type == eHIDDEN && B->layer_type == eHIDDEN) ||
	   (A->layer_type == eHIDDEN && B->layer_type == eOUTPUT)) {

		 B->i_back_err = A->o_back_err;
		 B->n_input = A->n_output;
		 B->input = A->output;
//		 printf("layer W-> %d\n",B->layer_type);
		 for(i = 0; i < B->n_output; i++) {
			 B->neuron[i].by_pass = false;
			 B->neuron[i].weight = malloc(B->n_input * sizeof(float));
			 //randomize weights()
			 for(j = 0; j < B->n_input; j++) { 
				 B->neuron[i].weight[j] = ((float)rand())/RAND_MAX;
//				 printf("%f ", B->neuron[i].weight[j]);
	//				 B->neuron[i].activate = relu;
	//				 B->neuron[i].deactivate = d_relu;
	//			 if (A->layer_type == eHIDDEN && B->layer_type == eHIDDEN) {
					 B->neuron[i].activate = sigmoid;
					 B->neuron[i].deactivate = d_sigmoid;
//		 if (A->layer_type == eINPUT && B->layer_type == eHIDDEN) {
	//		 B->neuron[i].activate = linear_actv;
	//		 B->neuron[i].deactivate = d_linear_actv;
//		 }
	//			B->neuron[i].activate = linear_actv;
	//			B->neuron[i].deactivate = d_linear_actv;
			 }
//					printf("\n");
		 }
	}
	else {
		//assert() // irregular sequence
	}
}

t_neural_network* create_neural_network(int n_in, t_layer_info hidden, int n_out)
{
	t_neural_network *temp = malloc(sizeof(t_neural_network));
	int i;
	time_t t = time(0);
	printf("seed : 0x%x\n", (unsigned int)t);
	srand(t);
	//srand(0x5e19cbd1);
//	srand(1);

	temp->eta = NN_NETA; //MACRO

	temp->i_layer = create_layer(eINPUT,n_in);
	temp->o_layer = create_layer(eOUTPUT,n_out);
	temp->h_layer = malloc(hidden.n_layers * sizeof(t_layer*));
	temp->n_hlayer = hidden.n_layers;

	for(i = 0; i < hidden.n_layers; i++) {
		temp->h_layer[i] = create_layer(eHIDDEN, hidden.layer_size[i]);
		if(i == 0)
			join_layers(temp->i_layer, temp->h_layer[i]);
		else
			join_layers(temp->h_layer[i-1], temp->h_layer[i]);
	}

	join_layers(temp->h_layer[i-1], temp->o_layer);

	return temp;
}

void destroy_layer(t_layer *l) 
{
	int i;

	switch(l->layer_type) {
		case eINPUT:
			free(l->input);
			free(l->o_back_err);
			break;
		case eOUTPUT:
			free(l->error);
		case eHIDDEN:
			for(i = 0; i < l->n_output; i++) { 
				free(l->neuron[i].weight);
				l->neuron[i].weight = NULL;
			}
			free(l->o_back_err);
			free(l->neuron);
			free(l->output);
			break;
		default:
			//assert();
			break;
	}

	l->input = l->output = l->error = NULL;
	l->i_back_err = l->o_back_err = NULL;
	l->neuron = NULL;
	l->n_output = l->n_input = 0;
	l->layer_type = eUNDEF;
	free(l);
	l = NULL;
}

void destroy_neural_network(t_neural_network *nwk)
{
	int i;

	destroy_layer(nwk->i_layer);
	destroy_layer(nwk->o_layer);

	for(i = 0; i < nwk->n_hlayer; i++) 
		destroy_layer(nwk->h_layer[i]);

	free(nwk->h_layer);
	nwk->h_layer = NULL;

	nwk->predict = nwk->train = NULL;

	free(nwk);
	nwk = NULL;
}


void print_layer(t_layer *l)
{
	int i,j;
	printf("LAYER --> %d \n",l->layer_type);
	printf("----------------------------\n");
#if 1
	for(i = 0; i < l->n_input; i++) 
		printf("[Ci/I] %f %f\n",l->i_back_err[i],l->input[i]); 
	for(i = 0; i < l->n_output; i++) {
		printf("[Co/O] %f ", l->o_back_err[i]); 
		printf("%f ",l->output[i]);
		if(l->layer_type == eOUTPUT)	
			printf("%f",l->error[i]);
		printf("\n");
#else
	for(i = 0; i < l->n_output; i++) {
	for(j = 0; j < l->n_input; j++) 
	  printf("W(%d,%d) %f\n",i,j,l->neuron[i].weight[j]);
	  printf("\n");
#endif
	}
	printf("****************************\n");

}

# if 0

#endif
