#include "os.h"
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

static f_actv_fn actv_fn[] = {linear_actv, relu, sigmoid};
static f_actv_fn dactv_fn[] = {d_linear_actv, d_relu, d_sigmoid};

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

t_layer* create_layer(t_lyrinfo l)
{
	t_layer *layer = malloc(sizeof(t_layer));
	
	layer->n_output = layer->n_input = 0;
	layer->input = layer->output = layer->error = NULL;
	layer->o_back_err = layer->i_back_err  = NULL;
	layer->neuron = NULL;
	layer->layer_type = l.layer_type;
	layer->actv = l.actv;

	switch(l.layer_type)
	{
		case eINPUT:
			layer->n_output = layer->n_input = l.size;
			layer->input = calloc(layer->n_input,sizeof(float));
			layer->output = layer->input;
			layer->o_back_err = calloc(l.size,sizeof(float));
			break;
		case eOUTPUT:
			layer->error = calloc(l.size,sizeof(float));

		case eHIDDEN:
			layer->neuron = calloc(l.size,sizeof(t_neuron));
			layer->output = calloc(l.size,sizeof(float));
			layer->o_back_err = calloc(l.size,sizeof(float));
			layer->n_output = l.size;
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

		 for (i = 0; i < B->n_output; i++) {
			 B->neuron[i].by_pass = false;
			 B->neuron[i].weight = malloc(B->n_input * sizeof(float));
			 //randomize weights()
			 for (j = 0; j < B->n_input; j++) { 
				 B->neuron[i].weight[j] = ((float)rand())/RAND_MAX;
				 B->neuron[i].activate = actv_fn[B->actv];
				 B->neuron[i].deactivate = dactv_fn[B->actv];
			 }
		 }
	}
	else {
		//assert() // irregular sequence
	}
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

void print_layer(t_layer *l)
{
#if 0
	int i,j;
	printf("LAYER --> %d \n",l->layer_type);
	printf("----------------------------\n");
	for(i = 0; i < l->n_input; i++) 
		printf("[Ci/I] %f %f\n",l->i_back_err[i],l->input[i]); 
	for(i = 0; i < l->n_output; i++) {
		printf("[Co/O] %f ", l->o_back_err[i]); 
		printf("%f ",l->output[i]);
		if(l->layer_type == eOUTPUT)	
			printf("%f",l->error[i]);
		printf("\n");
	}
#endif
	printf("****************************\n");

}

