#include "nn_def.h" 

void feed_forward_layer(t_layer *l, e_nntype type) 
{
	int i,j;
	float linear_out;

	for (i = 0; i < l->n_output; i++) {
		linear_out = 0;
		for (j = 0; j < l->n_input; j++) {
			linear_out += l->input[j] * l->neuron[i].weight[j];
		}
		if (l->layer_type == eOUTPUT && type == eREGRESS) {
			l->output[i] = linear_out;
		} else {
			l->output[i] = l->neuron[i].activate(linear_out);
		}
	}
}

void back_propogate_layer(t_layer *l, float eta, e_nntype type) 
{
	int i, j;
	float err_op;

	if (l->layer_type == eOUTPUT) {	
		for (i = 0; i < l->n_output; i++) {
			if (l->layer_type == eOUTPUT && type == eREGRESS) {
				l->o_back_err[i] = l->error[i];
			} else {
				l->o_back_err[i] = l->error[i] * l->neuron[i].deactivate(l->output[i]);
			}
		}
	}

	/* calculate error for the pevious layer */
	for (i = 0; i < l->n_input; i++) {
		for (err_op = j = 0; j < l->n_output; j++) {
			err_op += (l->neuron[j].weight[i] * l->o_back_err[j]);
		}
		l->i_back_err[i] = err_op * l->neuron[0].deactivate(l->input[i]);
	}

	for (i = 0; i < l->n_output; i++) {
		for(j = 0; j < l->n_input; j++) {
			l->neuron[i].weight[j] = l->neuron[i].weight[j] + (eta * l->o_back_err[i] * l->input[j]);
		}
	}
}

void feed_forward(t_neural_network *nwk, float *input)
{
	int i;
	
	for(i = 0; i < nwk->i_layer->n_input; i++) {
		nwk->i_layer->input[i] = input[i];
	}

	for(i = 0; i < nwk->n_hlayer; i++) {
		feed_forward_layer(nwk->h_layer[i], nwk->type);
	}
	
	feed_forward_layer(nwk->o_layer, nwk->type);

}


void back_propogate(t_neural_network *nwk, float *output)
{
	int i;

	for(i = 0; i < nwk->o_layer->n_output; i++) {
		nwk->o_layer->error[i] = output[i] - nwk->o_layer->output[i];
	}

//	print_layer(nwk->o_layer);
	back_propogate_layer(nwk->o_layer, nwk->eta, nwk->type);
//	print_layer(nwk->o_layer);
	i = nwk->n_hlayer;
	do {
		--i;
//		print_layer(nwk->h_layer[i]);
		back_propogate_layer(nwk->h_layer[i], nwk->eta, nwk->type);
//		print_layer(nwk->h_layer[i]);
	}while (i);
}

