/*! @file nn_algo.c
    @author jar3m
    @brief 
    Contains definitions of Neural network algos such as feedforward and backpropogation
*/
#include "nn_def.h" 

/*! @brief  
 *  Feed forward a layer
 *  @see https://en.wikipedia.org/wiki/Feedforward_neural_network#:~:text=its%20derivative%20is-,easily%20calculated,-%3A
 *  @param l 	- Pointer to layer
 *  @param type	- Type of layer
 *  @return     - NA
 * */
void feed_forward_layer(t_layer *l, e_nntype type) 
{
	int i,j;
	float linear_out;

	// Calculate the output for all neurons for given layer
	for (i = 0; i < l->n_output; i++) {
		linear_out = 0;
		for (j = 0; j < l->n_input; j++) {
			linear_out += l->input[j] * l->neuron[i].weight[j];
		}
		// Activate output only for Classification
		if (l->layer_type == eOUTPUT && type == eREGRESS) {
			l->output[i] = linear_out;
		} else {
			l->output[i] = l->neuron[i].activate(linear_out);
		}
	}
}

/*! @brief  
 *  Backpropogate the current layer
 *  @see https://en.wikipedia.org/wiki/Backpropagation#:~:text=The-,overall%20network,-is%20a%20combination
 *  @param l 	- Pointer to layer
 *  @param type	- Type of layer
 *  @return     - NA
 * */
void back_propogate_layer(t_layer *l, float eta, e_nntype type) 
{
	int i, j;
	float err_op;

	// For output layer calculate error
	if (l->layer_type == eOUTPUT) {	
		for (i = 0; i < l->n_output; i++) {
			// deActivate output only for Classification
			if (l->layer_type == eOUTPUT && type == eREGRESS) {
				l->o_back_err[i] = l->error[i];
			} else {
				l->o_back_err[i] = l->error[i] * l->neuron[i].deactivate(l->output[i]);
			}
		}
	}

	// calculate error for the previous layer 
	for (i = 0; i < l->n_input; i++) {
		for (err_op = j = 0; j < l->n_output; j++) {
			err_op += (l->neuron[j].weight[i] * l->o_back_err[j]);
		}
		l->i_back_err[i] = err_op * l->neuron[0].deactivate(l->input[i]);
	}
	
	// Update new weights 
	for (i = 0; i < l->n_output; i++) {
		for(j = 0; j < l->n_input; j++) {
			l->neuron[i].weight[j] = l->neuron[i].weight[j] + (eta * l->o_back_err[i] * l->input[j]);
		}
	}
}

/*! @brief  
 *  Run Backpropogation on neural network
 *  @see https://en.wikipedia.org/wiki/Feedforward_neural_network
 *  @param nwk 	  - Pointer to Neural network
 *  @param input  - Pointer to an input instance
 *  @return       - NA
 * */
void feed_forward(t_neural_network *nwk, float *input)
{
	int i;
	
	// Initialize inputs to input layer
	for(i = 0; i < nwk->i_layer->n_input; i++) {
		nwk->i_layer->input[i] = input[i];
	}

	// Feed forward hidden layers
	for(i = 0; i < nwk->n_hlayer; i++) {
		feed_forward_layer(nwk->h_layer[i], nwk->type);
	}
	
	// Feed forward Output layer
	feed_forward_layer(nwk->o_layer, nwk->type);

}

/*! @brief  
 *  Run Backpropogation on neural network
 *  @see https://en.wikipedia.org/wiki/Backpropagation
 *  @param nwk 	  - Pointer to layer
 *  @param output - Pointer to an output instance
 *  @return       - NA
 * */
void back_propogate(t_neural_network *nwk, float *output)
{
	int i;

	// Calculate error from expected output and neural network output
	for(i = 0; i < nwk->o_layer->n_output; i++) {
		nwk->o_layer->error[i] = output[i] - nwk->o_layer->output[i];
	}

	// Back prop output layer
	back_propogate_layer(nwk->o_layer, nwk->eta, nwk->type);
	
	// Back prop all Hidden layers 
	i = nwk->n_hlayer;
	do {
		--i;
		back_propogate_layer(nwk->h_layer[i], nwk->eta, nwk->type);
	}while (i);
}

