#include "os.h"
#include "nn_interface.h"


void* create_neural_network(t_nn_cfg config)
{
	t_neural_network *temp = malloc(sizeof(t_neural_network));
	int i;
	time_t t = time(0);
	t_lyrinfo linfo;

	printf("seed : 0x%x\n", (unsigned int)t);
	srand(t);
	//srand(0x5e19cbd1);
//	srand(1);

	temp->type = (e_nntype)config.type;
	temp->eta = config.eta; //MACRO

	// create_input layer
	linfo.layer_type = eINPUT;
	linfo.size = config.n_in;
	temp->i_layer = create_layer(linfo);

	// create_output layer
	linfo.layer_type = eOUTPUT;
	linfo.size = config.n_out;
	linfo.actv = config.oactv;
	temp->o_layer = create_layer(linfo);
	
	// create_hidden layer
	temp->n_hlayer = config.n_hdn;
	temp->h_layer = malloc(temp->n_hlayer * sizeof(t_layer*));

	for (i = 0; i < temp->n_hlayer; i++) {
		linfo.layer_type = eHIDDEN;
		linfo.size = config.hinfo[i].size;
		linfo.actv = config.hinfo[i].actv;
		temp->h_layer[i] = create_layer(linfo);
		if(i == 0) {
			join_layers(temp->i_layer, temp->h_layer[i]);
		}
		else {
			join_layers(temp->h_layer[i-1], temp->h_layer[i]);
		}
	}
	
	// join final hidden layer with output layer
	join_layers(temp->h_layer[i-1], temp->o_layer);

	return temp;
}

void destroy_neural_network(void *obj)
{
	int i;
	t_neural_network *nwk = (t_neural_network*)obj;
	

	destroy_layer(nwk->i_layer);
	destroy_layer(nwk->o_layer);

	for (i = 0; i < nwk->n_hlayer; i++) {
		destroy_layer(nwk->h_layer[i]);
	}
	free(nwk->h_layer);
	nwk->h_layer = NULL;

	nwk->predict = nwk->train = NULL;

	free(nwk);
	nwk = NULL;
}


void train_network(void *obj, t_sample train)
{
	int i;
	t_neural_network *nwk = (t_neural_network*)obj;
	feed_forward(nwk, train.in);
	back_propogate(nwk, train.out);
	for (i = 0; i < nwk->o_layer->n_output; i++) {
		train.error[i] = nwk->o_layer->error[i];
	}
//	printf("Expected output %f %f %f\n", train.in[3], train.out[2], nwk->o_layer->output[0]);
}


void predict_network(void *obj, t_sample test)
{		
	t_neural_network *nwk = (t_neural_network*)obj;
	int i;
	feed_forward(nwk, test.in);

	printf ("Expected output :");
	for (i = 0; i < nwk->o_layer->n_output; i++)
		printf( "(%f %f) ", test.out[i], nwk->o_layer->output[i]);
	printf("\n");
	
	for (i = 0; i < nwk->o_layer->n_output; i++) {
		test.out[i] = nwk->o_layer->output[i];
		test.error[i] = nwk->o_layer->error[i];
	}
}

