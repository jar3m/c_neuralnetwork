#include <stdio.h>
#include "../../neural_network/inc/nn_def.h" 

void train_network(t_neural_network *nwk,float *in, float *out) 
{
	feed_forward(nwk, in);
	back_propogate(nwk, out);
}

void predict_network(t_neural_network *nwk)
{
	float in[5],out[1];
	FILE *test_data;

	/* Get training data */
	test_data = fopen("test/src/test_data","r");
	if(!test_data) {
		printf("ERROR:\tfile not found\n");
		return ;
	}	
	
	while(fscanf(test_data,"%f%f%f%f%f%f", &in[0], &in[1], &in[2], &in[3], &in[4], &out[0])!=EOF) {
		feed_forward(nwk, in);
		printf("Expected output %f %f\n", out[0], nwk->o_layer->output[0]);
	}

}
int main()
{
	t_layer_info h = {2,{3,1}};
	t_neural_network *nn;
	FILE *train_data;
	float in[5], out[1];

	nn = create_neural_network(4, h, 3);

	/* Get training data */
	train_data= fopen("test/src/train_data","r");
	if(!train_data)
	{
		printf("ERROR:\tfile not found\n");
		return -1;
	}	
	
	while(fscanf(train_data,"%f%f%f%f%f%f", &in[0], &in[1], &in[2], &in[3], &in[4], &out[0])!=EOF)
	{
		train_network(nn,in,out);
	}
	predict_network(nn);

	destroy_neural_network(nn);

	return 1;
}
