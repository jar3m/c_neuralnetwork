#include <stdio.h>
#include "../../neural_network/inc/nn_def.h" 

void train_network(t_neural_network *nwk,float *in, float *out) 
{
	feed_forward(nwk, in);
	back_propogate(nwk, out);
}

void predict_network(t_neural_network *nwk)
{
	float in[4];
	FILE *test_data;
	int temp_out;

	/* Get training data */
	test_data = fopen("test/src/test_data","r");
	if(!test_data)
	{
		printf("ERROR:\tfile not found\n");
		return ;
	}	
	
	while(fscanf(test_data,"%f%f%f%f%d", &in[0], &in[1], &in[2],&in[3],&temp_out) != EOF){
		feed_forward(nwk, in);
		printf("Expected output %d\n", temp_out);
		printf("predicted output %f %f %f\n",nwk->o_layer->output[0],nwk->o_layer->output[1],nwk->o_layer->output[2]);
	}

}
int main()
{
	t_layer_info h = {1,{5,1}};
	t_neural_network *nn;
	FILE *train_data;
	float in[4], out[3];
	int i,temp_out;

	nn = create_neural_network(4, h, 3);

	/* Get training data */
	train_data= fopen("test/src/train_data","r");
	if(!train_data)
	{
		printf("ERROR:\tfile not found\n");
		return -1;
	}	
	
	while(fscanf(train_data,"%f%f%f%f%d", &in[0], &in[1], &in[2], &in[3],&temp_out)!=EOF)
	{
		for(i = 0; i < 3; i++){
			out[i] = 0;
			if(i == temp_out)
				out[i] = 1;;
		}
		train_network(nn,in,out);
	}
	predict_network(nn);

	destroy_neural_network(nn);

	return 1;
}
