#include <stdio.h>
#include <stdlib.h>
#include "test_config.h"

void train_network(t_neural_network *nwk,float *in, float *out) 
{
	feed_forward(nwk, in);
	back_propogate(nwk, out);
//	printf("Expected output %f %f\n", in[0], nwk->o_layer->output[0]);
}

void predict_network(t_neural_network *nwk, float *in, float *out, int n_out)
{
	int i;
	feed_forward(nwk, in);
	for(i = 0; i < 1; i++)
//		printf("Expected output %f %f\n", out[i], nwk->o_layer->output[i]);
		printf("Expected output %f %f %f\n", in[i], out[i], nwk->o_layer->output[i]);
}

int fetch_nn_config_params(t_nn_config *nn_cfg, FILE *cfg)
{
	int i, j;

	/* */
	fscanf(cfg, "%d", &nn_cfg->h.n_layers);
	printf("n_layers %d \n", nn_cfg->h.n_layers);
	for( i = 0; i < nn_cfg->h.n_layers;i++){
		fscanf(cfg, "%d", &nn_cfg->h.layer_size[i]);
		printf("%d ", nn_cfg->h.layer_size[i]);
	}
	printf("\n");
	fscanf(cfg, "%d%d", &nn_cfg->n_in, &nn_cfg->n_out);
	fscanf(cfg, "%d%d", &nn_cfg->n_train_set, &nn_cfg->n_test_set);
	fscanf(cfg, "%s%s", nn_cfg->train.file_name, nn_cfg->test.file_name);
	printf(" n_train_set %d n_test_set %d\n", nn_cfg->n_train_set, nn_cfg->n_test_set);
	printf("n_in : %d n_out : %d\n", nn_cfg->n_in, nn_cfg->n_out);
	printf("train : %s test : %s\n", nn_cfg->train.file_name, nn_cfg->test.file_name);

#if UNSCALED_DATA
	/* GET MEAN AND STANDARD DEVIATION OF EACH INPUT AND OUTPUT */
	nn_cfg->mean_in= malloc(nn_cfg->n_in* sizeof(float));
	nn_cfg->mean_out = malloc(nn_cfg->n_out* sizeof(float));

	nn_cfg->std_dev_in= malloc(nn_cfg->n_in * sizeof(float));
	nn_cfg->std_dev_out = malloc(nn_cfg->n_out * sizeof(float));

	for(i = 0; i < nn_cfg->n_in; i++) 
		fscanf(cfg, "%f", &nn_cfg->mean_in[i]);

	for(i = 0; i < nn_cfg->n_out; i++) 
		fscanf(cfg, "%f", &nn_cfg->mean_out[i]);

	for(i = 0; i < nn_cfg->n_in; i++) 
		fscanf(cfg, "%f", &nn_cfg->std_dev_in[i]);

	for(i = 0; i < nn_cfg->n_out; i++) 
		fscanf(cfg, "%f", &nn_cfg->std_dev_out[i]);
#endif
	/* GET NO OF TRAINING AND TESTING SETS */
	nn_cfg->train.in = malloc(nn_cfg->n_train_set * sizeof(float*));
	nn_cfg->train.out = malloc(nn_cfg->n_train_set * sizeof(float*));
	
	nn_cfg->test.in = malloc (nn_cfg->n_test_set * sizeof(float*));
	nn_cfg->test.out = malloc(nn_cfg->n_test_set * sizeof(float*));

	for(i = 0; i < nn_cfg->n_train_set; i++) {
		 nn_cfg->train.in[i]  = malloc(nn_cfg->n_in  * sizeof(float));
		 nn_cfg->train.out[i] = malloc(nn_cfg->n_out * sizeof(float));
	}
		
	for(i = 0; i < nn_cfg->n_test_set; i++) {
		 nn_cfg->test.in[i]  = malloc(nn_cfg->n_in  * sizeof(float));
		 nn_cfg->test.out[i] = malloc(nn_cfg->n_out * sizeof(float));
	}
	
	/* GENERATE THE TRAINING SET*/
	nn_cfg->train.fp = fopen(nn_cfg->train.file_name, "r");
	if(!nn_cfg->train.fp)
	{
		printf("ERROR:\ti%s: file not found\n",nn_cfg->train.file_name);
		return -1;
	}	

	for(i = 0; i < nn_cfg->n_train_set; i++) {
		for(j = 0; j < nn_cfg->n_in; j++) {
			fscanf(nn_cfg->train.fp, "%f" , &nn_cfg->train.in[i][j]); 
#if UNSCALED_DATA
			nn_cfg->train.in[i][j] =  (nn_cfg->train.in[i][j] -
					nn_cfg->mean_in[j]) / nn_cfg->std_dev_in[j];
#endif
		}
		for(j = 0; j < nn_cfg->n_out; j++) {
			fscanf(nn_cfg->train.fp, "%f" , &nn_cfg->train.out[i][j]);
#if UNSCALED_DATA
			nn_cfg->train.out[i][j] =  (nn_cfg->train.out[i][j] -
					nn_cfg->mean_out[j]) / nn_cfg->std_dev_out[j];
#endif
		}
	}
		
	/* GENERATE THE TESTING SET*/
	nn_cfg->test.fp = fopen(nn_cfg->test.file_name, "r");
	if(!nn_cfg->test.fp)
	{
		printf("ERROR:\ti%s: file not found\n",nn_cfg->test.file_name);
		return -1;
	}	

	for(i = 0; i < nn_cfg->n_test_set; i++) {
		for(j = 0; j < nn_cfg->n_in; j++) {
			fscanf(nn_cfg->test.fp, "%f" , &nn_cfg->test.in[i][j]);
#if UNSCALED_DATA
			nn_cfg->test.in[i][j] =  (nn_cfg->test.in[i][j] -
					nn_cfg->mean_in[j]) / nn_cfg->std_dev_in[j];
#endif
		}
		for(j = 0; j < nn_cfg->n_out; j++) 
			fscanf(nn_cfg->test.fp, "%f" , &nn_cfg->test.out[i][j]);
	}
	
	return 0;
}

void unload_nn_config(t_nn_config *nn_cfg, FILE *cfg)
{
	int i;

#if UNSCALED_DATA
	/* FREE MEAN AND STANDARD DEVIATION OF EACH INPUT AND OUTPUT */
	free(nn_cfg->mean_in);
	free(nn_cfg->mean_out);
	free(nn_cfg->std_dev_in);
	free(nn_cfg->std_dev_out);
	nn_cfg->mean_in = nn_cfg->mean_out = nn_cfg->std_dev_in =
		nn_cfg->std_dev_out = NULL;
#endif

	/* FREE TRAINING AND TESTING SETS */
	for(i = 0; i < nn_cfg->n_train_set; i++) {
		free(nn_cfg->train.in[i]);
		free(nn_cfg->train.out[i]);
		nn_cfg->train.in[i] = nn_cfg->train.out[i] = NULL;
	}
	free(nn_cfg->train.in );
	free(nn_cfg->train.out);
	nn_cfg->train.in = nn_cfg->train.out = NULL;
	fclose(nn_cfg->train.fp);
		
	for(i = 0; i < nn_cfg->n_test_set; i++) {
		free(nn_cfg->test.in[i]);
		free(nn_cfg->test.out[i]);
		nn_cfg->test.in[i] = nn_cfg->test.out[i] = NULL;
	}
	free(nn_cfg->test.in );
	free(nn_cfg->test.out);
	nn_cfg->test.in = nn_cfg->test.out = NULL;
	fclose(nn_cfg->test.fp);

	fclose(cfg);
}

int main(int argc, char *argv[])
{
	t_neural_network *nn;
	t_nn_config nn_config;
	int i;
	FILE *config;

	if(argc != 2 )
	{
		printf("USSAGE ERROR : ./nn test_fname");
		return -1;
	}	

	/* Get training data */
	config = fopen(argv[1],"r");
	if(!config)
	{
		printf("ERROR:\t%s: file not found\n",argv[1]);
		return -1;
	}	

	fetch_nn_config_params(&nn_config, config);
	
	nn = create_neural_network(nn_config.n_in, nn_config.h, nn_config.n_out);
	
	for(i = 0; i < nn_config.n_train_set; i++)
		train_network(nn, nn_config.train.in[i], nn_config.train.out[i]);

	for(i = 0; i < nn_config.n_test_set; i++)
		predict_network(nn, nn_config.test.in[i], nn_config.test.out[i],
				nn_config.n_out);
	
	destroy_neural_network(nn);

	unload_nn_config(&nn_config, config);
}
