#include "os.h"
#include "nn_utils.h"

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
