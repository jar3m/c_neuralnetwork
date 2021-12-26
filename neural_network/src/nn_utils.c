/*! @file nn_utils.c
    @author jar3m
    @brief 
    Contains definitions of utility functions used by Neural network algos 
*/
#include "os.h"
#include "nn_utils.h"

/*! @brief  
 *  Relu activation function
 *  @param x 	- Input of x
 *  @return     - relu(x)
 * */
float relu (float x)
{
	float temp = (x >= 0)? x : 0;
	return temp;
}

/*! @brief  
 *  Relu deactivation function
 *  @param x 	- Input of x
 *  @return     - derelu(x)
 * */
float d_relu (float x)
{
	float temp = (x > 0)? 1 : 0;
	return temp;
}

/*! @brief  
 *  sigmoid activation function
 *  @param x 	- Input
 *  @return     - sigmoid(x)
 * */
float sigmoid (float x)
{
	float temp = 1 / (1 + exp(-x));
	return temp;
}

/*! @brief  
 *  sigmoid deactivation function
 *  @param x 	- Input
 *  @return     - d_sigmoid(x)
 * */
float d_sigmoid (float x)
{
	float temp = sigmoid(x);

	temp = temp * (1 - temp);
	return temp;
}

/*! @brief  
 *  linear activation function
 *  @param x 	- Input 
 *  @return     - (x)
 * */
float linear_actv(float x)
{
	return x;
}

/*! @brief  
 *  linear deactivation function
 *  @param x 	- Input 
 *  @return     - 1
 * */
float d_linear_actv(float x)
{
	return 1;
}
