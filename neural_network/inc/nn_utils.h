/*! @file nn_utils.h
    @author jar3m
    @brief 
    Contains function declarations of util functions used by neural network
*/

#pragma once

float relu (float x);
float d_relu (float x);
float sigmoid (float x);
float d_sigmoid (float x);
float linear_actv(float x);
float d_linear_actv(float x);
