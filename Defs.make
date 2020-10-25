# #############################################################################
#
# #############################################################################
#  ======== Defs.make=============
#
#All defines to be explained in docs/testing_framework.txt

NN_REGRESS=0
NN_PREDICT=1
NN_LEARN_RATE=0.1
NN_PATH=$(PWD)

NN_SRC_DEFS= -DNN_NETA=$(NN_LEARN_RATE) -DNN_TYPE=$(NN_PREDICT) -DMAX_LAYERS=3 -DWEIGHT_UPPER_LIMIT=1 -DWEIGHT_LOWER_LIMIT=0

NN_TEST_DEFS= -DUNSCALED_INPUT=true -DUNSCALED_OUTPUT=false
MODULE=all

.show:
	@echo "NN_PATH= $(NN_PATH)"
	@ech0 "Defines= $(NN_SRC_DEFS)"
