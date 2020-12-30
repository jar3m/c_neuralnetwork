include  Defs.make

CC=cc


LDLIBS= -lm -pg
INCLUDES+=-I $(NN_PATH)/neural_network/inc
INCLUDES+=-I $(NN_PATH)/common/inc

nn_ARCHIVE=$(NN_PATH)/neural_network/bin/neural_network.so

ifeq ($(MODULE), neural_network)
MODULE_ARCHIVES += $(nn_ARCHIVE)
endif

ifeq ($(MODULE), all)
MODULE_ARCHIVES += $(nn_ARCHIVE)
endif

export

all: $(MODULE_ARCHIVES)
	@echo "All archives ($(MODULE_ARCHIVES)) created"

$(nn_ARCHIVE) :
	make -C neural_network/ all


.PHONY: clean help

clean:
	make -C neural_network/ clean
	find ${NN_PATH} -name "*.[ao]" -exec rm -v {} \;
	find ${NN_PATH} -name "*.[out]" -exec rm -v {} \;
	

help:
	@echo "Available build options"
	@echo "#> make <OPTIONS>"
	@echo
	@echo "OPTIONS:"
	@echo "all"
	@echo "neural_network"


