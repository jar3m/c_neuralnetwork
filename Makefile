include  Defs.make

CC=cc


LDLIBS= -lm -pg
INCLUDES+=-I $(NN_PATH)/neural_network/inc

nn_ARCHIVE=$(NN_PATH)/neural_network/bin/neural_network.a

ifeq ($(MODULE), neural_network)
MODULE_ARCHIVES += $(nn_ARCHIVE)
endif

ifeq ($(MODULE), all)
MODULE_ARCHIVES += $(nn_ARCHIVE)
endif

export

all: $(MODULE_ARCHIVES)
	@echo "All archives ($(MODULE_ARCHIVES)) created"
	$(CC) $(INCLUDES) $^ -o prometheus $(MODULE_ARCHIVES) $(LDLIBS) $(CFLAGS)

$(nn_ARCHIVE) :
	make -C neural_network/ all


.PHONY: clean help

clean:
	make -C neural_network/ clean
	make -C test/ clean
	find ${NN_PATH} -name "*.[ao]" -exec rm -v {} \;
	find ${NN_PATH} -name "*.[out]" -exec rm -v {} \;
	rm -rf ${NN_PATH}/prometheus

help:
	@echo "Available build options"
	@echo "#> make <OPTIONS>"
	@echo
	@echo "OPTIONS:"
	@echo "all"
	@echo "neural_network"
	@echo "test"

