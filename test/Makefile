export CC
export INCLUDES
export NN_PATH
export TEST_FILE_DIRECTORY

include $(NN_PATH)/Defs.make


LDLIBS += -pg

INCLUDES += -I $(NN_PATH)/test/inc/

CFLAGS += -g -Wall $(NN_TEST_DEFS)

test_SRC=src/test.c
test_OBJ=bin/test.o

test_ARCHIVE=$(NN_PATH)/test/bin/test.a

OBJS=$(test_OBJ)

$(test_OBJ): $(test_SRC)
	echo "----- $(INCLUDES)"
	$(CC) $(INCLUDES) -c $^ -o $@ $(CFLAGS)

all: $(OBJS)
	ar rcs $(test_ARCHIVE) $(OBJS)


.PHONY: clean

clean:
	rm -f $(test_ARCHIVE) $(OBJS)
