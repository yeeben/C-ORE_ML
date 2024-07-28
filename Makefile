IDIR = ./include
OBJDIR = ./obj

CC=gcc
CFLAGS=-w -I$(IDIR) -lm 

LIBS=-lm


apps:
	$(CC) minstNueralNetwork.c kaggleDatasetReader.c nueralBuildingBlocks.c $(CFLAGS) $(LIBS) -o obj/minstNueralNetwork.o
	./obj/minstNueralNetwork.o minst_dataset/train-images.idx3-ubyte minst_dataset/t10k-images.idx3-ubyte minst_dataset/train-labels.idx1-ubyte minst_dataset/t10k-labels.idx1-ubyte
test:
	$(CC) test.c kaggleDatasetReader.c nueralBuildingBlocks.c $(CFLAGS) $(LIBS) -o obj/test.o
	./obj/test.o

.PHONY: clean

clean:
	rm -f *.o obj/*.o
