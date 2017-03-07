IDIR = ../include
CC=nvcc
CFLAGS=-I$(IDIR)

ODIR=obj
dummy_build_folder := $(shell mkdir -p $(ODIR))

LIBS=-lm

_DEPS = idw.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = demo.o idw.o 
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))


$(ODIR)/%.o: %.cu $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

demo: $(OBJ)
	nvcc -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~ 
