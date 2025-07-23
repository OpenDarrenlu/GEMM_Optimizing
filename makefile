# REF  := MMult_cuBLAS_1
REF  := REF_MMult
# NEW := 00_cublas
# NEW := 01_naive
# NEW := 02_smem
# NEW := 03_stride
# NEW := 04_align
# NEW := 05_transposeLd
NEW := 06_pingpong
# NEW := 06_ptxPingpong
SMS ?= 86
BUILD_DIR := build

CC         := nvcc 
LINKER     := $(CC)
#CFLAGS     := -O0 -g -Wall
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
# CFLAGS     := -std=c++17 -O0 -g -G
CFLAGS     := -std=c++17 -O2 -Itest/utils
LDFLAGS    := -lm  -lcublas -lopenblas

UTIL       := $(addprefix $(BUILD_DIR)/, copy_matrix.o \
              compare_matrices.o \
              random_matrix.o \
              print_matrix.o)

TEST_OBJS  := $(addprefix $(BUILD_DIR)/, test_sgemm.o $(NEW).o $(REF).o)


$(BUILD_DIR)/%.o: test/sgemm/%.cpp
	$(CC) $(CFLAGS) $(GENCODE_FLAGS)  -c $< -o $@
	
$(BUILD_DIR)/%.o: test/sgemm/%.cu 
	$(CC) $(CFLAGS) $(GENCODE_FLAGS)  -c $< -o $@

$(BUILD_DIR)/%.o: test/utils/%.cpp
	$(CC) $(CFLAGS) $(GENCODE_FLAGS)  -c $< -o $@

$(BUILD_DIR)/%.o: kernels/sgemm/%.cu
	$(CC) $(CFLAGS) $(GENCODE_FLAGS)  -c $< -o $@

all:
	make clean
	make $(BUILD_DIR)/test_sgemm.x

$(BUILD_DIR)/test_sgemm.x: $(TEST_OBJS) $(UTIL)
	$(LINKER) $(TEST_OBJS) $(UTIL) $(LDFLAGS) \
        -o  $@ 

run:
	make all
	./$(BUILD_DIR)/test_sgemm.x > logs/$(NEW).log


clean:
	rm -rf build/*