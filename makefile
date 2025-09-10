REF  := MMult_cuBLAS_2
# REF  := REF_MMult
# NEW := 00_cublas
# NEW := 01_naive
# NEW := 02_smem
# NEW := 03_stride
# NEW := 04_align
# NEW := 05_transposeLd
# NEW := 06_pingpong
# NEW := 06_ptxPingpong
NEW := 07_wmma_naive
SMS ?= 86
BUILD_DIR := build

CC         := nvcc 
LINKER     := $(CC)
#CFLAGS     := -O0 -g -Wall
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
# CFLAGS     := -std=c++17 -O0 -g -G
CFLAGS     := -std=c++17 -O2 -Itest/utils
LDFLAGS    := -lm  -lcublas -lopenblas

TEST_OBJS  := $(addprefix $(BUILD_DIR)/, test_gemm.o $(NEW).o $(REF).o)


$(BUILD_DIR)/%.o: test/gemm/%.cpp
	$(CC) $(CFLAGS) $(GENCODE_FLAGS)  -c $< -o $@
	
$(BUILD_DIR)/%.o: test/gemm/%.cu
	$(CC) $(CFLAGS) $(GENCODE_FLAGS)  -c $< -o $@

$(BUILD_DIR)/%.o: test/utils/%.cpp
	$(CC) $(CFLAGS) $(GENCODE_FLAGS)  -c $< -o $@

$(BUILD_DIR)/%.o: kernels/sgemm/%.cu
	$(CC) $(CFLAGS) $(GENCODE_FLAGS)  -c $< -o $@

$(BUILD_DIR)/%.o: kernels/hgemm/%.cu
	$(CC) $(CFLAGS) $(GENCODE_FLAGS)  -c $< -o $@

all:
	make clean
	make $(BUILD_DIR)/test_gemm.x

$(BUILD_DIR)/test_gemm.x: $(TEST_OBJS) 
	$(LINKER) $(TEST_OBJS)  $(LDFLAGS) \
        -o  $@ 

run:
	make all
	./$(BUILD_DIR)/test_gemm.x > logs/$(NEW).log


clean:
	rm -rf build/*