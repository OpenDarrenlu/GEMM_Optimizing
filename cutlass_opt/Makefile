# ==============================================================================
# 通用配置
# ==============================================================================

# 目标输出目录
BUILD_DIR := build
# CUDA 编译器和调试器
NVCC := nvcc
CUDAGDB := cuda-gdb 

# 核心编译器标志 (不包含优化和调试级别)
NVCC_CORE_FLAGS := -std=c++17 --compiler-options -Wall

# 优化和调试配置
RELEASE_OPT_FLAGS := -O3 
DEBUG_OPT_FLAGS := -O0 -g # 关闭优化，启用调试符号

# 目标架构 (例如 Ampere)
ARCH := sm_80 
NVCC_ARCH_FLAG := -arch=$(ARCH)

# 依赖库：链接 cuBLAS
LDFLAGS := -lcublas 

# ==============================================================================
# CuTe/CUTLASS 配置
# ==============================================================================

# CuTe/CUTLASS 头文件路径检查
ifeq ($(CUTLASS_PATH),)
    $(error CUTLASS_PATH environment variable is not set. Please set it to the CUTLASS root directory.)
endif
CUTLASS_INCLUDE := -I$(CUTLASS_PATH)/include

# 所有的 Include 路径
INCLUDE_DIRS := $(CUTLASS_INCLUDE)

# ==============================================================================
# 文件查找与目标定义
# ==============================================================================

# 查找当前目录下所有的 .cu 源文件
SOURCES := $(wildcard *.cu)

# Release 目标 (build/file.out)
RELEASE_TARGETS := $(patsubst %.cu, $(BUILD_DIR)/%.out, $(SOURCES))

# Debug 目标 (build/file_debug.out)
DEBUG_TARGETS := $(patsubst %.cu, $(BUILD_DIR)/%_debug.out, $(SOURCES))

# 定义默认用于调试的目标 (第一个 Debug 文件)
CUDAGDB_TARGET := $(firstword $(DEBUG_TARGETS))

# ==============================================================================
# 规则
# ==============================================================================

.PHONY: all clean run setup debug cudagdb

# ----------------- 编译目标 -----------------

# 默认目标：编译所有 Release 版本
all: setup $(RELEASE_TARGETS)
	@echo "--------------------------------------------------------"
	@echo "所有 Release (Optimized) 程序已编译到 $(BUILD_DIR)/"
	@echo "--------------------------------------------------------"

# 编译所有 Debug 版本
debug: setup $(DEBUG_TARGETS)
	@echo "--------------------------------------------------------"
	@echo "所有 Debug (No Optimization, -g) 程序已编译到 $(BUILD_DIR)/"
	@echo "--------------------------------------------------------"

# Release 编译规则
$(BUILD_DIR)/%.out: %.cu | setup
	@echo "Compiling Release $< -> $@"
	$(NVCC) $(NVCC_CORE_FLAGS) $(RELEASE_OPT_FLAGS) $(NVCC_ARCH_FLAG) $(INCLUDE_DIRS) $< -o $@ $(LDFLAGS)

# Debug 编译规则
$(BUILD_DIR)/%_debug.out: %.cu | setup
	@echo "Compiling Debug $< -> $@"
	$(NVCC) $(NVCC_CORE_FLAGS) $(DEBUG_OPT_FLAGS) $(NVCC_ARCH_FLAG) $(INCLUDE_DIRS) $< -o $@ $(LDFLAGS)

# ----------------- 运行/调试目标 -----------------

# 运行所有 Release 版本
run: $(RELEASE_TARGETS)
	@echo "--------------------------------------------------------"
	@echo "Running all RELEASE targets in $(BUILD_DIR)..."
	@echo "--------------------------------------------------------"
	@for target in $(RELEASE_TARGETS); do \
	    echo "--- Running $$(basename $$target) ---"; \
	    $$target; \
	done

# 使用 cuda-gdb 调试第一个 Debug 目标
cudagdb: $(CUDAGDB_TARGET)
	@echo "--------------------------------------------------------"
	@echo "Starting CUDA-GDB for DEBUG target: $(notdir $(CUDAGDB_TARGET))"
	@echo "--------------------------------------------------------"
	$(CUDAGDB) $(CUDAGDB_TARGET)

# ----------------- 杂项目标 -----------------

# 目录设置：确保 build 目录存在
setup:
	@mkdir -p $(BUILD_DIR)

# 清理：删除 build 目录及其内容
clean:
	@echo "Cleaning up..."
	@rm -rf $(BUILD_DIR)