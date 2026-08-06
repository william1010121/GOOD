# =====================================================================
# HiPAC 隱藏題1 — SHA-256 部分碰撞搜尋
#
#   make              編譯（自動偵測 GPU 架構）
#   make ARCH=sm_90   手動指定架構
#   make run          用預設參數跑一次
#   make info         只印環境診斷，不編譯
#   make clean        清除產出
#
# 架構怎麼決定（依序）：
#   1. 你有指定 ARCH=  → 就用你指定的
#   2. nvidia-smi 問得到本機 GPU，且這個 nvcc 支援 → 用該架構（最快）
#   3. 都問不到（例如在沒有 GPU 的登入節點編譯）
#      → 編「多架構通用版」，涵蓋這個 nvcc 支援的常見架構，
#        換到任何一張卡上都能跑，代價是編譯久一點。
# =====================================================================

NVCC ?= nvcc

# ---- 環境探測（parse 階段就算好，後面只是拿來用）----
HAVE_NVCC  := $(shell command -v $(NVCC) 2>/dev/null)

# 這個 nvcc 支援哪些架構。不要用 CUDA 版本號去猜 ——
# 新版 CUDA 會「移除」舊架構（例如 CUDA 13 拿掉了 sm_70 / Volta），
# 版本高不代表支援多，只有問 nvcc 本人才準。
NVCC_CODES := $(shell $(NVCC) --list-gpu-code 2>/dev/null | tr -d ' ')

# 本機第一張 GPU 的 compute capability：9.0 -> 90
# grep 是必要的：舊驅動不支援 compute_cap 這個欄位時，
# nvidia-smi 會印一整句錯誤訊息而不是數字，不擋掉會被當成架構名。
GPU_CC := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
            | head -1 | tr -d ' .' | grep -E '^[0-9]+$$')
GPU_SM := $(if $(GPU_CC),sm_$(GPU_CC),)

# 通用版涵蓋的候選架構（V100 / T4 / A100 / RTX30 / RTX40 / H100·H200 / B200）
CANDIDATES := sm_70 sm_75 sm_80 sm_86 sm_89 sm_90 sm_100 sm_120
FAT_LIST   := $(filter $(CANDIDATES),$(NVCC_CODES))
FAT_FLAGS  := $(foreach a,$(FAT_LIST),-gencode arch=compute_$(patsubst sm_%,%,$(a)),code=$(a))

# ---- 決定這次要用的架構旗標 ----
ifdef ARCH
  ARCH_FLAGS := -arch=$(ARCH)
  ARCH_NOTE  := 使用指定架構 $(ARCH)
else ifneq ($(filter $(GPU_SM),$(NVCC_CODES)),)
  ARCH_FLAGS := -arch=$(GPU_SM)
  ARCH_NOTE  := 偵測到本機 GPU 架構 $(GPU_SM)
else
  ARCH_FLAGS := $(FAT_FLAGS)
  ARCH_NOTE  := 未鎖定單一架構，編多架構通用版（$(FAT_LIST)）
endif

# 這張卡的架構，這個 nvcc 不支援 —— 單獨標出來，錯誤訊息要講清楚
ifndef ARCH
  ifneq ($(GPU_SM),)
    ifeq ($(filter $(GPU_SM),$(NVCC_CODES)),)
      ARCH_MISMATCH := 1
    endif
  endif
endif

# 範本沒用到 OpenMP，但仍預先開啟：
# 「多 GPU」是本題的優化方向之一，最自然的寫法就是 OpenMP。
# 先開好，免得參賽者加了 #pragma omp 之後卡在連結錯誤。
CFLAGS := -O3 -Xcompiler -fopenmp

# 用 ?= 是為了讓「出題者專用/Makefile」能沿用上面整套架構偵測邏輯
# （它只覆寫這兩個變數再 include 本檔），偵測方式才不會兩邊各改各的走鐘。
TARGET  ?= collision
SOURCES ?= collision.cu

# =====================================================================

all: $(TARGET)

$(TARGET): $(SOURCES) | preflight
	@echo ">>> 編譯中（$(ARCH_NOTE)）…"
	$(NVCC) $(CFLAGS) $(ARCH_FLAGS) $(SOURCES) -o $(TARGET)
	@echo ""
	@echo ">>> 編譯完成。執行方式："
	@echo "      ./$(TARGET) [prefix]"
	@echo "    smoke 例： ./$(TARGET) --smoke hipac_demo 20000000"
	@echo ""

# 編譯前先擋掉三種會讓人看不懂錯誤訊息的情況
preflight:
	@if [ -z "$(HAVE_NVCC)" ]; then \
	  echo ""; \
	  echo "!! 找不到 nvcc（CUDA 編譯器）。"; \
	  echo "   請確認已載入 CUDA 模組，例如： module load cuda"; \
	  echo "   或指定完整路徑： make NVCC=/usr/local/cuda/bin/nvcc"; \
	  echo ""; \
	  exit 1; \
	fi
	@if [ -n "$(ARCH_MISMATCH)" ]; then \
	  echo ""; \
	  echo "!! 本機 GPU 是 $(GPU_SM)，但這個 nvcc 不支援這個架構。"; \
	  echo "   （新版 CUDA 會移除舊架構，例如 CUDA 13 已不支援 sm_70/Volta）"; \
	  echo ""; \
	  echo "   這個 nvcc 支援： $(FAT_LIST)"; \
	  echo "   請換一個 CUDA 版本，或換到支援的 GPU 節點。"; \
	  echo ""; \
	  exit 1; \
	fi
	@if [ -z "$(strip $(ARCH_FLAGS))" ]; then \
	  echo ""; \
	  echo "!! 無法決定 GPU 架構，也列不出 nvcc 支援的架構。"; \
	  echo "   請手動指定，例如： make ARCH=sm_90   （H200/H100=sm_90, A100=sm_80）"; \
	  echo ""; \
	  exit 1; \
	fi

# 編不過的時候，先跑這個把環境貼出來
info:
	@echo "nvcc           : $(if $(HAVE_NVCC),$(HAVE_NVCC),(找不到))"
	@echo "nvcc 支援架構  : $(if $(NVCC_CODES),$(NVCC_CODES),(列不出來))"
	@echo "偵測到的 GPU   : $(if $(GPU_SM),$(GPU_SM),(偵測不到))"
	@echo "本次編譯策略   : $(ARCH_NOTE)"
	@echo "實際旗標       : $(CFLAGS) $(ARCH_FLAGS)"

run: $(TARGET)
	./$(TARGET) --seconds 60 HiPAC2026crypto

bench: $(TARGET)
	./$(TARGET) --bench HiPAC2026crypto

clean:
	rm -f $(TARGET) solution_*.csv

.PHONY: all preflight info run bench clean
