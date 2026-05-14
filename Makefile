# =========================================================
# DSD Auto Regression Makefile
# =========================================================

VCS = vcs

VCS_FLAGS = -full64 \
            -f 01_rtl.f \
            +v2k \
            -debug_access+all \
            -R

# =========================================================
# Hard-coded pattern list
# =========================================================

PATTERNS = \
	noHazard \
	hasHazard \
	BrPred \
	Scaling \
	compression \
	compression_uncompressed \
	QSort_uncompressed \
	QSort \
	Conv \
	Conv_uncompressed \
	Mul \
	LFSR_HIST \
	LFSR_HIST_short

# =========================================================
# Default target
# =========================================================

all: regression summary

# =========================================================
# Regression
# =========================================================

regression:
	@echo ""
	@echo "========================================"
	@echo "Running Full Regression"
	@echo "========================================"

	@for p in $(PATTERNS); do \
		echo ""; \
		echo "----------------------------------------"; \
		echo "Running $$p"; \
		echo "----------------------------------------"; \
		$(VCS) $(VCS_FLAGS) +define+$$p | tee $$p.log; \
		grep "simulation time" $$p.log | tail -1 | awk '{print $$5}' > $$p.time; \
	done

# =========================================================
# Summary (UPDATED: PASS/FAIL detection)
# =========================================================

summary:
	@echo ""
	@echo "========================================"
	@echo "FINAL SUMMARY"
	@echo "========================================"
	@echo ""

	@for p in $(PATTERNS); do \
		time_file=$$p.time; \
		log_file=$$p.log; \
		time_val="N/A"; \
		status="UNKNOWN"; \
		\
		if [ -f $$time_file ]; then \
			time_val=$$(cat $$time_file); \
		fi; \
		\
		if grep -q "FAIL" $$log_file; then \
			status="FAIL"; \
		elif grep -q "CONGRATULATIONS" $$log_file; then \
			status="PASS"; \
		else \
			status="UNKNOWN"; \
		fi; \
		\
		printf "%-25s %-15s %s\n" $$p $$time_val $$status; \
	done

	@echo ""
	@echo "Unit: simulation time"
	@echo ""

# =========================================================
# Clean
# =========================================================

clean:
	-rm -rf csrc
	-rm -rf simv.daidir
	-rm -f simv
	-rm -f ucli.key
	-rm -f *.log
	-rm -f *.time
	-rm -rf *.fsdb
	-rm -rf novas*
	-rm -rf nWaveLog

# =========================================================
# Help
# =========================================================

help:
	@echo ""
	@echo "Usage:"
	@echo "  make"
	@echo "  make regression"
	@echo "  make clean"
	@echo ""

.PHONY: all regression summary clean help