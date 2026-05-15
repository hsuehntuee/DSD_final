# ==============================================================================
# Design Compiler Synthesis Script (Ultimate Optimized Version)
# Target: High-Performance & Area-Optimized RISC-V Processor
# ==============================================================================

# 1. Read RTL
# 讀取 01_RTL 底下的設計，由於剛才已經修復了 o_flush 的 Multiple Driver 錯誤，這裡將會順利通過。
read_verilog ../01_RTL/CHIP.v
current_design CHIP
link

# 2. Set Constraints (整合 SDC 規範)
# 這裡納入 SDC 的硬性規定，同時將 I/O Delay 設為 Cycle Time 的一半，完美對接 Memory 的半週期時序。
set cycle 3.5
create_clock -name CLK -period $cycle [get_ports clk]
set_fix_hold                          [get_clocks CLK]
set_dont_touch_network                [get_clocks CLK]
set_ideal_network                     [get_ports clk]
set_clock_uncertainty            0.1  [get_clocks CLK] 
set_clock_latency                0.5  [get_clocks CLK] 

set_max_fanout 6 [all_inputs] 

set_operating_conditions -min_library fast -min fast -max_library slow -max slow
set_wire_load_model -name tsmc13_wl10 -library slow  
set_drive        1     [all_inputs]
set_load         1     [all_outputs]

set t_in  [expr $cycle * 0.5]
set t_out [expr $cycle * 0.5]
set_input_delay  $t_in  -clock CLK [remove_from_collection [all_inputs] [get_ports clk]]
set_output_delay $t_out -clock CLK [all_outputs]


set_ideal_network [get_ports rst_n]
set_dont_touch_network [get_ports rst_n]


set_max_area 0

compile_ultra -area_high_effort_script

change_names -hierarchy -rules verilog

# 6. Write Outputs
write -format verilog -hierarchy -output ../03_GATE/CHIP_syn.v
write_sdf -version 2.1 ../03_GATE/CHIP_syn.sdf
write_sdc CHIP_syn.sdc

# 7. Generate Reports
report_area > area_report.txt
report_timing > timing_report.txt
report_power > power_report.txt
report_qor > qor_report.txt

echo "====================================================="
echo "✅ Synthesis Done! Netlist saved to 03_GATE/CHIP_syn.v"
echo "⚠️  Action Required: Open 'timing_report.txt' and search for 'slack'."
echo "   Make sure the slack is MET (positive). If it's VIOLATED,"
echo "   you may need to increase the 'set cycle 3.5' slightly."
echo "====================================================="
exit