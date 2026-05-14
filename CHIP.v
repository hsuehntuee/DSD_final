// -----------------------------------------------------------------------------
// DSD Final Project: Pipelined RISC-V Processor with I/D Caches, RV32M, RV32C
// -----------------------------------------------------------------------------

module CHIP (
    input               clk,
    input               rst_n,
    // ---------- for slow_memD ------------
    output              mem_read_D,
    output              mem_write_D,
    output      [31:4]  mem_addr_D,
    output      [127:0] mem_wdata_D,
    input       [127:0] mem_rdata_D,
    input               mem_ready_D,
    // ---------- for slow_memI ------------
    output              mem_read_I,
    output              mem_write_I,
    output      [31:4]  mem_addr_I,
    output      [127:0] mem_wdata_I,
    input       [127:0] mem_rdata_I,
    input               mem_ready_I,
    // ---------- for TestBed --------------
    output              o_done
);

    wire [31:0] icache_inst, icache_addr;
    wire        icache_valid, icache_stall;

    wire [31:0] dcache_rdata, dcache_wdata, dcache_addr;
    wire        dcache_ren, dcache_wen, dcache_valid, dcache_stall;
    wire        flush_req;

    RISCV_CORE core (
        .clk            (clk),
        .rst_n          (rst_n),
        .i_icache_inst  (icache_inst),
        .i_icache_valid (icache_valid),
        .i_icache_stall (icache_stall),
        .o_icache_addr  (icache_addr),
        .i_dcache_rdata (dcache_rdata),
        .i_dcache_valid (dcache_valid),
        .i_dcache_stall (dcache_stall),
        .o_dcache_wdata (dcache_wdata),
        .o_dcache_addr  (dcache_addr),
        .o_dcache_ren   (dcache_ren),
        .o_dcache_wen   (dcache_wen),
        .o_flush        (flush_req)
    );

    ICACHE icache (
        .clk            (clk),
        .rst_n          (rst_n),
        .i_cpu_addr     (icache_addr),
        .o_cpu_inst     (icache_inst),
        .o_cpu_valid    (icache_valid),
        .o_cpu_stall    (icache_stall),
        .o_mem_read     (mem_read_I),
        .o_mem_write    (mem_write_I),
        .o_mem_addr     (mem_addr_I),
        .o_mem_wdata    (mem_wdata_I),
        .i_mem_rdata    (mem_rdata_I),
        .i_mem_ready    (mem_ready_I)
    );

    DCACHE dcache (
        .clk            (clk),
        .rst_n          (rst_n),
        .i_cpu_ren      (dcache_ren),
        .i_cpu_wen      (dcache_wen),
        .i_cpu_addr     (dcache_addr),
        .i_cpu_wdata    (dcache_wdata),
        .o_cpu_rdata    (dcache_rdata),
        .o_cpu_valid    (dcache_valid),
        .o_cpu_stall    (dcache_stall),
        .i_flush        (flush_req),
        .o_flush_done   (o_done),
        .o_mem_read     (mem_read_D),
        .o_mem_write    (mem_write_D),
        .o_mem_addr     (mem_addr_D),
        .o_mem_wdata    (mem_wdata_D),
        .i_mem_rdata    (mem_rdata_D),
        .i_mem_ready    (mem_ready_D)
    );
endmodule

// =============================================================================
// RISC-V 5-Stage Pipeline Core
// =============================================================================
module RISCV_CORE (
    input         clk,
    input         rst_n,
    input  [31:0] i_icache_inst,
    input         i_icache_valid,
    input         i_icache_stall,
    output [31:0] o_icache_addr,
    input  [31:0] i_dcache_rdata,
    input         i_dcache_valid,
    input         i_dcache_stall,
    output [31:0] o_dcache_wdata,
    output [31:0] o_dcache_addr,
    output        o_dcache_ren,
    output        o_dcache_wen,
    output        o_flush
);

    wire [31:0] dcache_rdata_swapped = {i_dcache_rdata[7:0], i_dcache_rdata[15:8], i_dcache_rdata[23:16], i_dcache_rdata[31:24]};

    reg [31:0] IF_ID_pc, IF_ID_inst;
    reg        IF_ID_is_rvc;
    reg [31:0] ID_EX_pc, ID_EX_rs1_data, ID_EX_rs2_data, ID_EX_imm;
    reg [4:0]  ID_EX_rs1, ID_EX_rs2, ID_EX_rd;
    reg [3:0]  ID_EX_alu_op;
    reg        ID_EX_reg_write, ID_EX_mem_read, ID_EX_mem_write;
    reg        ID_EX_alu_src, ID_EX_mem_to_reg, ID_EX_pc_to_alu;
    reg        ID_EX_branch, ID_EX_jump, ID_EX_jalr, ID_EX_is_flush;
    reg        ID_EX_is_mult, ID_EX_is_rvc;

    reg [31:0] EX_MEM_alu_out, EX_MEM_rs2_data;
    reg [4:0]  EX_MEM_rd;
    reg        EX_MEM_reg_write, EX_MEM_mem_read, EX_MEM_mem_write;
    reg        EX_MEM_mem_to_reg, EX_MEM_is_flush;

    reg [31:0] MEM_WB_alu_out, MEM_WB_mem_rdata;
    reg [4:0]  MEM_WB_rd;
    reg        MEM_WB_reg_write, MEM_WB_mem_to_reg;

    wire load_use_stall;
    wire branch_taken;
    wire [31:0] branch_target;
    wire [31:0] wb_data = (MEM_WB_mem_to_reg) ? MEM_WB_mem_rdata : MEM_WB_alu_out;

    // -------------------------------------------------------------------------
    // IF Stage
    // -------------------------------------------------------------------------
    reg  [31:0] pc;
    reg state_if;
    parameter IF_NORMAL = 1'b0, IF_CROSS = 1'b1;
    reg [15:0] cross_word_lower;

    wire [31:0] fetch_pc = (state_if == IF_CROSS) ? {pc[31:2], 2'b00} + 4 : {pc[31:2], 2'b00};
    assign o_icache_addr = fetch_pc;
    
    wire [31:0] inst_swapped = {i_icache_inst[7:0], i_icache_inst[15:8], i_icache_inst[23:16], i_icache_inst[31:24]};
    
    wire pc_unaligned = pc[1];
    wire [15:0] inst_half = pc_unaligned ? inst_swapped[31:16] : inst_swapped[15:0];
    wire is_32bit_inst = (inst_half[1:0] == 2'b11);
    
    wire if_stall_req = (state_if == IF_NORMAL) && pc_unaligned && is_32bit_inst && i_icache_valid;
    
    wire [31:0] if_inst_raw = (state_if == IF_CROSS) ? {inst_swapped[15:0], cross_word_lower} :
                              (is_32bit_inst ? inst_swapped : {16'b0, inst_half});
                              
    wire [31:0] expanded_inst;
    RVC_EXPANDER rvc_exp(
        .inst_in(if_inst_raw),
        .inst_out(expanded_inst)
    );

    wire actual_is_32bit = (state_if == IF_CROSS) ? 1'b1 : is_32bit_inst;
    wire [31:0] pc_next_norm = pc + (actual_is_32bit ? 4 : 2);
    wire actual_is_rvc = !actual_is_32bit;

    wire mult_stall;
    wire mem_stall = i_icache_stall | i_dcache_stall;
    wire global_stall = mem_stall | mult_stall;
    wire front_stall = load_use_stall | if_stall_req;

    always @(posedge clk) begin
        if (!rst_n) begin
            state_if <= IF_NORMAL;
            cross_word_lower <= 16'b0;
        end else begin
            if (branch_taken) begin
                state_if <= IF_NORMAL;
            end else if (state_if == IF_NORMAL && if_stall_req && !mem_stall) begin
                state_if <= IF_CROSS;
                cross_word_lower <= inst_swapped[31:16];
            end else if (state_if == IF_CROSS && !global_stall && !front_stall) begin
                state_if <= IF_NORMAL;
            end
        end
    end

    always @(posedge clk) begin
        if (!rst_n) pc <= 32'b0;
        else if (!global_stall) begin
            if (branch_taken) pc <= branch_target;
            else if (!front_stall) pc <= pc_next_norm;
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            IF_ID_pc   <= 32'b0;
            IF_ID_inst <= 32'h00000013;
            IF_ID_is_rvc <= 0;
        end else if (!global_stall) begin
            if (branch_taken) begin
                IF_ID_pc   <= 32'b0;
                IF_ID_inst <= 32'h00000013;
                IF_ID_is_rvc <= 0;
            end else if (!front_stall) begin
                IF_ID_pc   <= pc;
                IF_ID_inst <= expanded_inst;
                IF_ID_is_rvc <= actual_is_rvc;
            end
        end
    end

    // -------------------------------------------------------------------------
    // ID Stage
    // -------------------------------------------------------------------------
    wire [6:0]  opcode = IF_ID_inst[6:0];
    wire [2:0]  funct3 = IF_ID_inst[14:12];
    wire [6:0]  funct7 = IF_ID_inst[31:25];
    wire [4:0]  rs1    = IF_ID_inst[19:15];
    wire [4:0]  rs2    = IF_ID_inst[24:20];
    wire [4:0]  rd     = IF_ID_inst[11:7];

    reg [31:0] regs [0:31];

    wire [31:0] rs1_data = (rs1 == 5'b0) ? 32'b0 : (MEM_WB_reg_write && MEM_WB_rd == rs1) ? wb_data : regs[rs1];
    wire [31:0] rs2_data = (rs2 == 5'b0) ? 32'b0 : (MEM_WB_reg_write && MEM_WB_rd == rs2) ? wb_data : regs[rs2];
    
    reg [31:0] imm;
    always @(*) begin
        case(opcode)
            7'b0010011, 7'b0000011, 7'b1100111: imm = {{20{IF_ID_inst[31]}}, IF_ID_inst[31:20]}; // I-Type
            7'b0100011: imm = {{20{IF_ID_inst[31]}}, IF_ID_inst[31:25], IF_ID_inst[11:7]};       // S-Type
            7'b1100011: imm = {{20{IF_ID_inst[31]}}, IF_ID_inst[7], IF_ID_inst[30:25], IF_ID_inst[11:8], 1'b0}; // B-Type
            7'b1101111: imm = {{12{IF_ID_inst[31]}}, IF_ID_inst[19:12], IF_ID_inst[20], IF_ID_inst[30:21], 1'b0}; // J-Type
            7'b0110111, 7'b0010111: imm = {IF_ID_inst[31:12], 12'b0}; // U-Type
            default: imm = 32'b0;
        endcase
    end

    reg ctrl_reg_write, ctrl_mem_read, ctrl_mem_write, ctrl_alu_src, ctrl_mem_to_reg;
    reg ctrl_branch, ctrl_jump, ctrl_jalr, ctrl_pc_to_alu, ctrl_is_mult;
    reg [3:0] ctrl_alu_op; 

    wire is_flush = (IF_ID_inst == 32'h00202007);

    always @(*) begin
        ctrl_reg_write = 0; ctrl_mem_read = 0; ctrl_mem_write = 0; 
        ctrl_alu_src = 0; ctrl_mem_to_reg = 0; ctrl_branch = 0; 
        ctrl_jump = 0; ctrl_jalr = 0; ctrl_alu_op = 4'b0000; ctrl_pc_to_alu = 0; ctrl_is_mult = 0;
        case(opcode)
            7'b0110011: begin 
                ctrl_reg_write = 1;
                if(funct7 == 7'b0000000) begin
                    if(funct3 == 3'b000) ctrl_alu_op = 4'd0;
                    else if(funct3 == 3'b111) ctrl_alu_op = 4'd2;
                    else if(funct3 == 3'b110) ctrl_alu_op = 4'd3;
                    else if(funct3 == 3'b100) ctrl_alu_op = 4'd4;
                    else if(funct3 == 3'b001) ctrl_alu_op = 4'd5;
                    else if(funct3 == 3'b101) ctrl_alu_op = 4'd6;
                    else if(funct3 == 3'b010) ctrl_alu_op = 4'd8;
                end else if (funct7 == 7'b0100000) begin
                    if(funct3 == 3'b000) ctrl_alu_op = 4'd1;
                    else if(funct3 == 3'b101) ctrl_alu_op = 4'd7;
                end else if (funct7 == 7'b0000001) begin // M-Type
                    ctrl_is_mult = 1;
                    if(funct3 == 3'b000) ctrl_alu_op = 4'd12; // MUL
                    else if(funct3 == 3'b001) ctrl_alu_op = 4'd13; // MULH
                    else if(funct3 == 3'b010) ctrl_alu_op = 4'd14; // MULHSU
                    else if(funct3 == 3'b011) ctrl_alu_op = 4'd15; // MULHU
                end
            end
            7'b0010011: begin // I-Type
                ctrl_reg_write = 1; ctrl_alu_src = 1;
                if(funct3 == 3'b000) ctrl_alu_op = 4'd0;
                else if(funct3 == 3'b111) ctrl_alu_op = 4'd2;
                else if(funct3 == 3'b110) ctrl_alu_op = 4'd3;
                else if(funct3 == 3'b100) ctrl_alu_op = 4'd4;
                else if(funct3 == 3'b001) ctrl_alu_op = 4'd5;
                else if(funct3 == 3'b101 && funct7 == 7'b0000000) ctrl_alu_op = 4'd6;
                else if(funct3 == 3'b101 && funct7 == 7'b0100000) ctrl_alu_op = 4'd7;
                else if(funct3 == 3'b010) ctrl_alu_op = 4'd8;
            end
            7'b0000011: begin // LW
                ctrl_reg_write = 1; ctrl_alu_src = 1; ctrl_mem_read = 1; ctrl_mem_to_reg = 1; ctrl_alu_op = 4'd0;
            end
            7'b0100011: begin // SW
                ctrl_alu_src = 1; ctrl_mem_write = 1; ctrl_alu_op = 4'd0;
            end
            7'b1100011: begin // Branch
                ctrl_branch = 1;
                if(funct3 == 3'b000) ctrl_alu_op = 4'd9;
                else if(funct3 == 3'b001) ctrl_alu_op = 4'd10;
            end
            7'b1101111: begin // JAL
                ctrl_jump = 1; ctrl_reg_write = 1;
            end
            7'b1100111: begin // JALR
                ctrl_jalr = 1; ctrl_reg_write = 1; ctrl_alu_src = 1; ctrl_alu_op = 4'd0;
            end
            7'b0110111: begin // LUI
                ctrl_reg_write = 1; ctrl_alu_src = 1; ctrl_alu_op = 4'd11; 
            end
            7'b0010111: begin // AUIPC
                ctrl_reg_write = 1; ctrl_alu_src = 1; ctrl_alu_op = 4'd0; ctrl_pc_to_alu = 1;
            end
        endcase
    end

    wire rs1_valid = (opcode == 7'b0110011) || (opcode == 7'b0010011) || (opcode == 7'b0000011) || 
                     (opcode == 7'b0100011) || (opcode == 7'b1100011) || (opcode == 7'b1100111);
    wire rs2_valid = (opcode == 7'b0110011) || (opcode == 7'b0100011) || (opcode == 7'b1100011);

    assign load_use_stall = (ID_EX_mem_read && ID_EX_rd != 0 &&
                            ((rs1_valid && ID_EX_rd == rs1) || (rs2_valid && ID_EX_rd == rs2)));

    always @(posedge clk) begin
        if (!rst_n) begin
            ID_EX_reg_write <= 0; ID_EX_mem_read <= 0; ID_EX_mem_write <= 0;
            ID_EX_branch <= 0; ID_EX_jump <= 0; ID_EX_jalr <= 0;
            ID_EX_is_flush <= 0; ID_EX_rd <= 0; ID_EX_pc_to_alu <= 0;
            ID_EX_is_mult <= 0; ID_EX_is_rvc <= 0;
        end else if (!global_stall) begin
            if (branch_taken || front_stall) begin
                ID_EX_reg_write <= 0; ID_EX_mem_read <= 0; ID_EX_mem_write <= 0;
                ID_EX_branch <= 0; ID_EX_jump <= 0; ID_EX_jalr <= 0;
                ID_EX_is_flush <= 0; ID_EX_rd <= 0; ID_EX_pc_to_alu <= 0;
                ID_EX_is_mult <= 0; ID_EX_is_rvc <= 0;
            end else begin
                ID_EX_pc       <= IF_ID_pc;
                ID_EX_rs1_data <= rs1_data;
                ID_EX_rs2_data <= rs2_data;
                ID_EX_imm      <= imm;
                ID_EX_rd       <= rd;
                ID_EX_rs1      <= rs1;
                ID_EX_rs2      <= rs2;
                ID_EX_alu_op   <= ctrl_alu_op;
                ID_EX_reg_write<= ctrl_reg_write;
                ID_EX_mem_read <= ctrl_mem_read;
                ID_EX_mem_write<= ctrl_mem_write;
                ID_EX_alu_src  <= ctrl_alu_src;
                ID_EX_mem_to_reg<= ctrl_mem_to_reg;
                ID_EX_branch   <= ctrl_branch;
                ID_EX_jump     <= ctrl_jump;
                ID_EX_jalr     <= ctrl_jalr;
                ID_EX_is_flush <= is_flush;
                ID_EX_pc_to_alu<= ctrl_pc_to_alu;
                ID_EX_is_mult  <= ctrl_is_mult;
                ID_EX_is_rvc   <= IF_ID_is_rvc;
            end
        end
    end

    // -------------------------------------------------------------------------
    // EX Stage
    // -------------------------------------------------------------------------
    wire [1:0] forward_A = (EX_MEM_reg_write && EX_MEM_rd != 0 && EX_MEM_rd == ID_EX_rs1) ? 2'b10 :
                           (MEM_WB_reg_write && MEM_WB_rd != 0 && MEM_WB_rd == ID_EX_rs1) ? 2'b01 : 2'b00;
                           
    wire [1:0] forward_B = (EX_MEM_reg_write && EX_MEM_rd != 0 && EX_MEM_rd == ID_EX_rs2) ? 2'b10 :
                           (MEM_WB_reg_write && MEM_WB_rd != 0 && MEM_WB_rd == ID_EX_rs2) ? 2'b01 : 2'b00;

    wire [31:0] ex_mem_fwd_data = (EX_MEM_mem_read) ? dcache_rdata_swapped : EX_MEM_alu_out;
    
    wire [31:0] alu_in1_base = (forward_A == 2'b10) ? ex_mem_fwd_data : (forward_A == 2'b01) ? wb_data : ID_EX_rs1_data;
    wire [31:0] alu_in1 = (ID_EX_pc_to_alu) ? ID_EX_pc : alu_in1_base;
    
    wire [31:0] fwd_in2 = (forward_B == 2'b10) ? ex_mem_fwd_data : (forward_B == 2'b01) ? wb_data : ID_EX_rs2_data;
    wire [31:0] alu_in2 = (ID_EX_alu_src) ? ID_EX_imm : fwd_in2;

    reg [31:0] alu_result;
    reg        alu_zero;
    always @(*) begin
        alu_zero = 0;
        case(ID_EX_alu_op)
            4'd0: alu_result = alu_in1 + alu_in2;             
            4'd1: alu_result = alu_in1 - alu_in2;             
            4'd2: alu_result = alu_in1 & alu_in2;             
            4'd3: alu_result = alu_in1 | alu_in2;             
            4'd4: alu_result = alu_in1 ^ alu_in2;             
            4'd5: alu_result = alu_in1 << alu_in2[4:0];       
            4'd6: alu_result = alu_in1 >> alu_in2[4:0];       
            4'd7: alu_result = $signed(alu_in1) >>> alu_in2[4:0]; 
            4'd8: alu_result = ($signed(alu_in1) < $signed(alu_in2)) ? 32'd1 : 32'd0; 
            4'd9:  begin alu_result = alu_in1 - alu_in2; alu_zero = (alu_result == 0); end 
            4'd10: begin alu_result = alu_in1 - alu_in2; alu_zero = (alu_result != 0); end 
            4'd11: alu_result = alu_in2; 
            default: alu_result = 32'b0;
        endcase
    end

    // Multiplier Logic (1 cycle stall = 2 pipeline stages total)
    reg mult_busy;
    reg [63:0] mult_res;
    wire signed [32:0] a_ext = (ID_EX_alu_op == 4'd13 || ID_EX_alu_op == 4'd14) ? {alu_in1[31], alu_in1} : {1'b0, alu_in1};
    wire signed [32:0] b_ext = (ID_EX_alu_op == 4'd13) ? {alu_in2[31], alu_in2} : {1'b0, alu_in2};
    wire signed [65:0] mul_tmp = a_ext * b_ext;

    always @(posedge clk) begin
        if (!rst_n) begin
            mult_busy <= 0;
            mult_res <= 0;
        end else if (ID_EX_is_mult && !mult_busy && !mem_stall) begin
            mult_busy <= 1;
            mult_res <= mul_tmp[63:0];
        end else if (mult_busy && !mem_stall) begin
            mult_busy <= 0;
        end
    end
    assign mult_stall = ID_EX_is_mult && !mult_busy;

    assign branch_target = (ID_EX_jalr) ? ((alu_in1_base + ID_EX_imm) & ~32'd1) : (ID_EX_pc + ID_EX_imm);
    assign branch_taken  = ID_EX_jump | ID_EX_jalr | (ID_EX_branch & alu_zero);

    always @(posedge clk) begin
        if (!rst_n) begin
            EX_MEM_reg_write <= 0; EX_MEM_mem_read <= 0; EX_MEM_mem_write <= 0; EX_MEM_is_flush <= 0;
            EX_MEM_rd <= 0;
        end else if (!global_stall) begin
            EX_MEM_reg_write <= ID_EX_reg_write;
            EX_MEM_mem_read  <= ID_EX_mem_read;
            EX_MEM_mem_write <= ID_EX_mem_write;
            EX_MEM_mem_to_reg<= ID_EX_mem_to_reg;
            EX_MEM_rd        <= ID_EX_rd;
            EX_MEM_alu_out   <= (ID_EX_is_mult) ? ((ID_EX_alu_op == 4'd12) ? mult_res[31:0] : mult_res[63:32]) :
                               (ID_EX_jump || ID_EX_jalr) ? (ID_EX_pc + (ID_EX_is_rvc ? 2 : 4)) : alu_result;
            EX_MEM_rs2_data  <= fwd_in2;
            EX_MEM_is_flush  <= ID_EX_is_flush;
        end
    end

    // -------------------------------------------------------------------------
    // MEM Stage
    // -------------------------------------------------------------------------
    assign o_dcache_ren   = EX_MEM_mem_read;
    assign o_dcache_wen   = EX_MEM_mem_write;
    assign o_dcache_addr  = EX_MEM_alu_out;
    assign o_flush        = EX_MEM_is_flush;
    
    assign o_dcache_wdata = {EX_MEM_rs2_data[7:0], EX_MEM_rs2_data[15:8], EX_MEM_rs2_data[23:16], EX_MEM_rs2_data[31:24]};

    always @(posedge clk) begin
        if (!rst_n) begin
            MEM_WB_reg_write <= 0; MEM_WB_rd <= 0;
        end else if (!global_stall) begin
            MEM_WB_reg_write <= EX_MEM_reg_write;
            MEM_WB_mem_to_reg<= EX_MEM_mem_to_reg;
            MEM_WB_rd        <= EX_MEM_rd;
            MEM_WB_alu_out   <= EX_MEM_alu_out;
            MEM_WB_mem_rdata <= dcache_rdata_swapped;
        end
    end

    // -------------------------------------------------------------------------
    // WB Stage
    // -------------------------------------------------------------------------
    integer i;
    always @(posedge clk) begin
        if (!rst_n) begin
            for(i=0; i<32; i=i+1) regs[i] <= 32'b0;
        end else if (MEM_WB_reg_write && MEM_WB_rd != 5'b0) begin
            regs[MEM_WB_rd] <= wb_data;
        end
    end

endmodule

module RVC_EXPANDER(
    input  [31:0] inst_in,
    output reg [31:0] inst_out
);
    wire [15:0] inst_c = inst_in[15:0];
    wire [1:0] op = inst_c[1:0];
    wire [2:0] funct3 = inst_c[15:13];

    wire [4:0] rs1_p = {2'b01, inst_c[9:7]};
    wire [4:0] rs2_p = {2'b01, inst_c[4:2]};

    always @(*) begin
        if (op == 2'b11) begin
            inst_out = inst_in; // 32-bit standard instruction
        end else begin
            inst_out = 32'h00000013; // default NOP
            case(op)
                2'b00: begin
                    case(funct3)
                        3'b000: begin // C.ADDI4SPN: addi rd', x2, nzuimm
                            if (inst_c[12:5] != 8'b0) begin
                                inst_out = {2'b00, inst_c[10:7], inst_c[12:11], inst_c[5], inst_c[6], 2'b00, 5'd2, 3'b000, rs2_p, 7'b0010011};
                            end
                        end
                        3'b010: begin // C.LW: lw rd', offset(rs1')
                            inst_out = {5'b0, inst_c[5], inst_c[12:10], inst_c[6], 2'b00, rs1_p, 3'b010, rs2_p, 7'b0000011};
                        end
                        3'b110: begin // C.SW: sw rs2', offset(rs1')
                            inst_out = {5'b0, inst_c[5], inst_c[12], rs2_p, rs1_p, 3'b010, inst_c[11:10], inst_c[6], 2'b00, 7'b0100011};
                        end
                    endcase
                end
                2'b01: begin
                    case(funct3)
                        3'b000: begin // C.ADDI: addi rd, rd, nzimm
                            if (inst_c[11:7] != 5'b0) begin
                                inst_out = {{6{inst_c[12]}}, inst_c[12], inst_c[6:2], inst_c[11:7], 3'b000, inst_c[11:7], 7'b0010011};
                            end
                        end
                        3'b001: begin // C.JAL: jal x1, offset
                            inst_out = {inst_c[12], inst_c[8], inst_c[10:9], inst_c[6], inst_c[7], inst_c[2], inst_c[11], inst_c[5:3], inst_c[12], {8{inst_c[12]}}, 5'd1, 7'b1101111};
                        end
                        3'b010: begin // C.LI: addi rd, x0, imm
                            if (inst_c[11:7] != 5'b0) begin
                                inst_out = {{6{inst_c[12]}}, inst_c[12], inst_c[6:2], 5'd0, 3'b000, inst_c[11:7], 7'b0010011};
                            end
                        end
                        3'b011: begin // C.LUI: lui rd, nzimm
                            if (inst_c[11:7] != 5'b0 && inst_c[11:7] != 5'd2) begin
                                inst_out = {{15{inst_c[12]}}, inst_c[6:2], inst_c[11:7], 7'b0110111};
                            end else if (inst_c[11:7] == 5'd2) begin // C.ADDI16SP
                                inst_out = {{3{inst_c[12]}}, inst_c[4:3], inst_c[5], inst_c[2], inst_c[6], 4'b0, 5'd2, 3'b000, 5'd2, 7'b0010011};
                            end
                        end
                        3'b100: begin 
                            case(inst_c[11:10])
                                2'b00: begin // C.SRLI
                                    inst_out = {7'b0000000, inst_c[6:2], rs1_p, 3'b101, rs1_p, 7'b0010011};
                                end
                                2'b01: begin // C.SRAI
                                    inst_out = {7'b0100000, inst_c[6:2], rs1_p, 3'b101, rs1_p, 7'b0010011};
                                end
                                2'b10: begin // C.ANDI
                                    inst_out = {{6{inst_c[12]}}, inst_c[12], inst_c[6:2], rs1_p, 3'b111, rs1_p, 7'b0010011};
                                end
                                2'b11: begin 
                                    case({inst_c[12], inst_c[6:5]})
                                        3'b000: inst_out = {7'b0100000, rs2_p, rs1_p, 3'b000, rs1_p, 7'b0110011}; // C.SUB
                                        3'b001: inst_out = {7'b0000000, rs2_p, rs1_p, 3'b100, rs1_p, 7'b0110011}; // C.XOR
                                        3'b010: inst_out = {7'b0000000, rs2_p, rs1_p, 3'b110, rs1_p, 7'b0110011}; // C.OR
                                        3'b011: inst_out = {7'b0000000, rs2_p, rs1_p, 3'b111, rs1_p, 7'b0110011}; // C.AND
                                    endcase
                                end
                            endcase
                        end
                        3'b101: begin // C.J: jal x0, offset
                            inst_out = {inst_c[12], inst_c[8], inst_c[10:9], inst_c[6], inst_c[7], inst_c[2], inst_c[11], inst_c[5:3], inst_c[12], {8{inst_c[12]}}, 5'd0, 7'b1101111};
                        end
                        3'b110: begin // C.BEQZ: beq rs1', x0, offset
                            inst_out = {inst_c[12], inst_c[12], inst_c[12], inst_c[12], inst_c[6:5], inst_c[2], 5'd0, rs1_p, 3'b000, inst_c[11:10], inst_c[4:3], inst_c[12], 7'b1100011};
                        end
                        3'b111: begin // C.BNEZ: bne rs1', x0, offset
                            inst_out = {inst_c[12], inst_c[12], inst_c[12], inst_c[12], inst_c[6:5], inst_c[2], 5'd0, rs1_p, 3'b001, inst_c[11:10], inst_c[4:3], inst_c[12], 7'b1100011};
                        end
                    endcase
                end
                2'b10: begin
                    case(funct3)
                        3'b000: begin // C.SLLI
                            inst_out = {7'b0000000, inst_c[6:2], inst_c[11:7], 3'b001, inst_c[11:7], 7'b0010011};
                        end
                        3'b010: begin // C.LWSP: lw rd, offset(x2)
                            if (inst_c[11:7] != 5'b0) begin
                                inst_out = {4'b0, inst_c[3:2], inst_c[12], inst_c[6:4], 2'b00, 5'd2, 3'b010, inst_c[11:7], 7'b0000011};
                            end
                        end
                        3'b100: begin
                            if (inst_c[12] == 0) begin
                                if (inst_c[6:2] == 0) begin // C.JR
                                    inst_out = {12'b0, inst_c[11:7], 3'b000, 5'd0, 7'b1100111};
                                end else begin // C.MV
                                    inst_out = {7'b0000000, inst_c[6:2], 5'd0, 3'b000, inst_c[11:7], 7'b0110011};
                                end
                            end else begin
                                if (inst_c[6:2] == 0) begin 
                                    if (inst_c[11:7] == 0) begin // C.EBREAK
                                        inst_out = 32'h00100073;
                                    end else begin // C.JALR
                                        inst_out = {12'b0, inst_c[11:7], 3'b000, 5'd1, 7'b1100111};
                                    end
                                end else begin // C.ADD
                                    inst_out = {7'b0000000, inst_c[6:2], inst_c[11:7], 3'b000, inst_c[11:7], 7'b0110011};
                                end
                            end
                        end
                        3'b110: begin // C.SWSP: sw rs2, offset(x2)
                            inst_out = {4'b0, inst_c[8:7], inst_c[12], inst_c[6:2], 5'd2, 3'b010, inst_c[11:9], 2'b00, 7'b0100011};
                        end
                    endcase
                end
            endcase
        end
    end
endmodule

// =============================================================================
// I-Cache (16 Sets, Normal Word Indexing)
// =============================================================================
module ICACHE (
    input          clk,
    input          rst_n,
    input   [31:0] i_cpu_addr,
    output  [31:0] o_cpu_inst,
    output         o_cpu_valid,
    output         o_cpu_stall,
    output         o_mem_read,
    output         o_mem_write,
    output  [31:4] o_mem_addr,
    output [127:0] o_mem_wdata,
    input  [127:0] i_mem_rdata,
    input          i_mem_ready
);

    reg         valid [0:15];
    reg [23:0]  tag   [0:15];
    reg [127:0] data  [0:15];

    wire [3:0]  idx = i_cpu_addr[7:4];
    wire [23:0] cur_tag = i_cpu_addr[31:8];
    wire [1:0]  word_offset = i_cpu_addr[3:2];

    wire cache_hit = valid[idx] && (tag[idx] == cur_tag);
    
    assign o_cpu_valid = cache_hit;
    assign o_cpu_stall = !cache_hit;

    wire [127:0] hit_block = data[idx];
    
    assign o_cpu_inst = (word_offset == 2'b00) ? hit_block[31:0] :
                        (word_offset == 2'b01) ? hit_block[63:32] :
                        (word_offset == 2'b10) ? hit_block[95:64] : hit_block[127:96];

    reg [1:0] state, next_state;
    parameter IDLE = 2'd0, FETCH = 2'd1, WAIT_MEM = 2'd2;

    always @(posedge clk) begin
        if (!rst_n) state <= IDLE;
        else        state <= next_state;
    end

    always @(*) begin
        case(state)
            IDLE:     next_state = (!cache_hit) ? FETCH : IDLE;
            FETCH:    next_state = (i_mem_ready) ? WAIT_MEM : FETCH;
            WAIT_MEM: next_state = IDLE; 
            default:  next_state = IDLE;
        endcase
    end

    assign o_mem_read  = (state == FETCH);
    assign o_mem_write = 1'b0;
    assign o_mem_wdata = 128'b0;
    assign o_mem_addr  = i_cpu_addr[31:4];

    integer i;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (i=0; i<16; i=i+1) valid[i] <= 0;
        end else if (state == FETCH && i_mem_ready) begin
            valid[idx] <= 1'b1;
            tag[idx]   <= cur_tag;
            data[idx]  <= i_mem_rdata;
        end
    end
endmodule

// =============================================================================
// D-Cache (16 Sets, Normal Word Indexing)
// =============================================================================
module DCACHE (
    input          clk,
    input          rst_n,
    input          i_cpu_ren,
    input          i_cpu_wen,
    input   [31:0] i_cpu_addr,
    input   [31:0] i_cpu_wdata,
    output  [31:0] o_cpu_rdata,
    output         o_cpu_valid,
    output         o_cpu_stall,
    input          i_flush,
    output reg     o_flush_done,
    output reg         o_mem_read,
    output reg         o_mem_write,
    output reg  [31:4] o_mem_addr,
    output reg [127:0] o_mem_wdata,
    input      [127:0] i_mem_rdata,
    input              i_mem_ready
);

    reg         valid [0:15];
    reg         dirty [0:15];
    reg [23:0]  tag   [0:15];
    reg [127:0] data  [0:15];

    wire [3:0]  idx = i_cpu_addr[7:4];
    wire [23:0] cur_tag = i_cpu_addr[31:8];
    wire [1:0]  word_offset = i_cpu_addr[3:2];

    wire hit = valid[idx] && (tag[idx] == cur_tag);
    wire cpu_req = i_cpu_ren | i_cpu_wen;

    parameter IDLE       = 3'd0;
    parameter WRITE_BACK = 3'd1;
    parameter WB_WAIT    = 3'd2; 
    parameter ALLOCATE   = 3'd3;
    parameter ALLOC_WAIT = 3'd4; 
    parameter FLUSH      = 3'd5;
    parameter FLUSH_WAIT = 3'd6; 
    
    reg [2:0] state, next_state;
    reg [4:0] flush_idx;

    always @(posedge clk) begin
        if (!rst_n) state <= IDLE;
        else        state <= next_state;
    end

    always @(*) begin
        case(state)
            IDLE: begin
                if (i_flush) next_state = FLUSH;
                else if (cpu_req && !hit) begin
                    if (valid[idx] && dirty[idx]) next_state = WRITE_BACK;
                    else next_state = ALLOCATE;
                end else next_state = IDLE;
            end
            WRITE_BACK: next_state = (i_mem_ready) ? WB_WAIT : WRITE_BACK;
            WB_WAIT:    next_state = ALLOCATE;
            ALLOCATE:   next_state = (i_mem_ready) ? ALLOC_WAIT : ALLOCATE;
            ALLOC_WAIT: next_state = IDLE;
            FLUSH: begin
                if (flush_idx == 16) next_state = IDLE;
                else if (valid[flush_idx] && dirty[flush_idx]) 
                    next_state = (i_mem_ready) ? FLUSH_WAIT : FLUSH;
                else 
                    next_state = FLUSH;
            end
            FLUSH_WAIT: next_state = FLUSH;
            default: next_state = IDLE;
        endcase
    end

    always @(*) begin
        o_mem_read  = 0;
        o_mem_write = 0;
        o_mem_addr  = 28'b0;
        o_mem_wdata = 128'b0;
        if (state == WRITE_BACK) begin
            o_mem_write = 1;
            o_mem_addr  = {tag[idx], idx};
            o_mem_wdata = data[idx];
        end else if (state == ALLOCATE) begin
            o_mem_read  = 1;
            o_mem_addr  = i_cpu_addr[31:4];
        end else if (state == FLUSH && flush_idx < 16 && valid[flush_idx] && dirty[flush_idx]) begin
            o_mem_write = 1;
            o_mem_addr  = {tag[flush_idx[3:0]], flush_idx[3:0]};
            o_mem_wdata = data[flush_idx];
        end
    end

    assign o_cpu_stall = (cpu_req && !hit) || (state != IDLE && state != FLUSH);
    assign o_cpu_valid = hit && (state == IDLE);
    
    wire [127:0] hit_block = data[idx];
    
    assign o_cpu_rdata = (word_offset == 2'b00) ? hit_block[31:0] :
                         (word_offset == 2'b01) ? hit_block[63:32] :
                         (word_offset == 2'b10) ? hit_block[95:64] : hit_block[127:96];

    integer i;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (i=0; i<16; i=i+1) begin
                valid[i] <= 0;
                dirty[i] <= 0;
            end
            flush_idx <= 0;
            o_flush_done <= 0;
        end else begin
            if (state == FLUSH && flush_idx == 16) o_flush_done <= 1;
            else o_flush_done <= 0;

            case(state)
                IDLE: begin
                    if (cpu_req && hit && i_cpu_wen) begin
                        dirty[idx] <= 1;
                        if (word_offset == 2'b00) data[idx][31:0]   <= i_cpu_wdata;
                        if (word_offset == 2'b01) data[idx][63:32]  <= i_cpu_wdata;
                        if (word_offset == 2'b10) data[idx][95:64]  <= i_cpu_wdata;
                        if (word_offset == 2'b11) data[idx][127:96] <= i_cpu_wdata;
                    end
                end
                ALLOCATE: begin
                    if (i_mem_ready) begin
                        valid[idx] <= 1;
                        dirty[idx] <= 0;
                        tag[idx]   <= cur_tag;
                        data[idx]  <= i_mem_rdata;
                    end
                end
                FLUSH: begin
                    if (flush_idx < 16) begin
                        if (!(valid[flush_idx] && dirty[flush_idx])) begin
                            flush_idx <= flush_idx + 1;
                        end else if (i_mem_ready) begin
                            dirty[flush_idx] <= 0;
                            flush_idx <= flush_idx + 1;
                        end
                    end
                end
            endcase
        end
    end
endmodule