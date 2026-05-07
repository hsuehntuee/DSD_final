// -----------------------------------------------------------------------------
// DSD Final Project: Pipelined RISC-V Processor with I/D Caches
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

    // 【關鍵修正 1】將 Memory 端讀取的 32-bit 資料進行 Byte Swap，符合 RISC-V Little-Endian 標準
    wire [31:0] inst_swapped         = {i_icache_inst[7:0],  i_icache_inst[15:8],  i_icache_inst[23:16],  i_icache_inst[31:24]};
    wire [31:0] dcache_rdata_swapped = {i_dcache_rdata[7:0], i_dcache_rdata[15:8], i_dcache_rdata[23:16], i_dcache_rdata[31:24]};

    reg [31:0] IF_ID_pc, IF_ID_inst;
    reg [31:0] ID_EX_pc, ID_EX_rs1_data, ID_EX_rs2_data, ID_EX_imm;
    reg [4:0]  ID_EX_rs1, ID_EX_rs2, ID_EX_rd;
    reg [3:0]  ID_EX_alu_op;
    reg        ID_EX_reg_write, ID_EX_mem_read, ID_EX_mem_write;
    reg        ID_EX_alu_src, ID_EX_mem_to_reg, ID_EX_pc_to_alu;
    reg        ID_EX_branch, ID_EX_jump, ID_EX_jalr, ID_EX_is_flush;

    reg [31:0] EX_MEM_alu_out, EX_MEM_rs2_data;
    reg [4:0]  EX_MEM_rd;
    reg        EX_MEM_reg_write, EX_MEM_mem_read, EX_MEM_mem_write;
    reg        EX_MEM_mem_to_reg, EX_MEM_is_flush;

    reg [31:0] MEM_WB_alu_out, MEM_WB_mem_rdata;
    reg [4:0]  MEM_WB_rd;
    reg        MEM_WB_reg_write, MEM_WB_mem_to_reg;

    wire stall_pipeline = i_icache_stall | i_dcache_stall;
    wire load_use_stall;
    wire branch_taken;
    wire [31:0] branch_target;
    wire [31:0] wb_data = (MEM_WB_mem_to_reg) ? MEM_WB_mem_rdata : MEM_WB_alu_out;

    // -------------------------------------------------------------------------
    // IF Stage
    // -------------------------------------------------------------------------
    reg  [31:0] pc;
    assign o_icache_addr = pc;

    // 【關鍵修正 2】嚴格的 PC Freeze：只要 stall_pipeline=1，PC 絕對不准提早跳躍！
    always @(posedge clk) begin
        if (!rst_n) pc <= 32'b0;
        else if (!stall_pipeline) begin
            if (branch_taken) pc <= branch_target;
            else if (!load_use_stall) pc <= pc + 4;
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            IF_ID_pc   <= 32'b0;
            IF_ID_inst <= 32'h00000013;
        end else if (!stall_pipeline) begin
            if (branch_taken) begin
                IF_ID_pc   <= 32'b0;
                IF_ID_inst <= 32'h00000013;
            end else if (!load_use_stall) begin
                IF_ID_pc   <= pc;
                IF_ID_inst <= inst_swapped; // 存入 Byte-swapped 正常的指令
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
    reg ctrl_branch, ctrl_jump, ctrl_jalr, ctrl_pc_to_alu;
    reg [3:0] ctrl_alu_op; 

    // FLUSH 指令被 Testbench Byte Swap 後會變成 0x00202007
    wire is_flush = (IF_ID_inst == 32'h00202007);

    always @(*) begin
        ctrl_reg_write = 0; ctrl_mem_read = 0; ctrl_mem_write = 0; 
        ctrl_alu_src = 0; ctrl_mem_to_reg = 0; ctrl_branch = 0; 
        ctrl_jump = 0; ctrl_jalr = 0; ctrl_alu_op = 4'b0000; ctrl_pc_to_alu = 0;
        case(opcode)
            7'b0110011: begin // R-Type
                ctrl_reg_write = 1;
                if(funct3 == 3'b000 && funct7 == 7'b0000000) ctrl_alu_op = 4'd0;
                else if(funct3 == 3'b000 && funct7 == 7'b0100000) ctrl_alu_op = 4'd1;
                else if(funct3 == 3'b111) ctrl_alu_op = 4'd2;
                else if(funct3 == 3'b110) ctrl_alu_op = 4'd3;
                else if(funct3 == 3'b100) ctrl_alu_op = 4'd4;
                else if(funct3 == 3'b001) ctrl_alu_op = 4'd5;
                else if(funct3 == 3'b101 && funct7 == 7'b0000000) ctrl_alu_op = 4'd6;
                else if(funct3 == 3'b101 && funct7 == 7'b0100000) ctrl_alu_op = 4'd7;
                else if(funct3 == 3'b010) ctrl_alu_op = 4'd8;
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
        end else if (!stall_pipeline) begin
            if (branch_taken || load_use_stall) begin
                ID_EX_reg_write <= 0; ID_EX_mem_read <= 0; ID_EX_mem_write <= 0;
                ID_EX_branch <= 0; ID_EX_jump <= 0; ID_EX_jalr <= 0;
                ID_EX_is_flush <= 0; ID_EX_rd <= 0; ID_EX_pc_to_alu <= 0;
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

    assign branch_target = (ID_EX_jalr) ? ((alu_in1_base + ID_EX_imm) & ~32'd1) : (ID_EX_pc + ID_EX_imm);
    assign branch_taken  = ID_EX_jump | ID_EX_jalr | (ID_EX_branch & alu_zero);

    always @(posedge clk) begin
        if (!rst_n) begin
            EX_MEM_reg_write <= 0; EX_MEM_mem_read <= 0; EX_MEM_mem_write <= 0; EX_MEM_is_flush <= 0;
            EX_MEM_rd <= 0;
        end else if (!stall_pipeline) begin
            EX_MEM_reg_write <= ID_EX_reg_write;
            EX_MEM_mem_read  <= ID_EX_mem_read;
            EX_MEM_mem_write <= ID_EX_mem_write;
            EX_MEM_mem_to_reg<= ID_EX_mem_to_reg;
            EX_MEM_rd        <= ID_EX_rd;
            EX_MEM_alu_out   <= (ID_EX_jump || ID_EX_jalr) ? (ID_EX_pc + 4) : alu_result;
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
    
    // 【關鍵修正 3】將寫入 Memory 的資料進行 Byte Swap
    assign o_dcache_wdata = {EX_MEM_rs2_data[7:0], EX_MEM_rs2_data[15:8], EX_MEM_rs2_data[23:16], EX_MEM_rs2_data[31:24]};

    always @(posedge clk) begin
        if (!rst_n) begin
            MEM_WB_reg_write <= 0; MEM_WB_rd <= 0;
        end else if (!stall_pipeline) begin
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
    
    // 取出正確位置的 32-bit (交給 Core 做 Endianness Swap)
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