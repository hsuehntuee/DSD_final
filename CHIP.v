module CHIP (
    input               clk,
    input               rst_n,
    output              mem_read_D,
    output              mem_write_D,
    output      [31:4]  mem_addr_D,
    output      [127:0] mem_wdata_D,
    input       [127:0] mem_rdata_D,
    input               mem_ready_D,
    output              mem_read_I,
    output              mem_write_I,
    output      [31:4]  mem_addr_I,
    output      [127:0] mem_wdata_I,
    input       [127:0] mem_rdata_I,
    input               mem_ready_I,
    output              o_done
);
    wire [127:0] icache_data;
    wire [31:0] icache_addr;
    wire        icache_valid, icache_stall;
    wire [31:0] dcache_rdata, dcache_wdata, dcache_addr;
    wire        dcache_ren, dcache_wen, dcache_valid, dcache_stall;
    wire        flush_req;

    RISCV_CORE core (
        .clk            (clk),
        .rst_n          (rst_n),
        .i_icache_data  (icache_data),
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
        .o_flush        (flush_req),
        .i_flush_done   (o_done)
    );

    ICACHE icache (
        .clk            (clk),
        .rst_n          (rst_n),
        .i_cpu_addr     (icache_addr),
        .o_cpu_data     (icache_data),
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
// RISC-V 5-Stage Pipeline Core (10-bit GHR, 1K PHT, 8 RSB)
// =============================================================================
module RISCV_CORE (
    input         clk,
    input         rst_n,
    input  [127:0] i_icache_data,
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
    output        o_flush,
    input         i_flush_done
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
    reg        ID_EX_pred_taken;
    reg [31:0] ID_EX_pred_target;
    
    reg [9:0] ID_EX_ghr;
    
    reg [31:0] ID_EX_branch_target;
    reg [31:0] ID_EX_fallthrough_pc;

    reg [31:0] EX_MEM_alu_out, EX_MEM_rs2_data;
    reg [4:0]  EX_MEM_rd;
    reg [3:0]  EX_MEM_alu_op;
    reg        EX_MEM_reg_write, EX_MEM_mem_read, EX_MEM_mem_write;
    reg        EX_MEM_mem_to_reg, EX_MEM_is_flush, EX_MEM_is_mult;

    reg [31:0] MEM_WB_alu_out, MEM_WB_mem_rdata;
    reg [4:0]  MEM_WB_rd;
    reg        MEM_WB_reg_write, MEM_WB_mem_to_reg;

    reg flush_done_reg;
    always @(posedge clk) begin
        if (!rst_n) flush_done_reg <= 0;
        else if (EX_MEM_is_flush && i_flush_done) flush_done_reg <= 1;
        else if (!EX_MEM_is_flush) flush_done_reg <= 0;
    end
    assign o_flush = EX_MEM_is_flush && !flush_done_reg;

    wire mem_stall = i_icache_stall | i_dcache_stall | (EX_MEM_is_flush && !flush_done_reg);
    wire global_stall = mem_stall;

    wire load_use_stall; 
    wire front_stall;    
    wire branch_taken;
    wire [31:0] correction_target;
    wire [31:0] wb_data = (MEM_WB_mem_to_reg) ? MEM_WB_mem_rdata : MEM_WB_alu_out;

    reg [31:0] pc;
    reg state_if;
    wire [127:0] current_line = {
        i_icache_data[103:96], i_icache_data[111:104], i_icache_data[119:112], i_icache_data[127:120],
        i_icache_data[71:64],  i_icache_data[79:72],   i_icache_data[87:80],   i_icache_data[95:88],
        i_icache_data[39:32],  i_icache_data[47:40],   i_icache_data[55:48],   i_icache_data[63:56],
        i_icache_data[7:0],    i_icache_data[15:8],    i_icache_data[23:16],   i_icache_data[31:24]
    };
    wire [31:0]  shifted_line = current_line >> (pc[3:1] * 16);
    wire [31:0]  raw_inst = shifted_line;

    // -------------------------------------------------------------------------
    // IF Stage
    // -------------------------------------------------------------------------
    reg [31:0] btb_tag_0 [0:15]; reg [31:0] btb_tag_1 [0:15]; reg [31:0] btb_tag_2 [0:15]; reg [31:0] btb_tag_3 [0:15];
    reg [31:0] btb_tgt_0 [0:15]; reg [31:0] btb_tgt_1 [0:15]; reg [31:0] btb_tgt_2 [0:15]; reg [31:0] btb_tgt_3 [0:15];
    reg        btb_val_0 [0:15]; reg        btb_val_1 [0:15]; reg        btb_val_2 [0:15]; reg        btb_val_3 [0:15];
    reg        btb_ret_0 [0:15]; reg        btb_ret_1 [0:15]; reg        btb_ret_2 [0:15]; reg        btb_ret_3 [0:15];
    reg        btb_cal_0 [0:15]; reg        btb_cal_1 [0:15]; reg        btb_cal_2 [0:15]; reg        btb_cal_3 [0:15];
    reg [2:0]  btb_plru  [0:15];
    
    reg [9:0] ghr;
    reg [1:0] pht [0:1023];
    reg [31:0] rsb_stack [0:7];
    reg [2:0]  rsb_ptr;

    wire [3:0] btb_idx = pc[5:2];
    wire [9:0] pht_idx = pc[11:2] ^ ghr;
    
    wire btb_hit0 = btb_val_0[btb_idx] && (btb_tag_0[btb_idx] == pc);
    wire btb_hit1 = btb_val_1[btb_idx] && (btb_tag_1[btb_idx] == pc);
    wire btb_hit2 = btb_val_2[btb_idx] && (btb_tag_2[btb_idx] == pc);
    wire btb_hit3 = btb_val_3[btb_idx] && (btb_tag_3[btb_idx] == pc);
    wire btb_hit = btb_hit0 || btb_hit1 || btb_hit2 || btb_hit3;
    
    wire is_call_hit = btb_hit0 ? btb_cal_0[btb_idx] : btb_hit1 ? btb_cal_1[btb_idx] : btb_hit2 ? btb_cal_2[btb_idx] : btb_hit3 ? btb_cal_3[btb_idx] : 1'b0;
    wire is_ret_hit  = btb_hit0 ? btb_ret_0[btb_idx] : btb_hit1 ? btb_ret_1[btb_idx] : btb_hit2 ? btb_ret_2[btb_idx] : btb_hit3 ? btb_ret_3[btb_idx] : 1'b0;
    wire [31:0] target_hit = btb_hit0 ? btb_tgt_0[btb_idx] : btb_hit1 ? btb_tgt_1[btb_idx] : btb_hit2 ? btb_tgt_2[btb_idx] : btb_hit3 ? btb_tgt_3[btb_idx] : 32'b0;

    wire predict_taken = btb_hit && (is_call_hit || is_ret_hit || (pht[pht_idx] >= 2'd2));
    wire [31:0] predict_target = (btb_hit && is_ret_hit) ? rsb_stack[(rsb_ptr-3'd1)] : target_hit;

    parameter IF_NORMAL = 1'b0, IF_CROSS = 1'b1;
    reg [15:0] cross_word_lower;
    reg        IF_ID_pred_taken;
    reg [31:0] IF_ID_pred_target;
    reg [9:0]  IF_ID_ghr;

    wire [31:0] fetch_pc = (state_if == IF_CROSS) ? {pc[31:4], 4'b0000} + 16 : {pc[31:4], 4'b0000};
    assign o_icache_addr = fetch_pc;
    
    wire is_32bit_inst = (raw_inst[1:0] == 2'b11);
    wire if_stall_req = (state_if == IF_NORMAL) && (pc[3:1] == 3'd7) && is_32bit_inst && i_icache_valid;
    wire [31:0] if_inst_raw = (state_if == IF_CROSS) ? {current_line[15:0], cross_word_lower} : (is_32bit_inst ? raw_inst : {16'b0, raw_inst[15:0]});
    wire [31:0] expanded_inst;
    RVC_EXPANDER rvc_exp(.inst_in(if_inst_raw), .inst_out(expanded_inst));

    wire actual_is_32bit = (state_if == IF_CROSS) ? 1'b1 : is_32bit_inst;
    wire [31:0] pc_next_norm = pc + (actual_is_32bit ? 4 : 2);
    wire [31:0] pc_next_pred = (predict_taken && !if_stall_req) ? predict_target : pc_next_norm;

    always @(posedge clk) begin
        if (!rst_n) begin
            state_if <= IF_NORMAL; cross_word_lower <= 16'b0;
            pc <= 32'b0; IF_ID_pc <= 32'b0; IF_ID_inst <= 32'h00000013;
            IF_ID_is_rvc <= 0; IF_ID_pred_taken <= 0; IF_ID_pred_target <= 32'b0;
        end else if (!global_stall) begin
            if (branch_taken) begin
                state_if <= IF_NORMAL; pc <= correction_target;
                IF_ID_inst <= 32'h00000013; IF_ID_is_rvc <= 0; IF_ID_pred_taken <= 0;
            end else begin
                if (state_if == IF_NORMAL && if_stall_req) begin
                    state_if <= IF_CROSS; cross_word_lower <= raw_inst[15:0];
                end else if (state_if == IF_CROSS && !front_stall) begin
                    state_if <= IF_NORMAL;
                end
                if (!front_stall) begin
                    pc <= pc_next_pred; IF_ID_pc <= pc; IF_ID_inst <= expanded_inst;
                    IF_ID_is_rvc <= !actual_is_32bit; IF_ID_pred_taken <= predict_taken;
                    IF_ID_pred_target <= predict_target; IF_ID_ghr <= ghr;
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // ID Stage
    // -------------------------------------------------------------------------
    wire [6:0] opcode = IF_ID_inst[6:0];
    wire [2:0] funct3 = IF_ID_inst[14:12];
    wire [6:0] funct7 = IF_ID_inst[31:25];
    wire [4:0] rs1 = IF_ID_inst[19:15];
    wire [4:0] rs2 = IF_ID_inst[24:20];
    wire [4:0] rd  = IF_ID_inst[11:7];

    reg [31:0] regs [0:31];
    wire [31:0] rs1_data = (rs1 == 5'b0) ? 32'b0 : (MEM_WB_reg_write && MEM_WB_rd == rs1) ? wb_data : regs[rs1];
    wire [31:0] rs2_data = (rs2 == 5'b0) ? 32'b0 : (MEM_WB_reg_write && MEM_WB_rd == rs2) ? wb_data : regs[rs2];
    
    wire rs1_valid = (opcode == 7'b0110011) || (opcode == 7'b0010011) || (opcode == 7'b0000011) || 
                     (opcode == 7'b0100011) || (opcode == 7'b1100011) || (opcode == 7'b1100111);
    wire rs2_valid = (opcode == 7'b0110011) || (opcode == 7'b0100011) || (opcode == 7'b1100011);

    assign load_use_stall = ((ID_EX_mem_read || ID_EX_is_mult) && ID_EX_rd != 0 &&
                            ((rs1_valid && (ID_EX_rd == rs1)) || (rs2_valid && (ID_EX_rd == rs2))));
    assign front_stall = load_use_stall | if_stall_req;

    reg [31:0] imm;
    always @(*) begin
        case(opcode)
            7'b0010011, 7'b0000011, 7'b1100111: imm = {{20{IF_ID_inst[31]}}, IF_ID_inst[31:20]};
            7'b0100011: imm = {{20{IF_ID_inst[31]}}, IF_ID_inst[31:25], IF_ID_inst[11:7]};
            7'b1100011: imm = {{20{IF_ID_inst[31]}}, IF_ID_inst[7], IF_ID_inst[30:25], IF_ID_inst[11:8], 1'b0};
            7'b1101111: imm = {{12{IF_ID_inst[31]}}, IF_ID_inst[19:12], IF_ID_inst[20], IF_ID_inst[30:21], 1'b0};
            7'b0110111, 7'b0010111: imm = {IF_ID_inst[31:12], 12'b0};
            default: imm = 32'b0;
        endcase
    end

    reg ctrl_reg_write, ctrl_mem_read, ctrl_mem_write, ctrl_alu_src, ctrl_mem_to_reg;
    reg ctrl_branch, ctrl_jump, ctrl_jalr, ctrl_pc_to_alu, ctrl_is_mult;
    reg [3:0] ctrl_alu_op; 
    wire is_flush = (IF_ID_inst == 32'h00202007);

    always @(*) begin
        ctrl_reg_write = 0; ctrl_mem_read = 0; ctrl_mem_write = 0; ctrl_alu_src = 0; 
        ctrl_mem_to_reg = 0; ctrl_branch = 0; ctrl_jump = 0; ctrl_jalr = 0; 
        ctrl_alu_op = 4'b0000; ctrl_pc_to_alu = 0; ctrl_is_mult = 0;
        case(opcode)
            7'b0110011: begin 
                ctrl_reg_write = 1;
                if(funct7 == 7'b0000000) begin
                    case(funct3)
                        3'b000: ctrl_alu_op = 4'd0; 3'b111: ctrl_alu_op = 4'd2;
                        3'b110: ctrl_alu_op = 4'd3; 3'b100: ctrl_alu_op = 4'd4;
                        3'b001: ctrl_alu_op = 4'd5; 3'b101: ctrl_alu_op = 4'd6;
                        3'b010: ctrl_alu_op = 4'd8;
                    endcase
                end else if (funct7 == 7'b0100000) begin
                    if(funct3 == 3'b000) ctrl_alu_op = 4'd1;
                    else if(funct3 == 3'b101) ctrl_alu_op = 4'd7;
                end else if (funct7 == 7'b0000001) begin // M-Type
                    ctrl_is_mult = 1;
                    case(funct3)
                        3'b000: ctrl_alu_op = 4'd12; 3'b001: ctrl_alu_op = 4'd13;
                        3'b010: ctrl_alu_op = 4'd14; 3'b011: ctrl_alu_op = 4'd15;
                    endcase
                end
            end
            7'b0010011: begin ctrl_reg_write = 1; ctrl_alu_src = 1;
                case(funct3)
                    3'b000: ctrl_alu_op = 4'd0; 3'b111: ctrl_alu_op = 4'd2;
                    3'b110: ctrl_alu_op = 4'd3; 3'b100: ctrl_alu_op = 4'd4;
                    3'b001: ctrl_alu_op = 4'd5; 3'b010: ctrl_alu_op = 4'd8;
                    3'b101: ctrl_alu_op = (funct7 == 7'b0100000) ? 4'd7 : 4'd6;
                endcase
            end
            7'b0000011: begin ctrl_reg_write = 1; ctrl_alu_src = 1; ctrl_mem_read = 1; ctrl_mem_to_reg = 1; end
            7'b0100011: begin ctrl_alu_src = 1; ctrl_mem_write = 1; end
            7'b1100011: begin ctrl_branch = 1; ctrl_alu_op = (funct3 == 3'b001) ? 4'd10 : 4'd9; end
            7'b1101111: begin ctrl_jump = 1; ctrl_reg_write = 1; end
            7'b1100111: begin ctrl_jalr = 1; ctrl_reg_write = 1; ctrl_alu_src = 1; end
            7'b0110111: begin ctrl_reg_write = 1; ctrl_alu_src = 1; ctrl_alu_op = 4'd11; end
            7'b0010111: begin ctrl_reg_write = 1; ctrl_alu_src = 1; ctrl_pc_to_alu = 1; end
        endcase
    end

    wire [31:0] id_branch_target_calc = IF_ID_pc + imm;
    wire [31:0] id_fallthrough_pc_calc = IF_ID_pc + (IF_ID_is_rvc ? 2 : 4);

    always @(posedge clk) begin
        if (!rst_n) begin
            ID_EX_reg_write <= 0; ID_EX_mem_read <= 0; ID_EX_mem_write <= 0;
            ID_EX_branch <= 0; ID_EX_jump <= 0; ID_EX_jalr <= 0; ID_EX_is_flush <= 0;
            ID_EX_rd <= 0; ID_EX_is_mult <= 0;
        end else if (!global_stall) begin
            if (branch_taken || front_stall) begin
                ID_EX_reg_write <= 0; ID_EX_mem_read <= 0; ID_EX_mem_write <= 0;
                ID_EX_branch <= 0; ID_EX_jump <= 0; ID_EX_jalr <= 0; ID_EX_is_flush <= 0;
                ID_EX_is_mult <= 0; ID_EX_rd <= 0; 
            end else begin
                ID_EX_pc <= IF_ID_pc; ID_EX_rs1_data <= rs1_data; ID_EX_rs2_data <= rs2_data;
                ID_EX_imm <= imm; ID_EX_rd <= rd; ID_EX_rs1 <= rs1; ID_EX_rs2 <= rs2;
                ID_EX_alu_op <= ctrl_alu_op; ID_EX_reg_write <= ctrl_reg_write;
                ID_EX_mem_read <= ctrl_mem_read; ID_EX_mem_write <= ctrl_mem_write;
                ID_EX_alu_src <= ctrl_alu_src; ID_EX_mem_to_reg <= ctrl_mem_to_reg;
                ID_EX_branch <= ctrl_branch; ID_EX_jump <= ctrl_jump; ID_EX_jalr <= ctrl_jalr;
                ID_EX_is_flush <= is_flush; ID_EX_pc_to_alu <= ctrl_pc_to_alu;
                ID_EX_is_mult <= ctrl_is_mult; ID_EX_is_rvc <= IF_ID_is_rvc;
                ID_EX_pred_taken <= IF_ID_pred_taken; ID_EX_pred_target <= IF_ID_pred_target;
                ID_EX_ghr <= IF_ID_ghr;
                
                ID_EX_branch_target <= id_branch_target_calc;
                ID_EX_fallthrough_pc <= id_fallthrough_pc_calc;
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

    wire [31:0] ex_mem_fwd_data = EX_MEM_alu_out;
    
    wire [31:0] alu_in1_base = (forward_A == 2'b10) ? ex_mem_fwd_data : (forward_A == 2'b01) ? wb_data : ID_EX_rs1_data;
    wire [31:0] alu_in1 = (ID_EX_pc_to_alu) ? ID_EX_pc : alu_in1_base;
    wire [31:0] fwd_in2 = (forward_B == 2'b10) ? ex_mem_fwd_data : (forward_B == 2'b01) ? wb_data : ID_EX_rs2_data;
    wire [31:0] alu_in2 = (ID_EX_alu_src) ? ID_EX_imm : fwd_in2;

    reg [31:0] alu_result;
    always @(*) begin
        case(ID_EX_alu_op)
            4'd0: alu_result = alu_in1 + alu_in2; 4'd1: alu_result = alu_in1 - alu_in2;
            4'd2: alu_result = alu_in1 & alu_in2; 4'd3: alu_result = alu_in1 | alu_in2;
            4'd4: alu_result = alu_in1 ^ alu_in2; 4'd5: alu_result = alu_in1 << alu_in2[4:0];
            4'd6: alu_result = alu_in1 >> alu_in2[4:0]; 4'd7: alu_result = $signed(alu_in1) >>> alu_in2[4:0];
            4'd8: alu_result = ($signed(alu_in1) < $signed(alu_in2)) ? 32'd1 : 32'd0;
            4'd9: alu_result = alu_in1 - alu_in2; 
            4'd10: alu_result = alu_in1 - alu_in2; 
            4'd11: alu_result = alu_in2; default: alu_result = 32'b0;
        endcase
    end

    wire branch_cmp_eq = (alu_in1 == alu_in2);
    
    reg actual_taken;
    always @(*) begin
        if (ID_EX_jump || ID_EX_jalr) actual_taken = 1'b1;
        else if (ID_EX_branch) begin
            if (ID_EX_alu_op == 4'd9) actual_taken = branch_cmp_eq;
            else if (ID_EX_alu_op == 4'd10) actual_taken = !branch_cmp_eq;
            else actual_taken = 1'b0;
        end else actual_taken = 1'b0;
    end

    wire signed [32:0] a_ext = (ID_EX_alu_op == 4'd13 || ID_EX_alu_op == 4'd14) ? {alu_in1[31], alu_in1} : {1'b0, alu_in1};
    wire signed [32:0] b_ext = (ID_EX_alu_op == 4'd13) ? {alu_in2[31], alu_in2} : {1'b0, alu_in2};
    wire [65:0] dw_mul_product;
    DW_mult_pipe #(
        .a_width(33), .b_width(33), .num_stages(2), .stall_mode(1), .rst_mode(1)
    ) u_dw_mult (
        .clk(clk), .rst_n(rst_n), .en(~global_stall), .tc(1'b1),
        .a(a_ext), .b(b_ext), .product(dw_mul_product)
    );
    wire [31:0] mul_mem_res = (EX_MEM_alu_op == 4'd12) ? dw_mul_product[31:0] : dw_mul_product[63:32];

    wire [31:0] branch_target = (ID_EX_jalr) ? ((alu_in1_base + ID_EX_imm) & ~32'd1) : ID_EX_branch_target;
    
    assign correction_target = (actual_taken) ? branch_target : ID_EX_fallthrough_pc;
    assign branch_taken = (ID_EX_branch || ID_EX_jump || ID_EX_jalr) && 
                          (actual_taken != ID_EX_pred_taken || (actual_taken && (branch_target != ID_EX_pred_target)));

    wire [3:0] ex_btb_idx = ID_EX_pc[5:2];
    wire [9:0] ex_pht_idx = ID_EX_pc[11:2] ^ ID_EX_ghr;
    wire ex_hit0 = btb_val_0[ex_btb_idx] && (btb_tag_0[ex_btb_idx] == ID_EX_pc);
    wire ex_hit1 = btb_val_1[ex_btb_idx] && (btb_tag_1[ex_btb_idx] == ID_EX_pc);
    wire ex_hit2 = btb_val_2[ex_btb_idx] && (btb_tag_2[ex_btb_idx] == ID_EX_pc);
    wire ex_hit3 = btb_val_3[ex_btb_idx] && (btb_tag_3[ex_btb_idx] == ID_EX_pc);
    wire ex_hit = ex_hit0 || ex_hit1 || ex_hit2 || ex_hit3;
    wire [1:0] ex_hit_way = ex_hit0 ? 2'd0 : ex_hit1 ? 2'd1 : ex_hit2 ? 2'd2 : 2'd3;
    
    wire is_call = (ID_EX_jump || ID_EX_jalr) && (ID_EX_rd == 5'd1 || ID_EX_rd == 5'd5);
    wire is_ret  = ID_EX_jalr && (ID_EX_rs1 == 5'd1 || ID_EX_rs1 == 5'd5) && (ID_EX_rd == 5'd0);

    integer j;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (j=0; j<16; j=j+1) begin
                btb_val_0[j] <= 0; btb_val_1[j] <= 0; btb_val_2[j] <= 0; btb_val_3[j] <= 0;
                btb_plru[j] <= 0;
            end
            for (j=0; j<1024; j=j+1) pht[j] <= 2'b01;
            ghr <= 0; rsb_ptr <= 0;
        end else if (!global_stall) begin
            if (ID_EX_branch || ID_EX_jump || ID_EX_jalr) begin
                if (ID_EX_branch) begin
                    if (actual_taken) begin if (pht[ex_pht_idx] != 2'd3) pht[ex_pht_idx] <= pht[ex_pht_idx] + 1; end
                    else begin if (pht[ex_pht_idx] != 2'd0) pht[ex_pht_idx] <= pht[ex_pht_idx] - 1; end
                    ghr <= {ghr[8:0], actual_taken};
                end
                if (ex_hit) begin
                    if (ex_hit_way == 2'd0) btb_tgt_0[ex_btb_idx] <= branch_target;
                    else if (ex_hit_way == 2'd1) btb_tgt_1[ex_btb_idx] <= branch_target;
                    else if (ex_hit_way == 2'd2) btb_tgt_2[ex_btb_idx] <= branch_target;
                    else btb_tgt_3[ex_btb_idx] <= branch_target;
                end
                if (is_call) begin rsb_stack[rsb_ptr] <= ID_EX_pc + (ID_EX_is_rvc ? 2 : 4); rsb_ptr <= rsb_ptr + 1; end
                else if (is_ret) rsb_ptr <= rsb_ptr - 1;
            end
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            EX_MEM_reg_write <= 0; EX_MEM_mem_read <= 0; EX_MEM_mem_write <= 0; 
            EX_MEM_is_flush <= 0; EX_MEM_rd <= 0; EX_MEM_is_mult <= 0;
        end else if (!global_stall) begin
            EX_MEM_reg_write <= ID_EX_reg_write; EX_MEM_mem_read <= ID_EX_mem_read;
            EX_MEM_mem_write <= ID_EX_mem_write; EX_MEM_mem_to_reg <= ID_EX_mem_to_reg;
            EX_MEM_rd <= ID_EX_rd; EX_MEM_alu_op <= ID_EX_alu_op;
            EX_MEM_alu_out <= (ID_EX_jump || ID_EX_jalr) ? (ID_EX_pc + (ID_EX_is_rvc ? 2 : 4)) : alu_result;
            EX_MEM_rs2_data <= fwd_in2; EX_MEM_is_flush <= ID_EX_is_flush; EX_MEM_is_mult <= ID_EX_is_mult;
        end
    end

    // -------------------------------------------------------------------------
    // MEM & WB Stage
    // -------------------------------------------------------------------------
    assign o_dcache_ren = EX_MEM_mem_read;
    assign o_dcache_wen = EX_MEM_mem_write;
    assign o_dcache_addr = EX_MEM_alu_out;
    assign o_dcache_wdata = {EX_MEM_rs2_data[7:0], EX_MEM_rs2_data[15:8], EX_MEM_rs2_data[23:16], EX_MEM_rs2_data[31:24]};

    always @(posedge clk) begin
        if (!rst_n) begin MEM_WB_reg_write <= 0; MEM_WB_rd <= 0; end
        else if (!global_stall) begin
            MEM_WB_reg_write <= EX_MEM_reg_write; MEM_WB_mem_to_reg <= EX_MEM_mem_to_reg;
            MEM_WB_rd <= EX_MEM_rd; MEM_WB_mem_rdata <= dcache_rdata_swapped;
            MEM_WB_alu_out <= EX_MEM_is_mult ? mul_mem_res : EX_MEM_alu_out;
        end
    end

    integer i;
    always @(posedge clk) begin
        if (!rst_n) for(i=0; i<32; i=i+1) regs[i] <= 32'b0;
        else if (MEM_WB_reg_write && MEM_WB_rd != 5'b0) regs[MEM_WB_rd] <= wb_data;
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
            inst_out = inst_in; 
        end else begin
            inst_out = 32'h00000013; 
            case(op)
                2'b00: begin
                    case(funct3)
                        3'b000: begin 
                            if (inst_c[12:5] != 8'b0) begin
                                inst_out = {2'b00, inst_c[10:7], inst_c[12:11], inst_c[5], inst_c[6], 2'b00, 5'd2, 3'b000, rs2_p, 7'b0010011};
                            end
                        end
                        3'b010: begin 
                            inst_out = {5'b0, inst_c[5], inst_c[12:10], inst_c[6], 2'b00, rs1_p, 3'b010, rs2_p, 7'b0000011};
                        end
                        3'b110: begin 
                            inst_out = {5'b0, inst_c[5], inst_c[12], rs2_p, rs1_p, 3'b010, inst_c[11:10], inst_c[6], 2'b00, 7'b0100011};
                        end
                    endcase
                end
                2'b01: begin
                    case(funct3)
                        3'b000: begin 
                            if (inst_c[11:7] != 5'b0) begin
                                inst_out = {{6{inst_c[12]}}, inst_c[12], inst_c[6:2], inst_c[11:7], 3'b000, inst_c[11:7], 7'b0010011};
                            end
                        end
                        3'b001: begin 
                            inst_out = {inst_c[12], inst_c[8], inst_c[10:9], inst_c[6], inst_c[7], inst_c[2], inst_c[11], inst_c[5:3], inst_c[12], {8{inst_c[12]}}, 5'd1, 7'b1101111};
                        end
                        3'b010: begin 
                            if (inst_c[11:7] != 5'b0) begin
                                inst_out = {{6{inst_c[12]}}, inst_c[12], inst_c[6:2], 5'd0, 3'b000, inst_c[11:7], 7'b0010011};
                            end
                        end
                        3'b011: begin 
                            if (inst_c[11:7] != 5'b0 && inst_c[11:7] != 5'd2) begin
                                inst_out = {{15{inst_c[12]}}, inst_c[6:2], inst_c[11:7], 7'b0110111};
                            end else if (inst_c[11:7] == 5'd2) begin 
                                inst_out = {{3{inst_c[12]}}, inst_c[4:3], inst_c[5], inst_c[2], inst_c[6], 4'b0, 5'd2, 3'b000, 5'd2, 7'b0010011};
                            end
                        end
                        3'b100: begin 
                            case(inst_c[11:10])
                                2'b00: begin 
                                    inst_out = {7'b0000000, inst_c[6:2], rs1_p, 3'b101, rs1_p, 7'b0010011};
                                end
                                2'b01: begin 
                                    inst_out = {7'b0100000, inst_c[6:2], rs1_p, 3'b101, rs1_p, 7'b0010011};
                                end
                                2'b10: begin 
                                    inst_out = {{6{inst_c[12]}}, inst_c[12], inst_c[6:2], rs1_p, 3'b111, rs1_p, 7'b0010011};
                                end
                                2'b11: begin 
                                    case({inst_c[12], inst_c[6:5]})
                                        3'b000: inst_out = {7'b0100000, rs2_p, rs1_p, 3'b000, rs1_p, 7'b0110011}; 
                                        3'b001: inst_out = {7'b0000000, rs2_p, rs1_p, 3'b100, rs1_p, 7'b0110011}; 
                                        3'b010: inst_out = {7'b0000000, rs2_p, rs1_p, 3'b110, rs1_p, 7'b0110011}; 
                                        3'b011: inst_out = {7'b0000000, rs2_p, rs1_p, 3'b111, rs1_p, 7'b0110011}; 
                                    endcase
                                end
                            endcase
                        end
                        3'b101: begin 
                            inst_out = {inst_c[12], inst_c[8], inst_c[10:9], inst_c[6], inst_c[7], inst_c[2], inst_c[11], inst_c[5:3], inst_c[12], {8{inst_c[12]}}, 5'd0, 7'b1101111};
                        end
                        3'b110: begin 
                            inst_out = {inst_c[12], inst_c[12], inst_c[12], inst_c[12], inst_c[6:5], inst_c[2], 5'd0, rs1_p, 3'b000, inst_c[11:10], inst_c[4:3], inst_c[12], 7'b1100011};
                        end
                        3'b111: begin 
                            inst_out = {inst_c[12], inst_c[12], inst_c[12], inst_c[12], inst_c[6:5], inst_c[2], 5'd0, rs1_p, 3'b001, inst_c[11:10], inst_c[4:3], inst_c[12], 7'b1100011};
                        end
                    endcase
                end
                2'b10: begin
                    case(funct3)
                        3'b000: begin 
                            inst_out = {7'b0000000, inst_c[6:2], inst_c[11:7], 3'b001, inst_c[11:7], 7'b0010011};
                        end
                        3'b010: begin 
                            if (inst_c[11:7] != 5'b0) begin
                                inst_out = {4'b0, inst_c[3:2], inst_c[12], inst_c[6:4], 2'b00, 5'd2, 3'b010, inst_c[11:7], 7'b0000011};
                            end
                        end
                        3'b100: begin
                            if (inst_c[12] == 0) begin
                                if (inst_c[6:2] == 0) begin 
                                    inst_out = {12'b0, inst_c[11:7], 3'b000, 5'd0, 7'b1100111};
                                end else begin 
                                    inst_out = {7'b0000000, inst_c[6:2], 5'd0, 3'b000, inst_c[11:7], 7'b0110011};
                                end
                            end else begin
                                if (inst_c[6:2] == 0) begin 
                                    if (inst_c[11:7] == 0) begin 
                                        inst_out = 32'h00100073;
                                    end else begin 
                                        inst_out = {12'b0, inst_c[11:7], 3'b000, 5'd1, 7'b1100111};
                                    end
                                end else begin 
                                    inst_out = {7'b0000000, inst_c[6:2], inst_c[11:7], 3'b000, inst_c[11:7], 7'b0110011};
                                end
                            end
                        end
                        3'b110: begin 
                            inst_out = {4'b0, inst_c[8:7], inst_c[12], inst_c[6:2], 5'd2, 3'b010, inst_c[11:9], 2'b00, 7'b0100011};
                        end
                    endcase
                end
            endcase
        end
    end
endmodule

// =============================================================================
// I-Cache (1KB, 16 Sets, 1D Array Flattened)
// =============================================================================
module ICACHE (
    input          clk,
    input          rst_n,
    input   [31:0] i_cpu_addr,
    output  [127:0] o_cpu_data,
    output         o_cpu_valid,
    output         o_cpu_stall,
    output         o_mem_read,
    output         o_mem_write,
    output  [31:4] o_mem_addr,
    output [127:0] o_mem_wdata,
    input  [127:0] i_mem_rdata,
    input          i_mem_ready
);
    reg         val_0 [0:15]; reg         val_1 [0:15]; reg         val_2 [0:15]; reg         val_3 [0:15];
    reg [23:0]  tag_0 [0:15]; reg [23:0]  tag_1 [0:15]; reg [23:0]  tag_2 [0:15]; reg [23:0]  tag_3 [0:15];
    reg [127:0] dat_0 [0:15]; reg [127:0] dat_1 [0:15]; reg [127:0] dat_2 [0:15]; reg [127:0] dat_3 [0:15];
    reg [2:0]   plru  [0:15];

    wire [3:0]  idx = i_cpu_addr[7:4];
    wire [23:0] cur_tag = i_cpu_addr[31:8];

    wire hit0 = val_0[idx] && (tag_0[idx] == cur_tag);
    wire hit1 = val_1[idx] && (tag_1[idx] == cur_tag);
    wire hit2 = val_2[idx] && (tag_2[idx] == cur_tag);
    wire hit3 = val_3[idx] && (tag_3[idx] == cur_tag);
    wire cache_hit = hit0 || hit1 || hit2 || hit3;
    wire [1:0] hit_way_idx = hit0 ? 2'd0 : hit1 ? 2'd1 : hit2 ? 2'd2 : 2'd3;
    
    assign o_cpu_valid = cache_hit;
    assign o_cpu_stall = !cache_hit;
    assign o_cpu_data = hit0 ? dat_0[idx] : hit1 ? dat_1[idx] : hit2 ? dat_2[idx] : dat_3[idx];

    reg [1:0] state, next_state;
    parameter IDLE = 2'd0, FETCH = 2'd1;

    reg [31:0] prefetch_addr;
    reg        prefetch_pending;
    reg [31:4] mem_addr_reg;
    reg        is_prefetch_reg;

    always @(posedge clk) begin
        if (!rst_n) state <= IDLE;
        else        state <= next_state;
    end

    always @(*) begin
        case(state)
            IDLE: begin
                if (!cache_hit) next_state = FETCH;
                else if (prefetch_pending && i_mem_ready) next_state = FETCH;
                else next_state = IDLE;
            end
            FETCH:    next_state = (i_mem_ready) ? IDLE : FETCH;
            default:  next_state = IDLE;
        endcase
    end

    assign o_mem_read  = (state == FETCH);
    assign o_mem_write = 1'b0;
    assign o_mem_wdata = 128'b0;
    assign o_mem_addr  = mem_addr_reg;
    wire [1:0] fill_way = (plru[mem_addr_reg[7:4]][0] == 0) ? (plru[mem_addr_reg[7:4]][1] == 0 ? 2'd0 : 2'd1) : (plru[mem_addr_reg[7:4]][2] == 0 ? 2'd2 : 2'd3);
    
    integer i;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (i=0; i<16; i=i+1) begin
                val_0[i] <= 0; val_1[i] <= 0; val_2[i] <= 0; val_3[i] <= 0;
                plru[i] <= 0;
            end
            prefetch_pending <= 0; prefetch_addr <= 0; mem_addr_reg <= 0; is_prefetch_reg <= 0;
        end else begin
            if (state == IDLE && cache_hit) begin
                if (hit_way_idx == 0) begin plru[idx][0] <= 1; plru[idx][1] <= 1; end
                else if (hit_way_idx == 1) begin plru[idx][0] <= 1; plru[idx][1] <= 0; end
                else if (hit_way_idx == 2) begin plru[idx][0] <= 0; plru[idx][2] <= 1; end
                else if (hit_way_idx == 3) begin plru[idx][0] <= 0; plru[idx][2] <= 0; end

                if (!prefetch_pending && prefetch_addr != (i_cpu_addr + 16)) begin
                    prefetch_addr <= (i_cpu_addr + 16); prefetch_pending <= 1;
                end
            end

            if (state == IDLE && next_state == FETCH) begin
                if (!cache_hit) begin mem_addr_reg <= i_cpu_addr[31:4]; is_prefetch_reg <= 0; end 
                else begin mem_addr_reg <= prefetch_addr[31:4]; is_prefetch_reg <= 1; end
            end

            if (state == FETCH && i_mem_ready) begin
                if (fill_way == 0) begin val_0[mem_addr_reg[7:4]] <= 1; tag_0[mem_addr_reg[7:4]] <= mem_addr_reg[31:8]; dat_0[mem_addr_reg[7:4]] <= i_mem_rdata; plru[mem_addr_reg[7:4]][0] <= 1; plru[mem_addr_reg[7:4]][1] <= 1; end
                else if (fill_way == 1) begin val_1[mem_addr_reg[7:4]] <= 1; tag_1[mem_addr_reg[7:4]] <= mem_addr_reg[31:8]; dat_1[mem_addr_reg[7:4]] <= i_mem_rdata; plru[mem_addr_reg[7:4]][0] <= 1; plru[mem_addr_reg[7:4]][1] <= 0; end
                else if (fill_way == 2) begin val_2[mem_addr_reg[7:4]] <= 1; tag_2[mem_addr_reg[7:4]] <= mem_addr_reg[31:8]; dat_2[mem_addr_reg[7:4]] <= i_mem_rdata; plru[mem_addr_reg[7:4]][0] <= 0; plru[mem_addr_reg[7:4]][2] <= 1; end
                else if (fill_way == 3) begin val_3[mem_addr_reg[7:4]] <= 1; tag_3[mem_addr_reg[7:4]] <= mem_addr_reg[31:8]; dat_3[mem_addr_reg[7:4]] <= i_mem_rdata; plru[mem_addr_reg[7:4]][0] <= 0; plru[mem_addr_reg[7:4]][2] <= 0; end
                
                if (prefetch_pending && prefetch_addr[31:4] == mem_addr_reg) prefetch_pending <= 0;
                else if (is_prefetch_reg) prefetch_pending <= 0;
            end
        end
    end
endmodule

// =============================================================================
// D-Cache (1KB, 16 Sets 支援 EB 背景寫回)
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
    reg         val_0 [0:15]; reg         val_1 [0:15]; reg         val_2 [0:15]; reg         val_3 [0:15];
    reg         dty_0 [0:15]; reg         dty_1 [0:15]; reg         dty_2 [0:15]; reg         dty_3 [0:15];
    reg [23:0]  tag_0 [0:15]; reg [23:0]  tag_1 [0:15]; reg [23:0]  tag_2 [0:15]; reg [23:0]  tag_3 [0:15];
    reg [127:0] dat_0 [0:15]; reg [127:0] dat_1 [0:15]; reg [127:0] dat_2 [0:15]; reg [127:0] dat_3 [0:15];
    reg [2:0]   plru  [0:15]; 

    wire [3:0]  idx = i_cpu_addr[7:4];
    wire [23:0] cur_tag = i_cpu_addr[31:8];
    wire [1:0]  word_offset = i_cpu_addr[3:2];

    wire hit0 = val_0[idx] && (tag_0[idx] == cur_tag);
    wire hit1 = val_1[idx] && (tag_1[idx] == cur_tag);
    wire hit2 = val_2[idx] && (tag_2[idx] == cur_tag);
    wire hit3 = val_3[idx] && (tag_3[idx] == cur_tag);
    wire hit = hit0 || hit1 || hit2 || hit3;
    wire [1:0] hit_way = hit0 ? 2'd0 : hit1 ? 2'd1 : hit2 ? 2'd2 : 2'd3;
    
    wire cpu_req = i_cpu_ren | i_cpu_wen;
    wire [1:0] evict_way = (plru[idx][0] == 0) ? (plru[idx][1] == 0 ? 2'd0 : 2'd1) : (plru[idx][2] == 0 ? 2'd2 : 2'd3);
    
    parameter IDLE     = 2'd0;
    parameter WAIT_EB  = 2'd1;
    parameter ALLOCATE = 2'd2; 
    parameter FLUSH    = 2'd3;
    
    reg [1:0] state, next_state;
    reg [6:0] flush_cnt; 
    reg [1:0] way_sel_reg;
    
    // 🔥 Background Eviction Buffer (EB)
    reg         eb_valid;
    reg [31:4]  eb_addr;
    reg [127:0] eb_data;

    wire eb_hazard = eb_valid && (eb_addr == i_cpu_addr[31:4]);
    
    wire evict_valid = (evict_way == 2'd0) ? val_0[idx] : (evict_way == 2'd1) ? val_1[idx] : (evict_way == 2'd2) ? val_2[idx] : val_3[idx];
    wire evict_dirty = (evict_way == 2'd0) ? dty_0[idx] : (evict_way == 2'd1) ? dty_1[idx] : (evict_way == 2'd2) ? dty_2[idx] : dty_3[idx];
    wire [23:0] evict_tag = (evict_way == 2'd0) ? tag_0[idx] : (evict_way == 2'd1) ? tag_1[idx] : (evict_way == 2'd2) ? tag_2[idx] : tag_3[idx];
    wire [127:0] evict_data = (evict_way == 2'd0) ? dat_0[idx] : (evict_way == 2'd1) ? dat_1[idx] : (evict_way == 2'd2) ? dat_2[idx] : dat_3[idx];

    wire [3:0] f_idx = flush_cnt[3:0];
    wire [1:0] f_way = flush_cnt[5:4];
    wire f_valid = (f_way == 2'd0) ? val_0[f_idx] : (f_way == 2'd1) ? val_1[f_idx] : (f_way == 2'd2) ? val_2[f_idx] : val_3[f_idx];
    wire f_dirty = (f_way == 2'd0) ? dty_0[f_idx] : (f_way == 2'd1) ? dty_1[f_idx] : (f_way == 2'd2) ? dty_2[f_idx] : dty_3[f_idx];
    wire [23:0] f_tag = (f_way == 2'd0) ? tag_0[f_idx] : (f_way == 2'd1) ? tag_1[f_idx] : (f_way == 2'd2) ? tag_2[f_idx] : tag_3[f_idx];
    wire [127:0] f_data = (f_way == 2'd0) ? dat_0[f_idx] : (f_way == 2'd1) ? dat_1[f_idx] : (f_way == 2'd2) ? dat_2[f_idx] : dat_3[f_idx];

    always @(posedge clk) begin
        if (!rst_n) state <= IDLE;
        else        state <= next_state;
    end

    // 🔥 修復 Critical Path: 將 FLUSH 動作改為透過 EB 暫存器輸出
    always @(*) begin
        case(state)
            IDLE: begin
                if (i_flush) begin
                    if (eb_valid) next_state = WAIT_EB; else next_state = FLUSH;
                end else if (cpu_req && !hit) begin
                    if (eb_valid || eb_hazard) next_state = WAIT_EB; 
                    else next_state = ALLOCATE;
                end else next_state = IDLE;
            end
            WAIT_EB: begin
                if (i_mem_ready) begin
                    if (i_flush) next_state = FLUSH;
                    else next_state = ALLOCATE;
                end else next_state = WAIT_EB;
            end
            ALLOCATE: next_state = (i_mem_ready) ? IDLE : ALLOCATE;
            FLUSH: begin
                if (flush_cnt >= 64) begin
                    next_state = IDLE;
                end else if (f_valid && f_dirty) begin
                    next_state = WAIT_EB; 
                end else begin
                    next_state = FLUSH;
                end
            end
            default: next_state = IDLE;
        endcase
    end

    always @(posedge clk) begin
        if (!rst_n) way_sel_reg <= 0;
        else if (state == IDLE && cpu_req && !hit) way_sel_reg <= evict_way;
    end

    // 🔥 乾淨的 Combinational 輸出，徹底切斷 flush_cnt 的負擔
    always @(*) begin
        o_mem_read  = 0; o_mem_write = 0; o_mem_addr  = 28'b0; o_mem_wdata = 128'b0;
        
        if (state == WAIT_EB) begin
            o_mem_write = 1; o_mem_addr  = eb_addr; o_mem_wdata = eb_data;
        end else if (state == IDLE && eb_valid && !i_flush) begin
            o_mem_write = 1; o_mem_addr  = eb_addr; o_mem_wdata = eb_data;
        end else if (state == ALLOCATE) begin
            o_mem_read  = 1; o_mem_addr  = i_cpu_addr[31:4];
        end
    end

    assign o_cpu_stall = (cpu_req && !hit) || (state != IDLE && state != FLUSH) || eb_hazard;
    assign o_cpu_valid = hit && (state == IDLE) && !eb_hazard;
    
    wire [127:0] hit_block = hit0 ? dat_0[idx] : hit1 ? dat_1[idx] : hit2 ? dat_2[idx] : dat_3[idx];
    assign o_cpu_rdata = (word_offset == 2'b00) ? hit_block[31:0] : (word_offset == 2'b01) ? hit_block[63:32] : (word_offset == 2'b10) ? hit_block[95:64] : hit_block[127:96];

    integer i;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (i=0; i<16; i=i+1) begin
                val_0[i] <= 0; val_1[i] <= 0; val_2[i] <= 0; val_3[i] <= 0;
                dty_0[i] <= 0; dty_1[i] <= 0; dty_2[i] <= 0; dty_3[i] <= 0;
                plru[i] <= 0;
            end
            flush_cnt <= 0; o_flush_done <= 0; eb_valid <= 0;
        end else begin
            if (o_mem_write && i_mem_ready) eb_valid <= 0;
            
            if (state == FLUSH && next_state == IDLE) o_flush_done <= 1;
            else o_flush_done <= 0;
            
            case(state)
                IDLE: begin
                    if (i_flush) flush_cnt <= 0;
                    if (cpu_req && hit && !eb_hazard) begin
                        if (hit_way == 0) begin plru[idx][0] <= 1; plru[idx][1] <= 1; end
                        else if (hit_way == 1) begin plru[idx][0] <= 1; plru[idx][1] <= 0; end
                        else if (hit_way == 2) begin plru[idx][0] <= 0; plru[idx][2] <= 1; end
                        else if (hit_way == 3) begin plru[idx][0] <= 0; plru[idx][2] <= 0; end

                        if (i_cpu_wen) begin
                            if (hit_way == 0) begin dty_0[idx] <= 1;
                                if (word_offset == 2'b00) dat_0[idx][31:0]   <= i_cpu_wdata;
                                if (word_offset == 2'b01) dat_0[idx][63:32]  <= i_cpu_wdata;
                                if (word_offset == 2'b10) dat_0[idx][95:64]  <= i_cpu_wdata;
                                if (word_offset == 2'b11) dat_0[idx][127:96] <= i_cpu_wdata;
                            end else if (hit_way == 1) begin dty_1[idx] <= 1;
                                if (word_offset == 2'b00) dat_1[idx][31:0]   <= i_cpu_wdata;
                                if (word_offset == 2'b01) dat_1[idx][63:32]  <= i_cpu_wdata;
                                if (word_offset == 2'b10) dat_1[idx][95:64]  <= i_cpu_wdata;
                                if (word_offset == 2'b11) dat_1[idx][127:96] <= i_cpu_wdata;
                            end else if (hit_way == 2) begin dty_2[idx] <= 1;
                                if (word_offset == 2'b00) dat_2[idx][31:0]   <= i_cpu_wdata;
                                if (word_offset == 2'b01) dat_2[idx][63:32]  <= i_cpu_wdata;
                                if (word_offset == 2'b10) dat_2[idx][95:64]  <= i_cpu_wdata;
                                if (word_offset == 2'b11) dat_2[idx][127:96] <= i_cpu_wdata;
                            end else if (hit_way == 3) begin dty_3[idx] <= 1;
                                if (word_offset == 2'b00) dat_3[idx][31:0]   <= i_cpu_wdata;
                                if (word_offset == 2'b01) dat_3[idx][63:32]  <= i_cpu_wdata;
                                if (word_offset == 2'b10) dat_3[idx][95:64]  <= i_cpu_wdata;
                                if (word_offset == 2'b11) dat_3[idx][127:96] <= i_cpu_wdata;
                            end
                        end
                    end
                    
                    if (cpu_req && !hit && !eb_valid && !eb_hazard && evict_dirty) begin
                        eb_valid <= 1; eb_addr <= {evict_tag, idx}; eb_data <= evict_data;
                    end
                end
                WAIT_EB: begin
                    // Does nothing here, written asynchronously
                end
                ALLOCATE: begin
                    if (i_mem_ready) begin
                        if (way_sel_reg == 0) begin val_0[idx] <= 1; dty_0[idx] <= 0; tag_0[idx] <= cur_tag; dat_0[idx] <= i_mem_rdata; plru[idx][0] <= 1; plru[idx][1] <= 1; end
                        else if (way_sel_reg == 1) begin val_1[idx] <= 1; dty_1[idx] <= 0; tag_1[idx] <= cur_tag; dat_1[idx] <= i_mem_rdata; plru[idx][0] <= 1; plru[idx][1] <= 0; end
                        else if (way_sel_reg == 2) begin val_2[idx] <= 1; dty_2[idx] <= 0; tag_2[idx] <= cur_tag; dat_2[idx] <= i_mem_rdata; plru[idx][0] <= 0; plru[idx][2] <= 1; end
                        else if (way_sel_reg == 3) begin val_3[idx] <= 1; dty_3[idx] <= 0; tag_3[idx] <= cur_tag; dat_3[idx] <= i_mem_rdata; plru[idx][0] <= 0; plru[idx][2] <= 0; end
                    end
                end
                FLUSH: begin
                    if (flush_cnt < 64) begin
                        if (f_valid && f_dirty) begin
                            eb_valid <= 1; eb_addr <= {f_tag, f_idx}; eb_data <= f_data;
                            if (f_way == 2'd0) dty_0[f_idx] <= 0;
                            else if (f_way == 2'd1) dty_1[f_idx] <= 0;
                            else if (f_way == 2'd2) dty_2[f_idx] <= 0;
                            else dty_3[f_idx] <= 0;
                            flush_cnt <= flush_cnt + 1;
                        end else begin
                            flush_cnt <= flush_cnt + 1;
                        end
                    end
                end
            endcase
        end
    end
endmodule