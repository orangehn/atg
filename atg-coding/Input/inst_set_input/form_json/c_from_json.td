class C_BEQZ_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<3> funct3 = 0b110;
	bits<9> imm;
	bits<3> rs1;
	bits<2> opcode = 0b01;
	let Inst{15-13} = funct3;
	let Inst{12-12} = imm{7-7};
	let Inst{11-10} = imm{3-2};
	let Inst{6-5} = imm{6-5};
	let Inst{4-3} = imm{1-0};
	let Inst{2-2} = imm{4-4};
	let Inst{9-7} = rs1;
	let Inst{1-0} = opcode;
}
def C_BEQZ: Instruction, C_BEQZ_FM, TSFlagTemplate<InstFormatCB>, Sched<[WriteJmp]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins GPRC:$rs1, simm9_lsb0:$imm);
	let AsmString = "c.beqz	$rs1, $imm";
	let Pattern = [];
	let Predicates = [HasStdExtC];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let isTerminator = 1;
	let isBranch = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class C_BNEZ_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<3> funct3 = 0b111;
	bits<9> imm;
	bits<3> rs1;
	bits<2> opcode = 0b01;
	let Inst{15-13} = funct3;
	let Inst{12-12} = imm{7-7};
	let Inst{11-10} = imm{3-2};
	let Inst{6-5} = imm{6-5};
	let Inst{4-3} = imm{1-0};
	let Inst{2-2} = imm{4-4};
	let Inst{9-7} = rs1;
	let Inst{1-0} = opcode;
}
def C_BNEZ: Instruction, C_BNEZ_FM, TSFlagTemplate<InstFormatCB>, Sched<[WriteJmp]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins GPRC:$rs1, simm9_lsb0:$imm);
	let AsmString = "c.bnez	$rs1, $imm";
	let Pattern = [];
	let Predicates = [HasStdExtC];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let isTerminator = 1;
	let isBranch = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class C_ADDI_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<3> funct3 = 0b000;
	bits<10> imm;
	bits<5> rd;
	bits<2> opcode = 0b01;
	let Inst{15-13} = funct3;
	let Inst{12-12} = imm{5-5};
	let Inst{6-2} = imm{4-0};
	let Inst{11-7} = rd;
	let Inst{1-0} = opcode;
}
def C_ADDI: Instruction, C_ADDI_FM, TSFlagTemplate<InstFormatCI>, Sched<[WriteIALU, ReadIALU]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPRNoX0:$rd_wb);
	let InOperandList = (ins GPRNoX0:$rd, simm6nonzero:$imm);
	let AsmString = "c.addi	$rd, $imm";
	let Pattern = [];
	let Predicates = [HasStdExtC];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "$rd = $rd_wb";
}
class C_ADDI16SP_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<3> funct3 = 0b011;
	bits<10> imm;
	bits<5> rd = 0b00010;
	bits<2> opcode = 0b01;
	let Inst{15-13} = funct3;
	let Inst{12-12} = imm{9-9};
	let Inst{6-6} = imm{4-4};
	let Inst{5-5} = imm{6-6};
	let Inst{4-3} = imm{8-7};
	let Inst{2-2} = imm{5-5};
	let Inst{11-7} = rd;
	let Inst{1-0} = opcode;
}
def C_ADDI16SP: Instruction, C_ADDI16SP_FM, TSFlagTemplate<InstFormatCI>, Sched<[WriteIALU, ReadIALU]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs SP:$rd_wb);
	let InOperandList = (ins SP:$rd, simm10_lsb0000nonzero:$imm);
	let AsmString = "c.addi16sp	$rd, $imm";
	let Pattern = [];
	let Predicates = [HasStdExtC];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "$rd = $rd_wb";
}
class C_ADDIW_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<3> funct3 = 0b001;
	bits<10> imm;
	bits<5> rd;
	bits<2> opcode = 0b01;
	let Inst{15-13} = funct3;
	let Inst{12-12} = imm{5-5};
	let Inst{6-2} = imm{4-0};
	let Inst{11-7} = rd;
	let Inst{1-0} = opcode;
}
def C_ADDIW: Instruction, C_ADDIW_FM, TSFlagTemplate<InstFormatCI>, Sched<[WriteIALU32, ReadIALU32]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPRNoX0:$rd_wb);
	let InOperandList = (ins GPRNoX0:$rd, simm6:$imm);
	let AsmString = "c.addiw	$rd, $imm";
	let Pattern = [];
	let Predicates = [HasStdExtC, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "$rd = $rd_wb";
}
class C_ANDI_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<3> funct3 = 0b100;
	bits<9> imm;
	bits<2> funct2 = 0b10;
	bits<3> rs1;
	bits<2> opcode = 0b01;
	let Inst{15-13} = funct3;
	let Inst{12-12} = imm{5-5};
	let Inst{6-2} = imm{4-0};
	let Inst{11-10} = funct2;
	let Inst{9-7} = rs1;
	let Inst{1-0} = opcode;
}
def C_ANDI: Instruction, C_ANDI_FM, TSFlagTemplate<InstFormatCI>, Sched<[WriteIALU, ReadIALU]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPRC:$rs1_wb);
	let InOperandList = (ins GPRC:$rs1, simm6:$imm);
	let AsmString = "c.andi	$rs1, $imm";
	let Pattern = [];
	let Predicates = [HasStdExtC];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "$rs1 = $rs1_wb";
}
class C_FLDSP_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<3> funct3 = 0b001;
	bits<10> imm;
	bits<5> rd;
	bits<2> opcode = 0b10;
	let Inst{15-13} = funct3;
	let Inst{12-12} = imm{5-5};
	let Inst{6-5} = imm{4-3};
	let Inst{4-2} = imm{8-6};
	let Inst{11-7} = rd;
	let Inst{1-0} = opcode;
}
def C_FLDSP: Instruction, C_FLDSP_FM, TSFlagTemplate<InstFormatCI>, Sched<[WriteFLD64, ReadMemBase]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR64:$rd);
	let InOperandList = (ins SP:$rs1, uimm9_lsb000:$imm);
	let AsmString = "c.fldsp	$rd, ${imm}(${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtC, HasStdExtD];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class C_FLWSP_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<3> funct3 = 0b011;
	bits<10> imm;
	bits<5> rd;
	bits<2> opcode = 0b10;
	let Inst{15-13} = funct3;
	let Inst{12-12} = imm{5-5};
	let Inst{6-4} = imm{4-2};
	let Inst{3-2} = imm{7-6};
	let Inst{11-7} = rd;
	let Inst{1-0} = opcode;
}
def C_FLWSP: Instruction, C_FLWSP_FM, TSFlagTemplate<InstFormatCI>, Sched<[WriteFLD32, ReadMemBase]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR32:$rd);
	let InOperandList = (ins SP:$rs1, uimm8_lsb00:$imm);
	let AsmString = "c.flwsp	$rd, ${imm}(${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtC, HasStdExtF, IsRV32];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "RISCV32Only_";
	let Constraints = "";
}
class C_LDSP_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<3> funct3 = 0b011;
	bits<10> imm;
	bits<5> rd;
	bits<2> opcode = 0b10;
	let Inst{15-13} = funct3;
	let Inst{12-12} = imm{5-5};
	let Inst{6-5} = imm{4-3};
	let Inst{4-2} = imm{8-6};
	let Inst{11-7} = rd;
	let Inst{1-0} = opcode;
}
def C_LDSP: Instruction, C_LDSP_FM, TSFlagTemplate<InstFormatCI>, Sched<[WriteLDD, ReadMemBase]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPRNoX0:$rd);
	let InOperandList = (ins SP:$rs1, uimm9_lsb000:$imm);
	let AsmString = "c.ldsp	$rd, ${imm}(${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtC, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class C_LI_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<3> funct3 = 0b010;
	bits<10> imm;
	bits<5> rd;
	bits<2> opcode = 0b01;
	let Inst{15-13} = funct3;
	let Inst{12-12} = imm{5-5};
	let Inst{6-2} = imm{4-0};
	let Inst{11-7} = rd;
	let Inst{1-0} = opcode;
}
def C_LI: Instruction, C_LI_FM, TSFlagTemplate<InstFormatCI>, Sched<[WriteIALU]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPRNoX0:$rd);
	let InOperandList = (ins simm6:$imm);
	let AsmString = "c.li	$rd, $imm";
	let Pattern = [];
	let Predicates = [HasStdExtC];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class C_LUI_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<3> funct3 = 0b011;
	bits<10> imm;
	bits<5> rd;
	bits<2> opcode = 0b01;
	let Inst{15-13} = funct3;
	let Inst{12-12} = imm{5-5};
	let Inst{6-2} = imm{4-0};
	let Inst{11-7} = rd;
	let Inst{1-0} = opcode;
}
def C_LUI: Instruction, C_LUI_FM, TSFlagTemplate<InstFormatCI>, Sched<[WriteIALU]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPRNoX0X2:$rd);
	let InOperandList = (ins c_lui_imm:$imm);
	let AsmString = "c.lui	$rd, $imm";
	let Pattern = [];
	let Predicates = [HasStdExtC];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class C_LWSP_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<3> funct3 = 0b010;
	bits<10> imm;
	bits<5> rd;
	bits<2> opcode = 0b10;
	let Inst{15-13} = funct3;
	let Inst{12-12} = imm{5-5};
	let Inst{6-4} = imm{4-2};
	let Inst{3-2} = imm{7-6};
	let Inst{11-7} = rd;
	let Inst{1-0} = opcode;
}
def C_LWSP: Instruction, C_LWSP_FM, TSFlagTemplate<InstFormatCI>, Sched<[WriteLDW, ReadMemBase]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPRNoX0:$rd);
	let InOperandList = (ins SP:$rs1, uimm8_lsb00:$imm);
	let AsmString = "c.lwsp	$rd, ${imm}(${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtC];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class C_NOP_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<3> funct3 = 0b000;
	bits<5> rd = 0b00000;
	bits<5> immediate = 0b00000;
	bits<2> opcode = 0b01;
	let Inst{15-13} = funct3;
	let Inst{12-12} = 0b0;
	let Inst{11-7} = rd;
	let Inst{6-2} = immediate;
	let Inst{1-0} = opcode;
}
def C_NOP: Instruction, C_NOP_FM, TSFlagTemplate<InstFormatCI>, Sched<[WriteNop]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins);
	let AsmString = "c.nop	";
	let Pattern = [];
	let Predicates = [HasStdExtC];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class C_SLLI_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<3> funct3 = 0b000;
	bits<10> imm;
	bits<5> rd;
	bits<2> opcode = 0b10;
	let Inst{15-13} = funct3;
	let Inst{12-12} = imm{5-5};
	let Inst{6-2} = imm{4-0};
	let Inst{11-7} = rd;
	let Inst{1-0} = opcode;
}
def C_SLLI: Instruction, C_SLLI_FM, TSFlagTemplate<InstFormatCI>, Sched<[WriteShift, ReadShift]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPRNoX0:$rd_wb);
	let InOperandList = (ins GPRNoX0:$rd, uimmlog2xlennonzero:$imm);
	let AsmString = "c.slli	$rd, $imm";
	let Pattern = [];
	let Predicates = [HasStdExtC];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "$rd = $rd_wb";
}
class C_SRAI_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<3> funct3 = 0b100;
	bits<9> imm;
	bits<2> funct2 = 0b01;
	bits<3> rs1;
	bits<2> opcode = 0b01;
	let Inst{15-13} = funct3;
	let Inst{12-12} = imm{5-5};
	let Inst{6-2} = imm{4-0};
	let Inst{11-10} = funct2;
	let Inst{9-7} = rs1;
	let Inst{1-0} = opcode;
}
def C_SRAI: Instruction, C_SRAI_FM, TSFlagTemplate<InstFormatCI>, Sched<[WriteShift, ReadShift]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPRC:$rs1_wb);
	let InOperandList = (ins GPRC:$rs1, uimmlog2xlennonzero:$imm);
	let AsmString = "c.srai	$rs1, $imm";
	let Pattern = [];
	let Predicates = [HasStdExtC];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "$rs1 = $rs1_wb";
}
class C_SRLI_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<3> funct3 = 0b100;
	bits<9> imm;
	bits<2> funct2 = 0b00;
	bits<3> rs1;
	bits<2> opcode = 0b01;
	let Inst{15-13} = funct3;
	let Inst{12-12} = imm{5-5};
	let Inst{6-2} = imm{4-0};
	let Inst{11-10} = funct2;
	let Inst{9-7} = rs1;
	let Inst{1-0} = opcode;
}
def C_SRLI: Instruction, C_SRLI_FM, TSFlagTemplate<InstFormatCI>, Sched<[WriteShift, ReadShift]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPRC:$rs1_wb);
	let InOperandList = (ins GPRC:$rs1, uimmlog2xlennonzero:$imm);
	let AsmString = "c.srli	$rs1, $imm";
	let Pattern = [];
	let Predicates = [HasStdExtC];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "$rs1 = $rs1_wb";
}
class C_ADDI4SPN_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<3> funct3 = 0b000;
	bits<10> imm;
	bits<3> rd;
	bits<2> opcode = 0b00;
	let Inst{15-13} = funct3;
	let Inst{12-11} = imm{5-4};
	let Inst{10-7} = imm{9-6};
	let Inst{6-6} = imm{2-2};
	let Inst{5-5} = imm{3-3};
	let Inst{4-2} = rd;
	let Inst{1-0} = opcode;
}
def C_ADDI4SPN: Instruction, C_ADDI4SPN_FM, TSFlagTemplate<InstFormatCIW>, Sched<[WriteIALU, ReadIALU]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPRC:$rd);
	let InOperandList = (ins SP:$rs1, uimm10_lsb00nonzero:$imm);
	let AsmString = "c.addi4spn	$rd, $rs1, $imm";
	let Pattern = [];
	let Predicates = [HasStdExtC];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [X2];
	let DecoderNamespace = "";
	let Constraints = "";
}
class C_J_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<3> funct3 = 0b101;
	bits<11> offset;
	bits<2> opcode = 0b01;
	let Inst{15-13} = funct3;
	let Inst{12-12} = offset{10-10};
	let Inst{11-11} = offset{3-3};
	let Inst{10-9} = offset{8-7};
	let Inst{8-8} = offset{9-9};
	let Inst{7-7} = offset{5-5};
	let Inst{6-6} = offset{6-6};
	let Inst{5-3} = offset{2-0};
	let Inst{2-2} = offset{4-4};
	let Inst{1-0} = opcode;
}
def C_J: Instruction, C_J_FM, TSFlagTemplate<InstFormatCJ>, Sched<[WriteJmp]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins simm12_lsb0:$offset);
	let AsmString = "c.j	$offset";
	let Pattern = [];
	let Predicates = [HasStdExtC];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let isBarrier = 1;
	let isTerminator = 1;
	let isBranch = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class C_JAL_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<3> funct3 = 0b001;
	bits<11> offset;
	bits<2> opcode = 0b01;
	let Inst{15-13} = funct3;
	let Inst{12-12} = offset{10-10};
	let Inst{11-11} = offset{3-3};
	let Inst{10-9} = offset{8-7};
	let Inst{8-8} = offset{9-9};
	let Inst{7-7} = offset{5-5};
	let Inst{6-6} = offset{6-6};
	let Inst{5-3} = offset{2-0};
	let Inst{2-2} = offset{4-4};
	let Inst{1-0} = opcode;
}
def C_JAL: Instruction, C_JAL_FM, TSFlagTemplate<InstFormatCJ>, Sched<[WriteJal]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins simm12_lsb0:$offset);
	let AsmString = "c.jal	$offset";
	let Pattern = [];
	let Predicates = [HasStdExtC, IsRV32];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let isCall = 1;
	let Defs = [X1];
	let Uses = [];
	let DecoderNamespace = "RISCV32Only_";
	let Constraints = "";
}
class C_FLD_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<3> funct3 = 0b001;
	bits<8> imm;
	bits<3> rs1;
	bits<3> rd;
	bits<2> opcode = 0b00;
	let Inst{15-13} = funct3;
	let Inst{12-10} = imm{5-3};
	let Inst{6-5} = imm{7-6};
	let Inst{9-7} = rs1;
	let Inst{4-2} = rd;
	let Inst{1-0} = opcode;
}
def C_FLD: Instruction, C_FLD_FM, TSFlagTemplate<InstFormatCL>, Sched<[WriteFLD64, ReadMemBase]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR64C:$rd);
	let InOperandList = (ins GPRC:$rs1, uimm8_lsb000:$imm);
	let AsmString = "c.fld	$rd, ${imm}(${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtC, HasStdExtD];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class C_FLW_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<3> funct3 = 0b011;
	bits<7> imm;
	bits<3> rs1;
	bits<3> rd;
	bits<2> opcode = 0b00;
	let Inst{15-13} = funct3;
	let Inst{12-10} = imm{5-3};
	let Inst{6-6} = imm{2-2};
	let Inst{5-5} = imm{6-6};
	let Inst{9-7} = rs1;
	let Inst{4-2} = rd;
	let Inst{1-0} = opcode;
}
def C_FLW: Instruction, C_FLW_FM, TSFlagTemplate<InstFormatCL>, Sched<[WriteFLD32, ReadMemBase]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR32C:$rd);
	let InOperandList = (ins GPRC:$rs1, uimm7_lsb00:$imm);
	let AsmString = "c.flw	$rd, ${imm}(${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtC, HasStdExtF, IsRV32];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "RISCV32Only_";
	let Constraints = "";
}
class C_LD_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<3> funct3 = 0b011;
	bits<8> imm;
	bits<3> rs1;
	bits<3> rd;
	bits<2> opcode = 0b00;
	let Inst{15-13} = funct3;
	let Inst{12-10} = imm{5-3};
	let Inst{6-5} = imm{7-6};
	let Inst{9-7} = rs1;
	let Inst{4-2} = rd;
	let Inst{1-0} = opcode;
}
def C_LD: Instruction, C_LD_FM, TSFlagTemplate<InstFormatCL>, Sched<[WriteLDD, ReadMemBase]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPRC:$rd);
	let InOperandList = (ins GPRC:$rs1, uimm8_lsb000:$imm);
	let AsmString = "c.ld	$rd, ${imm}(${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtC, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class C_LW_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<3> funct3 = 0b010;
	bits<7> imm;
	bits<3> rs1;
	bits<3> rd;
	bits<2> opcode = 0b00;
	let Inst{15-13} = funct3;
	let Inst{12-10} = imm{5-3};
	let Inst{6-6} = imm{2-2};
	let Inst{5-5} = imm{6-6};
	let Inst{9-7} = rs1;
	let Inst{4-2} = rd;
	let Inst{1-0} = opcode;
}
def C_LW: Instruction, C_LW_FM, TSFlagTemplate<InstFormatCL>, Sched<[WriteLDW, ReadMemBase]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPRC:$rd);
	let InOperandList = (ins GPRC:$rs1, uimm7_lsb00:$imm);
	let AsmString = "c.lw	$rd, ${imm}(${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtC];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class C_ADD_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<4> funct4 = 0b1001;
	bits<5> rs1;
	bits<5> rs2;
	bits<2> opcode = 0b10;
	let Inst{15-12} = funct4;
	let Inst{11-7} = rs1;
	let Inst{6-2} = rs2;
	let Inst{1-0} = opcode;
}
def C_ADD: Instruction, C_ADD_FM, TSFlagTemplate<InstFormatCR>, Sched<[WriteIALU, ReadIALU, ReadIALU]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPRNoX0:$rs1_wb);
	let InOperandList = (ins GPRNoX0:$rs1, GPRNoX0:$rs2);
	let AsmString = "c.add	$rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtC];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "$rs1 = $rs1_wb";
}
class C_ADDW_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<6> funct6 = 0b100111;
	bits<3> rd;
	bits<2> funct2 = 0b01;
	bits<3> rs2;
	bits<2> opcode = 0b01;
	let Inst{15-10} = funct6;
	let Inst{9-7} = rd;
	let Inst{6-5} = funct2;
	let Inst{4-2} = rs2;
	let Inst{1-0} = opcode;
}
def C_ADDW: Instruction, C_ADDW_FM, TSFlagTemplate<InstFormatCR>, Sched<[WriteIALU32, ReadIALU32, ReadIALU32]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPRC:$rd_wb);
	let InOperandList = (ins GPRC:$rd, GPRC:$rs2);
	let AsmString = "c.addw	$rd, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtC, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "$rd = $rd_wb";
}
class C_AND_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<6> funct6 = 0b100011;
	bits<3> rd;
	bits<2> funct2 = 0b11;
	bits<3> rs2;
	bits<2> opcode = 0b01;
	let Inst{15-10} = funct6;
	let Inst{9-7} = rd;
	let Inst{6-5} = funct2;
	let Inst{4-2} = rs2;
	let Inst{1-0} = opcode;
}
def C_AND: Instruction, C_AND_FM, TSFlagTemplate<InstFormatCR>, Sched<[WriteIALU, ReadIALU, ReadIALU]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPRC:$rd_wb);
	let InOperandList = (ins GPRC:$rd, GPRC:$rs2);
	let AsmString = "c.and	$rd, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtC];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "$rd = $rd_wb";
}
class C_EBREAK_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<4> funct4 = 0b1001;
	bits<5> rd = 0b00000;
	bits<5> rs2 = 0b00000;
	bits<2> opcode = 0b10;
	let Inst{15-12} = funct4;
	let Inst{11-7} = rd;
	let Inst{6-2} = rs2;
	let Inst{1-0} = opcode;
}
def C_EBREAK: Instruction, C_EBREAK_FM, TSFlagTemplate<InstFormatCR>, Sched<[]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins);
	let AsmString = "c.ebreak	";
	let Pattern = [];
	let Predicates = [HasStdExtC];
	let hasSideEffects = 1;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class C_JALR_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<4> funct4 = 0b1001;
	bits<5> rs1;
	bits<5> rd = 0b00000;
	bits<2> opcode = 0b10;
	let Inst{15-12} = funct4;
	let Inst{11-7} = rs1;
	let Inst{6-2} = rd;
	let Inst{1-0} = opcode;
}
def C_JALR: Instruction, C_JALR_FM, TSFlagTemplate<InstFormatCR>, Sched<[WriteJalr, ReadJalr]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins GPRNoX0:$rs1);
	let AsmString = "c.jalr	$rs1";
	let Pattern = [];
	let Predicates = [HasStdExtC];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let isCall = 1;
	let Defs = [X1];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class C_JR_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<4> funct4 = 0b1000;
	bits<5> rs1;
	bits<5> rd = 0b00000;
	bits<2> opcode = 0b10;
	let Inst{15-12} = funct4;
	let Inst{11-7} = rs1;
	let Inst{6-2} = rd;
	let Inst{1-0} = opcode;
}
def C_JR: Instruction, C_JR_FM, TSFlagTemplate<InstFormatCR>, Sched<[WriteJmpReg]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins GPRNoX0:$rs1);
	let AsmString = "c.jr	$rs1";
	let Pattern = [];
	let Predicates = [HasStdExtC];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let isBarrier = 1;
	let isTerminator = 1;
	let isBranch = 1;
	let isIndirectBranch = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class C_MV_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<4> funct4 = 0b1000;
	bits<5> rs1;
	bits<5> rs2;
	bits<2> opcode = 0b10;
	let Inst{15-12} = funct4;
	let Inst{11-7} = rs1;
	let Inst{6-2} = rs2;
	let Inst{1-0} = opcode;
}
def C_MV: Instruction, C_MV_FM, TSFlagTemplate<InstFormatCR>, Sched<[WriteIALU, ReadIALU]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPRNoX0:$rs1);
	let InOperandList = (ins GPRNoX0:$rs2);
	let AsmString = "c.mv	$rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtC];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let isAsCheapAsAMove = 1;
	let Defs = [];
	let Uses = [];
	let isMoveReg = 1;
	let DecoderNamespace = "";
	let Constraints = "";
}
class C_OR_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<6> funct6 = 0b100011;
	bits<3> rd;
	bits<2> funct2 = 0b10;
	bits<3> rs2;
	bits<2> opcode = 0b01;
	let Inst{15-10} = funct6;
	let Inst{9-7} = rd;
	let Inst{6-5} = funct2;
	let Inst{4-2} = rs2;
	let Inst{1-0} = opcode;
}
def C_OR: Instruction, C_OR_FM, TSFlagTemplate<InstFormatCR>, Sched<[WriteIALU, ReadIALU, ReadIALU]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPRC:$rd_wb);
	let InOperandList = (ins GPRC:$rd, GPRC:$rs2);
	let AsmString = "c.or	$rd, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtC];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "$rd = $rd_wb";
}
class C_SUB_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<6> funct6 = 0b100011;
	bits<3> rd;
	bits<2> funct2 = 0b00;
	bits<3> rs2;
	bits<2> opcode = 0b01;
	let Inst{15-10} = funct6;
	let Inst{9-7} = rd;
	let Inst{6-5} = funct2;
	let Inst{4-2} = rs2;
	let Inst{1-0} = opcode;
}
def C_SUB: Instruction, C_SUB_FM, TSFlagTemplate<InstFormatCR>, Sched<[WriteIALU, ReadIALU, ReadIALU]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPRC:$rd_wb);
	let InOperandList = (ins GPRC:$rd, GPRC:$rs2);
	let AsmString = "c.sub	$rd, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtC];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "$rd = $rd_wb";
}
class C_SUBW_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<6> funct6 = 0b100111;
	bits<3> rd;
	bits<2> funct2 = 0b00;
	bits<3> rs2;
	bits<2> opcode = 0b01;
	let Inst{15-10} = funct6;
	let Inst{9-7} = rd;
	let Inst{6-5} = funct2;
	let Inst{4-2} = rs2;
	let Inst{1-0} = opcode;
}
def C_SUBW: Instruction, C_SUBW_FM, TSFlagTemplate<InstFormatCR>, Sched<[WriteIALU32, ReadIALU32, ReadIALU32]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPRC:$rd_wb);
	let InOperandList = (ins GPRC:$rd, GPRC:$rs2);
	let AsmString = "c.subw	$rd, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtC, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "$rd = $rd_wb";
}
class C_XOR_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<6> funct6 = 0b100011;
	bits<3> rd;
	bits<2> funct2 = 0b01;
	bits<3> rs2;
	bits<2> opcode = 0b01;
	let Inst{15-10} = funct6;
	let Inst{9-7} = rd;
	let Inst{6-5} = funct2;
	let Inst{4-2} = rs2;
	let Inst{1-0} = opcode;
}
def C_XOR: Instruction, C_XOR_FM, TSFlagTemplate<InstFormatCR>, Sched<[WriteIALU, ReadIALU, ReadIALU]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPRC:$rd_wb);
	let InOperandList = (ins GPRC:$rd, GPRC:$rs2);
	let AsmString = "c.xor	$rd, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtC];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "$rd = $rd_wb";
}
class C_FSD_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<3> funct3 = 0b101;
	bits<8> imm;
	bits<3> rs1;
	bits<3> rs2;
	bits<2> opcode = 0b00;
	let Inst{15-13} = funct3;
	let Inst{12-10} = imm{5-3};
	let Inst{6-5} = imm{7-6};
	let Inst{9-7} = rs1;
	let Inst{4-2} = rs2;
	let Inst{1-0} = opcode;
}
def C_FSD: Instruction, C_FSD_FM, TSFlagTemplate<InstFormatCS>, Sched<[WriteFST64, ReadStoreData, ReadMemBase]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins FPR64C:$rs2, GPRC:$rs1, uimm8_lsb000:$imm);
	let AsmString = "c.fsd	$rs2, ${imm}(${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtC, HasStdExtD];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class C_FSW_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<3> funct3 = 0b111;
	bits<7> imm;
	bits<3> rs1;
	bits<3> rs2;
	bits<2> opcode = 0b00;
	let Inst{15-13} = funct3;
	let Inst{12-10} = imm{5-3};
	let Inst{6-6} = imm{2-2};
	let Inst{5-5} = imm{6-6};
	let Inst{9-7} = rs1;
	let Inst{4-2} = rs2;
	let Inst{1-0} = opcode;
}
def C_FSW: Instruction, C_FSW_FM, TSFlagTemplate<InstFormatCS>, Sched<[WriteFST32, ReadStoreData, ReadMemBase]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins FPR32C:$rs2, GPRC:$rs1, uimm7_lsb00:$imm);
	let AsmString = "c.fsw	$rs2, ${imm}(${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtC, HasStdExtF, IsRV32];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "RISCV32Only_";
	let Constraints = "";
}
class C_SD_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<3> funct3 = 0b111;
	bits<8> imm;
	bits<3> rs1;
	bits<3> rs2;
	bits<2> opcode = 0b00;
	let Inst{15-13} = funct3;
	let Inst{12-10} = imm{5-3};
	let Inst{6-5} = imm{7-6};
	let Inst{9-7} = rs1;
	let Inst{4-2} = rs2;
	let Inst{1-0} = opcode;
}
def C_SD: Instruction, C_SD_FM, TSFlagTemplate<InstFormatCS>, Sched<[WriteSTD, ReadStoreData, ReadMemBase]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins GPRC:$rs2, GPRC:$rs1, uimm8_lsb000:$imm);
	let AsmString = "c.sd	$rs2, ${imm}(${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtC, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class C_SW_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<3> funct3 = 0b110;
	bits<7> imm;
	bits<3> rs1;
	bits<3> rs2;
	bits<2> opcode = 0b00;
	let Inst{15-13} = funct3;
	let Inst{12-10} = imm{5-3};
	let Inst{6-6} = imm{2-2};
	let Inst{5-5} = imm{6-6};
	let Inst{9-7} = rs1;
	let Inst{4-2} = rs2;
	let Inst{1-0} = opcode;
}
def C_SW: Instruction, C_SW_FM, TSFlagTemplate<InstFormatCS>, Sched<[WriteSTW, ReadStoreData, ReadMemBase]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins GPRC:$rs2, GPRC:$rs1, uimm7_lsb00:$imm);
	let AsmString = "c.sw	$rs2, ${imm}(${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtC];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class C_FSDSP_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<3> funct3 = 0b101;
	bits<10> imm;
	bits<5> rs2;
	bits<2> opcode = 0b10;
	let Inst{15-13} = funct3;
	let Inst{12-10} = imm{5-3};
	let Inst{9-7} = imm{8-6};
	let Inst{6-2} = rs2;
	let Inst{1-0} = opcode;
}
def C_FSDSP: Instruction, C_FSDSP_FM, TSFlagTemplate<InstFormatCSS>, Sched<[WriteFST64, ReadStoreData, ReadMemBase]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins FPR64:$rs2, SP:$rs1, uimm9_lsb000:$imm);
	let AsmString = "c.fsdsp	$rs2, ${imm}(${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtC, HasStdExtD];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class C_FSWSP_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<3> funct3 = 0b111;
	bits<10> imm;
	bits<5> rs2;
	bits<2> opcode = 0b10;
	let Inst{15-13} = funct3;
	let Inst{12-9} = imm{5-2};
	let Inst{8-7} = imm{7-6};
	let Inst{6-2} = rs2;
	let Inst{1-0} = opcode;
}
def C_FSWSP: Instruction, C_FSWSP_FM, TSFlagTemplate<InstFormatCSS>, Sched<[WriteFST32, ReadStoreData, ReadMemBase]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins FPR32:$rs2, SP:$rs1, uimm8_lsb00:$imm);
	let AsmString = "c.fswsp	$rs2, ${imm}(${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtC, HasStdExtF, IsRV32];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "RISCV32Only_";
	let Constraints = "";
}
class C_SDSP_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<3> funct3 = 0b111;
	bits<10> imm;
	bits<5> rs2;
	bits<2> opcode = 0b10;
	let Inst{15-13} = funct3;
	let Inst{12-10} = imm{5-3};
	let Inst{9-7} = imm{8-6};
	let Inst{6-2} = rs2;
	let Inst{1-0} = opcode;
}
def C_SDSP: Instruction, C_SDSP_FM, TSFlagTemplate<InstFormatCSS>, Sched<[WriteSTD, ReadStoreData, ReadMemBase]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins GPR:$rs2, SP:$rs1, uimm9_lsb000:$imm);
	let AsmString = "c.sdsp	$rs2, ${imm}(${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtC, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class C_SWSP_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<3> funct3 = 0b110;
	bits<10> imm;
	bits<5> rs2;
	bits<2> opcode = 0b10;
	let Inst{15-13} = funct3;
	let Inst{12-9} = imm{5-2};
	let Inst{8-7} = imm{7-6};
	let Inst{6-2} = rs2;
	let Inst{1-0} = opcode;
}
def C_SWSP: Instruction, C_SWSP_FM, TSFlagTemplate<InstFormatCSS>, Sched<[WriteSTW, ReadStoreData, ReadMemBase]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins GPR:$rs2, SP:$rs1, uimm8_lsb00:$imm);
	let AsmString = "c.swsp	$rs2, ${imm}(${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtC];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class C_SLLI64_HINT_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<3> funct3 = 0b000;
	bits<5> rd;
	bits<2> Opcode = 0b10;
	let Inst{15-13} = funct3;
	let Inst{12-12} = 0b0;
	let Inst{11-7} = rd;
	let Inst{6-2} = 0b00000;
	let Inst{1-0} = Opcode;
}
def C_SLLI64_HINT: Instruction, C_SLLI64_HINT_FM, TSFlagTemplate<InstFormatCI>, Sched<[WriteShift, ReadShift]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd_wb);
	let InOperandList = (ins GPR:$rd);
	let AsmString = "c.slli64	$rd";
	let Pattern = [];
	let Predicates = [HasStdExtC, HasRVCHints];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "$rd = $rd_wb";
}
class C_SRLI64_HINT_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<3> funct3 = 0b100;
	bits<5> rd;
	bits<2> Opcode = 0b01;
	let Inst{15-13} = funct3;
	let Inst{12-10} = 0b000;
	let Inst{9-7} = rd{2-0};
	let Inst{6-2} = 0b00000;
	let Inst{1-0} = Opcode;
}
def C_SRLI64_HINT: Instruction, C_SRLI64_HINT_FM, TSFlagTemplate<InstFormatCI>, Sched<[WriteShift, ReadShift]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPRC:$rd_wb);
	let InOperandList = (ins GPRC:$rd);
	let AsmString = "c.srli64	$rd";
	let Pattern = [];
	let Predicates = [HasStdExtC, HasRVCHints];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "$rd = $rd_wb";
}
class C_SRAI64_HINT_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<3> funct3 = 0b100;
	bits<5> rd;
	bits<2> Opcode = 0b01;
	let Inst{15-13} = funct3;
	let Inst{12-10} = 0b001;
	let Inst{9-7} = rd{2-0};
	let Inst{6-2} = 0b00000;
	let Inst{1-0} = Opcode;
}
def C_SRAI64_HINT: Instruction, C_SRAI64_HINT_FM, TSFlagTemplate<InstFormatCI>, Sched<[WriteShift, ReadShift]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPRC:$rd_wb);
	let InOperandList = (ins GPRC:$rd);
	let AsmString = "c.srai64	$rd";
	let Pattern = [];
	let Predicates = [HasStdExtC, HasRVCHints];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "$rd = $rd_wb";
}