class ATGInst;
class TSFlagTemplate<InstFormat format, RISCVVConstraint rvv_constraint = NoConstraint>{
  bits<64> TSFlags = 0;
  let TSFlags{4-0} = format.Value;

  // Defaults
  RISCVVConstraint RVVConstraint = rvv_constraint;
  let TSFlags{7-5} = RVVConstraint.Value;

  bits<3> VLMul = 0;
  let TSFlags{10-8} = VLMul;

  bit HasDummyMask = 0;
  let TSFlags{11} = HasDummyMask;

  bit WritesElement0 = 0;
  let TSFlags{12} = WritesElement0;

  bit HasMergeOp = 0;
  let TSFlags{13} = HasMergeOp;

  bit HasSEWOp = 0;
  let TSFlags{14} = HasSEWOp;

  bit HasVLOp = 0;
  let TSFlags{15} = HasVLOp;
}
                  
class BEQ_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<12> imm12;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b000;
	bits<7> opcode = 0b1100011;
	let Inst{31-31} = imm12{11-11};
	let Inst{30-25} = imm12{9-4};
	let Inst{11-8} = imm12{3-0};
	let Inst{7-7} = imm12{10-10};
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{6-0} = opcode;
}
def BEQ: Instruction, BEQ_FM, TSFlagTemplate<InstFormatB>, Sched<[WriteJmp, ReadJmp, ReadJmp]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, simm13_lsb0:$imm12);
	let AsmString = "beq	$rs1, $rs2, $imm12";
	let Pattern = [];
	let Predicates = [];
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
class BGE_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<12> imm12;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b101;
	bits<7> opcode = 0b1100011;
	let Inst{31-31} = imm12{11-11};
	let Inst{30-25} = imm12{9-4};
	let Inst{11-8} = imm12{3-0};
	let Inst{7-7} = imm12{10-10};
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{6-0} = opcode;
}
def BGE: Instruction, BGE_FM, TSFlagTemplate<InstFormatB>, Sched<[WriteJmp, ReadJmp, ReadJmp]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, simm13_lsb0:$imm12);
	let AsmString = "bge	$rs1, $rs2, $imm12";
	let Pattern = [];
	let Predicates = [];
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
class BGEU_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<12> imm12;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b111;
	bits<7> opcode = 0b1100011;
	let Inst{31-31} = imm12{11-11};
	let Inst{30-25} = imm12{9-4};
	let Inst{11-8} = imm12{3-0};
	let Inst{7-7} = imm12{10-10};
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{6-0} = opcode;
}
def BGEU: Instruction, BGEU_FM, TSFlagTemplate<InstFormatB>, Sched<[WriteJmp, ReadJmp, ReadJmp]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, simm13_lsb0:$imm12);
	let AsmString = "bgeu	$rs1, $rs2, $imm12";
	let Pattern = [];
	let Predicates = [];
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
class BLT_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<12> imm12;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b100;
	bits<7> opcode = 0b1100011;
	let Inst{31-31} = imm12{11-11};
	let Inst{30-25} = imm12{9-4};
	let Inst{11-8} = imm12{3-0};
	let Inst{7-7} = imm12{10-10};
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{6-0} = opcode;
}
def BLT: Instruction, BLT_FM, TSFlagTemplate<InstFormatB>, Sched<[WriteJmp, ReadJmp, ReadJmp]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, simm13_lsb0:$imm12);
	let AsmString = "blt	$rs1, $rs2, $imm12";
	let Pattern = [];
	let Predicates = [];
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
class BLTU_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<12> imm12;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b110;
	bits<7> opcode = 0b1100011;
	let Inst{31-31} = imm12{11-11};
	let Inst{30-25} = imm12{9-4};
	let Inst{11-8} = imm12{3-0};
	let Inst{7-7} = imm12{10-10};
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{6-0} = opcode;
}
def BLTU: Instruction, BLTU_FM, TSFlagTemplate<InstFormatB>, Sched<[WriteJmp, ReadJmp, ReadJmp]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, simm13_lsb0:$imm12);
	let AsmString = "bltu	$rs1, $rs2, $imm12";
	let Pattern = [];
	let Predicates = [];
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
class BNE_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<12> imm12;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<7> opcode = 0b1100011;
	let Inst{31-31} = imm12{11-11};
	let Inst{30-25} = imm12{9-4};
	let Inst{11-8} = imm12{3-0};
	let Inst{7-7} = imm12{10-10};
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{6-0} = opcode;
}
def BNE: Instruction, BNE_FM, TSFlagTemplate<InstFormatB>, Sched<[WriteJmp, ReadJmp, ReadJmp]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, simm13_lsb0:$imm12);
	let AsmString = "bne	$rs1, $rs2, $imm12";
	let Pattern = [];
	let Predicates = [];
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
class ADDI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<12> imm12;
	bits<5> rs1;
	bits<3> funct3 = 0b000;
	bits<5> rd;
	bits<7> opcode = 0b0010011;
	let Inst{31-20} = imm12;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def ADDI: Instruction, ADDI_FM, TSFlagTemplate<InstFormatI>, Sched<[WriteIALU, ReadIALU]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, simm12:$imm12);
	let AsmString = "addi	$rd, $rs1, $imm12";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let isReMaterializable = 1;
	let isAsCheapAsAMove = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class ADDIW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<12> imm12;
	bits<5> rs1;
	bits<3> funct3 = 0b000;
	bits<5> rd;
	bits<7> opcode = 0b0011011;
	let Inst{31-20} = imm12;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def ADDIW: Instruction, ADDIW_FM, TSFlagTemplate<InstFormatI>, Sched<[WriteIALU32, ReadIALU32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, simm12:$imm12);
	let AsmString = "addiw	$rd, $rs1, $imm12";
	let Pattern = [];
	let Predicates = [IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class ANDI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<12> imm12;
	bits<5> rs1;
	bits<3> funct3 = 0b111;
	bits<5> rd;
	bits<7> opcode = 0b0010011;
	let Inst{31-20} = imm12;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def ANDI: Instruction, ANDI_FM, TSFlagTemplate<InstFormatI>, Sched<[WriteIALU, ReadIALU]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, simm12:$imm12);
	let AsmString = "andi	$rd, $rs1, $imm12";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class EBREAK_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<12> imm12 = 0b000000000001;
	bits<5> rs1 = 0b00000;
	bits<3> funct3 = 0b000;
	bits<5> rd = 0b00000;
	bits<7> opcode = 0b1110011;
	let Inst{31-20} = imm12;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def EBREAK: Instruction, EBREAK_FM, TSFlagTemplate<InstFormatI>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins);
	let AsmString = "ebreak	";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 1;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class ECALL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<12> imm12 = 0b000000000000;
	bits<5> rs1 = 0b00000;
	bits<3> funct3 = 0b000;
	bits<5> rd = 0b00000;
	bits<7> opcode = 0b1110011;
	let Inst{31-20} = imm12;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def ECALL: Instruction, ECALL_FM, TSFlagTemplate<InstFormatI>, Sched<[WriteJmp]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins);
	let AsmString = "ecall	";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 1;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FENCE_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> funct4 = 0b0000;
	bits<4> pred;
	bits<4> succ;
	bits<5> rs1 = 0b00000;
	bits<3> funct3 = 0b000;
	bits<5> rd = 0b00000;
	bits<7> opcode = 0b0001111;
	let Inst{31-28} = funct4;
	let Inst{27-24} = pred;
	let Inst{23-20} = succ;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def FENCE: Instruction, FENCE_FM, TSFlagTemplate<InstFormatI>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins fencearg:$pred, fencearg:$succ);
	let AsmString = "fence	$pred, $succ";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 1;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FENCE_I_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<12> imm12 = 0b000000000000;
	bits<5> rs1 = 0b00000;
	bits<3> funct3 = 0b001;
	bits<5> rd = 0b00000;
	bits<7> opcode = 0b0001111;
	let Inst{31-20} = imm12;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def FENCE_I: Instruction, FENCE_I_FM, TSFlagTemplate<InstFormatI>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins);
	let AsmString = "fence.i	";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 1;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FENCE_TSO_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<12> imm12 = 0b100000110011;
	bits<5> rs1 = 0b00000;
	bits<3> funct3 = 0b000;
	bits<5> rd = 0b00000;
	bits<7> opcode = 0b0001111;
	let Inst{31-20} = imm12;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def FENCE_TSO: Instruction, FENCE_TSO_FM, TSFlagTemplate<InstFormatI>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins);
	let AsmString = "fence.tso	";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 1;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class JALR_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<12> imm12;
	bits<5> rs1;
	bits<3> funct3 = 0b000;
	bits<5> rd;
	bits<7> opcode = 0b1100111;
	let Inst{31-20} = imm12;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def JALR: Instruction, JALR_FM, TSFlagTemplate<InstFormatI>, Sched<[WriteJalr, ReadJalr]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, simm12:$imm12);
	let AsmString = "jalr	$rd, ${imm12}(${rs1})";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let isCall = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class LB_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<12> imm12;
	bits<5> rs1;
	bits<3> funct3 = 0b000;
	bits<5> rd;
	bits<7> opcode = 0b0000011;
	let Inst{31-20} = imm12;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def LB: Instruction, LB_FM, TSFlagTemplate<InstFormatI>, Sched<[WriteLDB, ReadMemBase]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, simm12:$imm12);
	let AsmString = "lb	$rd, ${imm12}(${rs1})";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class LBU_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<12> imm12;
	bits<5> rs1;
	bits<3> funct3 = 0b100;
	bits<5> rd;
	bits<7> opcode = 0b0000011;
	let Inst{31-20} = imm12;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def LBU: Instruction, LBU_FM, TSFlagTemplate<InstFormatI>, Sched<[WriteLDB, ReadMemBase]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, simm12:$imm12);
	let AsmString = "lbu	$rd, ${imm12}(${rs1})";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class LD_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<12> imm12;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0000011;
	let Inst{31-20} = imm12;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def LD: Instruction, LD_FM, TSFlagTemplate<InstFormatI>, Sched<[WriteLDD, ReadMemBase]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, simm12:$imm12);
	let AsmString = "ld	$rd, ${imm12}(${rs1})";
	let Pattern = [];
	let Predicates = [IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class LH_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<12> imm12;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> opcode = 0b0000011;
	let Inst{31-20} = imm12;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def LH: Instruction, LH_FM, TSFlagTemplate<InstFormatI>, Sched<[WriteLDH, ReadMemBase]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, simm12:$imm12);
	let AsmString = "lh	$rd, ${imm12}(${rs1})";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class LHU_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<12> imm12;
	bits<5> rs1;
	bits<3> funct3 = 0b101;
	bits<5> rd;
	bits<7> opcode = 0b0000011;
	let Inst{31-20} = imm12;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def LHU: Instruction, LHU_FM, TSFlagTemplate<InstFormatI>, Sched<[WriteLDH, ReadMemBase]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, simm12:$imm12);
	let AsmString = "lhu	$rd, ${imm12}(${rs1})";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class LW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<12> imm12;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0000011;
	let Inst{31-20} = imm12;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def LW: Instruction, LW_FM, TSFlagTemplate<InstFormatI>, Sched<[WriteLDW, ReadMemBase]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, simm12:$imm12);
	let AsmString = "lw	$rd, ${imm12}(${rs1})";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class LWU_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<12> imm12;
	bits<5> rs1;
	bits<3> funct3 = 0b110;
	bits<5> rd;
	bits<7> opcode = 0b0000011;
	let Inst{31-20} = imm12;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def LWU: Instruction, LWU_FM, TSFlagTemplate<InstFormatI>, Sched<[WriteLDWU, ReadMemBase]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, simm12:$imm12);
	let AsmString = "lwu	$rd, ${imm12}(${rs1})";
	let Pattern = [];
	let Predicates = [IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class ORI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<12> imm12;
	bits<5> rs1;
	bits<3> funct3 = 0b110;
	bits<5> rd;
	bits<7> opcode = 0b0010011;
	let Inst{31-20} = imm12;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def ORI: Instruction, ORI_FM, TSFlagTemplate<InstFormatI>, Sched<[WriteIALU, ReadIALU]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, simm12:$imm12);
	let AsmString = "ori	$rd, $rs1, $imm12";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let isReMaterializable = 1;
	let isAsCheapAsAMove = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SLLI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000000;
	bits<6> shamt;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> opcode = 0b0010011;
	let Inst{31-26} = funct6;
	let Inst{25-20} = shamt;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def SLLI: Instruction, SLLI_FM, TSFlagTemplate<InstFormatI>, Sched<[WriteShift, ReadShift]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, uimmlog2xlen:$shamt);
	let AsmString = "slli	$rd, $rs1, $shamt";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SLLIW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct6 = 0b0000000;
	bits<5> shamt;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> opcode = 0b0011011;
	let Inst{31-25} = funct6;
	let Inst{24-20} = shamt;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def SLLIW: Instruction, SLLIW_FM, TSFlagTemplate<InstFormatI>, Sched<[WriteShift32, ReadShift32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, uimm5:$shamt);
	let AsmString = "slliw	$rd, $rs1, $shamt";
	let Pattern = [];
	let Predicates = [IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SLTI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<12> imm12;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0010011;
	let Inst{31-20} = imm12;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def SLTI: Instruction, SLTI_FM, TSFlagTemplate<InstFormatI>, Sched<[WriteIALU, ReadIALU]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, simm12:$imm12);
	let AsmString = "slti	$rd, $rs1, $imm12";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SLTIU_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<12> imm12;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0010011;
	let Inst{31-20} = imm12;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def SLTIU: Instruction, SLTIU_FM, TSFlagTemplate<InstFormatI>, Sched<[WriteIALU, ReadIALU]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, simm12:$imm12);
	let AsmString = "sltiu	$rd, $rs1, $imm12";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SRAI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010000;
	bits<6> shamt;
	bits<5> rs1;
	bits<3> funct3 = 0b101;
	bits<5> rd;
	bits<7> opcode = 0b0010011;
	let Inst{31-26} = funct6;
	let Inst{25-20} = shamt;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def SRAI: Instruction, SRAI_FM, TSFlagTemplate<InstFormatI>, Sched<[WriteShift, ReadShift]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, uimmlog2xlen:$shamt);
	let AsmString = "srai	$rd, $rs1, $shamt";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SRAIW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct6 = 0b0100000;
	bits<5> shamt;
	bits<5> rs1;
	bits<3> funct3 = 0b101;
	bits<5> rd;
	bits<7> opcode = 0b0011011;
	let Inst{31-25} = funct6;
	let Inst{24-20} = shamt;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def SRAIW: Instruction, SRAIW_FM, TSFlagTemplate<InstFormatI>, Sched<[WriteShift32, ReadShift32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, uimm5:$shamt);
	let AsmString = "sraiw	$rd, $rs1, $shamt";
	let Pattern = [];
	let Predicates = [IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SRLI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000000;
	bits<6> shamt;
	bits<5> rs1;
	bits<3> funct3 = 0b101;
	bits<5> rd;
	bits<7> opcode = 0b0010011;
	let Inst{31-26} = funct6;
	let Inst{25-20} = shamt;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def SRLI: Instruction, SRLI_FM, TSFlagTemplate<InstFormatI>, Sched<[WriteShift, ReadShift]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, uimmlog2xlen:$shamt);
	let AsmString = "srli	$rd, $rs1, $shamt";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SRLIW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct6 = 0b0000000;
	bits<5> shamt;
	bits<5> rs1;
	bits<3> funct3 = 0b101;
	bits<5> rd;
	bits<7> opcode = 0b0011011;
	let Inst{31-25} = funct6;
	let Inst{24-20} = shamt;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def SRLIW: Instruction, SRLIW_FM, TSFlagTemplate<InstFormatI>, Sched<[WriteShift32, ReadShift32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, uimm5:$shamt);
	let AsmString = "srliw	$rd, $rs1, $shamt";
	let Pattern = [];
	let Predicates = [IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class XORI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<12> imm12;
	bits<5> rs1;
	bits<3> funct3 = 0b100;
	bits<5> rd;
	bits<7> opcode = 0b0010011;
	let Inst{31-20} = imm12;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def XORI: Instruction, XORI_FM, TSFlagTemplate<InstFormatI>, Sched<[WriteIALU, ReadIALU]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, simm12:$imm12);
	let AsmString = "xori	$rd, $rs1, $imm12";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let isReMaterializable = 1;
	let isAsCheapAsAMove = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class JAL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<20> imm20;
	bits<5> rd;
	bits<7> opcode = 0b1101111;
	let Inst{31-31} = imm20{19-19};
	let Inst{30-21} = imm20{9-0};
	let Inst{20-20} = imm20{10-10};
	let Inst{19-12} = imm20{18-11};
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def JAL: Instruction, JAL_FM, TSFlagTemplate<InstFormatJ>, Sched<[WriteJal]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins simm21_lsb0_jal:$imm20);
	let AsmString = "jal	$rd, $imm20";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let isCall = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class ADD_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000000;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b000;
	bits<5> rd;
	bits<7> opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def ADD: Instruction, ADD_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteIALU, ReadIALU, ReadIALU]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "add	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class ADDW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000000;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b000;
	bits<5> rd;
	bits<7> opcode = 0b0111011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def ADDW: Instruction, ADDW_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteIALU32, ReadIALU32, ReadIALU32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "addw	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AND_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000000;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b111;
	bits<5> rd;
	bits<7> opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AND: Instruction, AND_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteIALU, ReadIALU, ReadIALU]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "and	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class MRET_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0011000;
	bits<5> rs2 = 0b00010;
	bits<5> rs1 = 0b00000;
	bits<3> funct3 = 0b000;
	bits<5> rd = 0b00000;
	bits<7> opcode = 0b1110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def MRET: Instruction, MRET_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "mret	";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 1;
	let mayLoad = 0;
	let mayStore = 0;
	let isBarrier = 1;
	let isReturn = 1;
	let isTerminator = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class OR_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000000;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b110;
	bits<5> rd;
	bits<7> opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def OR: Instruction, OR_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteIALU, ReadIALU, ReadIALU]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "or	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SFENCE_VMA_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0001001;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b000;
	bits<5> rd = 0b00000;
	bits<7> opcode = 0b1110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def SFENCE_VMA: Instruction, SFENCE_VMA_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "sfence.vma	$rs1, $rs2";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 1;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SLL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000000;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def SLL: Instruction, SLL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteIALU, ReadIALU, ReadIALU]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "sll	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SLLW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000000;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> opcode = 0b0111011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def SLLW: Instruction, SLLW_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteIALU32, ReadIALU32, ReadIALU32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "sllw	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SLT_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000000;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def SLT: Instruction, SLT_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteIALU, ReadIALU, ReadIALU]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "slt	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SLTU_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000000;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def SLTU: Instruction, SLTU_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteIALU, ReadIALU, ReadIALU]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "sltu	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SRA_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0100000;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b101;
	bits<5> rd;
	bits<7> opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def SRA: Instruction, SRA_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteIALU, ReadIALU, ReadIALU]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "sra	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SRAW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0100000;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b101;
	bits<5> rd;
	bits<7> opcode = 0b0111011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def SRAW: Instruction, SRAW_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteIALU32, ReadIALU32, ReadIALU32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "sraw	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SRET_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0001000;
	bits<5> rs2 = 0b00010;
	bits<5> rs1 = 0b00000;
	bits<3> funct3 = 0b000;
	bits<5> rd = 0b00000;
	bits<7> opcode = 0b1110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def SRET: Instruction, SRET_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "sret	";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 1;
	let mayLoad = 0;
	let mayStore = 0;
	let isBarrier = 1;
	let isReturn = 1;
	let isTerminator = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SRL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000000;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b101;
	bits<5> rd;
	bits<7> opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def SRL: Instruction, SRL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteIALU, ReadIALU, ReadIALU]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "srl	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SRLW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000000;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b101;
	bits<5> rd;
	bits<7> opcode = 0b0111011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def SRLW: Instruction, SRLW_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteIALU32, ReadIALU32, ReadIALU32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "srlw	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SUB_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0100000;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b000;
	bits<5> rd;
	bits<7> opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def SUB: Instruction, SUB_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteIALU, ReadIALU, ReadIALU]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "sub	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SUBW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0100000;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b000;
	bits<5> rd;
	bits<7> opcode = 0b0111011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def SUBW: Instruction, SUBW_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteIALU32, ReadIALU32, ReadIALU32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "subw	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class URET_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000000;
	bits<5> rs2 = 0b00010;
	bits<5> rs1 = 0b00000;
	bits<3> funct3 = 0b000;
	bits<5> rd = 0b00000;
	bits<7> opcode = 0b1110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def URET: Instruction, URET_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "uret	";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 1;
	let mayLoad = 0;
	let mayStore = 0;
	let isBarrier = 1;
	let isReturn = 1;
	let isTerminator = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class WFI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0001000;
	bits<5> rs2 = 0b00101;
	bits<5> rs1 = 0b00000;
	bits<3> funct3 = 0b000;
	bits<5> rd = 0b00000;
	bits<7> opcode = 0b1110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def WFI: Instruction, WFI_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "wfi	";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 1;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class XOR_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000000;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b100;
	bits<5> rd;
	bits<7> opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def XOR: Instruction, XOR_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteIALU, ReadIALU, ReadIALU]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "xor	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SB_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<12> imm12;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b000;
	bits<7> opcode = 0b0100011;
	let Inst{31-25} = imm12{11-5};
	let Inst{11-7} = imm12{4-0};
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{6-0} = opcode;
}
def SB: Instruction, SB_FM, TSFlagTemplate<InstFormatS>, Sched<[WriteSTB, ReadStoreData, ReadMemBase]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins GPR:$rs2, GPR:$rs1, simm12:$imm12);
	let AsmString = "sb	$rs2, ${imm12}(${rs1})";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SD_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<12> imm12;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<7> opcode = 0b0100011;
	let Inst{31-25} = imm12{11-5};
	let Inst{11-7} = imm12{4-0};
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{6-0} = opcode;
}
def SD: Instruction, SD_FM, TSFlagTemplate<InstFormatS>, Sched<[WriteSTD, ReadStoreData, ReadMemBase]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins GPR:$rs2, GPR:$rs1, simm12:$imm12);
	let AsmString = "sd	$rs2, ${imm12}(${rs1})";
	let Pattern = [];
	let Predicates = [IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SH_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<12> imm12;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<7> opcode = 0b0100011;
	let Inst{31-25} = imm12{11-5};
	let Inst{11-7} = imm12{4-0};
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{6-0} = opcode;
}
def SH: Instruction, SH_FM, TSFlagTemplate<InstFormatS>, Sched<[WriteSTH, ReadStoreData, ReadMemBase]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins GPR:$rs2, GPR:$rs1, simm12:$imm12);
	let AsmString = "sh	$rs2, ${imm12}(${rs1})";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<12> imm12;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<7> opcode = 0b0100011;
	let Inst{31-25} = imm12{11-5};
	let Inst{11-7} = imm12{4-0};
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{6-0} = opcode;
}
def SW: Instruction, SW_FM, TSFlagTemplate<InstFormatS>, Sched<[WriteSTW, ReadStoreData, ReadMemBase]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins GPR:$rs2, GPR:$rs1, simm12:$imm12);
	let AsmString = "sw	$rs2, ${imm12}(${rs1})";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AUIPC_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<20> imm20;
	bits<5> rd;
	bits<7> opcode = 0b0010111;
	let Inst{31-12} = imm20;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AUIPC: Instruction, AUIPC_FM, TSFlagTemplate<InstFormatU>, Sched<[WriteIALU]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins uimm20_auipc:$imm20);
	let AsmString = "auipc	$rd, $imm20";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class LUI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<20> imm20;
	bits<5> rd;
	bits<7> opcode = 0b0110111;
	let Inst{31-12} = imm20;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def LUI: Instruction, LUI_FM, TSFlagTemplate<InstFormatU>, Sched<[WriteIALU]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins uimm20_lui:$imm20);
	let AsmString = "lui	$rd, $imm20";
	let Pattern = [];
	let Predicates = [];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let isReMaterializable = 1;
	let isAsCheapAsAMove = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}