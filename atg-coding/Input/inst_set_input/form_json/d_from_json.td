class FLD_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<12> imm12;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0000111;
	let Inst{31-20} = imm12;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def FLD: Instruction, FLD_FM, TSFlagTemplate<InstFormatI>, Sched<[WriteFLD64, ReadFMemBase]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR64:$rd);
	let InOperandList = (ins GPR:$rs1, simm12:$imm12);
	let AsmString = "fld	$rd, ${imm12}(${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtD];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FADD_D_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000001;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3;
	bits<5> rd;
	bits<7> opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def FADD_D: Instruction, FADD_D_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFALU64, ReadFALU64, ReadFALU64]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR64:$rd);
	let InOperandList = (ins FPR64:$rs1, FPR64:$rs2, frmarg:$funct3);
	let AsmString = "fadd.d	$rd, $rs1, $rs2, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtD];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FCLASS_D_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1110001;
	bits<5> rs2 = 0b00000;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def FCLASS_D: Instruction, FCLASS_D_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFClass64, ReadFClass64]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins FPR64:$rs1);
	let AsmString = "fclass.d	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtD];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FCVT_D_L_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1101001;
	bits<5> rs2 = 0b00010;
	bits<5> rs1;
	bits<3> funct3;
	bits<5> rd;
	bits<7> opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def FCVT_D_L: Instruction, FCVT_D_L_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFCvtI64ToF64, ReadFCvtI64ToF64]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR64:$rd);
	let InOperandList = (ins GPR:$rs1, frmarg:$funct3);
	let AsmString = "fcvt.d.l	$rd, $rs1, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtD, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FCVT_D_LU_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1101001;
	bits<5> rs2 = 0b00011;
	bits<5> rs1;
	bits<3> funct3;
	bits<5> rd;
	bits<7> opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def FCVT_D_LU: Instruction, FCVT_D_LU_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFCvtI64ToF64, ReadFCvtI64ToF64]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR64:$rd);
	let InOperandList = (ins GPR:$rs1, frmarg:$funct3);
	let AsmString = "fcvt.d.lu	$rd, $rs1, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtD, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FCVT_D_S_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0100001;
	bits<5> rs2 = 0b00000;
	bits<5> rs1;
	bits<3> funct3 = 0b000;
	bits<5> rd;
	bits<7> opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def FCVT_D_S: Instruction, FCVT_D_S_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFCvtF32ToF64, ReadFCvtF32ToF64]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR64:$rd);
	let InOperandList = (ins FPR32:$rs1);
	let AsmString = "fcvt.d.s	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtD];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FCVT_D_W_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1101001;
	bits<5> rs2 = 0b00000;
	bits<5> rs1;
	bits<3> funct3 = 0b000;
	bits<5> rd;
	bits<7> opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def FCVT_D_W: Instruction, FCVT_D_W_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFCvtI32ToF64, ReadFCvtI32ToF64]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR64:$rd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "fcvt.d.w	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtD];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FCVT_D_WU_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1101001;
	bits<5> rs2 = 0b00001;
	bits<5> rs1;
	bits<3> funct3 = 0b000;
	bits<5> rd;
	bits<7> opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def FCVT_D_WU: Instruction, FCVT_D_WU_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFCvtI32ToF64, ReadFCvtI32ToF64]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR64:$rd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "fcvt.d.wu	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtD];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FCVT_L_D_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1100001;
	bits<5> rs2 = 0b00010;
	bits<5> rs1;
	bits<3> funct3;
	bits<5> rd;
	bits<7> opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def FCVT_L_D: Instruction, FCVT_L_D_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFCvtF64ToI64, ReadFCvtF64ToI64]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins FPR64:$rs1, frmarg:$funct3);
	let AsmString = "fcvt.l.d	$rd, $rs1, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtD, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FCVT_LU_D_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1100001;
	bits<5> rs2 = 0b00011;
	bits<5> rs1;
	bits<3> funct3;
	bits<5> rd;
	bits<7> opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def FCVT_LU_D: Instruction, FCVT_LU_D_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFCvtF64ToI64, ReadFCvtF64ToI64]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins FPR64:$rs1, frmarg:$funct3);
	let AsmString = "fcvt.lu.d	$rd, $rs1, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtD, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FCVT_W_D_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1100001;
	bits<5> rs2 = 0b00000;
	bits<5> rs1;
	bits<3> funct3;
	bits<5> rd;
	bits<7> opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def FCVT_W_D: Instruction, FCVT_W_D_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFCvtF64ToI32, ReadFCvtF64ToI32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins FPR64:$rs1, frmarg:$funct3);
	let AsmString = "fcvt.w.d	$rd, $rs1, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtD];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FCVT_WU_D_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1100001;
	bits<5> rs2 = 0b00001;
	bits<5> rs1;
	bits<3> funct3;
	bits<5> rd;
	bits<7> opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def FCVT_WU_D: Instruction, FCVT_WU_D_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFCvtF64ToI32, ReadFCvtF64ToI32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins FPR64:$rs1, frmarg:$funct3);
	let AsmString = "fcvt.wu.d	$rd, $rs1, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtD];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FDIV_D_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0001101;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3;
	bits<5> rd;
	bits<7> opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def FDIV_D: Instruction, FDIV_D_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFDiv64, ReadFDiv64, ReadFDiv64]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR64:$rd);
	let InOperandList = (ins FPR64:$rs1, FPR64:$rs2, frmarg:$funct3);
	let AsmString = "fdiv.d	$rd, $rs1, $rs2, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtD];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FEQ_D_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1010001;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def FEQ_D: Instruction, FEQ_D_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFCmp64, ReadFCmp64, ReadFCmp64]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins FPR64:$rs1, FPR64:$rs2);
	let AsmString = "feq.d	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtD];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FLE_D_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1010001;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b000;
	bits<5> rd;
	bits<7> opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def FLE_D: Instruction, FLE_D_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFCmp64, ReadFCmp64, ReadFCmp64]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins FPR64:$rs1, FPR64:$rs2);
	let AsmString = "fle.d	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtD];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FLT_D_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1010001;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def FLT_D: Instruction, FLT_D_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFCmp64, ReadFCmp64, ReadFCmp64]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins FPR64:$rs1, FPR64:$rs2);
	let AsmString = "flt.d	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtD];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FMAX_D_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0010101;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def FMAX_D: Instruction, FMAX_D_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFMinMax64, ReadFMinMax64, ReadFMinMax64]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR64:$rd);
	let InOperandList = (ins FPR64:$rs1, FPR64:$rs2);
	let AsmString = "fmax.d	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtD];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FMIN_D_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0010101;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b000;
	bits<5> rd;
	bits<7> opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def FMIN_D: Instruction, FMIN_D_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFMinMax64, ReadFMinMax64, ReadFMinMax64]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR64:$rd);
	let InOperandList = (ins FPR64:$rs1, FPR64:$rs2);
	let AsmString = "fmin.d	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtD];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FMUL_D_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0001001;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3;
	bits<5> rd;
	bits<7> opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def FMUL_D: Instruction, FMUL_D_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFMul64, ReadFMul64, ReadFMul64]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR64:$rd);
	let InOperandList = (ins FPR64:$rs1, FPR64:$rs2, frmarg:$funct3);
	let AsmString = "fmul.d	$rd, $rs1, $rs2, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtD];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FMV_D_X_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1111001;
	bits<5> rs2 = 0b00000;
	bits<5> rs1;
	bits<3> funct3 = 0b000;
	bits<5> rd;
	bits<7> opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def FMV_D_X: Instruction, FMV_D_X_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFMovI64ToF64, ReadFMovI64ToF64]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR64:$rd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "fmv.d.x	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtD, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FMV_X_D_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1110001;
	bits<5> rs2 = 0b00000;
	bits<5> rs1;
	bits<3> funct3 = 0b000;
	bits<5> rd;
	bits<7> opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def FMV_X_D: Instruction, FMV_X_D_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFMovF64ToI64, ReadFMovF64ToI64]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins FPR64:$rs1);
	let AsmString = "fmv.x.d	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtD, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FSGNJ_D_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0010001;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b000;
	bits<5> rd;
	bits<7> opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def FSGNJ_D: Instruction, FSGNJ_D_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFSGNJ64, ReadFSGNJ64, ReadFSGNJ64]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR64:$rd);
	let InOperandList = (ins FPR64:$rs1, FPR64:$rs2);
	let AsmString = "fsgnj.d	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtD];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FSGNJN_D_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0010001;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def FSGNJN_D: Instruction, FSGNJN_D_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFSGNJ64, ReadFSGNJ64, ReadFSGNJ64]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR64:$rd);
	let InOperandList = (ins FPR64:$rs1, FPR64:$rs2);
	let AsmString = "fsgnjn.d	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtD];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FSGNJX_D_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0010001;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def FSGNJX_D: Instruction, FSGNJX_D_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFSGNJ64, ReadFSGNJ64, ReadFSGNJ64]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR64:$rd);
	let InOperandList = (ins FPR64:$rs1, FPR64:$rs2);
	let AsmString = "fsgnjx.d	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtD];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FSQRT_D_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0101101;
	bits<5> rs2 = 0b00000;
	bits<5> rs1;
	bits<3> funct3;
	bits<5> rd;
	bits<7> opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def FSQRT_D: Instruction, FSQRT_D_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFSqrt64, ReadFSqrt64]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR64:$rd);
	let InOperandList = (ins FPR64:$rs1, frmarg:$funct3);
	let AsmString = "fsqrt.d	$rd, $rs1, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtD];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FSUB_D_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000101;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3;
	bits<5> rd;
	bits<7> opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def FSUB_D: Instruction, FSUB_D_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFALU64, ReadFALU64, ReadFALU64]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR64:$rd);
	let InOperandList = (ins FPR64:$rs1, FPR64:$rs2, frmarg:$funct3);
	let AsmString = "fsub.d	$rd, $rs1, $rs2, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtD];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FMADD_D_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> rs3;
	bits<2> funct2 = 0b01;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3;
	bits<5> rd;
	bits<7> opcode = 0b1000011;
	let Inst{31-27} = rs3;
	let Inst{26-25} = funct2;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def FMADD_D: Instruction, FMADD_D_FM, TSFlagTemplate<InstFormatR4>, Sched<[WriteFMulAdd64, ReadFMulAdd64, ReadFMulAdd64, ReadFMulAdd64]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR64:$rd);
	let InOperandList = (ins FPR64:$rs1, FPR64:$rs2, FPR64:$rs3, frmarg:$funct3);
	let AsmString = "fmadd.d	$rd, $rs1, $rs2, $rs3, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtD];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FMSUB_D_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> rs3;
	bits<2> funct2 = 0b01;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3;
	bits<5> rd;
	bits<7> opcode = 0b1000111;
	let Inst{31-27} = rs3;
	let Inst{26-25} = funct2;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def FMSUB_D: Instruction, FMSUB_D_FM, TSFlagTemplate<InstFormatR4>, Sched<[WriteFMulSub64, ReadFMulSub64, ReadFMulSub64, ReadFMulSub64]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR64:$rd);
	let InOperandList = (ins FPR64:$rs1, FPR64:$rs2, FPR64:$rs3, frmarg:$funct3);
	let AsmString = "fmsub.d	$rd, $rs1, $rs2, $rs3, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtD];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FNMADD_D_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> rs3;
	bits<2> funct2 = 0b01;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3;
	bits<5> rd;
	bits<7> opcode = 0b1001111;
	let Inst{31-27} = rs3;
	let Inst{26-25} = funct2;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def FNMADD_D: Instruction, FNMADD_D_FM, TSFlagTemplate<InstFormatR4>, Sched<[WriteFMulAdd64, ReadFMulAdd64, ReadFMulAdd64, ReadFMulAdd64]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR64:$rd);
	let InOperandList = (ins FPR64:$rs1, FPR64:$rs2, FPR64:$rs3, frmarg:$funct3);
	let AsmString = "fnmadd.d	$rd, $rs1, $rs2, $rs3, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtD];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FNMSUB_D_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> rs3;
	bits<2> funct2 = 0b01;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3;
	bits<5> rd;
	bits<7> opcode = 0b1001011;
	let Inst{31-27} = rs3;
	let Inst{26-25} = funct2;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def FNMSUB_D: Instruction, FNMSUB_D_FM, TSFlagTemplate<InstFormatR4>, Sched<[WriteFMulSub64, ReadFMulSub64, ReadFMulSub64, ReadFMulSub64]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR64:$rd);
	let InOperandList = (ins FPR64:$rs1, FPR64:$rs2, FPR64:$rs3, frmarg:$funct3);
	let AsmString = "fnmsub.d	$rd, $rs1, $rs2, $rs3, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtD];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FSD_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<12> imm12;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<7> opcode = 0b0100111;
	let Inst{31-25} = imm12{11-5};
	let Inst{11-7} = imm12{4-0};
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{6-0} = opcode;
}
def FSD: Instruction, FSD_FM, TSFlagTemplate<InstFormatS>, Sched<[WriteFST64, ReadStoreData, ReadFMemBase]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins FPR64:$rs2, GPR:$rs1, simm12:$imm12);
	let AsmString = "fsd	$rs2, ${imm12}(${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtD];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FCVT_S_D_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0100000;
	bits<5> rs2 = 0b00001;
	bits<5> rs1;
	bits<3> funct3;
	bits<5> rd;
	bits<7> opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def FCVT_S_D: Instruction, FCVT_S_D_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFCvtF64ToF32, ReadFCvtF64ToF32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR32:$rd);
	let InOperandList = (ins FPR64:$rs1, frmarg:$funct3);
	let AsmString = "fcvt.s.d	$rd, $rs1, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtD];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}