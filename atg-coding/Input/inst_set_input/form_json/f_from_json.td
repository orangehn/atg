class FLW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<12> imm12;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0000111;
	let Inst{31-20} = imm12;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def FLW: Instruction, FLW_FM, TSFlagTemplate<InstFormatI>, Sched<[WriteFLD32, ReadFMemBase]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR32:$rd);
	let InOperandList = (ins GPR:$rs1, simm12:$imm12);
	let AsmString = "flw	$rd, ${imm12}(${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FADD_S_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000000;
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
def FADD_S: Instruction, FADD_S_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFALU32, ReadFALU32, ReadFALU32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR32:$rd);
	let InOperandList = (ins FPR32:$rs1, FPR32:$rs2, frmarg:$funct3);
	let AsmString = "fadd.s	$rd, $rs1, $rs2, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FCLASS_S_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1110000;
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
def FCLASS_S: Instruction, FCLASS_S_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFClass32, ReadFClass32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins FPR32:$rs1);
	let AsmString = "fclass.s	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FCVT_L_S_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1100000;
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
def FCVT_L_S: Instruction, FCVT_L_S_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFCvtF32ToI64, ReadFCvtF32ToI64]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins FPR32:$rs1, frmarg:$funct3);
	let AsmString = "fcvt.l.s	$rd, $rs1, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtF, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FCVT_LU_S_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1100000;
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
def FCVT_LU_S: Instruction, FCVT_LU_S_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFCvtF32ToI64, ReadFCvtF32ToI64]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins FPR32:$rs1, frmarg:$funct3);
	let AsmString = "fcvt.lu.s	$rd, $rs1, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtF, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FCVT_S_L_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1101000;
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
def FCVT_S_L: Instruction, FCVT_S_L_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFCvtI64ToF32, ReadFCvtI64ToF32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR32:$rd);
	let InOperandList = (ins GPR:$rs1, frmarg:$funct3);
	let AsmString = "fcvt.s.l	$rd, $rs1, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtF, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FCVT_S_LU_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1101000;
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
def FCVT_S_LU: Instruction, FCVT_S_LU_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFCvtI64ToF32, ReadFCvtI64ToF32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR32:$rd);
	let InOperandList = (ins GPR:$rs1, frmarg:$funct3);
	let AsmString = "fcvt.s.lu	$rd, $rs1, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtF, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FCVT_S_W_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1101000;
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
def FCVT_S_W: Instruction, FCVT_S_W_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFCvtI32ToF32, ReadFCvtI32ToF32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR32:$rd);
	let InOperandList = (ins GPR:$rs1, frmarg:$funct3);
	let AsmString = "fcvt.s.w	$rd, $rs1, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FCVT_S_WU_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1101000;
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
def FCVT_S_WU: Instruction, FCVT_S_WU_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFCvtI32ToF32, ReadFCvtI32ToF32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR32:$rd);
	let InOperandList = (ins GPR:$rs1, frmarg:$funct3);
	let AsmString = "fcvt.s.wu	$rd, $rs1, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FCVT_W_S_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1100000;
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
def FCVT_W_S: Instruction, FCVT_W_S_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFCvtF32ToI32, ReadFCvtF32ToI32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins FPR32:$rs1, frmarg:$funct3);
	let AsmString = "fcvt.w.s	$rd, $rs1, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FCVT_WU_S_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1100000;
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
def FCVT_WU_S: Instruction, FCVT_WU_S_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFCvtF32ToI32, ReadFCvtF32ToI32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins FPR32:$rs1, frmarg:$funct3);
	let AsmString = "fcvt.wu.s	$rd, $rs1, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FDIV_S_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0001100;
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
def FDIV_S: Instruction, FDIV_S_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFDiv32, ReadFDiv32, ReadFDiv32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR32:$rd);
	let InOperandList = (ins FPR32:$rs1, FPR32:$rs2, frmarg:$funct3);
	let AsmString = "fdiv.s	$rd, $rs1, $rs2, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FEQ_S_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1010000;
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
def FEQ_S: Instruction, FEQ_S_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFCmp32, ReadFCmp32, ReadFCmp32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins FPR32:$rs1, FPR32:$rs2);
	let AsmString = "feq.s	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FLE_S_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1010000;
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
def FLE_S: Instruction, FLE_S_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFCmp32, ReadFCmp32, ReadFCmp32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins FPR32:$rs1, FPR32:$rs2);
	let AsmString = "fle.s	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FLT_S_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1010000;
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
def FLT_S: Instruction, FLT_S_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFCmp32, ReadFCmp32, ReadFCmp32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins FPR32:$rs1, FPR32:$rs2);
	let AsmString = "flt.s	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FMAX_S_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0010100;
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
def FMAX_S: Instruction, FMAX_S_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFMinMax32, ReadFMinMax32, ReadFMinMax32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR32:$rd);
	let InOperandList = (ins FPR32:$rs1, FPR32:$rs2);
	let AsmString = "fmax.s	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FMIN_S_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0010100;
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
def FMIN_S: Instruction, FMIN_S_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFMinMax32, ReadFMinMax32, ReadFMinMax32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR32:$rd);
	let InOperandList = (ins FPR32:$rs1, FPR32:$rs2);
	let AsmString = "fmin.s	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FMUL_S_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0001000;
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
def FMUL_S: Instruction, FMUL_S_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFMul32, ReadFMul32, ReadFMul32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR32:$rd);
	let InOperandList = (ins FPR32:$rs1, FPR32:$rs2, frmarg:$funct3);
	let AsmString = "fmul.s	$rd, $rs1, $rs2, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FMV_W_X_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1111000;
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
def FMV_W_X: Instruction, FMV_W_X_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFMovI32ToF32, ReadFMovI32ToF32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR32:$rd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "fmv.w.x	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FMV_X_W_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1110000;
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
def FMV_X_W: Instruction, FMV_X_W_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFMovF32ToI32, ReadFMovF32ToI32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins FPR32:$rs1);
	let AsmString = "fmv.x.w	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FSGNJ_S_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0010000;
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
def FSGNJ_S: Instruction, FSGNJ_S_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFSGNJ32, ReadFSGNJ32, ReadFSGNJ32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR32:$rd);
	let InOperandList = (ins FPR32:$rs1, FPR32:$rs2);
	let AsmString = "fsgnj.s	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FSGNJN_S_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0010000;
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
def FSGNJN_S: Instruction, FSGNJN_S_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFSGNJ32, ReadFSGNJ32, ReadFSGNJ32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR32:$rd);
	let InOperandList = (ins FPR32:$rs1, FPR32:$rs2);
	let AsmString = "fsgnjn.s	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FSGNJX_S_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0010000;
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
def FSGNJX_S: Instruction, FSGNJX_S_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFSGNJ32, ReadFSGNJ32, ReadFSGNJ32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR32:$rd);
	let InOperandList = (ins FPR32:$rs1, FPR32:$rs2);
	let AsmString = "fsgnjx.s	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FSQRT_S_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0101100;
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
def FSQRT_S: Instruction, FSQRT_S_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFSqrt32, ReadFSqrt32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR32:$rd);
	let InOperandList = (ins FPR32:$rs1, frmarg:$funct3);
	let AsmString = "fsqrt.s	$rd, $rs1, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FSUB_S_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000100;
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
def FSUB_S: Instruction, FSUB_S_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteFALU32, ReadFALU32, ReadFALU32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR32:$rd);
	let InOperandList = (ins FPR32:$rs1, FPR32:$rs2, frmarg:$funct3);
	let AsmString = "fsub.s	$rd, $rs1, $rs2, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FMADD_S_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> rs3;
	bits<2> funct2 = 0b00;
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
def FMADD_S: Instruction, FMADD_S_FM, TSFlagTemplate<InstFormatR4>, Sched<[WriteFMulAdd32, ReadFMulAdd32, ReadFMulAdd32, ReadFMulAdd32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR32:$rd);
	let InOperandList = (ins FPR32:$rs1, FPR32:$rs2, FPR32:$rs3, frmarg:$funct3);
	let AsmString = "fmadd.s	$rd, $rs1, $rs2, $rs3, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FMSUB_S_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> rs3;
	bits<2> funct2 = 0b00;
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
def FMSUB_S: Instruction, FMSUB_S_FM, TSFlagTemplate<InstFormatR4>, Sched<[WriteFMulSub32, ReadFMulSub32, ReadFMulSub32, ReadFMulSub32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR32:$rd);
	let InOperandList = (ins FPR32:$rs1, FPR32:$rs2, FPR32:$rs3, frmarg:$funct3);
	let AsmString = "fmsub.s	$rd, $rs1, $rs2, $rs3, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FNMADD_S_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> rs3;
	bits<2> funct2 = 0b00;
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
def FNMADD_S: Instruction, FNMADD_S_FM, TSFlagTemplate<InstFormatR4>, Sched<[WriteFMulAdd32, ReadFMulAdd32, ReadFMulAdd32, ReadFMulAdd32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR32:$rd);
	let InOperandList = (ins FPR32:$rs1, FPR32:$rs2, FPR32:$rs3, frmarg:$funct3);
	let AsmString = "fnmadd.s	$rd, $rs1, $rs2, $rs3, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FNMSUB_S_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> rs3;
	bits<2> funct2 = 0b00;
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
def FNMSUB_S: Instruction, FNMSUB_S_FM, TSFlagTemplate<InstFormatR4>, Sched<[WriteFMulSub32, ReadFMulSub32, ReadFMulSub32, ReadFMulSub32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR32:$rd);
	let InOperandList = (ins FPR32:$rs1, FPR32:$rs2, FPR32:$rs3, frmarg:$funct3);
	let AsmString = "fnmsub.s	$rd, $rs1, $rs2, $rs3, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FSW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<12> imm12;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<7> opcode = 0b0100111;
	let Inst{31-25} = imm12{11-5};
	let Inst{11-7} = imm12{4-0};
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{6-0} = opcode;
}
def FSW: Instruction, FSW_FM, TSFlagTemplate<InstFormatS>, Sched<[WriteFST32, ReadStoreData, ReadFMemBase]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins FPR32:$rs2, GPR:$rs1, simm12:$imm12);
	let AsmString = "fsw	$rs2, ${imm12}(${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}