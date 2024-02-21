class FSH_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<12> imm12;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<7> Opcode = 0b0100111;
	let Inst{31-25} = imm12{11-5};
	let Inst{11-7} = imm12{4-0};
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{6-0} = Opcode;
}
def FSH: Instruction, FSH_FM, TSFlagTemplate<InstFormatS>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins FPR16:$rs2, GPR:$rs1, simm12:$imm12);
	let AsmString = "fsh	$rs2, ${imm12}(${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtZfh];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FMADD_H_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> rs3;
	bits<2> funct2 = 0b10;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3;
	bits<5> rd;
	bits<7> Opcode = 0b1000011;
	let Inst{31-27} = rs3;
	let Inst{26-25} = funct2;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FMADD_H: Instruction, FMADD_H_FM, TSFlagTemplate<InstFormatR4>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR16:$rd);
	let InOperandList = (ins FPR16:$rs1, FPR16:$rs2, FPR16:$rs3, frmarg:$funct3);
	let AsmString = "fmadd.h	$rd, $rs1, $rs2, $rs3, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtZfh];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FMSUB_H_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> rs3;
	bits<2> funct2 = 0b10;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3;
	bits<5> rd;
	bits<7> Opcode = 0b1000111;
	let Inst{31-27} = rs3;
	let Inst{26-25} = funct2;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FMSUB_H: Instruction, FMSUB_H_FM, TSFlagTemplate<InstFormatR4>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR16:$rd);
	let InOperandList = (ins FPR16:$rs1, FPR16:$rs2, FPR16:$rs3, frmarg:$funct3);
	let AsmString = "fmsub.h	$rd, $rs1, $rs2, $rs3, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtZfh];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FNMSUB_H_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> rs3;
	bits<2> funct2 = 0b10;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3;
	bits<5> rd;
	bits<7> Opcode = 0b1001011;
	let Inst{31-27} = rs3;
	let Inst{26-25} = funct2;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FNMSUB_H: Instruction, FNMSUB_H_FM, TSFlagTemplate<InstFormatR4>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR16:$rd);
	let InOperandList = (ins FPR16:$rs1, FPR16:$rs2, FPR16:$rs3, frmarg:$funct3);
	let AsmString = "fnmsub.h	$rd, $rs1, $rs2, $rs3, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtZfh];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FNMADD_H_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> rs3;
	bits<2> funct2 = 0b10;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3;
	bits<5> rd;
	bits<7> Opcode = 0b1001111;
	let Inst{31-27} = rs3;
	let Inst{26-25} = funct2;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FNMADD_H: Instruction, FNMADD_H_FM, TSFlagTemplate<InstFormatR4>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR16:$rd);
	let InOperandList = (ins FPR16:$rs1, FPR16:$rs2, FPR16:$rs3, frmarg:$funct3);
	let AsmString = "fnmadd.h	$rd, $rs1, $rs2, $rs3, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtZfh];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FADD_H_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000010;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3;
	bits<5> rd;
	bits<7> Opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FADD_H: Instruction, FADD_H_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR16:$rd);
	let InOperandList = (ins FPR16:$rs1, FPR16:$rs2, frmarg:$funct3);
	let AsmString = "fadd.h	$rd, $rs1, $rs2, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtZfh];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FSUB_H_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000110;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3;
	bits<5> rd;
	bits<7> Opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FSUB_H: Instruction, FSUB_H_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR16:$rd);
	let InOperandList = (ins FPR16:$rs1, FPR16:$rs2, frmarg:$funct3);
	let AsmString = "fsub.h	$rd, $rs1, $rs2, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtZfh];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FMUL_H_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0001010;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3;
	bits<5> rd;
	bits<7> Opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FMUL_H: Instruction, FMUL_H_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR16:$rd);
	let InOperandList = (ins FPR16:$rs1, FPR16:$rs2, frmarg:$funct3);
	let AsmString = "fmul.h	$rd, $rs1, $rs2, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtZfh];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FDIV_H_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0001110;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3;
	bits<5> rd;
	bits<7> Opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FDIV_H: Instruction, FDIV_H_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR16:$rd);
	let InOperandList = (ins FPR16:$rs1, FPR16:$rs2, frmarg:$funct3);
	let AsmString = "fdiv.h	$rd, $rs1, $rs2, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtZfh];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FSQRT_H_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0101110;
	bits<5> rs2 = 0b00000;
	bits<5> rs1;
	bits<3> funct3;
	bits<5> rd;
	bits<7> Opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FSQRT_H: Instruction, FSQRT_H_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR16:$rd);
	let InOperandList = (ins FPR16:$rs1, frmarg:$funct3);
	let AsmString = "fsqrt.h	$rd, $rs1, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtZfh];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FSGNJ_H_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0010010;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b000;
	bits<5> rd;
	bits<7> Opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FSGNJ_H: Instruction, FSGNJ_H_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR16:$rd);
	let InOperandList = (ins FPR16:$rs1, FPR16:$rs2);
	let AsmString = "fsgnj.h	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZfh];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FSGNJN_H_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0010010;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FSGNJN_H: Instruction, FSGNJN_H_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR16:$rd);
	let InOperandList = (ins FPR16:$rs1, FPR16:$rs2);
	let AsmString = "fsgnjn.h	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZfh];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FSGNJX_H_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0010010;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> Opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FSGNJX_H: Instruction, FSGNJX_H_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR16:$rd);
	let InOperandList = (ins FPR16:$rs1, FPR16:$rs2);
	let AsmString = "fsgnjx.h	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZfh];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FMIN_H_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0010110;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b000;
	bits<5> rd;
	bits<7> Opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FMIN_H: Instruction, FMIN_H_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR16:$rd);
	let InOperandList = (ins FPR16:$rs1, FPR16:$rs2);
	let AsmString = "fmin.h	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZfh];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FMAX_H_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0010110;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FMAX_H: Instruction, FMAX_H_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR16:$rd);
	let InOperandList = (ins FPR16:$rs1, FPR16:$rs2);
	let AsmString = "fmax.h	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZfh];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FCVT_W_H_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1100010;
	bits<5> rs2 = 0b00000;
	bits<5> rs1;
	bits<3> funct3;
	bits<5> rd;
	bits<7> Opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FCVT_W_H: Instruction, FCVT_W_H_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins FPR16:$rs1, frmarg:$funct3);
	let AsmString = "fcvt.w.h	$rd, $rs1, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtZfh];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FCVT_WU_H_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1100010;
	bits<5> rs2 = 0b00001;
	bits<5> rs1;
	bits<3> funct3;
	bits<5> rd;
	bits<7> Opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FCVT_WU_H: Instruction, FCVT_WU_H_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins FPR16:$rs1, frmarg:$funct3);
	let AsmString = "fcvt.wu.h	$rd, $rs1, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtZfh];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FCVT_H_W_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1101010;
	bits<5> rs2 = 0b00000;
	bits<5> rs1;
	bits<3> funct3;
	bits<5> rd;
	bits<7> Opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FCVT_H_W: Instruction, FCVT_H_W_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR16:$rd);
	let InOperandList = (ins GPR:$rs1, frmarg:$funct3);
	let AsmString = "fcvt.h.w	$rd, $rs1, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtZfh];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FCVT_H_WU_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1101010;
	bits<5> rs2 = 0b00001;
	bits<5> rs1;
	bits<3> funct3;
	bits<5> rd;
	bits<7> Opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FCVT_H_WU: Instruction, FCVT_H_WU_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR16:$rd);
	let InOperandList = (ins GPR:$rs1, frmarg:$funct3);
	let AsmString = "fcvt.h.wu	$rd, $rs1, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtZfh];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FCVT_H_S_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0100010;
	bits<5> rs2 = 0b00000;
	bits<5> rs1;
	bits<3> funct3;
	bits<5> rd;
	bits<7> Opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FCVT_H_S: Instruction, FCVT_H_S_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR16:$rd);
	let InOperandList = (ins FPR32:$rs1, frmarg:$funct3);
	let AsmString = "fcvt.h.s	$rd, $rs1, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtZfh];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FCVT_S_H_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0100000;
	bits<5> rs2 = 0b00010;
	bits<5> rs1;
	bits<3> funct3 = 0b000;
	bits<5> rd;
	bits<7> Opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FCVT_S_H: Instruction, FCVT_S_H_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR32:$rd);
	let InOperandList = (ins FPR16:$rs1);
	let AsmString = "fcvt.s.h	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtZfh];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FMV_X_H_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1110010;
	bits<5> rs2 = 0b00000;
	bits<5> rs1;
	bits<3> funct3 = 0b000;
	bits<5> rd;
	bits<7> Opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FMV_X_H: Instruction, FMV_X_H_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins FPR16:$rs1);
	let AsmString = "fmv.x.h	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtZfh];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FMV_H_X_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1111010;
	bits<5> rs2 = 0b00000;
	bits<5> rs1;
	bits<3> funct3 = 0b000;
	bits<5> rd;
	bits<7> Opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FMV_H_X: Instruction, FMV_H_X_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR16:$rd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "fmv.h.x	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtZfh];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FEQ_H_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1010010;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> Opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FEQ_H: Instruction, FEQ_H_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins FPR16:$rs1, FPR16:$rs2);
	let AsmString = "feq.h	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZfh];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FLT_H_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1010010;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FLT_H: Instruction, FLT_H_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins FPR16:$rs1, FPR16:$rs2);
	let AsmString = "flt.h	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZfh];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FLE_H_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1010010;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b000;
	bits<5> rd;
	bits<7> Opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FLE_H: Instruction, FLE_H_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins FPR16:$rs1, FPR16:$rs2);
	let AsmString = "fle.h	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZfh];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FCLASS_H_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1110010;
	bits<5> rs2 = 0b00000;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FCLASS_H: Instruction, FCLASS_H_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins FPR16:$rs1);
	let AsmString = "fclass.h	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtZfh];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FCVT_L_H_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1100010;
	bits<5> rs2 = 0b00010;
	bits<5> rs1;
	bits<3> funct3;
	bits<5> rd;
	bits<7> Opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FCVT_L_H: Instruction, FCVT_L_H_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins FPR16:$rs1, frmarg:$funct3);
	let AsmString = "fcvt.l.h	$rd, $rs1, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtZfh, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FCVT_LU_H_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1100010;
	bits<5> rs2 = 0b00011;
	bits<5> rs1;
	bits<3> funct3;
	bits<5> rd;
	bits<7> Opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FCVT_LU_H: Instruction, FCVT_LU_H_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins FPR16:$rs1, frmarg:$funct3);
	let AsmString = "fcvt.lu.h	$rd, $rs1, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtZfh, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FCVT_H_L_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1101010;
	bits<5> rs2 = 0b00010;
	bits<5> rs1;
	bits<3> funct3;
	bits<5> rd;
	bits<7> Opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FCVT_H_L: Instruction, FCVT_H_L_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR16:$rd);
	let InOperandList = (ins GPR:$rs1, frmarg:$funct3);
	let AsmString = "fcvt.h.l	$rd, $rs1, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtZfh, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FCVT_H_LU_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b1101010;
	bits<5> rs2 = 0b00011;
	bits<5> rs1;
	bits<3> funct3;
	bits<5> rd;
	bits<7> Opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FCVT_H_LU: Instruction, FCVT_H_LU_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR16:$rd);
	let InOperandList = (ins GPR:$rs1, frmarg:$funct3);
	let AsmString = "fcvt.h.lu	$rd, $rs1, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtZfh, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FCVT_H_D_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0100010;
	bits<5> rs2 = 0b00001;
	bits<5> rs1;
	bits<3> funct3;
	bits<5> rd;
	bits<7> Opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FCVT_H_D: Instruction, FCVT_H_D_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR16:$rd);
	let InOperandList = (ins FPR64:$rs1, frmarg:$funct3);
	let AsmString = "fcvt.h.d	$rd, $rs1, $funct3";
	let Pattern = [];
	let Predicates = [HasStdExtZfh, HasStdExtD];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FCVT_D_H_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0100001;
	bits<5> rs2 = 0b00010;
	bits<5> rs1;
	bits<3> funct3 = 0b000;
	bits<5> rd;
	bits<7> Opcode = 0b1010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FCVT_D_H: Instruction, FCVT_D_H_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR64:$rd);
	let InOperandList = (ins FPR16:$rs1);
	let AsmString = "fcvt.d.h	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtZfh, HasStdExtD];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FLH_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<12> imm12;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0000111;
	let Inst{31-20} = imm12;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FLH: Instruction, FLH_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR16:$rd);
	let InOperandList = (ins GPR:$rs1, simm12:$imm12);
	let AsmString = "flh	$rd, ${imm12}(${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtZfh];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}