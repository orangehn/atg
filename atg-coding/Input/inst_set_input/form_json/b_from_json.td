class ADDUW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000100;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b000;
	bits<5> rd;
	bits<7> Opcode = 0b0111011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def ADDUW: Instruction, ADDUW_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "add.uw	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZba, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class ANDN_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0100000;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b111;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def ANDN: Instruction, ANDN_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "andn	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbbOrZbp];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class BCLR_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0100100;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def BCLR: Instruction, BCLR_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "bclr	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbs];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class BCLRI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b01001;
	bits<6> shamt;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0010011;
	let Inst{31-27} = funct5;
	let Inst{26-26} = 0b0;
	let Inst{25-20} = shamt;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def BCLRI: Instruction, BCLRI_FM, TSFlagTemplate<InstFormatI>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, uimmlog2xlen:$shamt);
	let AsmString = "bclri	$rd, $rs1, $shamt";
	let Pattern = [];
	let Predicates = [HasStdExtZbs];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class BCLRIW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0100100;
	bits<5> shamt;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0011011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = shamt;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def BCLRIW: Instruction, BCLRIW_FM, TSFlagTemplate<InstFormatI>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, uimm5:$shamt);
	let AsmString = "bclriw	$rd, $rs1, $shamt";
	let Pattern = [];
	let Predicates = [HasStdExtZbs, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class BCLRW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0100100;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0111011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def BCLRW: Instruction, BCLRW_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "bclrw	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbs, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class BEXT_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0100100;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b101;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def BEXT: Instruction, BEXT_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "bext	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbs];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class BEXTI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b01001;
	bits<6> shamt;
	bits<5> rs1;
	bits<3> funct3 = 0b101;
	bits<5> rd;
	bits<7> Opcode = 0b0010011;
	let Inst{31-27} = funct5;
	let Inst{26-26} = 0b0;
	let Inst{25-20} = shamt;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def BEXTI: Instruction, BEXTI_FM, TSFlagTemplate<InstFormatI>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, uimmlog2xlen:$shamt);
	let AsmString = "bexti	$rd, $rs1, $shamt";
	let Pattern = [];
	let Predicates = [HasStdExtZbs];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class BEXTW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0100100;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b101;
	bits<5> rd;
	bits<7> Opcode = 0b0111011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def BEXTW: Instruction, BEXTW_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "bextw	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbs, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class BFP_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0100100;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b111;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def BFP: Instruction, BFP_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "bfp	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbf];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class BFPW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0100100;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b111;
	bits<5> rd;
	bits<7> Opcode = 0b0111011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def BFPW: Instruction, BFPW_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "bfpw	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbf, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class BINV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0110100;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def BINV: Instruction, BINV_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "binv	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbs];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class BINVI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b01101;
	bits<6> shamt;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0010011;
	let Inst{31-27} = funct5;
	let Inst{26-26} = 0b0;
	let Inst{25-20} = shamt;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def BINVI: Instruction, BINVI_FM, TSFlagTemplate<InstFormatI>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, uimmlog2xlen:$shamt);
	let AsmString = "binvi	$rd, $rs1, $shamt";
	let Pattern = [];
	let Predicates = [HasStdExtZbs];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class BINVIW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0110100;
	bits<5> shamt;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0011011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = shamt;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def BINVIW: Instruction, BINVIW_FM, TSFlagTemplate<InstFormatI>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, uimm5:$shamt);
	let AsmString = "binviw	$rd, $rs1, $shamt";
	let Pattern = [];
	let Predicates = [HasStdExtZbs, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class BINVW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0110100;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0111011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def BINVW: Instruction, BINVW_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "binvw	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbs, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class BMATFLIP_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0110000;
	bits<5> funct5 = 0b00011;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = funct5;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def BMATFLIP: Instruction, BMATFLIP_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "bmatflip	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtZbm, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class BMATOR_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000100;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def BMATOR: Instruction, BMATOR_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "bmator	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbm, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class BMATXOR_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0100100;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def BMATXOR: Instruction, BMATXOR_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "bmatxor	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbm, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class BSET_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0010100;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def BSET: Instruction, BSET_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "bset	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbs];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class BSETI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00101;
	bits<6> shamt;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0010011;
	let Inst{31-27} = funct5;
	let Inst{26-26} = 0b0;
	let Inst{25-20} = shamt;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def BSETI: Instruction, BSETI_FM, TSFlagTemplate<InstFormatI>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, uimmlog2xlen:$shamt);
	let AsmString = "bseti	$rd, $rs1, $shamt";
	let Pattern = [];
	let Predicates = [HasStdExtZbs];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class BSETIW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0010100;
	bits<5> shamt;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0011011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = shamt;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def BSETIW: Instruction, BSETIW_FM, TSFlagTemplate<InstFormatI>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, uimm5:$shamt);
	let AsmString = "bsetiw	$rd, $rs1, $shamt";
	let Pattern = [];
	let Predicates = [HasStdExtZbs, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class BSETW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0010100;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0111011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def BSETW: Instruction, BSETW_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "bsetw	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbs, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class CLMUL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000101;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def CLMUL: Instruction, CLMUL_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "clmul	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbc];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class CLMULH_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000101;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def CLMULH: Instruction, CLMULH_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "clmulh	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbc];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class CLMULR_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000101;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def CLMULR: Instruction, CLMULR_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "clmulr	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbc];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class CLZ_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0110000;
	bits<5> funct5 = 0b00000;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = funct5;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def CLZ: Instruction, CLZ_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "clz	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtZbb];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class CLZW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0110000;
	bits<5> funct5 = 0b00000;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0011011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = funct5;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def CLZW: Instruction, CLZW_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "clzw	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtZbb, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class CMIX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> rs3;
	bits<2> funct2 = 0b11;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3_b = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-27} = rs3;
	let Inst{26-25} = funct2;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3_b;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def CMIX: Instruction, CMIX_FM, TSFlagTemplate<InstFormatR4>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, GPR:$rs3);
	let AsmString = "cmix	$rd, $rs2, $rs1, $rs3";
	let Pattern = [];
	let Predicates = [HasStdExtZbt];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class CMOV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> rs3;
	bits<2> funct2 = 0b11;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3_b = 0b101;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-27} = rs3;
	let Inst{26-25} = funct2;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3_b;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def CMOV: Instruction, CMOV_FM, TSFlagTemplate<InstFormatR4>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, GPR:$rs3);
	let AsmString = "cmov	$rd, $rs2, $rs1, $rs3";
	let Pattern = [];
	let Predicates = [HasStdExtZbt];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class CPOP_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0110000;
	bits<5> funct5 = 0b00010;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = funct5;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def CPOP: Instruction, CPOP_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "cpop	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtZbb];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class CPOPW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0110000;
	bits<5> funct5 = 0b00010;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0011011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = funct5;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def CPOPW: Instruction, CPOPW_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "cpopw	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtZbb, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class CRC32B_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0110000;
	bits<5> funct5 = 0b10000;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = funct5;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def CRC32B: Instruction, CRC32B_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "crc32.b	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtZbr];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class CRC32CB_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0110000;
	bits<5> funct5 = 0b11000;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = funct5;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def CRC32CB: Instruction, CRC32CB_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "crc32c.b	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtZbr];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class CRC32CD_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0110000;
	bits<5> funct5 = 0b11011;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = funct5;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def CRC32CD: Instruction, CRC32CD_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "crc32c.d	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtZbr, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class CRC32CH_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0110000;
	bits<5> funct5 = 0b11001;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = funct5;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def CRC32CH: Instruction, CRC32CH_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "crc32c.h	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtZbr];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class CRC32CW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0110000;
	bits<5> funct5 = 0b11010;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = funct5;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def CRC32CW: Instruction, CRC32CW_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "crc32c.w	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtZbr];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class CRC32D_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0110000;
	bits<5> funct5 = 0b10011;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = funct5;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def CRC32D: Instruction, CRC32D_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "crc32.d	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtZbr, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class CRC32H_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0110000;
	bits<5> funct5 = 0b10001;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = funct5;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def CRC32H: Instruction, CRC32H_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "crc32.h	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtZbr];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class CRC32W_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0110000;
	bits<5> funct5 = 0b10010;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = funct5;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def CRC32W: Instruction, CRC32W_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "crc32.w	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtZbr];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class CTZ_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0110000;
	bits<5> funct5 = 0b00001;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = funct5;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def CTZ: Instruction, CTZ_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "ctz	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtZbb];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class CTZW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0110000;
	bits<5> funct5 = 0b00001;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0011011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = funct5;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def CTZW: Instruction, CTZW_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "ctzw	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtZbb, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FSL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> rs3;
	bits<2> funct2 = 0b10;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3_b = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-27} = rs3;
	let Inst{26-25} = funct2;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3_b;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FSL: Instruction, FSL_FM, TSFlagTemplate<InstFormatR4>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, GPR:$rs3);
	let AsmString = "fsl	$rd, $rs1, $rs3, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbt];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FSLW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> rs3;
	bits<2> funct2 = 0b10;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3_b = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0111011;
	let Inst{31-27} = rs3;
	let Inst{26-25} = funct2;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3_b;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FSLW: Instruction, FSLW_FM, TSFlagTemplate<InstFormatR4>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, GPR:$rs3);
	let AsmString = "fslw	$rd, $rs1, $rs3, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbt, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FSR_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> rs3;
	bits<2> funct2 = 0b10;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3_b = 0b101;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-27} = rs3;
	let Inst{26-25} = funct2;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3_b;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FSR: Instruction, FSR_FM, TSFlagTemplate<InstFormatR4>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, GPR:$rs3);
	let AsmString = "fsr	$rd, $rs1, $rs3, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbt];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FSRI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> rs3;
	bits<1> immediate = 0b1;
	bits<6> shamt;
	bits<5> rs1;
	bits<3> funct3_b = 0b101;
	bits<5> rd;
	bits<7> Opcode = 0b0010011;
	let Inst{31-27} = rs3;
	let Inst{26-26} = immediate;
	let Inst{25-20} = shamt;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3_b;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FSRI: Instruction, FSRI_FM, TSFlagTemplate<InstFormatR4>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs3, uimmlog2xlen:$shamt);
	let AsmString = "fsri	$rd, $rs1, $rs3, $shamt";
	let Pattern = [];
	let Predicates = [HasStdExtZbt];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FSRIW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> rs3;
	bits<2> funct2 = 0b10;
	bits<5> shamt;
	bits<5> rs1;
	bits<3> funct3_b = 0b101;
	bits<5> rd;
	bits<7> Opcode = 0b0011011;
	let Inst{31-27} = rs3;
	let Inst{26-25} = funct2;
	let Inst{24-20} = shamt;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3_b;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FSRIW: Instruction, FSRIW_FM, TSFlagTemplate<InstFormatR4>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs3, uimm5:$shamt);
	let AsmString = "fsriw	$rd, $rs1, $rs3, $shamt";
	let Pattern = [];
	let Predicates = [HasStdExtZbt, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class FSRW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> rs3;
	bits<2> funct2 = 0b10;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3_b = 0b101;
	bits<5> rd;
	bits<7> Opcode = 0b0111011;
	let Inst{31-27} = rs3;
	let Inst{26-25} = funct2;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3_b;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def FSRW: Instruction, FSRW_FM, TSFlagTemplate<InstFormatR4>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, GPR:$rs3);
	let AsmString = "fsrw	$rd, $rs1, $rs3, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbt, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class GORC_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0010100;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b101;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def GORC: Instruction, GORC_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "gorc	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbp];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class GORCI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00101;
	bits<6> shamt;
	bits<5> rs1;
	bits<3> funct3 = 0b101;
	bits<5> rd;
	bits<7> Opcode = 0b0010011;
	let Inst{31-27} = funct5;
	let Inst{26-26} = 0b0;
	let Inst{25-20} = shamt;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def GORCI: Instruction, GORCI_FM, TSFlagTemplate<InstFormatI>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, uimmlog2xlen:$shamt);
	let AsmString = "gorci	$rd, $rs1, $shamt";
	let Pattern = [];
	let Predicates = [HasStdExtZbp];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class GORCIW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0010100;
	bits<5> shamt;
	bits<5> rs1;
	bits<3> funct3 = 0b101;
	bits<5> rd;
	bits<7> Opcode = 0b0011011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = shamt;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def GORCIW: Instruction, GORCIW_FM, TSFlagTemplate<InstFormatI>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, uimm5:$shamt);
	let AsmString = "gorciw	$rd, $rs1, $shamt";
	let Pattern = [];
	let Predicates = [HasStdExtZbp, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class GORCW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0010100;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b101;
	bits<5> rd;
	bits<7> Opcode = 0b0111011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def GORCW: Instruction, GORCW_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "gorcw	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbp, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class GREV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0110100;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b101;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def GREV: Instruction, GREV_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "grev	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbp];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class GREVI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b01101;
	bits<6> shamt;
	bits<5> rs1;
	bits<3> funct3 = 0b101;
	bits<5> rd;
	bits<7> Opcode = 0b0010011;
	let Inst{31-27} = funct5;
	let Inst{26-26} = 0b0;
	let Inst{25-20} = shamt;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def GREVI: Instruction, GREVI_FM, TSFlagTemplate<InstFormatI>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, uimmlog2xlen:$shamt);
	let AsmString = "grevi	$rd, $rs1, $shamt";
	let Pattern = [];
	let Predicates = [HasStdExtZbp];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class GREVIW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0110100;
	bits<5> shamt;
	bits<5> rs1;
	bits<3> funct3 = 0b101;
	bits<5> rd;
	bits<7> Opcode = 0b0011011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = shamt;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def GREVIW: Instruction, GREVIW_FM, TSFlagTemplate<InstFormatI>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, uimm5:$shamt);
	let AsmString = "greviw	$rd, $rs1, $shamt";
	let Pattern = [];
	let Predicates = [HasStdExtZbp, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class GREVW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0110100;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b101;
	bits<5> rd;
	bits<7> Opcode = 0b0111011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def GREVW: Instruction, GREVW_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "grevw	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbp, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class MAX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000101;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b110;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def MAX: Instruction, MAX_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "max	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbb];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class MAXU_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000101;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b111;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def MAXU: Instruction, MAXU_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "maxu	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbb];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class MIN_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000101;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b100;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def MIN: Instruction, MIN_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "min	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbb];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class MINU_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000101;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b101;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def MINU: Instruction, MINU_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "minu	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbb];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class ORCB_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<12> immediate = 0b001010000111;
	bits<5> rs1;
	bits<3> funct3 = 0b101;
	bits<5> rd;
	bits<7> Opcode = 0b0010011;
	let Inst{31-20} = immediate;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def ORCB: Instruction, ORCB_FM, TSFlagTemplate<InstFormatI>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "orc.b	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtZbbOrZbp];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class ORN_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0100000;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b110;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def ORN: Instruction, ORN_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "orn	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbbOrZbp];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class PACKUW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0100100;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b100;
	bits<5> rd;
	bits<7> Opcode = 0b0111011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def PACKUW: Instruction, PACKUW_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "packuw	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbp, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class PACKW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000100;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b100;
	bits<5> rd;
	bits<7> Opcode = 0b0111011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def PACKW: Instruction, PACKW_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "packw	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbp, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class REV8_RV32_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<12> immediate = 0b011010011000;
	bits<5> rs1;
	bits<3> funct3 = 0b101;
	bits<5> rd;
	bits<7> Opcode = 0b0010011;
	let Inst{31-20} = immediate;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def REV8_RV32: Instruction, REV8_RV32_FM, TSFlagTemplate<InstFormatI>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "rev8	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtZbbOrZbp, IsRV32];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class REV8_RV64_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<12> immediate = 0b011010111000;
	bits<5> rs1;
	bits<3> funct3 = 0b101;
	bits<5> rd;
	bits<7> Opcode = 0b0010011;
	let Inst{31-20} = immediate;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def REV8_RV64: Instruction, REV8_RV64_FM, TSFlagTemplate<InstFormatI>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "rev8	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtZbbOrZbp, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class ROL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0110000;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def ROL: Instruction, ROL_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "rol	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbbOrZbp];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class ROLW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0110000;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0111011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def ROLW: Instruction, ROLW_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "rolw	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbbOrZbp, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class ROR_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0110000;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b101;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def ROR: Instruction, ROR_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "ror	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbbOrZbp];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class RORI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b01100;
	bits<6> shamt;
	bits<5> rs1;
	bits<3> funct3 = 0b101;
	bits<5> rd;
	bits<7> Opcode = 0b0010011;
	let Inst{31-27} = funct5;
	let Inst{26-26} = 0b0;
	let Inst{25-20} = shamt;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def RORI: Instruction, RORI_FM, TSFlagTemplate<InstFormatI>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, uimmlog2xlen:$shamt);
	let AsmString = "rori	$rd, $rs1, $shamt";
	let Pattern = [];
	let Predicates = [HasStdExtZbbOrZbp];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class RORIW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0110000;
	bits<5> shamt;
	bits<5> rs1;
	bits<3> funct3 = 0b101;
	bits<5> rd;
	bits<7> Opcode = 0b0011011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = shamt;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def RORIW: Instruction, RORIW_FM, TSFlagTemplate<InstFormatI>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, uimm5:$shamt);
	let AsmString = "roriw	$rd, $rs1, $shamt";
	let Pattern = [];
	let Predicates = [HasStdExtZbbOrZbp, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class RORW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0110000;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b101;
	bits<5> rd;
	bits<7> Opcode = 0b0111011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def RORW: Instruction, RORW_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "rorw	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbbOrZbp, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SEXTB_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0110000;
	bits<5> funct5 = 0b00100;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = funct5;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def SEXTB: Instruction, SEXTB_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "sext.b	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtZbb];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SEXTH_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0110000;
	bits<5> funct5 = 0b00101;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0010011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = funct5;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def SEXTH: Instruction, SEXTH_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "sext.h	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtZbb];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SH1ADD_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0010000;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def SH1ADD: Instruction, SH1ADD_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "sh1add	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZba];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SH1ADDUW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0010000;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> Opcode = 0b0111011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def SH1ADDUW: Instruction, SH1ADDUW_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "sh1add.uw	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZba, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SH2ADD_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0010000;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b100;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def SH2ADD: Instruction, SH2ADD_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "sh2add	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZba];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SH2ADDUW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0010000;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b100;
	bits<5> rd;
	bits<7> Opcode = 0b0111011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def SH2ADDUW: Instruction, SH2ADDUW_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "sh2add.uw	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZba, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SH3ADD_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0010000;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b110;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def SH3ADD: Instruction, SH3ADD_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "sh3add	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZba];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SH3ADDUW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0010000;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b110;
	bits<5> rd;
	bits<7> Opcode = 0b0111011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def SH3ADDUW: Instruction, SH3ADDUW_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "sh3add.uw	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZba, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SHFL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000100;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def SHFL: Instruction, SHFL_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "shfl	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbp];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SHFLI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000010;
	bits<6> shamt;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0010011;
	let Inst{31-26} = funct6;
	let Inst{25-20} = shamt;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def SHFLI: Instruction, SHFLI_FM, TSFlagTemplate<InstFormatI>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, shfl_uimm:$shamt);
	let AsmString = "shfli	$rd, $rs1, $shamt";
	let Pattern = [];
	let Predicates = [HasStdExtZbp];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SHFLW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000100;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0111011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def SHFLW: Instruction, SHFLW_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "shflw	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbp, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SLLIUW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00001;
	bits<6> shamt;
	bits<5> rs1;
	bits<3> funct3 = 0b001;
	bits<5> rd;
	bits<7> Opcode = 0b0011011;
	let Inst{31-27} = funct5;
	let Inst{26-26} = 0b0;
	let Inst{25-20} = shamt;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def SLLIUW: Instruction, SLLIUW_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, uimmlog2xlen:$shamt);
	let AsmString = "slli.uw	$rd, $rs1, $shamt";
	let Pattern = [];
	let Predicates = [HasStdExtZba, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class UNSHFL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000100;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b101;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def UNSHFL: Instruction, UNSHFL_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "unshfl	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbp];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class UNSHFLI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000010;
	bits<6> shamt;
	bits<5> rs1;
	bits<3> funct3 = 0b101;
	bits<5> rd;
	bits<7> Opcode = 0b0010011;
	let Inst{31-26} = funct6;
	let Inst{25-20} = shamt;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def UNSHFLI: Instruction, UNSHFLI_FM, TSFlagTemplate<InstFormatI>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, shfl_uimm:$shamt);
	let AsmString = "unshfli	$rd, $rs1, $shamt";
	let Pattern = [];
	let Predicates = [HasStdExtZbp];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class UNSHFLW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000100;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b101;
	bits<5> rd;
	bits<7> Opcode = 0b0111011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def UNSHFLW: Instruction, UNSHFLW_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "unshflw	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbp, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class XNOR_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0100000;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b100;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def XNOR: Instruction, XNOR_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "xnor	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbbOrZbp];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class ZEXTH_RV32_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000100;
	bits<5> rs2 = 0b00000;
	bits<5> rs1;
	bits<3> funct3 = 0b100;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def ZEXTH_RV32: Instruction, ZEXTH_RV32_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "zext.h	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtZbbOrZbp, IsRV32];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class ZEXTH_RV64_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000100;
	bits<5> rs2 = 0b00000;
	bits<5> rs1;
	bits<3> funct3 = 0b100;
	bits<5> rd;
	bits<7> Opcode = 0b0111011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def ZEXTH_RV64: Instruction, ZEXTH_RV64_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "zext.h	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtZbbOrZbp, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class BCOMPRESS_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000100;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b110;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def BCOMPRESS: Instruction, BCOMPRESS_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "bcompress	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbe];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class BCOMPRESSW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000100;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b110;
	bits<5> rd;
	bits<7> Opcode = 0b0111011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def BCOMPRESSW: Instruction, BCOMPRESSW_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "bcompressw	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbe, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class BDECOMPRESS_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0100100;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b110;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def BDECOMPRESS: Instruction, BDECOMPRESS_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "bdecompress	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbe];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class BDECOMPRESSW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0100100;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b110;
	bits<5> rd;
	bits<7> Opcode = 0b0111011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def BDECOMPRESSW: Instruction, BDECOMPRESSW_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "bdecompressw	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbe, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class PACK_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000100;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b100;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def PACK: Instruction, PACK_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "pack	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbp];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class PACKH_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000100;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b111;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def PACKH: Instruction, PACKH_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "packh	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbp];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class PACKU_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0100100;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b100;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def PACKU: Instruction, PACKU_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "packu	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbp];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class XPERMB_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0010100;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b100;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def XPERMB: Instruction, XPERMB_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "xperm.b	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbp];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class XPERMH_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0010100;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b110;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def XPERMH: Instruction, XPERMH_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "xperm.h	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbp];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class XPERMN_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0010100;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def XPERMN: Instruction, XPERMN_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "xperm.n	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbp];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class XPERMW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0010100;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b000;
	bits<5> rd;
	bits<7> Opcode = 0b0110011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = Opcode;
}
def XPERMW: Instruction, XPERMW_FM, TSFlagTemplate<InstFormatR>, Sched<[]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "xperm.w	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtZbp, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class C_NEG_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<4> immediate1 = 0b0110;
	bits<2> funct2 = 0b01;
	bits<3> rs;
	bits<7> immediate = 0b0000001;
	let Inst{15-12} = immediate1;
	let Inst{11-10} = funct2;
	let Inst{9-7} = rs;
	let Inst{6-0} = immediate;
}
def C_NEG: Instruction, C_NEG_FM, TSFlagTemplate<InstFormatCR>, Sched<[]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPRC:$rs_wb);
	let InOperandList = (ins GPRC:$rs);
	let AsmString = "c.neg	$rs";
	let Pattern = [];
	let Predicates = [HasStdExtZbproposedc, HasStdExtC];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "RVBC";
	let Constraints = "$rs = $rs_wb";
}
class C_NOT_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<4> immediate1 = 0b0110;
	bits<2> funct2 = 0b00;
	bits<3> rs;
	bits<7> immediate = 0b0000001;
	let Inst{15-12} = immediate1;
	let Inst{11-10} = funct2;
	let Inst{9-7} = rs;
	let Inst{6-0} = immediate;
}
def C_NOT: Instruction, C_NOT_FM, TSFlagTemplate<InstFormatCR>, Sched<[]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPRC:$rs_wb);
	let InOperandList = (ins GPRC:$rs);
	let AsmString = "c.not	$rs";
	let Pattern = [];
	let Predicates = [HasStdExtZbproposedc, HasStdExtC];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "RVBC";
	let Constraints = "$rs = $rs_wb";
}
class C_ZEXTW_FM: ATGInst{
	bits<16> SoftFail=0;
	bits<16> Inst;
	bits<4> immediate1 = 0b0110;
	bits<2> funct2 = 0b10;
	bits<3> rs;
	bits<7> immediate = 0b0000001;
	let Inst{15-12} = immediate1;
	let Inst{11-10} = funct2;
	let Inst{9-7} = rs;
	let Inst{6-0} = immediate;
}
def C_ZEXTW: Instruction, C_ZEXTW_FM, TSFlagTemplate<InstFormatCR>, Sched<[]>{
	let Size = 2;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPRC:$rs_wb);
	let InOperandList = (ins GPRC:$rs);
	let AsmString = "c.zext.w	$rs";
	let Pattern = [];
	let Predicates = [HasStdExtZbproposedc, HasStdExtZba, HasStdExtC, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "RVBC";
	let Constraints = "$rs = $rs_wb";
}