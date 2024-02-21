class DIV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000001;
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
def DIV: Instruction, DIV_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteIDiv, ReadIDiv, ReadIDiv]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "div	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtM];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class DIVU_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000001;
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
def DIVU: Instruction, DIVU_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteIDiv, ReadIDiv, ReadIDiv]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "divu	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtM];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class DIVUW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000001;
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
def DIVUW: Instruction, DIVUW_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteIDiv32, ReadIDiv32, ReadIDiv32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "divuw	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtM, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class DIVW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000001;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b100;
	bits<5> rd;
	bits<7> opcode = 0b0111011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def DIVW: Instruction, DIVW_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteIDiv32, ReadIDiv32, ReadIDiv32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "divw	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtM, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class MUL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000001;
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
def MUL: Instruction, MUL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteIMul, ReadIMul, ReadIMul]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "mul	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtM];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class MULH_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000001;
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
def MULH: Instruction, MULH_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteIMul, ReadIMul, ReadIMul]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "mulh	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtM];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class MULHSU_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000001;
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
def MULHSU: Instruction, MULHSU_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteIMul, ReadIMul, ReadIMul]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "mulhsu	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtM];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class MULHU_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000001;
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
def MULHU: Instruction, MULHU_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteIMul, ReadIMul, ReadIMul]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "mulhu	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtM];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class MULW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000001;
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
def MULW: Instruction, MULW_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteIMul32, ReadIMul32, ReadIMul32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "mulw	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtM, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class REM_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000001;
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
def REM: Instruction, REM_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteIDiv, ReadIDiv, ReadIDiv]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "rem	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtM];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class REMU_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000001;
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
def REMU: Instruction, REMU_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteIDiv, ReadIDiv, ReadIDiv]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "remu	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtM];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class REMUW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000001;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b111;
	bits<5> rd;
	bits<7> opcode = 0b0111011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def REMUW: Instruction, REMUW_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteIDiv32, ReadIDiv32, ReadIDiv32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "remuw	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtM, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class REMW_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<7> funct7 = 0b0000001;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b110;
	bits<5> rd;
	bits<7> opcode = 0b0111011;
	let Inst{31-25} = funct7;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def REMW: Instruction, REMW_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteIDiv32, ReadIDiv32, ReadIDiv32]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "remw	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtM, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}