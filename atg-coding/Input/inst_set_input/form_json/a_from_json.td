class AMOADD_D_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00000;
	bits<1> aq = 0b0;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOADD_D: Instruction, AMOADD_D_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoadd.d	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOADD_D_AQ_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00000;
	bits<1> aq = 0b1;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOADD_D_AQ: Instruction, AMOADD_D_AQ_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoadd.d.aq	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOADD_D_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00000;
	bits<1> aq = 0b0;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOADD_D_RL: Instruction, AMOADD_D_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoadd.d.rl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOADD_D_AQ_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00000;
	bits<1> aq = 0b1;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOADD_D_AQ_RL: Instruction, AMOADD_D_AQ_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoadd.d.aqrl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOADD_W_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00000;
	bits<1> aq = 0b0;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOADD_W: Instruction, AMOADD_W_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoadd.w	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOADD_W_AQ_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00000;
	bits<1> aq = 0b1;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOADD_W_AQ: Instruction, AMOADD_W_AQ_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoadd.w.aq	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOADD_W_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00000;
	bits<1> aq = 0b0;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOADD_W_RL: Instruction, AMOADD_W_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoadd.w.rl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOADD_W_AQ_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00000;
	bits<1> aq = 0b1;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOADD_W_AQ_RL: Instruction, AMOADD_W_AQ_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoadd.w.aqrl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOAND_D_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b01100;
	bits<1> aq = 0b0;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOAND_D: Instruction, AMOAND_D_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoand.d	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOAND_D_AQ_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b01100;
	bits<1> aq = 0b1;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOAND_D_AQ: Instruction, AMOAND_D_AQ_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoand.d.aq	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOAND_D_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b01100;
	bits<1> aq = 0b0;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOAND_D_RL: Instruction, AMOAND_D_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoand.d.rl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOAND_D_AQ_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b01100;
	bits<1> aq = 0b1;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOAND_D_AQ_RL: Instruction, AMOAND_D_AQ_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoand.d.aqrl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOAND_W_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b01100;
	bits<1> aq = 0b0;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOAND_W: Instruction, AMOAND_W_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoand.w	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOAND_W_AQ_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b01100;
	bits<1> aq = 0b1;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOAND_W_AQ: Instruction, AMOAND_W_AQ_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoand.w.aq	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOAND_W_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b01100;
	bits<1> aq = 0b0;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOAND_W_RL: Instruction, AMOAND_W_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoand.w.rl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOAND_W_AQ_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b01100;
	bits<1> aq = 0b1;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOAND_W_AQ_RL: Instruction, AMOAND_W_AQ_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoand.w.aqrl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOMAX_D_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b10100;
	bits<1> aq = 0b0;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOMAX_D: Instruction, AMOMAX_D_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amomax.d	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOMAX_D_AQ_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b10100;
	bits<1> aq = 0b1;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOMAX_D_AQ: Instruction, AMOMAX_D_AQ_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amomax.d.aq	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOMAX_D_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b10100;
	bits<1> aq = 0b0;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOMAX_D_RL: Instruction, AMOMAX_D_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amomax.d.rl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOMAX_D_AQ_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b10100;
	bits<1> aq = 0b1;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOMAX_D_AQ_RL: Instruction, AMOMAX_D_AQ_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amomax.d.aqrl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOMAX_W_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b10100;
	bits<1> aq = 0b0;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOMAX_W: Instruction, AMOMAX_W_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amomax.w	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOMAX_W_AQ_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b10100;
	bits<1> aq = 0b1;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOMAX_W_AQ: Instruction, AMOMAX_W_AQ_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amomax.w.aq	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOMAX_W_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b10100;
	bits<1> aq = 0b0;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOMAX_W_RL: Instruction, AMOMAX_W_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amomax.w.rl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOMAX_W_AQ_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b10100;
	bits<1> aq = 0b1;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOMAX_W_AQ_RL: Instruction, AMOMAX_W_AQ_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amomax.w.aqrl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOMAXU_D_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b11100;
	bits<1> aq = 0b0;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOMAXU_D: Instruction, AMOMAXU_D_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amomaxu.d	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOMAXU_D_AQ_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b11100;
	bits<1> aq = 0b1;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOMAXU_D_AQ: Instruction, AMOMAXU_D_AQ_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amomaxu.d.aq	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOMAXU_D_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b11100;
	bits<1> aq = 0b0;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOMAXU_D_RL: Instruction, AMOMAXU_D_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amomaxu.d.rl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOMAXU_D_AQ_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b11100;
	bits<1> aq = 0b1;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOMAXU_D_AQ_RL: Instruction, AMOMAXU_D_AQ_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amomaxu.d.aqrl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOMAXU_W_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b11100;
	bits<1> aq = 0b0;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOMAXU_W: Instruction, AMOMAXU_W_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amomaxu.w	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOMAXU_W_AQ_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b11100;
	bits<1> aq = 0b1;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOMAXU_W_AQ: Instruction, AMOMAXU_W_AQ_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amomaxu.w.aq	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOMAXU_W_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b11100;
	bits<1> aq = 0b0;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOMAXU_W_RL: Instruction, AMOMAXU_W_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amomaxu.w.rl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOMAXU_W_AQ_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b11100;
	bits<1> aq = 0b1;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOMAXU_W_AQ_RL: Instruction, AMOMAXU_W_AQ_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amomaxu.w.aqrl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOMIN_D_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b10000;
	bits<1> aq = 0b0;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOMIN_D: Instruction, AMOMIN_D_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amomin.d	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOMIN_D_AQ_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b10000;
	bits<1> aq = 0b1;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOMIN_D_AQ: Instruction, AMOMIN_D_AQ_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amomin.d.aq	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOMIN_D_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b10000;
	bits<1> aq = 0b0;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOMIN_D_RL: Instruction, AMOMIN_D_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amomin.d.rl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOMIN_D_AQ_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b10000;
	bits<1> aq = 0b1;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOMIN_D_AQ_RL: Instruction, AMOMIN_D_AQ_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amomin.d.aqrl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOMIN_W_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b10000;
	bits<1> aq = 0b0;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOMIN_W: Instruction, AMOMIN_W_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amomin.w	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOMIN_W_AQ_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b10000;
	bits<1> aq = 0b1;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOMIN_W_AQ: Instruction, AMOMIN_W_AQ_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amomin.w.aq	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOMIN_W_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b10000;
	bits<1> aq = 0b0;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOMIN_W_RL: Instruction, AMOMIN_W_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amomin.w.rl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOMIN_W_AQ_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b10000;
	bits<1> aq = 0b1;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOMIN_W_AQ_RL: Instruction, AMOMIN_W_AQ_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amomin.w.aqrl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOMINU_D_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b11000;
	bits<1> aq = 0b0;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOMINU_D: Instruction, AMOMINU_D_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amominu.d	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOMINU_D_AQ_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b11000;
	bits<1> aq = 0b1;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOMINU_D_AQ: Instruction, AMOMINU_D_AQ_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amominu.d.aq	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOMINU_D_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b11000;
	bits<1> aq = 0b0;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOMINU_D_RL: Instruction, AMOMINU_D_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amominu.d.rl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOMINU_D_AQ_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b11000;
	bits<1> aq = 0b1;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOMINU_D_AQ_RL: Instruction, AMOMINU_D_AQ_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amominu.d.aqrl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOMINU_W_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b11000;
	bits<1> aq = 0b0;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOMINU_W: Instruction, AMOMINU_W_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amominu.w	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOMINU_W_AQ_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b11000;
	bits<1> aq = 0b1;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOMINU_W_AQ: Instruction, AMOMINU_W_AQ_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amominu.w.aq	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOMINU_W_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b11000;
	bits<1> aq = 0b0;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOMINU_W_RL: Instruction, AMOMINU_W_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amominu.w.rl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOMINU_W_AQ_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b11000;
	bits<1> aq = 0b1;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOMINU_W_AQ_RL: Instruction, AMOMINU_W_AQ_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amominu.w.aqrl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOOR_D_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b01000;
	bits<1> aq = 0b0;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOOR_D: Instruction, AMOOR_D_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoor.d	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOOR_D_AQ_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b01000;
	bits<1> aq = 0b1;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOOR_D_AQ: Instruction, AMOOR_D_AQ_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoor.d.aq	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOOR_D_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b01000;
	bits<1> aq = 0b0;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOOR_D_RL: Instruction, AMOOR_D_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoor.d.rl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOOR_D_AQ_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b01000;
	bits<1> aq = 0b1;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOOR_D_AQ_RL: Instruction, AMOOR_D_AQ_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoor.d.aqrl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOOR_W_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b01000;
	bits<1> aq = 0b0;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOOR_W: Instruction, AMOOR_W_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoor.w	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOOR_W_AQ_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b01000;
	bits<1> aq = 0b1;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOOR_W_AQ: Instruction, AMOOR_W_AQ_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoor.w.aq	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOOR_W_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b01000;
	bits<1> aq = 0b0;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOOR_W_RL: Instruction, AMOOR_W_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoor.w.rl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOOR_W_AQ_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b01000;
	bits<1> aq = 0b1;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOOR_W_AQ_RL: Instruction, AMOOR_W_AQ_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoor.w.aqrl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOXOR_D_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00100;
	bits<1> aq = 0b0;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOXOR_D: Instruction, AMOXOR_D_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoxor.d	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOXOR_D_AQ_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00100;
	bits<1> aq = 0b1;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOXOR_D_AQ: Instruction, AMOXOR_D_AQ_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoxor.d.aq	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOXOR_D_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00100;
	bits<1> aq = 0b0;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOXOR_D_RL: Instruction, AMOXOR_D_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoxor.d.rl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOXOR_D_AQ_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00100;
	bits<1> aq = 0b1;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOXOR_D_AQ_RL: Instruction, AMOXOR_D_AQ_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoxor.d.aqrl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOXOR_W_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00100;
	bits<1> aq = 0b0;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOXOR_W: Instruction, AMOXOR_W_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoxor.w	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOXOR_W_AQ_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00100;
	bits<1> aq = 0b1;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOXOR_W_AQ: Instruction, AMOXOR_W_AQ_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoxor.w.aq	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOXOR_W_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00100;
	bits<1> aq = 0b0;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOXOR_W_RL: Instruction, AMOXOR_W_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoxor.w.rl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOXOR_W_AQ_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00100;
	bits<1> aq = 0b1;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOXOR_W_AQ_RL: Instruction, AMOXOR_W_AQ_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoxor.w.aqrl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class LR_D_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00010;
	bits<1> aq = 0b0;
	bits<1> rl = 0b0;
	bits<5> rs2 = 0b00000;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def LR_D: Instruction, LR_D_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicLDD, ReadAtomicLDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1);
	let AsmString = "lr.d	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class LR_D_AQ_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00010;
	bits<1> aq = 0b1;
	bits<1> rl = 0b0;
	bits<5> rs2 = 0b00000;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def LR_D_AQ: Instruction, LR_D_AQ_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicLDD, ReadAtomicLDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1);
	let AsmString = "lr.d.aq	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class LR_D_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00010;
	bits<1> aq = 0b0;
	bits<1> rl = 0b1;
	bits<5> rs2 = 0b00000;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def LR_D_RL: Instruction, LR_D_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicLDD, ReadAtomicLDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1);
	let AsmString = "lr.d.rl	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class LR_D_AQ_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00010;
	bits<1> aq = 0b1;
	bits<1> rl = 0b1;
	bits<5> rs2 = 0b00000;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def LR_D_AQ_RL: Instruction, LR_D_AQ_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicLDD, ReadAtomicLDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1);
	let AsmString = "lr.d.aqrl	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class LR_W_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00010;
	bits<1> aq = 0b0;
	bits<1> rl = 0b0;
	bits<5> rs2 = 0b00000;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def LR_W: Instruction, LR_W_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicLDW, ReadAtomicLDW]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1);
	let AsmString = "lr.w	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class LR_W_AQ_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00010;
	bits<1> aq = 0b1;
	bits<1> rl = 0b0;
	bits<5> rs2 = 0b00000;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def LR_W_AQ: Instruction, LR_W_AQ_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicLDW, ReadAtomicLDW]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1);
	let AsmString = "lr.w.aq	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class LR_W_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00010;
	bits<1> aq = 0b0;
	bits<1> rl = 0b1;
	bits<5> rs2 = 0b00000;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def LR_W_RL: Instruction, LR_W_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicLDW, ReadAtomicLDW]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1);
	let AsmString = "lr.w.rl	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class LR_W_AQ_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00010;
	bits<1> aq = 0b1;
	bits<1> rl = 0b1;
	bits<5> rs2 = 0b00000;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def LR_W_AQ_RL: Instruction, LR_W_AQ_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicLDW, ReadAtomicLDW]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1);
	let AsmString = "lr.w.aqrl	$rd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SC_D_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00011;
	bits<1> aq = 0b0;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def SC_D: Instruction, SC_D_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicSTD, ReadAtomicSTD, ReadAtomicSTD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "sc.d	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SC_D_AQ_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00011;
	bits<1> aq = 0b1;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def SC_D_AQ: Instruction, SC_D_AQ_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicSTD, ReadAtomicSTD, ReadAtomicSTD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "sc.d.aq	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SC_D_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00011;
	bits<1> aq = 0b0;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def SC_D_RL: Instruction, SC_D_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicSTD, ReadAtomicSTD, ReadAtomicSTD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "sc.d.rl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SC_D_AQ_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00011;
	bits<1> aq = 0b1;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def SC_D_AQ_RL: Instruction, SC_D_AQ_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicSTD, ReadAtomicSTD, ReadAtomicSTD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "sc.d.aqrl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SC_W_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00011;
	bits<1> aq = 0b0;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def SC_W: Instruction, SC_W_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicSTW, ReadAtomicSTW, ReadAtomicSTW]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "sc.w	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SC_W_AQ_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00011;
	bits<1> aq = 0b1;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def SC_W_AQ: Instruction, SC_W_AQ_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicSTW, ReadAtomicSTW, ReadAtomicSTW]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "sc.w.aq	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SC_W_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00011;
	bits<1> aq = 0b0;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def SC_W_RL: Instruction, SC_W_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicSTW, ReadAtomicSTW, ReadAtomicSTW]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "sc.w.rl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class SC_W_AQ_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00011;
	bits<1> aq = 0b1;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def SC_W_AQ_RL: Instruction, SC_W_AQ_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicSTW, ReadAtomicSTW, ReadAtomicSTW]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "sc.w.aqrl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOSWAP_D_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00001;
	bits<1> aq = 0b0;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOSWAP_D: Instruction, AMOSWAP_D_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoswap.d	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOSWAP_D_AQ_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00001;
	bits<1> aq = 0b1;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOSWAP_D_AQ: Instruction, AMOSWAP_D_AQ_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoswap.d.aq	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOSWAP_D_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00001;
	bits<1> aq = 0b0;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOSWAP_D_RL: Instruction, AMOSWAP_D_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoswap.d.rl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOSWAP_D_AQ_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00001;
	bits<1> aq = 0b1;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b011;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOSWAP_D_AQ_RL: Instruction, AMOSWAP_D_AQ_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoswap.d.aqrl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA, IsRV64];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOSWAP_W_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00001;
	bits<1> aq = 0b0;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOSWAP_W: Instruction, AMOSWAP_W_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoswap.w	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOSWAP_W_AQ_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00001;
	bits<1> aq = 0b1;
	bits<1> rl = 0b0;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOSWAP_W_AQ: Instruction, AMOSWAP_W_AQ_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoswap.w.aq	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOSWAP_W_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00001;
	bits<1> aq = 0b0;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOSWAP_W_RL: Instruction, AMOSWAP_W_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoswap.w.rl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class AMOSWAP_W_AQ_RL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> funct5 = 0b00001;
	bits<1> aq = 0b1;
	bits<1> rl = 0b1;
	bits<5> rs2;
	bits<5> rs1;
	bits<3> funct3 = 0b010;
	bits<5> rd;
	bits<7> opcode = 0b0101111;
	let Inst{31-27} = funct5;
	let Inst{26-26} = aq;
	let Inst{25-25} = rl;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = funct3;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def AMOSWAP_W_AQ_RL: Instruction, AMOSWAP_W_AQ_RL_FM, TSFlagTemplate<InstFormatR>, Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPRMemAtomic:$rs1, GPR:$rs2);
	let AsmString = "amoswap.w.aqrl	$rd, $rs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtA];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}