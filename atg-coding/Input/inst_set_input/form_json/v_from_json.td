class VWADDU_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b110000;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VWADDU_VV: Instruction, VWADDU_VV_FM, TSFlagTemplate<InstFormatR, WidenV>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vwaddu.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VMADC_VI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010001;
	bits<1> vm = 0b1;
	bits<5> vs2;
	bits<5> imm;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = imm;
	let Inst{14-12} = 0b011;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMADC_VI: Instruction, VMADC_VI_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, simm5:$imm);
	let AsmString = "vmadc.vi	$vd, $vs2, $imm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VWADDU_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b110000;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b110;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VWADDU_VX: Instruction, VWADDU_VX_FM, TSFlagTemplate<InstFormatR, WidenV>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vwaddu.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VWADDU_WX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b110100;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b110;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VWADDU_WX: Instruction, VWADDU_WX_FM, TSFlagTemplate<InstFormatR, WidenW>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vwaddu.wx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VWADDU_WV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b110100;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VWADDU_WV: Instruction, VWADDU_WV_FM, TSFlagTemplate<InstFormatR, WidenW>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vwaddu.wv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VSADD_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100001;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSADD_VV: Instruction, VSADD_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vsadd.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSADD_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100001;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSADD_VX: Instruction, VSADD_VX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vsadd.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSADD_VI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100001;
	bits<1> vm;
	bits<5> vs2;
	bits<5> imm;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = imm;
	let Inst{14-12} = 0b011;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSADD_VI: Instruction, VSADD_VI_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, simm5:$imm, VMaskOp:$vm);
	let AsmString = "vsadd.vi	$vd, $vs2, $imm$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSADDU_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100000;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSADDU_VV: Instruction, VSADDU_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vsaddu.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VAADDU_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b001000;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VAADDU_VV: Instruction, VAADDU_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vaaddu.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSADDU_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100000;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSADDU_VX: Instruction, VSADDU_VX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vsaddu.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VAADDU_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b001000;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b110;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VAADDU_VX: Instruction, VAADDU_VX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vaaddu.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSADDU_VI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100000;
	bits<1> vm;
	bits<5> vs2;
	bits<5> imm;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = imm;
	let Inst{14-12} = 0b011;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSADDU_VI: Instruction, VSADDU_VI_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, simm5:$imm, VMaskOp:$vm);
	let AsmString = "vsaddu.vi	$vd, $vs2, $imm$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VADD_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000000;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VADD_VV: Instruction, VADD_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vadd.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VADD_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000000;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VADD_VX: Instruction, VADD_VX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vadd.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VWADD_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b110001;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VWADD_VV: Instruction, VWADD_VV_FM, TSFlagTemplate<InstFormatR, WidenV>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vwadd.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VWADD_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b110001;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b110;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VWADD_VX: Instruction, VWADD_VX_FM, TSFlagTemplate<InstFormatR, WidenV>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vwadd.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VWADD_WV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b110101;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VWADD_WV: Instruction, VWADD_WV_FM, TSFlagTemplate<InstFormatR, WidenW>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vwadd.wv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VWADD_WX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b110101;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b110;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VWADD_WX: Instruction, VWADD_WX_FM, TSFlagTemplate<InstFormatR, WidenW>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vwadd.wx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VADC_VVM_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010000;
	bits<1> vm = 0b0;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VADC_VVM: Instruction, VADC_VVM_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMV0:$v0);
	let AsmString = "vadc.vvm	$vd, $vs2, $vs1, v0";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VADC_VXM_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010000;
	bits<1> vm = 0b0;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VADC_VXM: Instruction, VADC_VXM_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMV0:$v0);
	let AsmString = "vadc.vxm	$vd, $vs2, $rs1, v0";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VADC_VIM_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010000;
	bits<1> vm = 0b0;
	bits<5> vs2;
	bits<5> imm;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = imm;
	let Inst{14-12} = 0b011;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VADC_VIM: Instruction, VADC_VIM_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, simm5:$imm, VMV0:$v0);
	let AsmString = "vadc.vim	$vd, $vs2, $imm, v0";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VAADD_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b001001;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VAADD_VV: Instruction, VAADD_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vaadd.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VAADD_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b001001;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b110;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VAADD_VX: Instruction, VAADD_VX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vaadd.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VADD_VI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000000;
	bits<1> vm;
	bits<5> vs2;
	bits<5> imm;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = imm;
	let Inst{14-12} = 0b011;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VADD_VI: Instruction, VADD_VI_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, simm5:$imm, VMaskOp:$vm);
	let AsmString = "vadd.vi	$vd, $vs2, $imm$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VAND_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b001001;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VAND_VV: Instruction, VAND_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vand.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VAND_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b001001;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VAND_VX: Instruction, VAND_VX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vand.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VREDAND_VS_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000001;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VREDAND_VS: Instruction, VREDAND_VS_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vredand.vs	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMAND_MM_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b011001;
	bits<1> vm = 0b1;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMAND_MM: Instruction, VMAND_MM_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1);
	let AsmString = "vmand.mm	$vd, $vs2, $vs1";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMNAND_MM_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b011101;
	bits<1> vm = 0b1;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMNAND_MM: Instruction, VMNAND_MM_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1);
	let AsmString = "vmnand.mm	$vd, $vs2, $vs1";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMANDNOT_MM_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b011000;
	bits<1> vm = 0b1;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMANDNOT_MM: Instruction, VMANDNOT_MM_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1);
	let AsmString = "vmandnot.mm	$vd, $vs2, $vs1";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VID_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010100;
	bits<1> vm;
	bits<5> vs2 = 0b00000;
	bits<5> vs1 = 0b10001;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VID_V: Instruction, VID_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VMaskOp:$vm);
	let AsmString = "vid.v	$vd$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VAND_VI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b001001;
	bits<1> vm;
	bits<5> vs2;
	bits<5> imm;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = imm;
	let Inst{14-12} = 0b011;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VAND_VI: Instruction, VAND_VI_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, simm5:$imm, VMaskOp:$vm);
	let AsmString = "vand.vi	$vd, $vs2, $imm$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VRSUB_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000011;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VRSUB_VX: Instruction, VRSUB_VX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vrsub.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMSBC_VVM_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010011;
	bits<1> vm = 0b0;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMSBC_VVM: Instruction, VMSBC_VVM_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMV0:$v0);
	let AsmString = "vmsbc.vvm	$vd, $vs2, $vs1, v0";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VSBC_VVM_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010010;
	bits<1> vm = 0b0;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSBC_VVM: Instruction, VSBC_VVM_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMV0:$v0);
	let AsmString = "vsbc.vvm	$vd, $vs2, $vs1, v0";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSBC_VXM_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010010;
	bits<1> vm = 0b0;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSBC_VXM: Instruction, VSBC_VXM_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMV0:$v0);
	let AsmString = "vsbc.vxm	$vd, $vs2, $rs1, v0";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VWMACCUS_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b111110;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b110;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VWMACCUS_VX: Instruction, VWMACCUS_VX_FM, TSFlagTemplate<InstFormatR, WidenV>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vwmaccus.vx	$vd, $rs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VASUB_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b001011;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VASUB_VV: Instruction, VASUB_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vasub.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VASUB_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b001011;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b110;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VASUB_VX: Instruction, VASUB_VX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vasub.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VWSUBU_WV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b110110;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VWSUBU_WV: Instruction, VWSUBU_WV_FM, TSFlagTemplate<InstFormatR, WidenW>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vwsubu.wv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VMSBC_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010011;
	bits<1> vm = 0b1;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMSBC_VX: Instruction, VMSBC_VX_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1);
	let AsmString = "vmsbc.vx	$vd, $vs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VMSBC_VXM_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010011;
	bits<1> vm = 0b0;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMSBC_VXM: Instruction, VMSBC_VXM_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMV0:$v0);
	let AsmString = "vmsbc.vxm	$vd, $vs2, $rs1, v0";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VMSBC_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010011;
	bits<1> vm = 0b1;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMSBC_VV: Instruction, VMSBC_VV_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1);
	let AsmString = "vmsbc.vv	$vd, $vs2, $vs1";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VASUBU_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b001010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VASUBU_VV: Instruction, VASUBU_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vasubu.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VASUBU_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b001010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b110;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VASUBU_VX: Instruction, VASUBU_VX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vasubu.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VRGATHER_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b001100;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VRGATHER_VV: Instruction, VRGATHER_VV_FM, TSFlagTemplate<InstFormatR, Vrgather>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vrgather.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VRGATHEREI16_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b001110;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VRGATHEREI16_VV: Instruction, VRGATHEREI16_VV_FM, TSFlagTemplate<InstFormatR, Vrgather>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vrgatherei16.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VRGATHER_VI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b001100;
	bits<1> vm;
	bits<5> vs2;
	bits<5> imm;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = imm;
	let Inst{14-12} = 0b011;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VRGATHER_VI: Instruction, VRGATHER_VI_FM, TSFlagTemplate<InstFormatR, Vrgather>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, uimm5:$imm, VMaskOp:$vm);
	let AsmString = "vrgather.vi	$vd, $vs2, $imm$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VMSBF_M_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010100;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1 = 0b00001;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMSBF_M: Instruction, VMSBF_M_FM, TSFlagTemplate<InstFormatR, Iota>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VMaskOp:$vm);
	let AsmString = "vmsbf.m	$vd, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VMSIF_M_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010100;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1 = 0b00011;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMSIF_M: Instruction, VMSIF_M_FM, TSFlagTemplate<InstFormatR, Iota>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VMaskOp:$vm);
	let AsmString = "vmsif.m	$vd, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VRGATHER_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b001100;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VRGATHER_VX: Instruction, VRGATHER_VX_FM, TSFlagTemplate<InstFormatR, Vrgather>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vrgather.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VSETVL_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	bits<7> opcode = 0b1010111;
	let Inst{31-31} = 0b1;
	let Inst{30-25} = 0b000000;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = 0b111;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def VSETVL: Instruction, VSETVL_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2);
	let AsmString = "vsetvl	$rd, $rs1, $rs2";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 1;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [VTYPE, VL];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSETVLI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<11> vtypei;
	bits<5> rs1;
	bits<5> rd;
	bits<7> opcode = 0b1010111;
	let Inst{31-31} = 0b0;
	let Inst{30-20} = vtypei;
	let Inst{19-15} = rs1;
	let Inst{14-12} = 0b111;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def VSETVLI: Instruction, VSETVLI_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins GPR:$rs1, VTypeIOp:$vtypei);
	let AsmString = "vsetvli	$rd, $rs1, $vtypei";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 1;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [VTYPE, VL];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFIRST_M_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010000;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1 = 0b10001;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFIRST_M: Instruction, VFIRST_M_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$vd);
	let InOperandList = (ins VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfirst.m	$vd, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSETIVLI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<10> vtypei;
	bits<5> uimm;
	bits<5> rd;
	bits<7> opcode = 0b1010111;
	let Inst{31-31} = 0b1;
	let Inst{30-30} = 0b1;
	let Inst{29-20} = vtypei;
	let Inst{19-15} = uimm;
	let Inst{14-12} = 0b111;
	let Inst{11-7} = rd;
	let Inst{6-0} = opcode;
}
def VSETIVLI: Instruction, VSETIVLI_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$rd);
	let InOperandList = (ins uimm5:$uimm, VTypeIOp:$vtypei);
	let AsmString = "vsetivli	$rd, $uimm, $vtypei";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 1;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [VTYPE, VL];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMSEQ_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b011000;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMSEQ_VV: Instruction, VMSEQ_VV_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vmseq.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMSNE_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b011001;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMSNE_VX: Instruction, VMSNE_VX_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vmsne.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMSEQ_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b011000;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMSEQ_VX: Instruction, VMSEQ_VX_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vmseq.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMSNE_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b011001;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMSNE_VV: Instruction, VMSNE_VV_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vmsne.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMSEQ_VI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b011000;
	bits<1> vm;
	bits<5> vs2;
	bits<5> imm;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = imm;
	let Inst{14-12} = 0b011;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMSEQ_VI: Instruction, VMSEQ_VI_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, simm5:$imm, VMaskOp:$vm);
	let AsmString = "vmseq.vi	$vd, $vs2, $imm$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMSNE_VI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b011001;
	bits<1> vm;
	bits<5> vs2;
	bits<5> imm;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = imm;
	let Inst{14-12} = 0b011;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMSNE_VI: Instruction, VMSNE_VI_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, simm5:$imm, VMaskOp:$vm);
	let AsmString = "vmsne.vi	$vd, $vs2, $imm$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMSLE_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b011101;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMSLE_VV: Instruction, VMSLE_VV_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vmsle.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMSLE_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b011101;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMSLE_VX: Instruction, VMSLE_VX_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vmsle.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMSLEU_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b011100;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMSLEU_VV: Instruction, VMSLEU_VV_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vmsleu.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMSLEU_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b011100;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMSLEU_VX: Instruction, VMSLEU_VX_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vmsleu.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMSLE_VI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b011101;
	bits<1> vm;
	bits<5> vs2;
	bits<5> imm;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = imm;
	let Inst{14-12} = 0b011;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMSLE_VI: Instruction, VMSLE_VI_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, simm5:$imm, VMaskOp:$vm);
	let AsmString = "vmsle.vi	$vd, $vs2, $imm$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMSLEU_VI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b011100;
	bits<1> vm;
	bits<5> vs2;
	bits<5> imm;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = imm;
	let Inst{14-12} = 0b011;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMSLEU_VI: Instruction, VMSLEU_VI_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, simm5:$imm, VMaskOp:$vm);
	let AsmString = "vmsleu.vi	$vd, $vs2, $imm$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMSLT_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b011011;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMSLT_VV: Instruction, VMSLT_VV_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vmslt.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMSLT_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b011011;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMSLT_VX: Instruction, VMSLT_VX_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vmslt.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMSGT_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b011111;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMSGT_VX: Instruction, VMSGT_VX_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vmsgt.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMSLTU_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b011010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMSLTU_VV: Instruction, VMSLTU_VV_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vmsltu.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMSLTU_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b011010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMSLTU_VX: Instruction, VMSLTU_VX_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vmsltu.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMSGTU_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b011110;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMSGTU_VX: Instruction, VMSGTU_VX_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vmsgtu.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMSGTU_VI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b011110;
	bits<1> vm;
	bits<5> vs2;
	bits<5> imm;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = imm;
	let Inst{14-12} = 0b011;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMSGTU_VI: Instruction, VMSGTU_VI_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, simm5:$imm, VMaskOp:$vm);
	let AsmString = "vmsgtu.vi	$vd, $vs2, $imm$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMSGT_VI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b011111;
	bits<1> vm;
	bits<5> vs2;
	bits<5> imm;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = imm;
	let Inst{14-12} = 0b011;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMSGT_VI: Instruction, VMSGT_VI_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, simm5:$imm, VMaskOp:$vm);
	let AsmString = "vmsgt.vi	$vd, $vs2, $imm$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VDIV_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100001;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VDIV_VV: Instruction, VDIV_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vdiv.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VDIV_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100001;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b110;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VDIV_VX: Instruction, VDIV_VX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vdiv.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VDIVU_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100000;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VDIVU_VV: Instruction, VDIVU_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vdivu.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VDIVU_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100000;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b110;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VDIVU_VX: Instruction, VDIVU_VX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vdivu.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFADD_VF_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000000;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFADD_VF: Instruction, VFADD_VF_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, FPR32:$rs1, VMaskOp:$vm);
	let AsmString = "vfadd.vf	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFWADD_VF_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b110000;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFWADD_VF: Instruction, VFWADD_VF_FM, TSFlagTemplate<InstFormatR, WidenV>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, FPR32:$rs1, VMaskOp:$vm);
	let AsmString = "vfwadd.vf	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VFWADD_WF_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b110100;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFWADD_WF: Instruction, VFWADD_WF_FM, TSFlagTemplate<InstFormatR, WidenW>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, FPR32:$rs1, VMaskOp:$vm);
	let AsmString = "vfwadd.wf	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VFADD_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000000;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFADD_VV: Instruction, VFADD_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vfadd.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFWADD_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b110000;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFWADD_VV: Instruction, VFWADD_VV_FM, TSFlagTemplate<InstFormatR, WidenV>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vfwadd.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VFWADD_WV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b110100;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFWADD_WV: Instruction, VFWADD_WV_FM, TSFlagTemplate<InstFormatR, WidenW>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vfwadd.wv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VMFEQ_VF_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b011000;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMFEQ_VF: Instruction, VMFEQ_VF_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, FPR32:$rs1, VMaskOp:$vm);
	let AsmString = "vmfeq.vf	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMFEQ_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b011000;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMFEQ_VV: Instruction, VMFEQ_VV_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vmfeq.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFCLASS_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010011;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1 = 0b10000;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFCLASS_V: Instruction, VFCLASS_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfclass.v	$vd, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMFLE_VF_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b011001;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMFLE_VF: Instruction, VMFLE_VF_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, FPR32:$rs1, VMaskOp:$vm);
	let AsmString = "vmfle.vf	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMFGE_VF_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b011111;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMFGE_VF: Instruction, VMFGE_VF_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, FPR32:$rs1, VMaskOp:$vm);
	let AsmString = "vmfge.vf	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMFLE_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b011001;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMFLE_VV: Instruction, VMFLE_VV_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vmfle.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMFLT_VF_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b011011;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMFLT_VF: Instruction, VMFLT_VF_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, FPR32:$rs1, VMaskOp:$vm);
	let AsmString = "vmflt.vf	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMFGT_VF_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b011101;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMFGT_VF: Instruction, VMFGT_VF_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, FPR32:$rs1, VMaskOp:$vm);
	let AsmString = "vmfgt.vf	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMFLT_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b011011;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMFLT_VV: Instruction, VMFLT_VV_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vmflt.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMFNE_VF_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b011100;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMFNE_VF: Instruction, VMFNE_VF_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, FPR32:$rs1, VMaskOp:$vm);
	let AsmString = "vmfne.vf	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMFNE_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b011100;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMFNE_VV: Instruction, VMFNE_VV_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vmfne.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFDIV_VF_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100000;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFDIV_VF: Instruction, VFDIV_VF_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, FPR32:$rs1, VMaskOp:$vm);
	let AsmString = "vfdiv.vf	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFRDIV_VF_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100001;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFRDIV_VF: Instruction, VFRDIV_VF_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, FPR32:$rs1, VMaskOp:$vm);
	let AsmString = "vfrdiv.vf	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFDIV_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100000;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFDIV_VV: Instruction, VFDIV_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vfdiv.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFWCVT_F_X_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1 = 0b01011;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFWCVT_F_X_V: Instruction, VFWCVT_F_X_V_FM, TSFlagTemplate<InstFormatR, WidenCvt>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfwcvt.f.x.v	$vd, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VFCVT_RTZ_XU_F_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1 = 0b00110;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFCVT_RTZ_XU_F_V: Instruction, VFCVT_RTZ_XU_F_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfcvt.rtz.xu.f.v	$vd, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFCVT_RTZ_X_F_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1 = 0b00111;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFCVT_RTZ_X_F_V: Instruction, VFCVT_RTZ_X_F_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfcvt.rtz.x.f.v	$vd, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFCVT_F_X_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1 = 0b00011;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFCVT_F_X_V: Instruction, VFCVT_F_X_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfcvt.f.x.v	$vd, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFWCVT_RTZ_XU_F_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1 = 0b01110;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFWCVT_RTZ_XU_F_V: Instruction, VFWCVT_RTZ_XU_F_V_FM, TSFlagTemplate<InstFormatR, WidenCvt>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfwcvt.rtz.xu.f.v	$vd, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VFWCVT_RTZ_X_F_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1 = 0b01111;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFWCVT_RTZ_X_F_V: Instruction, VFWCVT_RTZ_X_F_V_FM, TSFlagTemplate<InstFormatR, WidenCvt>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfwcvt.rtz.x.f.v	$vd, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VFWCVT_F_F_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1 = 0b01100;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFWCVT_F_F_V: Instruction, VFWCVT_F_F_V_FM, TSFlagTemplate<InstFormatR, WidenCvt>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfwcvt.f.f.v	$vd, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VFNCVT_RTZ_XU_F_W_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1 = 0b10110;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFNCVT_RTZ_XU_F_W: Instruction, VFNCVT_RTZ_XU_F_W_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfncvt.rtz.xu.f.w	$vd, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VFNCVT_RTZ_X_F_W_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1 = 0b10111;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFNCVT_RTZ_X_F_W: Instruction, VFNCVT_RTZ_X_F_W_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfncvt.rtz.x.f.w	$vd, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VFNCVT_F_X_W_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1 = 0b10011;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFNCVT_F_X_W: Instruction, VFNCVT_F_X_W_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfncvt.f.x.w	$vd, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VFNCVT_ROD_F_F_W_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1 = 0b10101;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFNCVT_ROD_F_F_W: Instruction, VFNCVT_ROD_F_F_W_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfncvt.rod.f.f.w	$vd, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VFWCVT_F_XU_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1 = 0b01010;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFWCVT_F_XU_V: Instruction, VFWCVT_F_XU_V_FM, TSFlagTemplate<InstFormatR, WidenCvt>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfwcvt.f.xu.v	$vd, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VFCVT_F_XU_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1 = 0b00010;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFCVT_F_XU_V: Instruction, VFCVT_F_XU_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfcvt.f.xu.v	$vd, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFNCVT_F_XU_W_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1 = 0b10010;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFNCVT_F_XU_W: Instruction, VFNCVT_F_XU_W_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfncvt.f.xu.w	$vd, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VFNCVT_F_F_W_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1 = 0b10100;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFNCVT_F_F_W: Instruction, VFNCVT_F_F_W_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfncvt.f.f.w	$vd, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VFMACC_VF_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101100;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFMACC_VF: Instruction, VFMACC_VF_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins FPR32:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfmacc.vf	$vd, $rs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFNMACC_VF_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101101;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFNMACC_VF: Instruction, VFNMACC_VF_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins FPR32:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfnmacc.vf	$vd, $rs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFMADD_VF_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101000;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFMADD_VF: Instruction, VFMADD_VF_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins FPR32:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfmadd.vf	$vd, $rs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFNMADD_VF_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101001;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFNMADD_VF: Instruction, VFNMADD_VF_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins FPR32:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfnmadd.vf	$vd, $rs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFWMACC_VF_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b111100;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFWMACC_VF: Instruction, VFWMACC_VF_FM, TSFlagTemplate<InstFormatR, WidenV>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins FPR32:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfwmacc.vf	$vd, $rs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VFWNMACC_VF_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b111101;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFWNMACC_VF: Instruction, VFWNMACC_VF_FM, TSFlagTemplate<InstFormatR, WidenV>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins FPR32:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfwnmacc.vf	$vd, $rs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VFMACC_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101100;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFMACC_VV: Instruction, VFMACC_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfmacc.vv	$vd, $vs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFNMACC_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101101;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFNMACC_VV: Instruction, VFNMACC_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfnmacc.vv	$vd, $vs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFMADD_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101000;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFMADD_VV: Instruction, VFMADD_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfmadd.vv	$vd, $vs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFNMADD_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101001;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFNMADD_VV: Instruction, VFNMADD_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfnmadd.vv	$vd, $vs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFWMACC_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b111100;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFWMACC_VV: Instruction, VFWMACC_VV_FM, TSFlagTemplate<InstFormatR, WidenV>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfwmacc.vv	$vd, $vs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VFWNMACC_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b111101;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFWNMACC_VV: Instruction, VFWNMACC_VV_FM, TSFlagTemplate<InstFormatR, WidenV>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfwnmacc.vv	$vd, $vs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VFMAX_VF_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000110;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFMAX_VF: Instruction, VFMAX_VF_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, FPR32:$rs1, VMaskOp:$vm);
	let AsmString = "vfmax.vf	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFMAX_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000110;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFMAX_VV: Instruction, VFMAX_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vfmax.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFMIN_VF_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000100;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFMIN_VF: Instruction, VFMIN_VF_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, FPR32:$rs1, VMaskOp:$vm);
	let AsmString = "vfmin.vf	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFMIN_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000100;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFMIN_VV: Instruction, VFMIN_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vfmin.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFMSAC_VF_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101110;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFMSAC_VF: Instruction, VFMSAC_VF_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins FPR32:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfmsac.vf	$vd, $rs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFNMSAC_VF_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101111;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFNMSAC_VF: Instruction, VFNMSAC_VF_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins FPR32:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfnmsac.vf	$vd, $rs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFMSUB_VF_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFMSUB_VF: Instruction, VFMSUB_VF_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins FPR32:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfmsub.vf	$vd, $rs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFNMSUB_VF_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101011;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFNMSUB_VF: Instruction, VFNMSUB_VF_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins FPR32:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfnmsub.vf	$vd, $rs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFWMSAC_VF_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b111110;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFWMSAC_VF: Instruction, VFWMSAC_VF_FM, TSFlagTemplate<InstFormatR, WidenV>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins FPR32:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfwmsac.vf	$vd, $rs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VFWNMSAC_VF_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b111111;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFWNMSAC_VF: Instruction, VFWNMSAC_VF_FM, TSFlagTemplate<InstFormatR, WidenV>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins FPR32:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfwnmsac.vf	$vd, $rs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VFMSAC_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101110;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFMSAC_VV: Instruction, VFMSAC_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfmsac.vv	$vd, $vs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFNMSAC_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101111;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFNMSAC_VV: Instruction, VFNMSAC_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfnmsac.vv	$vd, $vs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFMSUB_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFMSUB_VV: Instruction, VFMSUB_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfmsub.vv	$vd, $vs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFNMSUB_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101011;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFNMSUB_VV: Instruction, VFNMSUB_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfnmsub.vv	$vd, $vs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFWMSAC_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b111110;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFWMSAC_VV: Instruction, VFWMSAC_VV_FM, TSFlagTemplate<InstFormatR, WidenV>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfwmsac.vv	$vd, $vs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VFWNMSAC_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b111111;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFWNMSAC_VV: Instruction, VFWNMSAC_VV_FM, TSFlagTemplate<InstFormatR, WidenV>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfwnmsac.vv	$vd, $vs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VFMUL_VF_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100100;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFMUL_VF: Instruction, VFMUL_VF_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, FPR32:$rs1, VMaskOp:$vm);
	let AsmString = "vfmul.vf	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFWMUL_VF_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b111000;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFWMUL_VF: Instruction, VFWMUL_VF_FM, TSFlagTemplate<InstFormatR, WidenV>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, FPR32:$rs1, VMaskOp:$vm);
	let AsmString = "vfwmul.vf	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VFMUL_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100100;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFMUL_VV: Instruction, VFMUL_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vfmul.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFWMUL_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b111000;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFWMUL_VV: Instruction, VFWMUL_VV_FM, TSFlagTemplate<InstFormatR, WidenV>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vfwmul.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VFREC7_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010011;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1 = 0b00101;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFREC7_V: Instruction, VFREC7_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfrec7.v	$vd, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFRSQRT7_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010011;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1 = 0b00100;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFRSQRT7_V: Instruction, VFRSQRT7_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfrsqrt7.v	$vd, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFSQRT_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010011;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1 = 0b00000;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFSQRT_V: Instruction, VFSQRT_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfsqrt.v	$vd, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFSUB_VF_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFSUB_VF: Instruction, VFSUB_VF_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, FPR32:$rs1, VMaskOp:$vm);
	let AsmString = "vfsub.vf	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFWSUB_VF_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b110010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFWSUB_VF: Instruction, VFWSUB_VF_FM, TSFlagTemplate<InstFormatR, WidenV>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, FPR32:$rs1, VMaskOp:$vm);
	let AsmString = "vfwsub.vf	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VFWSUB_WF_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b110110;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFWSUB_WF: Instruction, VFWSUB_WF_FM, TSFlagTemplate<InstFormatR, WidenW>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, FPR32:$rs1, VMaskOp:$vm);
	let AsmString = "vfwsub.wf	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VFSUB_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFSUB_VV: Instruction, VFSUB_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vfsub.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFRSUB_VF_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100111;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFRSUB_VF: Instruction, VFRSUB_VF_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, FPR32:$rs1, VMaskOp:$vm);
	let AsmString = "vfrsub.vf	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFWSUB_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b110010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFWSUB_VV: Instruction, VFWSUB_VV_FM, TSFlagTemplate<InstFormatR, WidenV>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vfwsub.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VFWSUB_WV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b110110;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFWSUB_WV: Instruction, VFWSUB_WV_FM, TSFlagTemplate<InstFormatR, WidenW>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vfwsub.wv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VFWCVT_X_F_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1 = 0b01001;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFWCVT_X_F_V: Instruction, VFWCVT_X_F_V_FM, TSFlagTemplate<InstFormatR, WidenCvt>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfwcvt.x.f.v	$vd, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VFCVT_X_F_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1 = 0b00001;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFCVT_X_F_V: Instruction, VFCVT_X_F_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfcvt.x.f.v	$vd, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFNCVT_X_F_W_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1 = 0b10001;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFNCVT_X_F_W: Instruction, VFNCVT_X_F_W_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfncvt.x.f.w	$vd, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VFWCVT_XU_F_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1 = 0b01000;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFWCVT_XU_F_V: Instruction, VFWCVT_XU_F_V_FM, TSFlagTemplate<InstFormatR, WidenCvt>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfwcvt.xu.f.v	$vd, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VFCVT_XU_F_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1 = 0b00000;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFCVT_XU_F_V: Instruction, VFCVT_XU_F_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfcvt.xu.f.v	$vd, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFNCVT_XU_F_W_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1 = 0b10000;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFNCVT_XU_F_W: Instruction, VFNCVT_XU_F_W_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VMaskOp:$vm);
	let AsmString = "vfncvt.xu.f.w	$vd, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VWSUBU_WX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b110110;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b110;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VWSUBU_WX: Instruction, VWSUBU_WX_FM, TSFlagTemplate<InstFormatR, WidenW>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vwsubu.wx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VWSUBU_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b110010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VWSUBU_VV: Instruction, VWSUBU_VV_FM, TSFlagTemplate<InstFormatR, WidenV>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vwsubu.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VWSUBU_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b110010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b110;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VWSUBU_VX: Instruction, VWSUBU_VX_FM, TSFlagTemplate<InstFormatR, WidenV>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vwsubu.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VLE8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b000;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b00000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLE8_V: Instruction, VLE8_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vle8.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLE1_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b000;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b01011;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLE1_V: Instruction, VLE1_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "vle1.v	$vd, (${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSE8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b000;
	bits<2> MOPLDStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSE8_V: Instruction, VLSE8_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vlse8.v	$vd, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLUXEI8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b000;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLUXEI8_V: Instruction, VLUXEI8_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vluxei8.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLOXEI8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b000;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLOXEI8_V: Instruction, VLOXEI8_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vloxei8.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLE8FF_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b000;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b10000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLE8FF_V: Instruction, VLE8FF_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vle8ff.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG2E8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b001;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b00000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG2E8_V: Instruction, VLSEG2E8_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg2e8.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG3E8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b010;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b00000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG3E8_V: Instruction, VLSEG3E8_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg3e8.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG4E8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b011;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b00000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG4E8_V: Instruction, VLSEG4E8_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg4e8.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG5E8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b100;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b00000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG5E8_V: Instruction, VLSEG5E8_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg5e8.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG6E8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b101;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b00000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG6E8_V: Instruction, VLSEG6E8_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg6e8.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG7E8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b110;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b00000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG7E8_V: Instruction, VLSEG7E8_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg7e8.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG8E8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b111;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b00000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG8E8_V: Instruction, VLSEG8E8_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg8e8.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG2E8FF_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b001;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b10000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG2E8FF_V: Instruction, VLSEG2E8FF_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg2e8ff.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG3E8FF_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b010;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b10000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG3E8FF_V: Instruction, VLSEG3E8FF_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg3e8ff.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG4E8FF_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b011;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b10000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG4E8FF_V: Instruction, VLSEG4E8FF_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg4e8ff.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG5E8FF_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b100;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b10000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG5E8FF_V: Instruction, VLSEG5E8FF_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg5e8ff.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG6E8FF_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b101;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b10000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG6E8FF_V: Instruction, VLSEG6E8FF_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg6e8ff.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG7E8FF_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b110;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b10000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG7E8FF_V: Instruction, VLSEG7E8FF_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg7e8ff.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG8E8FF_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b111;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b10000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG8E8FF_V: Instruction, VLSEG8E8FF_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg8e8ff.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSSEG2E8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b001;
	bits<2> MOPLDStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSSEG2E8_V: Instruction, VLSSEG2E8_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vlsseg2e8.v	$vd, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSSEG3E8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b010;
	bits<2> MOPLDStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSSEG3E8_V: Instruction, VLSSEG3E8_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vlsseg3e8.v	$vd, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSSEG4E8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b011;
	bits<2> MOPLDStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSSEG4E8_V: Instruction, VLSSEG4E8_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vlsseg4e8.v	$vd, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSSEG5E8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b100;
	bits<2> MOPLDStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSSEG5E8_V: Instruction, VLSSEG5E8_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vlsseg5e8.v	$vd, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSSEG6E8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b101;
	bits<2> MOPLDStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSSEG6E8_V: Instruction, VLSSEG6E8_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vlsseg6e8.v	$vd, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSSEG7E8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b110;
	bits<2> MOPLDStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSSEG7E8_V: Instruction, VLSSEG7E8_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vlsseg7e8.v	$vd, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSSEG8E8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b111;
	bits<2> MOPLDStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSSEG8E8_V: Instruction, VLSSEG8E8_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vlsseg8e8.v	$vd, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLUXSEG2EI8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b001;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLUXSEG2EI8_V: Instruction, VLUXSEG2EI8_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vluxseg2ei8.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLUXSEG3EI8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b010;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLUXSEG3EI8_V: Instruction, VLUXSEG3EI8_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vluxseg3ei8.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLUXSEG4EI8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b011;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLUXSEG4EI8_V: Instruction, VLUXSEG4EI8_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vluxseg4ei8.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLUXSEG5EI8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b100;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLUXSEG5EI8_V: Instruction, VLUXSEG5EI8_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vluxseg5ei8.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLUXSEG6EI8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b101;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLUXSEG6EI8_V: Instruction, VLUXSEG6EI8_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vluxseg6ei8.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLUXSEG7EI8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b110;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLUXSEG7EI8_V: Instruction, VLUXSEG7EI8_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vluxseg7ei8.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLUXSEG8EI8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b111;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLUXSEG8EI8_V: Instruction, VLUXSEG8EI8_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vluxseg8ei8.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLOXSEG2EI8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b001;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLOXSEG2EI8_V: Instruction, VLOXSEG2EI8_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vloxseg2ei8.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLOXSEG3EI8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b010;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLOXSEG3EI8_V: Instruction, VLOXSEG3EI8_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vloxseg3ei8.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLOXSEG4EI8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b011;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLOXSEG4EI8_V: Instruction, VLOXSEG4EI8_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vloxseg4ei8.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLOXSEG5EI8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b100;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLOXSEG5EI8_V: Instruction, VLOXSEG5EI8_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vloxseg5ei8.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLOXSEG6EI8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b101;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLOXSEG6EI8_V: Instruction, VLOXSEG6EI8_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vloxseg6ei8.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLOXSEG7EI8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b110;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLOXSEG7EI8_V: Instruction, VLOXSEG7EI8_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vloxseg7ei8.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLOXSEG8EI8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b111;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLOXSEG8EI8_V: Instruction, VLOXSEG8EI8_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vloxseg8ei8.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VL1RE8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b000;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm = 0b1;
	bits<5> lumop = 0b01000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VL1RE8_V: Instruction, VL1RE8_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "vl1re8.v	$vd, (${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VL2RE8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b001;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm = 0b1;
	bits<5> lumop = 0b01000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VL2RE8_V: Instruction, VL2RE8_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "vl2re8.v	$vd, (${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VL4RE8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b011;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm = 0b1;
	bits<5> lumop = 0b01000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VL4RE8_V: Instruction, VL4RE8_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "vl4re8.v	$vd, (${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VL8RE8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b111;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm = 0b1;
	bits<5> lumop = 0b01000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VL8RE8_V: Instruction, VL8RE8_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "vl8re8.v	$vd, (${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLE64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b000;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b00000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLE64_V: Instruction, VLE64_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vle64.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSE64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b000;
	bits<2> MOPLDStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSE64_V: Instruction, VLSE64_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vlse64.v	$vd, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLUXEI64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b000;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLUXEI64_V: Instruction, VLUXEI64_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vluxei64.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLOXEI64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b000;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLOXEI64_V: Instruction, VLOXEI64_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vloxei64.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLE64FF_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b000;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b10000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLE64FF_V: Instruction, VLE64FF_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vle64ff.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG2E64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b001;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b00000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG2E64_V: Instruction, VLSEG2E64_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg2e64.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG3E64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b010;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b00000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG3E64_V: Instruction, VLSEG3E64_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg3e64.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG4E64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b011;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b00000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG4E64_V: Instruction, VLSEG4E64_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg4e64.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG5E64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b100;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b00000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG5E64_V: Instruction, VLSEG5E64_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg5e64.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG6E64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b101;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b00000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG6E64_V: Instruction, VLSEG6E64_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg6e64.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG7E64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b110;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b00000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG7E64_V: Instruction, VLSEG7E64_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg7e64.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG8E64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b111;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b00000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG8E64_V: Instruction, VLSEG8E64_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg8e64.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG2E64FF_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b001;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b10000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG2E64FF_V: Instruction, VLSEG2E64FF_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg2e64ff.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG3E64FF_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b010;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b10000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG3E64FF_V: Instruction, VLSEG3E64FF_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg3e64ff.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG4E64FF_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b011;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b10000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG4E64FF_V: Instruction, VLSEG4E64FF_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg4e64ff.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG5E64FF_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b100;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b10000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG5E64FF_V: Instruction, VLSEG5E64FF_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg5e64ff.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG6E64FF_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b101;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b10000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG6E64FF_V: Instruction, VLSEG6E64FF_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg6e64ff.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG7E64FF_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b110;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b10000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG7E64FF_V: Instruction, VLSEG7E64FF_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg7e64ff.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG8E64FF_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b111;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b10000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG8E64FF_V: Instruction, VLSEG8E64FF_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg8e64ff.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSSEG2E64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b001;
	bits<2> MOPLDStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSSEG2E64_V: Instruction, VLSSEG2E64_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vlsseg2e64.v	$vd, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSSEG3E64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b010;
	bits<2> MOPLDStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSSEG3E64_V: Instruction, VLSSEG3E64_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vlsseg3e64.v	$vd, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSSEG4E64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b011;
	bits<2> MOPLDStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSSEG4E64_V: Instruction, VLSSEG4E64_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vlsseg4e64.v	$vd, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSSEG5E64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b100;
	bits<2> MOPLDStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSSEG5E64_V: Instruction, VLSSEG5E64_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vlsseg5e64.v	$vd, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSSEG6E64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b101;
	bits<2> MOPLDStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSSEG6E64_V: Instruction, VLSSEG6E64_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vlsseg6e64.v	$vd, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSSEG7E64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b110;
	bits<2> MOPLDStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSSEG7E64_V: Instruction, VLSSEG7E64_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vlsseg7e64.v	$vd, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSSEG8E64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b111;
	bits<2> MOPLDStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSSEG8E64_V: Instruction, VLSSEG8E64_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vlsseg8e64.v	$vd, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLUXSEG2EI64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b001;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLUXSEG2EI64_V: Instruction, VLUXSEG2EI64_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vluxseg2ei64.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLUXSEG3EI64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b010;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLUXSEG3EI64_V: Instruction, VLUXSEG3EI64_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vluxseg3ei64.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLUXSEG4EI64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b011;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLUXSEG4EI64_V: Instruction, VLUXSEG4EI64_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vluxseg4ei64.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLUXSEG5EI64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b100;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLUXSEG5EI64_V: Instruction, VLUXSEG5EI64_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vluxseg5ei64.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLUXSEG6EI64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b101;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLUXSEG6EI64_V: Instruction, VLUXSEG6EI64_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vluxseg6ei64.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLUXSEG7EI64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b110;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLUXSEG7EI64_V: Instruction, VLUXSEG7EI64_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vluxseg7ei64.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLUXSEG8EI64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b111;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLUXSEG8EI64_V: Instruction, VLUXSEG8EI64_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vluxseg8ei64.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLOXSEG2EI64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b001;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLOXSEG2EI64_V: Instruction, VLOXSEG2EI64_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vloxseg2ei64.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLOXSEG3EI64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b010;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLOXSEG3EI64_V: Instruction, VLOXSEG3EI64_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vloxseg3ei64.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLOXSEG4EI64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b011;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLOXSEG4EI64_V: Instruction, VLOXSEG4EI64_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vloxseg4ei64.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLOXSEG5EI64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b100;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLOXSEG5EI64_V: Instruction, VLOXSEG5EI64_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vloxseg5ei64.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLOXSEG6EI64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b101;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLOXSEG6EI64_V: Instruction, VLOXSEG6EI64_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vloxseg6ei64.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLOXSEG7EI64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b110;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLOXSEG7EI64_V: Instruction, VLOXSEG7EI64_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vloxseg7ei64.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLOXSEG8EI64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b111;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLOXSEG8EI64_V: Instruction, VLOXSEG8EI64_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vloxseg8ei64.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VL1RE64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b000;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm = 0b1;
	bits<5> lumop = 0b01000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VL1RE64_V: Instruction, VL1RE64_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "vl1re64.v	$vd, (${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VL2RE64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b001;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm = 0b1;
	bits<5> lumop = 0b01000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VL2RE64_V: Instruction, VL2RE64_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "vl2re64.v	$vd, (${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VL4RE64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b011;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm = 0b1;
	bits<5> lumop = 0b01000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VL4RE64_V: Instruction, VL4RE64_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "vl4re64.v	$vd, (${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VL8RE64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b111;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm = 0b1;
	bits<5> lumop = 0b01000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VL8RE64_V: Instruction, VL8RE64_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "vl8re64.v	$vd, (${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLE16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b000;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b00000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLE16_V: Instruction, VLE16_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vle16.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSE16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b000;
	bits<2> MOPLDStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSE16_V: Instruction, VLSE16_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vlse16.v	$vd, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLUXEI16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b000;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLUXEI16_V: Instruction, VLUXEI16_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vluxei16.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLOXEI16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b000;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLOXEI16_V: Instruction, VLOXEI16_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vloxei16.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLE16FF_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b000;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b10000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLE16FF_V: Instruction, VLE16FF_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vle16ff.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG2E16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b001;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b00000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG2E16_V: Instruction, VLSEG2E16_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg2e16.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG3E16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b010;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b00000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG3E16_V: Instruction, VLSEG3E16_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg3e16.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG4E16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b011;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b00000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG4E16_V: Instruction, VLSEG4E16_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg4e16.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG5E16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b100;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b00000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG5E16_V: Instruction, VLSEG5E16_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg5e16.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG6E16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b101;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b00000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG6E16_V: Instruction, VLSEG6E16_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg6e16.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG7E16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b110;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b00000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG7E16_V: Instruction, VLSEG7E16_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg7e16.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG8E16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b111;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b00000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG8E16_V: Instruction, VLSEG8E16_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg8e16.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG2E16FF_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b001;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b10000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG2E16FF_V: Instruction, VLSEG2E16FF_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg2e16ff.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG3E16FF_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b010;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b10000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG3E16FF_V: Instruction, VLSEG3E16FF_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg3e16ff.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG4E16FF_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b011;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b10000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG4E16FF_V: Instruction, VLSEG4E16FF_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg4e16ff.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG5E16FF_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b100;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b10000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG5E16FF_V: Instruction, VLSEG5E16FF_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg5e16ff.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG6E16FF_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b101;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b10000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG6E16FF_V: Instruction, VLSEG6E16FF_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg6e16ff.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG7E16FF_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b110;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b10000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG7E16FF_V: Instruction, VLSEG7E16FF_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg7e16ff.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG8E16FF_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b111;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b10000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG8E16FF_V: Instruction, VLSEG8E16FF_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg8e16ff.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSSEG2E16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b001;
	bits<2> MOPLDStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSSEG2E16_V: Instruction, VLSSEG2E16_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vlsseg2e16.v	$vd, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSSEG3E16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b010;
	bits<2> MOPLDStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSSEG3E16_V: Instruction, VLSSEG3E16_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vlsseg3e16.v	$vd, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSSEG4E16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b011;
	bits<2> MOPLDStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSSEG4E16_V: Instruction, VLSSEG4E16_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vlsseg4e16.v	$vd, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSSEG5E16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b100;
	bits<2> MOPLDStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSSEG5E16_V: Instruction, VLSSEG5E16_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vlsseg5e16.v	$vd, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSSEG6E16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b101;
	bits<2> MOPLDStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSSEG6E16_V: Instruction, VLSSEG6E16_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vlsseg6e16.v	$vd, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSSEG7E16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b110;
	bits<2> MOPLDStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSSEG7E16_V: Instruction, VLSSEG7E16_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vlsseg7e16.v	$vd, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSSEG8E16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b111;
	bits<2> MOPLDStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSSEG8E16_V: Instruction, VLSSEG8E16_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vlsseg8e16.v	$vd, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLUXSEG2EI16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b001;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLUXSEG2EI16_V: Instruction, VLUXSEG2EI16_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vluxseg2ei16.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLUXSEG3EI16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b010;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLUXSEG3EI16_V: Instruction, VLUXSEG3EI16_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vluxseg3ei16.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLUXSEG4EI16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b011;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLUXSEG4EI16_V: Instruction, VLUXSEG4EI16_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vluxseg4ei16.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLUXSEG5EI16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b100;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLUXSEG5EI16_V: Instruction, VLUXSEG5EI16_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vluxseg5ei16.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLUXSEG6EI16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b101;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLUXSEG6EI16_V: Instruction, VLUXSEG6EI16_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vluxseg6ei16.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLUXSEG7EI16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b110;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLUXSEG7EI16_V: Instruction, VLUXSEG7EI16_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vluxseg7ei16.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLUXSEG8EI16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b111;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLUXSEG8EI16_V: Instruction, VLUXSEG8EI16_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vluxseg8ei16.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLOXSEG2EI16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b001;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLOXSEG2EI16_V: Instruction, VLOXSEG2EI16_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vloxseg2ei16.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLOXSEG3EI16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b010;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLOXSEG3EI16_V: Instruction, VLOXSEG3EI16_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vloxseg3ei16.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLOXSEG4EI16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b011;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLOXSEG4EI16_V: Instruction, VLOXSEG4EI16_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vloxseg4ei16.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLOXSEG5EI16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b100;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLOXSEG5EI16_V: Instruction, VLOXSEG5EI16_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vloxseg5ei16.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLOXSEG6EI16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b101;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLOXSEG6EI16_V: Instruction, VLOXSEG6EI16_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vloxseg6ei16.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLOXSEG7EI16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b110;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLOXSEG7EI16_V: Instruction, VLOXSEG7EI16_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vloxseg7ei16.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLOXSEG8EI16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b111;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLOXSEG8EI16_V: Instruction, VLOXSEG8EI16_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vloxseg8ei16.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VL1RE16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b000;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm = 0b1;
	bits<5> lumop = 0b01000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VL1RE16_V: Instruction, VL1RE16_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "vl1re16.v	$vd, (${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VL2RE16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b001;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm = 0b1;
	bits<5> lumop = 0b01000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VL2RE16_V: Instruction, VL2RE16_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "vl2re16.v	$vd, (${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VL4RE16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b011;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm = 0b1;
	bits<5> lumop = 0b01000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VL4RE16_V: Instruction, VL4RE16_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "vl4re16.v	$vd, (${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VL8RE16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b111;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm = 0b1;
	bits<5> lumop = 0b01000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VL8RE16_V: Instruction, VL8RE16_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "vl8re16.v	$vd, (${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLE32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b000;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b00000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLE32_V: Instruction, VLE32_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vle32.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSE32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b000;
	bits<2> MOPLDStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSE32_V: Instruction, VLSE32_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vlse32.v	$vd, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLUXEI32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b000;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLUXEI32_V: Instruction, VLUXEI32_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vluxei32.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLOXEI32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b000;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLOXEI32_V: Instruction, VLOXEI32_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vloxei32.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLE32FF_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b000;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b10000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLE32FF_V: Instruction, VLE32FF_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vle32ff.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG2E32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b001;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b00000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG2E32_V: Instruction, VLSEG2E32_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg2e32.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG3E32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b010;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b00000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG3E32_V: Instruction, VLSEG3E32_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg3e32.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG4E32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b011;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b00000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG4E32_V: Instruction, VLSEG4E32_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg4e32.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG5E32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b100;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b00000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG5E32_V: Instruction, VLSEG5E32_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg5e32.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG6E32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b101;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b00000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG6E32_V: Instruction, VLSEG6E32_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg6e32.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG7E32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b110;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b00000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG7E32_V: Instruction, VLSEG7E32_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg7e32.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG8E32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b111;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b00000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG8E32_V: Instruction, VLSEG8E32_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg8e32.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG2E32FF_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b001;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b10000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG2E32FF_V: Instruction, VLSEG2E32FF_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg2e32ff.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG3E32FF_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b010;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b10000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG3E32FF_V: Instruction, VLSEG3E32FF_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg3e32ff.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG4E32FF_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b011;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b10000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG4E32FF_V: Instruction, VLSEG4E32FF_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg4e32ff.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG5E32FF_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b100;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b10000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG5E32FF_V: Instruction, VLSEG5E32FF_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg5e32ff.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG6E32FF_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b101;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b10000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG6E32FF_V: Instruction, VLSEG6E32FF_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg6e32ff.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG7E32FF_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b110;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b10000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG7E32FF_V: Instruction, VLSEG7E32FF_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg7e32ff.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSEG8E32FF_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b111;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm;
	bits<5> lumop = 0b10000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSEG8E32FF_V: Instruction, VLSEG8E32FF_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vlseg8e32ff.v	$vd, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSSEG2E32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b001;
	bits<2> MOPLDStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSSEG2E32_V: Instruction, VLSSEG2E32_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vlsseg2e32.v	$vd, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSSEG3E32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b010;
	bits<2> MOPLDStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSSEG3E32_V: Instruction, VLSSEG3E32_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vlsseg3e32.v	$vd, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSSEG4E32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b011;
	bits<2> MOPLDStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSSEG4E32_V: Instruction, VLSSEG4E32_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vlsseg4e32.v	$vd, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSSEG5E32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b100;
	bits<2> MOPLDStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSSEG5E32_V: Instruction, VLSSEG5E32_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vlsseg5e32.v	$vd, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSSEG6E32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b101;
	bits<2> MOPLDStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSSEG6E32_V: Instruction, VLSSEG6E32_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vlsseg6e32.v	$vd, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSSEG7E32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b110;
	bits<2> MOPLDStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSSEG7E32_V: Instruction, VLSSEG7E32_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vlsseg7e32.v	$vd, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLSSEG8E32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b111;
	bits<2> MOPLDStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLSSEG8E32_V: Instruction, VLSSEG8E32_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vlsseg8e32.v	$vd, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLUXSEG2EI32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b001;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLUXSEG2EI32_V: Instruction, VLUXSEG2EI32_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vluxseg2ei32.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLUXSEG3EI32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b010;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLUXSEG3EI32_V: Instruction, VLUXSEG3EI32_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vluxseg3ei32.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLUXSEG4EI32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b011;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLUXSEG4EI32_V: Instruction, VLUXSEG4EI32_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vluxseg4ei32.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLUXSEG5EI32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b100;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLUXSEG5EI32_V: Instruction, VLUXSEG5EI32_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vluxseg5ei32.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLUXSEG6EI32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b101;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLUXSEG6EI32_V: Instruction, VLUXSEG6EI32_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vluxseg6ei32.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLUXSEG7EI32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b110;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLUXSEG7EI32_V: Instruction, VLUXSEG7EI32_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vluxseg7ei32.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLUXSEG8EI32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b111;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLUXSEG8EI32_V: Instruction, VLUXSEG8EI32_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vluxseg8ei32.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLOXSEG2EI32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b001;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLOXSEG2EI32_V: Instruction, VLOXSEG2EI32_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vloxseg2ei32.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLOXSEG3EI32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b010;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLOXSEG3EI32_V: Instruction, VLOXSEG3EI32_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vloxseg3ei32.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLOXSEG4EI32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b011;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLOXSEG4EI32_V: Instruction, VLOXSEG4EI32_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vloxseg4ei32.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLOXSEG5EI32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b100;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLOXSEG5EI32_V: Instruction, VLOXSEG5EI32_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vloxseg5ei32.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLOXSEG6EI32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b101;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLOXSEG6EI32_V: Instruction, VLOXSEG6EI32_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vloxseg6ei32.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLOXSEG7EI32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b110;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLOXSEG7EI32_V: Instruction, VLOXSEG7EI32_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vloxseg7ei32.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VLOXSEG8EI32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b111;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VLOXSEG8EI32_V: Instruction, VLOXSEG8EI32_V_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vloxseg8ei32.v	$vd, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VL1RE32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b000;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm = 0b1;
	bits<5> lumop = 0b01000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VL1RE32_V: Instruction, VL1RE32_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "vl1re32.v	$vd, (${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VL2RE32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b001;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm = 0b1;
	bits<5> lumop = 0b01000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VL2RE32_V: Instruction, VL2RE32_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "vl2re32.v	$vd, (${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VL4RE32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b011;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm = 0b1;
	bits<5> lumop = 0b01000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VL4RE32_V: Instruction, VL4RE32_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "vl4re32.v	$vd, (${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VL8RE32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b111;
	bits<2> MOPLDUnitStride = 0b00;
	bits<1> vm = 0b1;
	bits<5> lumop = 0b01000;
	bits<5> rs1;
	bits<5> vd;
	bits<7> opcode = 0b0000111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPLDUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = lumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VL8RE32_V: Instruction, VL8RE32_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "vl8re32.v	$vd, (${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 1;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMACC_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101101;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMACC_VV: Instruction, VMACC_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vmacc.vv	$vd, $vs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VWMACCU_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b111100;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VWMACCU_VV: Instruction, VWMACCU_VV_FM, TSFlagTemplate<InstFormatR, WidenV>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vwmaccu.vv	$vd, $vs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VWMACC_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b111101;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VWMACC_VV: Instruction, VWMACC_VV_FM, TSFlagTemplate<InstFormatR, WidenV>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vwmacc.vv	$vd, $vs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VWMACCSU_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b111111;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VWMACCSU_VV: Instruction, VWMACCSU_VV_FM, TSFlagTemplate<InstFormatR, WidenV>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vwmaccsu.vv	$vd, $vs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VMADD_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101001;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b110;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMADD_VX: Instruction, VMADD_VX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vmadd.vx	$vd, $rs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMACC_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101101;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b110;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMACC_VX: Instruction, VMACC_VX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vmacc.vx	$vd, $rs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VWMACCU_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b111100;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b110;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VWMACCU_VX: Instruction, VWMACCU_VX_FM, TSFlagTemplate<InstFormatR, WidenV>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vwmaccu.vx	$vd, $rs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VWMACC_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b111101;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b110;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VWMACC_VX: Instruction, VWMACC_VX_FM, TSFlagTemplate<InstFormatR, WidenV>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vwmacc.vx	$vd, $rs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VWMACCSU_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b111111;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b110;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VWMACCSU_VX: Instruction, VWMACCSU_VX_FM, TSFlagTemplate<InstFormatR, WidenV>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vwmaccsu.vx	$vd, $rs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VMADC_VVM_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010001;
	bits<1> vm = 0b0;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMADC_VVM: Instruction, VMADC_VVM_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMV0:$v0);
	let AsmString = "vmadc.vvm	$vd, $vs2, $vs1, v0";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VMADC_VXM_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010001;
	bits<1> vm = 0b0;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMADC_VXM: Instruction, VMADC_VXM_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMV0:$v0);
	let AsmString = "vmadc.vxm	$vd, $vs2, $rs1, v0";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VMADC_VIM_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010001;
	bits<1> vm = 0b0;
	bits<5> vs2;
	bits<5> imm;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = imm;
	let Inst{14-12} = 0b011;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMADC_VIM: Instruction, VMADC_VIM_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, simm5:$imm, VMV0:$v0);
	let AsmString = "vmadc.vim	$vd, $vs2, $imm, v0";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VMADC_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010001;
	bits<1> vm = 0b1;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMADC_VV: Instruction, VMADC_VV_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1);
	let AsmString = "vmadc.vv	$vd, $vs2, $vs1";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VMADC_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010001;
	bits<1> vm = 0b1;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMADC_VX: Instruction, VMADC_VX_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1);
	let AsmString = "vmadc.vx	$vd, $vs2, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VMADD_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101001;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMADD_VV: Instruction, VMADD_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vmadd.vv	$vd, $vs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VREDMAXU_VS_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000110;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VREDMAXU_VS: Instruction, VREDMAXU_VS_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vredmaxu.vs	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VREDMAX_VS_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000111;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VREDMAX_VS: Instruction, VREDMAX_VS_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vredmax.vs	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFREDMAX_VS_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000111;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFREDMAX_VS: Instruction, VFREDMAX_VS_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vfredmax.vs	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMAX_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000111;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMAX_VV: Instruction, VMAX_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vmax.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMAX_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000111;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMAX_VX: Instruction, VMAX_VX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vmax.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMAXU_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000110;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMAXU_VV: Instruction, VMAXU_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vmaxu.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMAXU_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000110;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMAXU_VX: Instruction, VMAXU_VX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vmaxu.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VREDMINU_VS_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000100;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VREDMINU_VS: Instruction, VREDMINU_VS_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vredminu.vs	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VREDMIN_VS_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000101;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VREDMIN_VS: Instruction, VREDMIN_VS_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vredmin.vs	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFREDMIN_VS_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000101;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFREDMIN_VS: Instruction, VFREDMIN_VS_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vfredmin.vs	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMIN_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000101;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMIN_VV: Instruction, VMIN_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vmin.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMIN_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000101;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMIN_VX: Instruction, VMIN_VX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vmin.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMINU_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000100;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMINU_VV: Instruction, VMINU_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vminu.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMINU_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000100;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMINU_VX: Instruction, VMINU_VX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vminu.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VREM_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100011;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VREM_VV: Instruction, VREM_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vrem.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VREM_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100011;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b110;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VREM_VX: Instruction, VREM_VX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vrem.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VREMU_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VREMU_VV: Instruction, VREMU_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vremu.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VREMU_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b110;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VREMU_VX: Instruction, VREMU_VX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vremu.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMV_V_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010111;
	bits<1> vm = 0b1;
	bits<5> vs2 = 0b00000;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMV_V_V: Instruction, VMV_V_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs1);
	let AsmString = "vmv.v.v	$vd, $vs1";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMV_V_X_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010111;
	bits<1> vm = 0b1;
	bits<5> vs2 = 0b00000;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMV_V_X: Instruction, VMV_V_X_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1);
	let AsmString = "vmv.v.x	$vd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMV_V_I_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010111;
	bits<1> vm = 0b1;
	bits<5> vs2 = 0b00000;
	bits<5> imm;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = imm;
	let Inst{14-12} = 0b011;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMV_V_I: Instruction, VMV_V_I_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins simm5:$imm);
	let AsmString = "vmv.v.i	$vd, $imm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFMV_V_F_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010111;
	bits<1> vm = 0b1;
	bits<5> vs2 = 0b00000;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFMV_V_F: Instruction, VFMV_V_F_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins FPR32:$rs1);
	let AsmString = "vfmv.v.f	$vd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMV_X_S_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010000;
	bits<1> vm = 0b1;
	bits<5> vs2;
	bits<5> vs1 = 0b00000;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMV_X_S: Instruction, VMV_X_S_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$vd);
	let InOperandList = (ins VR:$vs2);
	let AsmString = "vmv.x.s	$vd, $vs2";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMV_S_X_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010000;
	bits<1> vm = 0b1;
	bits<5> vs2 = 0b00000;
	bits<5> rs1;
	bits<3> opv = 0b110;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMV_S_X: Instruction, VMV_S_X_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd_wb);
	let InOperandList = (ins VR:$vd, GPR:$rs1);
	let AsmString = "vmv.s.x	$vd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "$vd = $vd_wb";
}
class VFMV_F_S_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010000;
	bits<1> vm = 0b1;
	bits<5> vs2;
	bits<5> vs1 = 0b00000;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFMV_F_S: Instruction, VFMV_F_S_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs FPR32:$vd);
	let InOperandList = (ins VR:$vs2);
	let AsmString = "vfmv.f.s	$vd, $vs2";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFMV_S_F_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010000;
	bits<1> vm = 0b1;
	bits<5> vs2 = 0b00000;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFMV_S_F: Instruction, VFMV_S_F_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd_wb);
	let InOperandList = (ins VR:$vd, FPR32:$rs1);
	let AsmString = "vfmv.s.f	$vd, $rs1";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "$vd = $vd_wb";
}
class VMV1R_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100111;
	bits<1> vm = 0b1;
	bits<5> vs2;
	bits<5> vs1 = 0b00000;
	bits<3> opv = 0b011;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMV1R_V: Instruction, VMV1R_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2);
	let AsmString = "vmv1r.v	$vd, $vs2";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMV2R_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100111;
	bits<1> vm = 0b1;
	bits<5> vs2;
	bits<5> vs1 = 0b00001;
	bits<3> opv = 0b011;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMV2R_V: Instruction, VMV2R_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2);
	let AsmString = "vmv2r.v	$vd, $vs2";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMV4R_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100111;
	bits<1> vm = 0b1;
	bits<5> vs2;
	bits<5> vs1 = 0b00011;
	bits<3> opv = 0b011;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMV4R_V: Instruction, VMV4R_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2);
	let AsmString = "vmv4r.v	$vd, $vs2";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMV8R_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100111;
	bits<1> vm = 0b1;
	bits<5> vs2;
	bits<5> vs1 = 0b00111;
	bits<3> opv = 0b011;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMV8R_V: Instruction, VMV8R_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2);
	let AsmString = "vmv8r.v	$vd, $vs2";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VNMSAC_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101111;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VNMSAC_VV: Instruction, VNMSAC_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vnmsac.vv	$vd, $vs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VNMSUB_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101011;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b110;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VNMSUB_VX: Instruction, VNMSUB_VX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vnmsub.vx	$vd, $rs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VNMSAC_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101111;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b110;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VNMSAC_VX: Instruction, VNMSAC_VX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vnmsac.vx	$vd, $rs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VNMSUB_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101011;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VNMSUB_VV: Instruction, VNMSUB_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vnmsub.vv	$vd, $vs1, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMUL_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100101;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMUL_VV: Instruction, VMUL_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vmul.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMULHU_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100100;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMULHU_VV: Instruction, VMULHU_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vmulhu.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VWMUL_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b111011;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VWMUL_VV: Instruction, VWMUL_VV_FM, TSFlagTemplate<InstFormatR, WidenV>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vwmul.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VWMULU_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b111000;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VWMULU_VV: Instruction, VWMULU_VV_FM, TSFlagTemplate<InstFormatR, WidenV>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vwmulu.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VWMULSU_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b111010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VWMULSU_VV: Instruction, VWMULSU_VV_FM, TSFlagTemplate<InstFormatR, WidenV>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vwmulsu.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VSMUL_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100111;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSMUL_VV: Instruction, VSMUL_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vsmul.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMULH_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100111;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b110;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMULH_VX: Instruction, VMULH_VX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vmulh.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMULHSU_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100110;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b110;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMULHSU_VX: Instruction, VMULHSU_VX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vmulhsu.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMUL_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100101;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b110;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMUL_VX: Instruction, VMUL_VX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vmul.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMULHU_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100100;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b110;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMULHU_VX: Instruction, VMULHU_VX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vmulhu.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VWMUL_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b111011;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b110;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VWMUL_VX: Instruction, VWMUL_VX_FM, TSFlagTemplate<InstFormatR, WidenV>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vwmul.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VWMULU_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b111000;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b110;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VWMULU_VX: Instruction, VWMULU_VX_FM, TSFlagTemplate<InstFormatR, WidenV>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vwmulu.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VWMULSU_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b111010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b110;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VWMULSU_VX: Instruction, VWMULSU_VX_FM, TSFlagTemplate<InstFormatR, WidenV>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vwmulsu.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VSMUL_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100111;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSMUL_VX: Instruction, VSMUL_VX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vsmul.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMULH_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100111;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMULH_VV: Instruction, VMULH_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vmulh.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMULHSU_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100110;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMULHSU_VV: Instruction, VMULHSU_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vmulhsu.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMSOF_M_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010100;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1 = 0b00010;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMSOF_M: Instruction, VMSOF_M_FM, TSFlagTemplate<InstFormatR, Iota>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VMaskOp:$vm);
	let AsmString = "vmsof.m	$vd, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VIOTA_M_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010100;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1 = 0b10000;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VIOTA_M: Instruction, VIOTA_M_FM, TSFlagTemplate<InstFormatR, Iota>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VMaskOp:$vm);
	let AsmString = "viota.m	$vd, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VOR_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b001010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VOR_VV: Instruction, VOR_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vor.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VOR_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b001010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VOR_VX: Instruction, VOR_VX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vor.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VREDOR_VS_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VREDOR_VS: Instruction, VREDOR_VS_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vredor.vs	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMOR_MM_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b011010;
	bits<1> vm = 0b1;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMOR_MM: Instruction, VMOR_MM_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1);
	let AsmString = "vmor.mm	$vd, $vs2, $vs1";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMNOR_MM_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b011110;
	bits<1> vm = 0b1;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMNOR_MM: Instruction, VMNOR_MM_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1);
	let AsmString = "vmnor.mm	$vd, $vs2, $vs1";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMORNOT_MM_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b011100;
	bits<1> vm = 0b1;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMORNOT_MM: Instruction, VMORNOT_MM_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1);
	let AsmString = "vmornot.mm	$vd, $vs2, $vs1";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VOR_VI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b001010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> imm;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = imm;
	let Inst{14-12} = 0b011;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VOR_VI: Instruction, VOR_VI_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, simm5:$imm, VMaskOp:$vm);
	let AsmString = "vor.vi	$vd, $vs2, $imm$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VPOPC_M_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010000;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1 = 0b10000;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VPOPC_M: Instruction, VPOPC_M_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs GPR:$vd);
	let InOperandList = (ins VR:$vs2, VMaskOp:$vm);
	let AsmString = "vpopc.m	$vd, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSLIDEUP_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b001110;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSLIDEUP_VX: Instruction, VSLIDEUP_VX_FM, TSFlagTemplate<InstFormatR, SlideUp>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vslideup.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VSLIDEDOWN_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b001111;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSLIDEDOWN_VX: Instruction, VSLIDEDOWN_VX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vslidedown.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSLIDEUP_VI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b001110;
	bits<1> vm;
	bits<5> vs2;
	bits<5> imm;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = imm;
	let Inst{14-12} = 0b011;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSLIDEUP_VI: Instruction, VSLIDEUP_VI_FM, TSFlagTemplate<InstFormatR, SlideUp>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, uimm5:$imm, VMaskOp:$vm);
	let AsmString = "vslideup.vi	$vd, $vs2, $imm$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VSLIDE1UP_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b001110;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b110;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSLIDE1UP_VX: Instruction, VSLIDE1UP_VX_FM, TSFlagTemplate<InstFormatR, SlideUp>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vslide1up.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VSLIDEDOWN_VI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b001111;
	bits<1> vm;
	bits<5> vs2;
	bits<5> imm;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = imm;
	let Inst{14-12} = 0b011;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSLIDEDOWN_VI: Instruction, VSLIDEDOWN_VI_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, uimm5:$imm, VMaskOp:$vm);
	let AsmString = "vslidedown.vi	$vd, $vs2, $imm$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFSLIDE1DOWN_VF_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b001111;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFSLIDE1DOWN_VF: Instruction, VFSLIDE1DOWN_VF_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, FPR32:$rs1, VMaskOp:$vm);
	let AsmString = "vfslide1down.vf	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSLIDE1DOWN_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b001111;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b110;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSLIDE1DOWN_VX: Instruction, VSLIDE1DOWN_VX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vslide1down.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFSLIDE1UP_VF_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b001110;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFSLIDE1UP_VF: Instruction, VFSLIDE1UP_VF_FM, TSFlagTemplate<InstFormatR, SlideUp>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, FPR32:$rs1, VMaskOp:$vm);
	let AsmString = "vfslide1up.vf	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VSLL_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100101;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSLL_VV: Instruction, VSLL_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vsll.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSLL_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100101;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSLL_VX: Instruction, VSLL_VX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vsll.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSLL_VI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100101;
	bits<1> vm;
	bits<5> vs2;
	bits<5> imm;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = imm;
	let Inst{14-12} = 0b011;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSLL_VI: Instruction, VSLL_VI_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, uimm5:$imm, VMaskOp:$vm);
	let AsmString = "vsll.vi	$vd, $vs2, $imm$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VZEXT_VF2_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1 = 0b00110;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VZEXT_VF2: Instruction, VZEXT_VF2_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VMaskOp:$vm);
	let AsmString = "vzext.vf2	$vd, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSEXT_VF2_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1 = 0b00111;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSEXT_VF2: Instruction, VSEXT_VF2_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsext.vf2	$vd, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VZEXT_VF4_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1 = 0b00100;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VZEXT_VF4: Instruction, VZEXT_VF4_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VMaskOp:$vm);
	let AsmString = "vzext.vf4	$vd, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSEXT_VF4_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1 = 0b00101;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSEXT_VF4: Instruction, VSEXT_VF4_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsext.vf4	$vd, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VZEXT_VF8_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1 = 0b00010;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VZEXT_VF8: Instruction, VZEXT_VF8_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VMaskOp:$vm);
	let AsmString = "vzext.vf8	$vd, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSEXT_VF8_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1 = 0b00011;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSEXT_VF8: Instruction, VSEXT_VF8_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsext.vf8	$vd, $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSRA_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101001;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSRA_VV: Instruction, VSRA_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vsra.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VNSRA_WX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101101;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VNSRA_WX: Instruction, VNSRA_WX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vnsra.wx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VSRA_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101001;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSRA_VX: Instruction, VSRA_VX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vsra.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VNSRA_WV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101101;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VNSRA_WV: Instruction, VNSRA_WV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vnsra.wv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VSRA_VI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101001;
	bits<1> vm;
	bits<5> vs2;
	bits<5> imm;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = imm;
	let Inst{14-12} = 0b011;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSRA_VI: Instruction, VSRA_VI_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, uimm5:$imm, VMaskOp:$vm);
	let AsmString = "vsra.vi	$vd, $vs2, $imm$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VNSRA_WI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101101;
	bits<1> vm;
	bits<5> vs2;
	bits<5> imm;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = imm;
	let Inst{14-12} = 0b011;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VNSRA_WI: Instruction, VNSRA_WI_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, uimm5:$imm, VMaskOp:$vm);
	let AsmString = "vnsra.wi	$vd, $vs2, $imm$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VSSRA_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101011;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSSRA_VV: Instruction, VSSRA_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vssra.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSRA_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101011;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSSRA_VX: Instruction, VSSRA_VX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vssra.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSRA_VI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101011;
	bits<1> vm;
	bits<5> vs2;
	bits<5> imm;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = imm;
	let Inst{14-12} = 0b011;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSSRA_VI: Instruction, VSSRA_VI_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, uimm5:$imm, VMaskOp:$vm);
	let AsmString = "vssra.vi	$vd, $vs2, $imm$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSRL_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101000;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSRL_VV: Instruction, VSRL_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vsrl.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VNSRL_WX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101100;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VNSRL_WX: Instruction, VNSRL_WX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vnsrl.wx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VSRL_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101000;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSRL_VX: Instruction, VSRL_VX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vsrl.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VNSRL_WV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101100;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VNSRL_WV: Instruction, VNSRL_WV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vnsrl.wv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VSRL_VI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101000;
	bits<1> vm;
	bits<5> vs2;
	bits<5> imm;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = imm;
	let Inst{14-12} = 0b011;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSRL_VI: Instruction, VSRL_VI_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, uimm5:$imm, VMaskOp:$vm);
	let AsmString = "vsrl.vi	$vd, $vs2, $imm$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VNSRL_WI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101100;
	bits<1> vm;
	bits<5> vs2;
	bits<5> imm;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = imm;
	let Inst{14-12} = 0b011;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VNSRL_WI: Instruction, VNSRL_WI_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, uimm5:$imm, VMaskOp:$vm);
	let AsmString = "vnsrl.wi	$vd, $vs2, $imm$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VSSRL_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSSRL_VV: Instruction, VSSRL_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vssrl.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSRL_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSSRL_VX: Instruction, VSSRL_VX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vssrl.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSRL_VI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> imm;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = imm;
	let Inst{14-12} = 0b011;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSSRL_VI: Instruction, VSSRL_VI_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, uimm5:$imm, VMaskOp:$vm);
	let AsmString = "vssrl.vi	$vd, $vs2, $imm$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSE8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b000;
	bits<2> MOPSTUnitStride = 0b00;
	bits<1> vm;
	bits<5> sumop = 0b00000;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = sumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSE8_V: Instruction, VSE8_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vse8.v	$vs3, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSE8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b000;
	bits<2> MOPSTStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSE8_V: Instruction, VSSE8_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vsse8.v	$vs3, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSUXEI8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b000;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSUXEI8_V: Instruction, VSUXEI8_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsuxei8.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSOXEI8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b000;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSOXEI8_V: Instruction, VSOXEI8_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsoxei8.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSEG2E8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b001;
	bits<2> MOPSTUnitStride = 0b00;
	bits<1> vm;
	bits<5> sumop = 0b00000;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = sumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSEG2E8_V: Instruction, VSSEG2E8_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vsseg2e8.v	$vs3, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSEG3E8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b010;
	bits<2> MOPSTUnitStride = 0b00;
	bits<1> vm;
	bits<5> sumop = 0b00000;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = sumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSEG3E8_V: Instruction, VSSEG3E8_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vsseg3e8.v	$vs3, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSEG4E8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b011;
	bits<2> MOPSTUnitStride = 0b00;
	bits<1> vm;
	bits<5> sumop = 0b00000;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = sumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSEG4E8_V: Instruction, VSSEG4E8_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vsseg4e8.v	$vs3, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSEG5E8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b100;
	bits<2> MOPSTUnitStride = 0b00;
	bits<1> vm;
	bits<5> sumop = 0b00000;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = sumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSEG5E8_V: Instruction, VSSEG5E8_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vsseg5e8.v	$vs3, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSEG6E8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b101;
	bits<2> MOPSTUnitStride = 0b00;
	bits<1> vm;
	bits<5> sumop = 0b00000;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = sumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSEG6E8_V: Instruction, VSSEG6E8_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vsseg6e8.v	$vs3, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSEG7E8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b110;
	bits<2> MOPSTUnitStride = 0b00;
	bits<1> vm;
	bits<5> sumop = 0b00000;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = sumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSEG7E8_V: Instruction, VSSEG7E8_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vsseg7e8.v	$vs3, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSEG8E8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b111;
	bits<2> MOPSTUnitStride = 0b00;
	bits<1> vm;
	bits<5> sumop = 0b00000;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = sumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSEG8E8_V: Instruction, VSSEG8E8_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vsseg8e8.v	$vs3, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSSEG2E8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b001;
	bits<2> MOPSTStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSSEG2E8_V: Instruction, VSSSEG2E8_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vssseg2e8.v	$vs3, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSSEG3E8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b010;
	bits<2> MOPSTStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSSEG3E8_V: Instruction, VSSSEG3E8_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vssseg3e8.v	$vs3, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSSEG4E8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b011;
	bits<2> MOPSTStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSSEG4E8_V: Instruction, VSSSEG4E8_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vssseg4e8.v	$vs3, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSSEG5E8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b100;
	bits<2> MOPSTStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSSEG5E8_V: Instruction, VSSSEG5E8_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vssseg5e8.v	$vs3, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSSEG6E8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b101;
	bits<2> MOPSTStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSSEG6E8_V: Instruction, VSSSEG6E8_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vssseg6e8.v	$vs3, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSSEG7E8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b110;
	bits<2> MOPSTStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSSEG7E8_V: Instruction, VSSSEG7E8_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vssseg7e8.v	$vs3, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSSEG8E8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b111;
	bits<2> MOPSTStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSSEG8E8_V: Instruction, VSSSEG8E8_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vssseg8e8.v	$vs3, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSUXSEG2EI8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b001;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSUXSEG2EI8_V: Instruction, VSUXSEG2EI8_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsuxseg2ei8.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSUXSEG3EI8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b010;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSUXSEG3EI8_V: Instruction, VSUXSEG3EI8_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsuxseg3ei8.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSUXSEG4EI8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b011;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSUXSEG4EI8_V: Instruction, VSUXSEG4EI8_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsuxseg4ei8.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSUXSEG5EI8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b100;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSUXSEG5EI8_V: Instruction, VSUXSEG5EI8_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsuxseg5ei8.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSUXSEG6EI8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b101;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSUXSEG6EI8_V: Instruction, VSUXSEG6EI8_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsuxseg6ei8.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSUXSEG7EI8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b110;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSUXSEG7EI8_V: Instruction, VSUXSEG7EI8_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsuxseg7ei8.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSUXSEG8EI8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b111;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSUXSEG8EI8_V: Instruction, VSUXSEG8EI8_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsuxseg8ei8.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSOXSEG2EI8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b001;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSOXSEG2EI8_V: Instruction, VSOXSEG2EI8_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsoxseg2ei8.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSOXSEG3EI8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b010;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSOXSEG3EI8_V: Instruction, VSOXSEG3EI8_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsoxseg3ei8.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSOXSEG4EI8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b011;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSOXSEG4EI8_V: Instruction, VSOXSEG4EI8_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsoxseg4ei8.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSOXSEG5EI8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b100;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSOXSEG5EI8_V: Instruction, VSOXSEG5EI8_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsoxseg5ei8.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSOXSEG6EI8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b101;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSOXSEG6EI8_V: Instruction, VSOXSEG6EI8_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsoxseg6ei8.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSOXSEG7EI8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b110;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSOXSEG7EI8_V: Instruction, VSOXSEG7EI8_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsoxseg7ei8.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSOXSEG8EI8_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b111;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSOXSEG8EI8_V: Instruction, VSOXSEG8EI8_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsoxseg8ei8.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VS1R_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<3> nf = 0b000;
	bits<1> mew = 0b0;
	bits<2> MOPSTUnitStride = 0b00;
	bits<1> vm = 0b1;
	bits<5> sumop = 0b01000;
	bits<5> rs1;
	bits<3> width = 0b000;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{31-29} = nf;
	let Inst{28-28} = mew;
	let Inst{27-26} = MOPSTUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = sumop;
	let Inst{19-15} = rs1;
	let Inst{14-12} = width;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VS1R_V: Instruction, VS1R_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1);
	let AsmString = "vs1r.v	$vs3, (${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSE1_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0000;
	bits<3> nf = 0b000;
	bits<2> MOPSTUnitStride = 0b00;
	bits<1> vm;
	bits<5> sumop = 0b01011;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = sumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSE1_V: Instruction, VSE1_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1);
	let AsmString = "vse1.v	$vs3, (${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSE64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b000;
	bits<2> MOPSTUnitStride = 0b00;
	bits<1> vm;
	bits<5> sumop = 0b00000;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = sumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSE64_V: Instruction, VSE64_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vse64.v	$vs3, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSE64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b000;
	bits<2> MOPSTStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSE64_V: Instruction, VSSE64_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vsse64.v	$vs3, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSUXEI64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b000;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSUXEI64_V: Instruction, VSUXEI64_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsuxei64.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSOXEI64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b000;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSOXEI64_V: Instruction, VSOXEI64_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsoxei64.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSEG2E64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b001;
	bits<2> MOPSTUnitStride = 0b00;
	bits<1> vm;
	bits<5> sumop = 0b00000;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = sumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSEG2E64_V: Instruction, VSSEG2E64_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vsseg2e64.v	$vs3, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSEG3E64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b010;
	bits<2> MOPSTUnitStride = 0b00;
	bits<1> vm;
	bits<5> sumop = 0b00000;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = sumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSEG3E64_V: Instruction, VSSEG3E64_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vsseg3e64.v	$vs3, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSEG4E64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b011;
	bits<2> MOPSTUnitStride = 0b00;
	bits<1> vm;
	bits<5> sumop = 0b00000;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = sumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSEG4E64_V: Instruction, VSSEG4E64_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vsseg4e64.v	$vs3, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSEG5E64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b100;
	bits<2> MOPSTUnitStride = 0b00;
	bits<1> vm;
	bits<5> sumop = 0b00000;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = sumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSEG5E64_V: Instruction, VSSEG5E64_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vsseg5e64.v	$vs3, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSEG6E64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b101;
	bits<2> MOPSTUnitStride = 0b00;
	bits<1> vm;
	bits<5> sumop = 0b00000;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = sumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSEG6E64_V: Instruction, VSSEG6E64_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vsseg6e64.v	$vs3, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSEG7E64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b110;
	bits<2> MOPSTUnitStride = 0b00;
	bits<1> vm;
	bits<5> sumop = 0b00000;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = sumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSEG7E64_V: Instruction, VSSEG7E64_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vsseg7e64.v	$vs3, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSEG8E64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b111;
	bits<2> MOPSTUnitStride = 0b00;
	bits<1> vm;
	bits<5> sumop = 0b00000;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = sumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSEG8E64_V: Instruction, VSSEG8E64_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vsseg8e64.v	$vs3, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSSEG2E64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b001;
	bits<2> MOPSTStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSSEG2E64_V: Instruction, VSSSEG2E64_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vssseg2e64.v	$vs3, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSSEG3E64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b010;
	bits<2> MOPSTStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSSEG3E64_V: Instruction, VSSSEG3E64_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vssseg3e64.v	$vs3, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSSEG4E64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b011;
	bits<2> MOPSTStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSSEG4E64_V: Instruction, VSSSEG4E64_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vssseg4e64.v	$vs3, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSSEG5E64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b100;
	bits<2> MOPSTStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSSEG5E64_V: Instruction, VSSSEG5E64_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vssseg5e64.v	$vs3, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSSEG6E64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b101;
	bits<2> MOPSTStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSSEG6E64_V: Instruction, VSSSEG6E64_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vssseg6e64.v	$vs3, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSSEG7E64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b110;
	bits<2> MOPSTStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSSEG7E64_V: Instruction, VSSSEG7E64_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vssseg7e64.v	$vs3, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSSEG8E64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b111;
	bits<2> MOPSTStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSSEG8E64_V: Instruction, VSSSEG8E64_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vssseg8e64.v	$vs3, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSUXSEG2EI64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b001;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSUXSEG2EI64_V: Instruction, VSUXSEG2EI64_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsuxseg2ei64.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSUXSEG3EI64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b010;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSUXSEG3EI64_V: Instruction, VSUXSEG3EI64_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsuxseg3ei64.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSUXSEG4EI64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b011;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSUXSEG4EI64_V: Instruction, VSUXSEG4EI64_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsuxseg4ei64.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSUXSEG5EI64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b100;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSUXSEG5EI64_V: Instruction, VSUXSEG5EI64_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsuxseg5ei64.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSUXSEG6EI64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b101;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSUXSEG6EI64_V: Instruction, VSUXSEG6EI64_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsuxseg6ei64.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSUXSEG7EI64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b110;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSUXSEG7EI64_V: Instruction, VSUXSEG7EI64_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsuxseg7ei64.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSUXSEG8EI64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b111;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSUXSEG8EI64_V: Instruction, VSUXSEG8EI64_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsuxseg8ei64.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSOXSEG2EI64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b001;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSOXSEG2EI64_V: Instruction, VSOXSEG2EI64_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsoxseg2ei64.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSOXSEG3EI64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b010;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSOXSEG3EI64_V: Instruction, VSOXSEG3EI64_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsoxseg3ei64.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSOXSEG4EI64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b011;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSOXSEG4EI64_V: Instruction, VSOXSEG4EI64_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsoxseg4ei64.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSOXSEG5EI64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b100;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSOXSEG5EI64_V: Instruction, VSOXSEG5EI64_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsoxseg5ei64.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSOXSEG6EI64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b101;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSOXSEG6EI64_V: Instruction, VSOXSEG6EI64_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsoxseg6ei64.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSOXSEG7EI64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b110;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSOXSEG7EI64_V: Instruction, VSOXSEG7EI64_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsoxseg7ei64.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSOXSEG8EI64_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0111;
	bits<3> nf = 0b111;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSOXSEG8EI64_V: Instruction, VSOXSEG8EI64_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsoxseg8ei64.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VS8R_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<3> nf = 0b111;
	bits<1> mew = 0b0;
	bits<2> MOPSTUnitStride = 0b00;
	bits<1> vm = 0b1;
	bits<5> sumop = 0b01000;
	bits<5> rs1;
	bits<3> width = 0b000;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{31-29} = nf;
	let Inst{28-28} = mew;
	let Inst{27-26} = MOPSTUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = sumop;
	let Inst{19-15} = rs1;
	let Inst{14-12} = width;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VS8R_V: Instruction, VS8R_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1);
	let AsmString = "vs8r.v	$vs3, (${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSE16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b000;
	bits<2> MOPSTUnitStride = 0b00;
	bits<1> vm;
	bits<5> sumop = 0b00000;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = sumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSE16_V: Instruction, VSE16_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vse16.v	$vs3, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSE16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b000;
	bits<2> MOPSTStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSE16_V: Instruction, VSSE16_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vsse16.v	$vs3, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSUXEI16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b000;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSUXEI16_V: Instruction, VSUXEI16_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsuxei16.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSOXEI16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b000;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSOXEI16_V: Instruction, VSOXEI16_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsoxei16.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSEG2E16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b001;
	bits<2> MOPSTUnitStride = 0b00;
	bits<1> vm;
	bits<5> sumop = 0b00000;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = sumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSEG2E16_V: Instruction, VSSEG2E16_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vsseg2e16.v	$vs3, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSEG3E16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b010;
	bits<2> MOPSTUnitStride = 0b00;
	bits<1> vm;
	bits<5> sumop = 0b00000;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = sumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSEG3E16_V: Instruction, VSSEG3E16_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vsseg3e16.v	$vs3, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSEG4E16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b011;
	bits<2> MOPSTUnitStride = 0b00;
	bits<1> vm;
	bits<5> sumop = 0b00000;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = sumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSEG4E16_V: Instruction, VSSEG4E16_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vsseg4e16.v	$vs3, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSEG5E16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b100;
	bits<2> MOPSTUnitStride = 0b00;
	bits<1> vm;
	bits<5> sumop = 0b00000;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = sumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSEG5E16_V: Instruction, VSSEG5E16_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vsseg5e16.v	$vs3, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSEG6E16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b101;
	bits<2> MOPSTUnitStride = 0b00;
	bits<1> vm;
	bits<5> sumop = 0b00000;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = sumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSEG6E16_V: Instruction, VSSEG6E16_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vsseg6e16.v	$vs3, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSEG7E16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b110;
	bits<2> MOPSTUnitStride = 0b00;
	bits<1> vm;
	bits<5> sumop = 0b00000;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = sumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSEG7E16_V: Instruction, VSSEG7E16_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vsseg7e16.v	$vs3, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSEG8E16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b111;
	bits<2> MOPSTUnitStride = 0b00;
	bits<1> vm;
	bits<5> sumop = 0b00000;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = sumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSEG8E16_V: Instruction, VSSEG8E16_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vsseg8e16.v	$vs3, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSSEG2E16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b001;
	bits<2> MOPSTStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSSEG2E16_V: Instruction, VSSSEG2E16_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vssseg2e16.v	$vs3, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSSEG3E16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b010;
	bits<2> MOPSTStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSSEG3E16_V: Instruction, VSSSEG3E16_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vssseg3e16.v	$vs3, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSSEG4E16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b011;
	bits<2> MOPSTStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSSEG4E16_V: Instruction, VSSSEG4E16_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vssseg4e16.v	$vs3, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSSEG5E16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b100;
	bits<2> MOPSTStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSSEG5E16_V: Instruction, VSSSEG5E16_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vssseg5e16.v	$vs3, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSSEG6E16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b101;
	bits<2> MOPSTStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSSEG6E16_V: Instruction, VSSSEG6E16_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vssseg6e16.v	$vs3, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSSEG7E16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b110;
	bits<2> MOPSTStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSSEG7E16_V: Instruction, VSSSEG7E16_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vssseg7e16.v	$vs3, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSSEG8E16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b111;
	bits<2> MOPSTStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSSEG8E16_V: Instruction, VSSSEG8E16_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vssseg8e16.v	$vs3, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSUXSEG2EI16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b001;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSUXSEG2EI16_V: Instruction, VSUXSEG2EI16_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsuxseg2ei16.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSUXSEG3EI16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b010;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSUXSEG3EI16_V: Instruction, VSUXSEG3EI16_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsuxseg3ei16.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSUXSEG4EI16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b011;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSUXSEG4EI16_V: Instruction, VSUXSEG4EI16_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsuxseg4ei16.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSUXSEG5EI16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b100;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSUXSEG5EI16_V: Instruction, VSUXSEG5EI16_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsuxseg5ei16.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSUXSEG6EI16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b101;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSUXSEG6EI16_V: Instruction, VSUXSEG6EI16_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsuxseg6ei16.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSUXSEG7EI16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b110;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSUXSEG7EI16_V: Instruction, VSUXSEG7EI16_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsuxseg7ei16.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSUXSEG8EI16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b111;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSUXSEG8EI16_V: Instruction, VSUXSEG8EI16_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsuxseg8ei16.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSOXSEG2EI16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b001;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSOXSEG2EI16_V: Instruction, VSOXSEG2EI16_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsoxseg2ei16.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSOXSEG3EI16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b010;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSOXSEG3EI16_V: Instruction, VSOXSEG3EI16_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsoxseg3ei16.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSOXSEG4EI16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b011;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSOXSEG4EI16_V: Instruction, VSOXSEG4EI16_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsoxseg4ei16.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSOXSEG5EI16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b100;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSOXSEG5EI16_V: Instruction, VSOXSEG5EI16_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsoxseg5ei16.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSOXSEG6EI16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b101;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSOXSEG6EI16_V: Instruction, VSOXSEG6EI16_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsoxseg6ei16.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSOXSEG7EI16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b110;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSOXSEG7EI16_V: Instruction, VSOXSEG7EI16_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsoxseg7ei16.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSOXSEG8EI16_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0101;
	bits<3> nf = 0b111;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSOXSEG8EI16_V: Instruction, VSOXSEG8EI16_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsoxseg8ei16.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VS2R_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<3> nf = 0b001;
	bits<1> mew = 0b0;
	bits<2> MOPSTUnitStride = 0b00;
	bits<1> vm = 0b1;
	bits<5> sumop = 0b01000;
	bits<5> rs1;
	bits<3> width = 0b000;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{31-29} = nf;
	let Inst{28-28} = mew;
	let Inst{27-26} = MOPSTUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = sumop;
	let Inst{19-15} = rs1;
	let Inst{14-12} = width;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VS2R_V: Instruction, VS2R_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1);
	let AsmString = "vs2r.v	$vs3, (${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSE32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b000;
	bits<2> MOPSTUnitStride = 0b00;
	bits<1> vm;
	bits<5> sumop = 0b00000;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = sumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSE32_V: Instruction, VSE32_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vse32.v	$vs3, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSE32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b000;
	bits<2> MOPSTStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSE32_V: Instruction, VSSE32_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vsse32.v	$vs3, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSUXEI32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b000;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSUXEI32_V: Instruction, VSUXEI32_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsuxei32.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSOXEI32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b000;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSOXEI32_V: Instruction, VSOXEI32_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsoxei32.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSEG2E32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b001;
	bits<2> MOPSTUnitStride = 0b00;
	bits<1> vm;
	bits<5> sumop = 0b00000;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = sumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSEG2E32_V: Instruction, VSSEG2E32_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vsseg2e32.v	$vs3, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSEG3E32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b010;
	bits<2> MOPSTUnitStride = 0b00;
	bits<1> vm;
	bits<5> sumop = 0b00000;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = sumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSEG3E32_V: Instruction, VSSEG3E32_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vsseg3e32.v	$vs3, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSEG4E32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b011;
	bits<2> MOPSTUnitStride = 0b00;
	bits<1> vm;
	bits<5> sumop = 0b00000;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = sumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSEG4E32_V: Instruction, VSSEG4E32_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vsseg4e32.v	$vs3, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSEG5E32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b100;
	bits<2> MOPSTUnitStride = 0b00;
	bits<1> vm;
	bits<5> sumop = 0b00000;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = sumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSEG5E32_V: Instruction, VSSEG5E32_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vsseg5e32.v	$vs3, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSEG6E32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b101;
	bits<2> MOPSTUnitStride = 0b00;
	bits<1> vm;
	bits<5> sumop = 0b00000;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = sumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSEG6E32_V: Instruction, VSSEG6E32_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vsseg6e32.v	$vs3, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSEG7E32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b110;
	bits<2> MOPSTUnitStride = 0b00;
	bits<1> vm;
	bits<5> sumop = 0b00000;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = sumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSEG7E32_V: Instruction, VSSEG7E32_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vsseg7e32.v	$vs3, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSEG8E32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b111;
	bits<2> MOPSTUnitStride = 0b00;
	bits<1> vm;
	bits<5> sumop = 0b00000;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = sumop;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSEG8E32_V: Instruction, VSSEG8E32_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vsseg8e32.v	$vs3, (${rs1})$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSSEG2E32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b001;
	bits<2> MOPSTStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSSEG2E32_V: Instruction, VSSSEG2E32_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vssseg2e32.v	$vs3, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSSEG3E32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b010;
	bits<2> MOPSTStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSSEG3E32_V: Instruction, VSSSEG3E32_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vssseg3e32.v	$vs3, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSSEG4E32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b011;
	bits<2> MOPSTStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSSEG4E32_V: Instruction, VSSSEG4E32_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vssseg4e32.v	$vs3, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSSEG5E32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b100;
	bits<2> MOPSTStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSSEG5E32_V: Instruction, VSSSEG5E32_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vssseg5e32.v	$vs3, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSSEG6E32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b101;
	bits<2> MOPSTStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSSEG6E32_V: Instruction, VSSSEG6E32_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vssseg6e32.v	$vs3, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSSEG7E32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b110;
	bits<2> MOPSTStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSSEG7E32_V: Instruction, VSSSEG7E32_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vssseg7e32.v	$vs3, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSSEG8E32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b111;
	bits<2> MOPSTStrided = 0b10;
	bits<1> vm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = MOPSTStrided;
	let Inst{25-25} = vm;
	let Inst{24-20} = rs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSSSEG8E32_V: Instruction, VSSSEG8E32_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, GPR:$rs2, VMaskOp:$vm);
	let AsmString = "vssseg8e32.v	$vs3, (${rs1}), $rs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSUXSEG2EI32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b001;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSUXSEG2EI32_V: Instruction, VSUXSEG2EI32_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsuxseg2ei32.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSUXSEG3EI32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b010;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSUXSEG3EI32_V: Instruction, VSUXSEG3EI32_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsuxseg3ei32.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSUXSEG4EI32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b011;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSUXSEG4EI32_V: Instruction, VSUXSEG4EI32_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsuxseg4ei32.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSUXSEG5EI32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b100;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSUXSEG5EI32_V: Instruction, VSUXSEG5EI32_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsuxseg5ei32.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSUXSEG6EI32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b101;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSUXSEG6EI32_V: Instruction, VSUXSEG6EI32_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsuxseg6ei32.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSUXSEG7EI32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b110;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSUXSEG7EI32_V: Instruction, VSUXSEG7EI32_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsuxseg7ei32.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSUXSEG8EI32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b111;
	bits<2> mop = 0b01;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSUXSEG8EI32_V: Instruction, VSUXSEG8EI32_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsuxseg8ei32.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSOXSEG2EI32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b001;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSOXSEG2EI32_V: Instruction, VSOXSEG2EI32_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsoxseg2ei32.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSOXSEG3EI32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b010;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSOXSEG3EI32_V: Instruction, VSOXSEG3EI32_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsoxseg3ei32.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSOXSEG4EI32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b011;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSOXSEG4EI32_V: Instruction, VSOXSEG4EI32_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsoxseg4ei32.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSOXSEG5EI32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b100;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSOXSEG5EI32_V: Instruction, VSOXSEG5EI32_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsoxseg5ei32.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSOXSEG6EI32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b101;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSOXSEG6EI32_V: Instruction, VSOXSEG6EI32_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsoxseg6ei32.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSOXSEG7EI32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b110;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSOXSEG7EI32_V: Instruction, VSOXSEG7EI32_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsoxseg7ei32.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSOXSEG8EI32_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<4> width = 0b0110;
	bits<3> nf = 0b111;
	bits<2> mop = 0b11;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{28-28} = width{3-3};
	let Inst{14-12} = width{2-0};
	let Inst{31-29} = nf;
	let Inst{27-26} = mop;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VSOXSEG8EI32_V: Instruction, VSOXSEG8EI32_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm);
	let AsmString = "vsoxseg8ei32.v	$vs3, (${rs1}), $vs2$vm";
	let Pattern = [];
	let Predicates = [HasStdExtZvlsseg];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VS4R_V_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<3> nf = 0b011;
	bits<1> mew = 0b0;
	bits<2> MOPSTUnitStride = 0b00;
	bits<1> vm = 0b1;
	bits<5> sumop = 0b01000;
	bits<5> rs1;
	bits<3> width = 0b000;
	bits<5> vs3;
	bits<7> opcode = 0b0100111;
	let Inst{31-29} = nf;
	let Inst{28-28} = mew;
	let Inst{27-26} = MOPSTUnitStride;
	let Inst{25-25} = vm;
	let Inst{24-20} = sumop;
	let Inst{19-15} = rs1;
	let Inst{14-12} = width;
	let Inst{11-7} = vs3;
	let Inst{6-0} = opcode;
}
def VS4R_V: Instruction, VS4R_V_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs);
	let InOperandList = (ins VR:$vs3, GPR:$rs1);
	let AsmString = "vs4r.v	$vs3, (${rs1})";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 1;
	let Defs = [];
	let Uses = [];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSUBU_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSSUBU_VV: Instruction, VSSUBU_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vssubu.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSUBU_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSSUBU_VX: Instruction, VSSUBU_VX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vssubu.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSUB_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSUB_VV: Instruction, VSUB_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vsub.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VWSUB_WV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b110111;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VWSUB_WV: Instruction, VWSUB_WV_FM, TSFlagTemplate<InstFormatR, WidenW>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vwsub.wv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VWSUB_WX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b110111;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b110;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VWSUB_WX: Instruction, VWSUB_WX_FM, TSFlagTemplate<InstFormatR, WidenW>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vwsub.wx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VWSUB_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b110011;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VWSUB_VV: Instruction, VWSUB_VV_FM, TSFlagTemplate<InstFormatR, WidenV>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vwsub.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VSUB_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSUB_VX: Instruction, VSUB_VX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vsub.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VWSUB_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b110011;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b110;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VWSUB_VX: Instruction, VWSUB_VX_FM, TSFlagTemplate<InstFormatR, WidenV>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vwsub.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VSSUB_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100011;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSSUB_VV: Instruction, VSSUB_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vssub.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VSSUB_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b100011;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VSSUB_VX: Instruction, VSSUB_VX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vssub.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VRSUB_VI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000011;
	bits<1> vm;
	bits<5> vs2;
	bits<5> imm;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = imm;
	let Inst{14-12} = 0b011;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VRSUB_VI: Instruction, VRSUB_VI_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, simm5:$imm, VMaskOp:$vm);
	let AsmString = "vrsub.vi	$vd, $vs2, $imm$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VXOR_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b001011;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VXOR_VV: Instruction, VXOR_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vxor.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VXOR_VX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b001011;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VXOR_VX: Instruction, VXOR_VX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vxor.vx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VREDXOR_VS_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000011;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VREDXOR_VS: Instruction, VREDXOR_VS_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vredxor.vs	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMXOR_MM_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b011011;
	bits<1> vm = 0b1;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMXOR_MM: Instruction, VMXOR_MM_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1);
	let AsmString = "vmxor.mm	$vd, $vs2, $vs1";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMXNOR_MM_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b011111;
	bits<1> vm = 0b1;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMXNOR_MM: Instruction, VMXNOR_MM_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1);
	let AsmString = "vmxnor.mm	$vd, $vs2, $vs1";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VXOR_VI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b001011;
	bits<1> vm;
	bits<5> vs2;
	bits<5> imm;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = imm;
	let Inst{14-12} = 0b011;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VXOR_VI: Instruction, VXOR_VI_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, simm5:$imm, VMaskOp:$vm);
	let AsmString = "vxor.vi	$vd, $vs2, $imm$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMERGE_VVM_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010111;
	bits<1> vm = 0b0;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMERGE_VVM: Instruction, VMERGE_VVM_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMV0:$v0);
	let AsmString = "vmerge.vvm	$vd, $vs2, $vs1, v0";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMERGE_VXM_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010111;
	bits<1> vm = 0b0;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMERGE_VXM: Instruction, VMERGE_VXM_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMV0:$v0);
	let AsmString = "vmerge.vxm	$vd, $vs2, $rs1, v0";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VMERGE_VIM_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010111;
	bits<1> vm = 0b0;
	bits<5> vs2;
	bits<5> imm;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = imm;
	let Inst{14-12} = 0b011;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VMERGE_VIM: Instruction, VMERGE_VIM_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, simm5:$imm, VMV0:$v0);
	let AsmString = "vmerge.vim	$vd, $vs2, $imm, v0";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VNCLIPU_WV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101110;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VNCLIPU_WV: Instruction, VNCLIPU_WV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vnclipu.wv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VNCLIPU_WX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101110;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VNCLIPU_WX: Instruction, VNCLIPU_WX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vnclipu.wx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VNCLIPU_WI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101110;
	bits<1> vm;
	bits<5> vs2;
	bits<5> imm;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = imm;
	let Inst{14-12} = 0b011;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VNCLIPU_WI: Instruction, VNCLIPU_WI_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, uimm5:$imm, VMaskOp:$vm);
	let AsmString = "vnclipu.wi	$vd, $vs2, $imm$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VNCLIP_WV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101111;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VNCLIP_WV: Instruction, VNCLIP_WV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vnclip.wv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VNCLIP_WX_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101111;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b100;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VNCLIP_WX: Instruction, VNCLIP_WX_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm);
	let AsmString = "vnclip.wx	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VNCLIP_WI_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b101111;
	bits<1> vm;
	bits<5> vs2;
	bits<5> imm;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = imm;
	let Inst{14-12} = 0b011;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VNCLIP_WI: Instruction, VNCLIP_WI_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, uimm5:$imm, VMaskOp:$vm);
	let AsmString = "vnclip.wi	$vd, $vs2, $imm$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VFSGNJ_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b001000;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFSGNJ_VV: Instruction, VFSGNJ_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vfsgnj.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFSGNJ_VF_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b001000;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFSGNJ_VF: Instruction, VFSGNJ_VF_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, FPR32:$rs1, VMaskOp:$vm);
	let AsmString = "vfsgnj.vf	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFSGNJN_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b001001;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFSGNJN_VV: Instruction, VFSGNJN_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vfsgnjn.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFSGNJN_VF_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b001001;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFSGNJN_VF: Instruction, VFSGNJN_VF_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, FPR32:$rs1, VMaskOp:$vm);
	let AsmString = "vfsgnjn.vf	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFSGNJX_VV_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b001010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFSGNJX_VV: Instruction, VFSGNJX_VV_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vfsgnjx.vv	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFSGNJX_VF_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b001010;
	bits<1> vm;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFSGNJX_VF: Instruction, VFSGNJX_VF_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, FPR32:$rs1, VMaskOp:$vm);
	let AsmString = "vfsgnjx.vf	$vd, $vs2, $rs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFMERGE_VFM_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010111;
	bits<1> vm = 0b0;
	bits<5> vs2;
	bits<5> rs1;
	bits<3> opv = 0b101;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = rs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFMERGE_VFM: Instruction, VFMERGE_VFM_FM, TSFlagTemplate<InstFormatR, VMConstraint>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, FPR32:$rs1, VMV0:$v0);
	let AsmString = "vfmerge.vfm	$vd, $vs2, $rs1, v0";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VREDSUM_VS_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000000;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VREDSUM_VS: Instruction, VREDSUM_VS_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vredsum.vs	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VWREDSUMU_VS_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b110000;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VWREDSUMU_VS: Instruction, VWREDSUMU_VS_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vwredsumu.vs	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VWREDSUM_VS_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b110001;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b000;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VWREDSUM_VS: Instruction, VWREDSUM_VS_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vwredsum.vs	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VFREDSUM_VS_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000001;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFREDSUM_VS: Instruction, VFREDSUM_VS_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vfredsum.vs	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFWREDSUM_VS_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b110001;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFWREDSUM_VS: Instruction, VFWREDSUM_VS_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vfwredsum.vs	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VCOMPRESS_VM_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b010111;
	bits<1> vm = 0b1;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b010;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VCOMPRESS_VM: Instruction, VCOMPRESS_VM_FM, TSFlagTemplate<InstFormatR, Vcompress>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1);
	let AsmString = "vcompress.vm	$vd, $vs2, $vs1";
	let Pattern = [];
	let Predicates = [HasStdExtV];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}
class VFREDOSUM_VS_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b000011;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFREDOSUM_VS: Instruction, VFREDOSUM_VS_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vfredosum.vs	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "";
}
class VFWREDOSUM_VS_FM: ATGInst{
	bits<32> SoftFail=0;
	bits<32> Inst;
	bits<6> funct6 = 0b110011;
	bits<1> vm;
	bits<5> vs2;
	bits<5> vs1;
	bits<3> opv = 0b001;
	bits<5> vd;
	bits<7> opcode = 0b1010111;
	let Inst{31-26} = funct6;
	let Inst{25-25} = vm;
	let Inst{24-20} = vs2;
	let Inst{19-15} = vs1;
	let Inst{14-12} = opv;
	let Inst{11-7} = vd;
	let Inst{6-0} = opcode;
}
def VFWREDOSUM_VS: Instruction, VFWREDOSUM_VS_FM, TSFlagTemplate<InstFormatR>{
	let Size = 4;
	let Namespace = "RISCV";
	let OutOperandList = (outs VR:$vd);
	let InOperandList = (ins VR:$vs2, VR:$vs1, VMaskOp:$vm);
	let AsmString = "vfwredosum.vs	$vd, $vs2, $vs1$vm";
	let Pattern = [];
	let Predicates = [HasStdExtV, HasStdExtF];
	let hasSideEffects = 0;
	let mayLoad = 0;
	let mayStore = 0;
	let Defs = [];
	let Uses = [VTYPE, VL];
	let DecoderNamespace = "";
	let Constraints = "@earlyclobber $vd";
}