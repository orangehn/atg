//===-- RISCVInstrInfoV.td - RISC-V 'V' instructions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file describes the RISC-V instructions from the standard 'V' Vector
/// extension, version 0.10.
/// This version is still experimental as the 'V' extension hasn't been
/// ratified yet.
///
//===----------------------------------------------------------------------===//

include "RISCVInstrFormatsV.td"

//===----------------------------------------------------------------------===//
// Operand and SDNode transformation definitions.
//===----------------------------------------------------------------------===//

def VTypeIAsmOperand : AsmOperandClass {
  let Name = "VTypeI";
  let ParserMethod = "parseVTypeI";
  let DiagnosticType = "InvalidVTypeI";
}

def VTypeIOp : Operand<XLenVT> {
  let ParserMatchClass = VTypeIAsmOperand;
  let PrintMethod = "printVTypeI";
  let DecoderMethod = "decodeUImmOperand<11>";
}

def VMaskAsmOperand : AsmOperandClass {
  let Name = "RVVMaskRegOpOperand";
  let RenderMethod = "addRegOperands";
  let PredicateMethod = "isV0Reg";
  let ParserMethod = "parseMaskReg";
  let IsOptional = 1;
  let DefaultMethod = "defaultMaskRegOp";
  let DiagnosticType = "InvalidVMaskRegister";
}

def VMaskOp : RegisterOperand<VMV0> {
  let ParserMatchClass = VMaskAsmOperand;
  let PrintMethod = "printVMaskReg";
  let EncoderMethod = "getVMaskReg";
  let DecoderMethod = "decodeVMaskReg";
}

def simm5 : Operand<XLenVT>, ImmLeaf<XLenVT, [{return isInt<5>(Imm);}]> {
  let ParserMatchClass = SImmAsmOperand<5>;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeSImmOperand<5>";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (MCOp.evaluateAsConstantImm(Imm))
      return isInt<5>(Imm);
    return MCOp.isBareSymbolRef();
  }];
}

def SImm5Plus1AsmOperand : AsmOperandClass {
  let Name = "SImm5Plus1";
  let RenderMethod = "addImmOperands";
  let DiagnosticType = "InvalidSImm5Plus1";
}

def simm5_plus1 : Operand<XLenVT>, ImmLeaf<XLenVT,
                                           [{return isInt<5>(Imm - 1);}]> {
  let ParserMatchClass = SImm5Plus1AsmOperand;
  let MCOperandPredicate = [{
    int64_t Imm;
    if (MCOp.evaluateAsConstantImm(Imm))
      return isInt<5>(Imm - 1);
    return MCOp.isBareSymbolRef();
  }];
}

//===----------------------------------------------------------------------===//
// Instruction class templates
//===----------------------------------------------------------------------===//
let hasSideEffects = 0, mayLoad = 1, mayStore = 1 in {
// vamo vd, (rs1), vs2, vd, vm
class VAMOWd<RISCVAMOOP amoop, RISCVWidth width, string opcodestr>
    : RVInstVAMO<amoop, width.Value{2-0}, (outs VR:$vd_wd),
            (ins GPR:$rs1, VR:$vs2, VR:$vd, VMaskOp:$vm),
            opcodestr, "$vd_wd, (${rs1}), $vs2, $vd$vm"> {
    let Constraints = "$vd_wd = $vd";
    let wd = 1;
    bits<5> vd;
    let Inst{11-7} = vd;
}

// vamo x0, (rs1), vs2, vs3, vm
class VAMONoWd<RISCVAMOOP amoop, RISCVWidth width, string opcodestr>
    : RVInstVAMO<amoop, width.Value{2-0}, (outs),
            (ins GPR:$rs1, VR:$vs2, VR:$vs3, VMaskOp:$vm),
            opcodestr, "x0, (${rs1}), $vs2, $vs3$vm"> {
    bits<5> vs3;
    let Inst{11-7} = vs3;
}

} // hasSideEffects = 0, mayLoad = 1, mayStore = 1
multiclass VAMO<RISCVAMOOP amoop, RISCVWidth width, string opcodestr> {
  def _WD : VAMOWd<amoop, width, opcodestr>;
  def _UNWD : VAMONoWd<amoop, width, opcodestr>;
}

include "gen/RISCVInstrInfoV2.td"


//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtV] in {

def : InstAlias<"vl1r.v $vd, (${rs1})", (VL1RE8_V VR:$vd, GPR:$rs1)>;
def : InstAlias<"vl2r.v $vd, (${rs1})", (VL2RE8_V VR:$vd, GPR:$rs1)>;
def : InstAlias<"vl4r.v $vd, (${rs1})", (VL4RE8_V VR:$vd, GPR:$rs1)>;
def : InstAlias<"vl8r.v $vd, (${rs1})", (VL8RE8_V VR:$vd, GPR:$rs1)>;

def : InstAlias<"vneg.v $vd, $vs$vm", (VRSUB_VX VR:$vd, VR:$vs, X0, VMaskOp:$vm)>;

def : InstAlias<"vwcvt.x.x.v $vd, $vs$vm",
                (VWADD_VX VR:$vd, VR:$vs, X0, VMaskOp:$vm)>;
def : InstAlias<"vwcvtu.x.x.v $vd, $vs$vm",
                (VWADDU_VX VR:$vd, VR:$vs, X0, VMaskOp:$vm)>;

def : InstAlias<"vnot.v $vd, $vs$vm",
                (VXOR_VI VR:$vd, VR:$vs, -1, VMaskOp:$vm)>;

def : InstAlias<"vncvt.x.x.w $vd, $vs$vm",
                (VNSRL_WX VR:$vd, VR:$vs, X0, VMaskOp:$vm)>;

def : InstAlias<"vmsgtu.vv $vd, $va, $vb$vm",
                (VMSLTU_VV VR:$vd, VR:$vb, VR:$va, VMaskOp:$vm), 0>;
def : InstAlias<"vmsgt.vv $vd, $va, $vb$vm",
                (VMSLT_VV VR:$vd, VR:$vb, VR:$va, VMaskOp:$vm), 0>;
def : InstAlias<"vmsgeu.vv $vd, $va, $vb$vm",
                (VMSLEU_VV VR:$vd, VR:$vb, VR:$va, VMaskOp:$vm), 0>;
def : InstAlias<"vmsge.vv $vd, $va, $vb$vm",
                (VMSLE_VV VR:$vd, VR:$vb, VR:$va, VMaskOp:$vm), 0>;

let isCodeGenOnly = 0, isAsmParserOnly = 1, hasSideEffects = 0, mayLoad = 0,
    mayStore = 0 in {
// For unsigned comparisons we need to special case 0 immediate to maintain
// the always true/false semantics we would invert if we just decremented the
// immediate like we do for signed. To match the GNU assembler we will use
// vmseq/vmsne.vv with the same register for both operands which we can't do
// from an InstAlias.
def PseudoVMSGEU_VI : Pseudo<(outs VR:$vd),
                             (ins VR:$vs2, simm5_plus1:$imm, VMaskOp:$vm),
                             [], "vmsgeu.vi", "$vd, $vs2, $imm$vm">;
def PseudoVMSLTU_VI : Pseudo<(outs VR:$vd),
                             (ins VR:$vs2, simm5_plus1:$imm, VMaskOp:$vm),
                             [], "vmsltu.vi", "$vd, $vs2, $imm$vm">;
// Handle signed with pseudos as well for more consistency in the
// implementation.
def PseudoVMSGE_VI : Pseudo<(outs VR:$vd),
                            (ins VR:$vs2, simm5_plus1:$imm, VMaskOp:$vm),
                            [], "vmsge.vi", "$vd, $vs2, $imm$vm">;
def PseudoVMSLT_VI : Pseudo<(outs VR:$vd),
                            (ins VR:$vs2, simm5_plus1:$imm, VMaskOp:$vm),
                            [], "vmslt.vi", "$vd, $vs2, $imm$vm">;
}

let isCodeGenOnly = 0, isAsmParserOnly = 1, hasSideEffects = 0, mayLoad = 0,
    mayStore = 0 in {
def PseudoVMSGEU_VX : Pseudo<(outs VR:$vd),
                             (ins VR:$vs2, GPR:$rs1),
                             [], "vmsgeu.vx", "$vd, $vs2, $rs1">;
def PseudoVMSGE_VX : Pseudo<(outs VR:$vd),
                            (ins VR:$vs2, GPR:$rs1),
                            [], "vmsge.vx", "$vd, $vs2, $rs1">;
def PseudoVMSGEU_VX_M : Pseudo<(outs VRNoV0:$vd),
                               (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm),
                               [], "vmsgeu.vx", "$vd, $vs2, $rs1$vm">;
def PseudoVMSGE_VX_M : Pseudo<(outs VRNoV0:$vd),
                              (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm),
                              [], "vmsge.vx", "$vd, $vs2, $rs1$vm">;
def PseudoVMSGEU_VX_M_T : Pseudo<(outs VMV0:$vd, VR:$scratch),
                                 (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm),
                                 [], "vmsgeu.vx", "$vd, $vs2, $rs1$vm, $scratch">;
def PseudoVMSGE_VX_M_T : Pseudo<(outs VMV0:$vd, VR:$scratch),
                                (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm),
                                [], "vmsge.vx", "$vd, $vs2, $rs1$vm, $scratch">;
}

} // Predicates = [HasStdExtV]

let Predicates = [HasStdExtV, HasStdExtF] in {
def : InstAlias<"vfneg.v $vd, $vs$vm",
                (VFSGNJN_VV VR:$vd, VR:$vs, VR:$vs, VMaskOp:$vm)>;

def : InstAlias<"vmfgt.vv $vd, $va, $vb$vm",
                (VMFLT_VV VR:$vd, VR:$vb, VR:$va, VMaskOp:$vm), 0>;
def : InstAlias<"vmfge.vv $vd, $va, $vb$vm",
                (VMFLE_VV VR:$vd, VR:$vb, VR:$va, VMaskOp:$vm), 0>;
} // Predicates = [HasStdExtV, HasStdExtF]

let Predicates = [HasStdExtV] in {

def : InstAlias<"vmmv.m $vd, $vs",
                (VMAND_MM VR:$vd, VR:$vs, VR:$vs)>;
def : InstAlias<"vmclr.m $vd",
                (VMXOR_MM VR:$vd, VR:$vd, VR:$vd)>;
def : InstAlias<"vmset.m $vd",
                (VMXNOR_MM VR:$vd, VR:$vd, VR:$vd)>;
def : InstAlias<"vmnot.m $vd, $vs",
                (VMNAND_MM VR:$vd, VR:$vs, VR:$vs)>;
} // Predicates = [HasStdExtV]

// not in instruction manuel
let Predicates = [HasStdExtZvamo, HasStdExtA] in {
  defm VAMOSWAPEI8 : VAMO<AMOOPVamoSwap, LSWidth8, "vamoswapei8.v">;
  defm VAMOSWAPEI16 : VAMO<AMOOPVamoSwap, LSWidth16, "vamoswapei16.v">;
  defm VAMOSWAPEI32 : VAMO<AMOOPVamoSwap, LSWidth32, "vamoswapei32.v">;

  defm VAMOADDEI8 : VAMO<AMOOPVamoAdd, LSWidth8, "vamoaddei8.v">;
  defm VAMOADDEI16 : VAMO<AMOOPVamoAdd, LSWidth16, "vamoaddei16.v">;
  defm VAMOADDEI32 : VAMO<AMOOPVamoAdd, LSWidth32, "vamoaddei32.v">;

  defm VAMOXOREI8 : VAMO<AMOOPVamoXor, LSWidth8, "vamoxorei8.v">;
  defm VAMOXOREI16 : VAMO<AMOOPVamoXor, LSWidth16, "vamoxorei16.v">;
  defm VAMOXOREI32 : VAMO<AMOOPVamoXor, LSWidth32, "vamoxorei32.v">;

  defm VAMOANDEI8 : VAMO<AMOOPVamoAnd, LSWidth8, "vamoandei8.v">;
  defm VAMOANDEI16 : VAMO<AMOOPVamoAnd, LSWidth16, "vamoandei16.v">;
  defm VAMOANDEI32 : VAMO<AMOOPVamoAnd, LSWidth32, "vamoandei32.v">;

  defm VAMOOREI8 : VAMO<AMOOPVamoOr, LSWidth8, "vamoorei8.v">;
  defm VAMOOREI16 : VAMO<AMOOPVamoOr, LSWidth16, "vamoorei16.v">;
  defm VAMOOREI32 : VAMO<AMOOPVamoOr, LSWidth32, "vamoorei32.v">;

  defm VAMOMINEI8 : VAMO<AMOOPVamoMin, LSWidth8, "vamominei8.v">;
  defm VAMOMINEI16 : VAMO<AMOOPVamoMin, LSWidth16, "vamominei16.v">;
  defm VAMOMINEI32 : VAMO<AMOOPVamoMin, LSWidth32, "vamominei32.v">;

  defm VAMOMAXEI8 : VAMO<AMOOPVamoMax, LSWidth8, "vamomaxei8.v">;
  defm VAMOMAXEI16 : VAMO<AMOOPVamoMax, LSWidth16, "vamomaxei16.v">;
  defm VAMOMAXEI32 : VAMO<AMOOPVamoMax, LSWidth32, "vamomaxei32.v">;

  defm VAMOMINUEI8 : VAMO<AMOOPVamoMinu, LSWidth8, "vamominuei8.v">;
  defm VAMOMINUEI16 : VAMO<AMOOPVamoMinu, LSWidth16, "vamominuei16.v">;
  defm VAMOMINUEI32 : VAMO<AMOOPVamoMinu, LSWidth32, "vamominuei32.v">;

  defm VAMOMAXUEI8 : VAMO<AMOOPVamoMaxu, LSWidth8, "vamomaxuei8.v">;
  defm VAMOMAXUEI16 : VAMO<AMOOPVamoMaxu, LSWidth16, "vamomaxuei16.v">;
  defm VAMOMAXUEI32 : VAMO<AMOOPVamoMaxu, LSWidth32, "vamomaxuei32.v">;
} // Predicates = [HasStdExtZvamo, HasStdExtA]

let Predicates = [HasStdExtZvamo, HasStdExtA, IsRV64] in {
  defm VAMOSWAPEI64 : VAMO<AMOOPVamoSwap, LSWidth64, "vamoswapei64.v">;
  defm VAMOADDEI64 : VAMO<AMOOPVamoAdd, LSWidth64, "vamoaddei64.v">;
  defm VAMOXOREI64 : VAMO<AMOOPVamoXor, LSWidth64, "vamoxorei64.v">;
  defm VAMOANDEI64 : VAMO<AMOOPVamoAnd, LSWidth64, "vamoandei64.v">;
  defm VAMOOREI64 : VAMO<AMOOPVamoOr, LSWidth64, "vamoorei64.v">;
  defm VAMOMINEI64 : VAMO<AMOOPVamoMin, LSWidth64, "vamominei64.v">;
  defm VAMOMAXEI64 : VAMO<AMOOPVamoMax, LSWidth64, "vamomaxei64.v">;
  defm VAMOMINUEI64 : VAMO<AMOOPVamoMinu, LSWidth64, "vamominuei64.v">;
  defm VAMOMAXUEI64 : VAMO<AMOOPVamoMaxu, LSWidth64, "vamomaxuei64.v">;
} // Predicates = [HasStdExtZvamo, HasStdExtA, IsRV64]

include "RISCVInstrInfoVPseudos.td"
