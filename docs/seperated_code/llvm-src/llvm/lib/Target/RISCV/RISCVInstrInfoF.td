//===-- RISCVInstrInfoF.td - RISC-V 'F' instructions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard 'F',
// Single-Precision Floating-Point instruction set extension.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// RISC-V specific DAG Nodes.
//===----------------------------------------------------------------------===//

def SDT_RISCVFMV_W_X_RV64
    : SDTypeProfile<1, 1, [SDTCisVT<0, f32>, SDTCisVT<1, i64>]>;
def SDT_RISCVFMV_X_ANYEXTW_RV64
    : SDTypeProfile<1, 1, [SDTCisVT<0, i64>, SDTCisVT<1, f32>]>;

def riscv_fmv_w_x_rv64
    : SDNode<"RISCVISD::FMV_W_X_RV64", SDT_RISCVFMV_W_X_RV64>;
def riscv_fmv_x_anyextw_rv64
    : SDNode<"RISCVISD::FMV_X_ANYEXTW_RV64", SDT_RISCVFMV_X_ANYEXTW_RV64>;

//===----------------------------------------------------------------------===//
// Operand and SDNode transformation definitions.
//===----------------------------------------------------------------------===//

// Floating-point rounding mode

def FRMArg : AsmOperandClass {
  let Name = "FRMArg";
  let RenderMethod = "addFRMArgOperands";
  let DiagnosticType = "InvalidFRMArg";
}

def frmarg : Operand<XLenVT> {
  let ParserMatchClass = FRMArg;
  let PrintMethod = "printFRMArg";
  let DecoderMethod = "decodeFRMArg";
}

//===----------------------------------------------------------------------===//
// Instruction class templates
//===----------------------------------------------------------------------===//
include "gen/RISCVInstrInfoF2.td"

// class FPFMASDynFrmAlias<FPFMAS_rrr_frm Inst, string OpcodeStr>
//     : InstAlias<OpcodeStr#" $rd, $rs1, $rs2, $rs3",
//                 (Inst FPR32:$rd, FPR32:$rs1, FPR32:$rs2, FPR32:$rs3, 0b111)>;

// class FPALUSDynFrmAlias<FPALUS_rr_frm Inst, string OpcodeStr>
//     : InstAlias<OpcodeStr#" $rd, $rs1, $rs2",
//                 (Inst FPR32:$rd, FPR32:$rs1, FPR32:$rs2, 0b111)>;

// class FPUnaryOpDynFrmAlias<FPUnaryOp_r_frm Inst, string OpcodeStr,
//                            RegisterClass rdty, RegisterClass rs1ty>
//     : InstAlias<OpcodeStr#" $rd, $rs1",
//                 (Inst rdty:$rd, rs1ty:$rs1, 0b111)>;

class FPFMASDynFrmAlias<ATGInst Inst, string OpcodeStr>
    : InstAlias<OpcodeStr#" $rd, $rs1, $rs2, $rs3",
                (Inst FPR32:$rd, FPR32:$rs1, FPR32:$rs2, FPR32:$rs3, 0b111)>;

class FPALUSDynFrmAlias<ATGInst Inst, string OpcodeStr>
    : InstAlias<OpcodeStr#" $rd, $rs1, $rs2",
                (Inst FPR32:$rd, FPR32:$rs1, FPR32:$rs2, 0b111)>;

class FPUnaryOpDynFrmAlias<ATGInst Inst, string OpcodeStr,
                           RegisterClass rdty, RegisterClass rs1ty>
    : InstAlias<OpcodeStr#" $rd, $rs1",
                (Inst rdty:$rd, rs1ty:$rs1, 0b111)>;

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtF] in {

def          : FPFMASDynFrmAlias<FMADD_S, "fmadd.s">;
def          : FPFMASDynFrmAlias<FMSUB_S, "fmsub.s">;
def          : FPFMASDynFrmAlias<FNMSUB_S, "fnmsub.s">;
def          : FPFMASDynFrmAlias<FNMADD_S, "fnmadd.s">;

def        : FPALUSDynFrmAlias<FADD_S, "fadd.s">;
def        : FPALUSDynFrmAlias<FSUB_S, "fsub.s">;
def        : FPALUSDynFrmAlias<FMUL_S, "fmul.s">;
def        : FPALUSDynFrmAlias<FDIV_S, "fdiv.s">;

def         : FPUnaryOpDynFrmAlias<FSQRT_S, "fsqrt.s", FPR32, FPR32>;

def          : FPUnaryOpDynFrmAlias<FCVT_W_S, "fcvt.w.s", GPR, FPR32>;

def           : FPUnaryOpDynFrmAlias<FCVT_WU_S, "fcvt.wu.s", GPR, FPR32>;

def          : FPUnaryOpDynFrmAlias<FCVT_S_W, "fcvt.s.w", FPR32, GPR>;

def           : FPUnaryOpDynFrmAlias<FCVT_S_WU, "fcvt.s.wu", FPR32, GPR>;


} // Predicates = [HasStdExtF]

let Predicates = [HasStdExtF, IsRV64] in {

def           : FPUnaryOpDynFrmAlias<FCVT_L_S, "fcvt.l.s", GPR, FPR32>;

def            : FPUnaryOpDynFrmAlias<FCVT_LU_S, "fcvt.lu.s", GPR, FPR32>;

def          : FPUnaryOpDynFrmAlias<FCVT_S_L, "fcvt.s.l", FPR32, GPR>;

def           : FPUnaryOpDynFrmAlias<FCVT_S_LU, "fcvt.s.lu", FPR32, GPR>;
} // Predicates = [HasStdExtF, IsRV64]

//===----------------------------------------------------------------------===//
// Assembler Pseudo Instructions (User-Level ISA, Version 2.2, Chapter 20)
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtF] in {
def : InstAlias<"flw $rd, (${rs1})",  (FLW FPR32:$rd,  GPR:$rs1, 0), 0>;
def : InstAlias<"fsw $rs2, (${rs1})", (FSW FPR32:$rs2, GPR:$rs1, 0), 0>;

def : InstAlias<"fmv.s $rd, $rs",  (FSGNJ_S  FPR32:$rd, FPR32:$rs, FPR32:$rs)>;
def : InstAlias<"fabs.s $rd, $rs", (FSGNJX_S FPR32:$rd, FPR32:$rs, FPR32:$rs)>;
def : InstAlias<"fneg.s $rd, $rs", (FSGNJN_S FPR32:$rd, FPR32:$rs, FPR32:$rs)>;

// fgt.s/fge.s are recognised by the GNU assembler but the canonical
// flt.s/fle.s forms will always be printed. Therefore, set a zero weight.
def : InstAlias<"fgt.s $rd, $rs, $rt",
                (FLT_S GPR:$rd, FPR32:$rt, FPR32:$rs), 0>;
def : InstAlias<"fge.s $rd, $rs, $rt",
                (FLE_S GPR:$rd, FPR32:$rt, FPR32:$rs), 0>;

// The following csr instructions actually alias instructions from the base ISA.
// However, it only makes sense to support them when the F extension is enabled.
// NOTE: "frcsr", "frrm", and "frflags" are more specialized version of "csrr".
def : InstAlias<"frcsr $rd",      (CSRRS GPR:$rd, FCSR.Encoding, X0), 2>;
def : InstAlias<"fscsr $rd, $rs", (CSRRW GPR:$rd, FCSR.Encoding, GPR:$rs)>;
def : InstAlias<"fscsr $rs",      (CSRRW      X0, FCSR.Encoding, GPR:$rs), 2>;

// frsr, fssr are obsolete aliases replaced by frcsr, fscsr, so give them
// zero weight.
def : InstAlias<"frsr $rd",       (CSRRS GPR:$rd, FCSR.Encoding, X0), 0>;
def : InstAlias<"fssr $rd, $rs",  (CSRRW GPR:$rd, FCSR.Encoding, GPR:$rs), 0>;
def : InstAlias<"fssr $rs",       (CSRRW      X0, FCSR.Encoding, GPR:$rs), 0>;

def : InstAlias<"frrm $rd",        (CSRRS  GPR:$rd, FRM.Encoding, X0), 2>;
def : InstAlias<"fsrm $rd, $rs",   (CSRRW  GPR:$rd, FRM.Encoding, GPR:$rs)>;
def : InstAlias<"fsrm $rs",        (CSRRW       X0, FRM.Encoding, GPR:$rs), 2>;
def : InstAlias<"fsrmi $rd, $imm", (CSRRWI GPR:$rd, FRM.Encoding, uimm5:$imm)>;
def : InstAlias<"fsrmi $imm",      (CSRRWI      X0, FRM.Encoding, uimm5:$imm), 2>;

def : InstAlias<"frflags $rd",        (CSRRS  GPR:$rd, FFLAGS.Encoding, X0), 2>;
def : InstAlias<"fsflags $rd, $rs",   (CSRRW  GPR:$rd, FFLAGS.Encoding, GPR:$rs)>;
def : InstAlias<"fsflags $rs",        (CSRRW       X0, FFLAGS.Encoding, GPR:$rs), 2>;
def : InstAlias<"fsflagsi $rd, $imm", (CSRRWI GPR:$rd, FFLAGS.Encoding, uimm5:$imm)>;
def : InstAlias<"fsflagsi $imm",      (CSRRWI      X0, FFLAGS.Encoding, uimm5:$imm), 2>;

// fmv.w.x and fmv.x.w were previously known as fmv.s.x and fmv.x.s. Both
// spellings should be supported by standard tools.
def : MnemonicAlias<"fmv.s.x", "fmv.w.x">;
def : MnemonicAlias<"fmv.x.s", "fmv.x.w">;

def PseudoFLW  : PseudoFloatLoad<"flw", FPR32>;
def PseudoFSW  : PseudoStore<"fsw", FPR32>;
} // Predicates = [HasStdExtF]

//===----------------------------------------------------------------------===//
// Pseudo-instructions and codegen patterns
//===----------------------------------------------------------------------===//

/// Floating point constants
def fpimm0 : PatLeaf<(fpimm), [{ return N->isExactlyValue(+0.0); }]>;

/// Generic pattern classes
// class PatFpr32Fpr32<SDPatternOperator OpNode, RVInstR Inst>
class PatFpr32Fpr32<SDPatternOperator OpNode, ATGInst Inst>
    : Pat<(OpNode FPR32:$rs1, FPR32:$rs2), (Inst $rs1, $rs2)>;

// class PatFpr32Fpr32DynFrm<SDPatternOperator OpNode, RVInstRFrm Inst>
class PatFpr32Fpr32DynFrm<SDPatternOperator OpNode, ATGInst Inst>
    : Pat<(OpNode FPR32:$rs1, FPR32:$rs2), (Inst $rs1, $rs2, 0b111)>;

let Predicates = [HasStdExtF] in {

/// Float constants
def : Pat<(f32 (fpimm0)), (FMV_W_X X0)>;

/// Float conversion operations

// [u]int32<->float conversion patterns must be gated on IsRV32 or IsRV64, so
// are defined later.

/// Float arithmetic operations

def : PatFpr32Fpr32DynFrm<fadd, FADD_S>;
def : PatFpr32Fpr32DynFrm<fsub, FSUB_S>;
def : PatFpr32Fpr32DynFrm<fmul, FMUL_S>;
def : PatFpr32Fpr32DynFrm<fdiv, FDIV_S>;

def : Pat<(fsqrt FPR32:$rs1), (FSQRT_S FPR32:$rs1, 0b111)>;

def : Pat<(fneg FPR32:$rs1), (FSGNJN_S $rs1, $rs1)>;
def : Pat<(fabs FPR32:$rs1), (FSGNJX_S $rs1, $rs1)>;

def : PatFpr32Fpr32<fcopysign, FSGNJ_S>;
def : Pat<(fcopysign FPR32:$rs1, (fneg FPR32:$rs2)), (FSGNJN_S $rs1, $rs2)>;

// fmadd: rs1 * rs2 + rs3
def : Pat<(fma FPR32:$rs1, FPR32:$rs2, FPR32:$rs3),
          (FMADD_S $rs1, $rs2, $rs3, 0b111)>;

// fmsub: rs1 * rs2 - rs3
def : Pat<(fma FPR32:$rs1, FPR32:$rs2, (fneg FPR32:$rs3)),
          (FMSUB_S FPR32:$rs1, FPR32:$rs2, FPR32:$rs3, 0b111)>;

// fnmsub: -rs1 * rs2 + rs3
def : Pat<(fma (fneg FPR32:$rs1), FPR32:$rs2, FPR32:$rs3),
          (FNMSUB_S FPR32:$rs1, FPR32:$rs2, FPR32:$rs3, 0b111)>;

// fnmadd: -rs1 * rs2 - rs3
def : Pat<(fma (fneg FPR32:$rs1), FPR32:$rs2, (fneg FPR32:$rs3)),
          (FNMADD_S FPR32:$rs1, FPR32:$rs2, FPR32:$rs3, 0b111)>;

// The RISC-V 2.2 user-level ISA spec defines fmin and fmax as returning the
// canonical NaN when given a signaling NaN. This doesn't match the LLVM
// behaviour (see https://bugs.llvm.org/show_bug.cgi?id=27363). However, the
// draft 2.3 ISA spec changes the definition of fmin and fmax in a way that
// matches LLVM's fminnum and fmaxnum
// <https://github.com/riscv/riscv-isa-manual/commit/cd20cee7efd9bac7c5aa127ec3b451749d2b3cce>.
def : PatFpr32Fpr32<fminnum, FMIN_S>;
def : PatFpr32Fpr32<fmaxnum, FMAX_S>;

/// Setcc

def : PatFpr32Fpr32<seteq, FEQ_S>;
def : PatFpr32Fpr32<setoeq, FEQ_S>;
def : PatFpr32Fpr32<setlt, FLT_S>;
def : PatFpr32Fpr32<setolt, FLT_S>;
def : PatFpr32Fpr32<setle, FLE_S>;
def : PatFpr32Fpr32<setole, FLE_S>;

def Select_FPR32_Using_CC_GPR : SelectCC_rrirr<FPR32, GPR>;

/// Loads

defm : LdPat<load, FLW>;

/// Stores

defm : StPat<store, FSW, FPR32>;

} // Predicates = [HasStdExtF]

let Predicates = [HasStdExtF, IsRV32] in {
// Moves (no conversion)
def : Pat<(bitconvert GPR:$rs1), (FMV_W_X GPR:$rs1)>;
def : Pat<(bitconvert FPR32:$rs1), (FMV_X_W FPR32:$rs1)>;

// float->[u]int. Round-to-zero must be used.
def : Pat<(fp_to_sint FPR32:$rs1), (FCVT_W_S $rs1, 0b001)>;
def : Pat<(fp_to_uint FPR32:$rs1), (FCVT_WU_S $rs1, 0b001)>;

// [u]int->float. Match GCC and default to using dynamic rounding mode.
def : Pat<(sint_to_fp GPR:$rs1), (FCVT_S_W $rs1, 0b111)>;
def : Pat<(uint_to_fp GPR:$rs1), (FCVT_S_WU $rs1, 0b111)>;
} // Predicates = [HasStdExtF, IsRV32]

let Predicates = [HasStdExtF, IsRV64] in {
// Moves (no conversion)
def : Pat<(riscv_fmv_w_x_rv64 GPR:$src), (FMV_W_X GPR:$src)>;
def : Pat<(riscv_fmv_x_anyextw_rv64 FPR32:$src), (FMV_X_W FPR32:$src)>;
def : Pat<(sext_inreg (riscv_fmv_x_anyextw_rv64 FPR32:$src), i32),
          (FMV_X_W FPR32:$src)>;

// FP->[u]int32 is mostly handled by the FP->[u]int64 patterns. This is safe
// because fpto[u|s]i produces poison if the value can't fit into the target.
// We match the single case below because fcvt.wu.s sign-extends its result so
// is cheaper than fcvt.lu.s+sext.w.
def : Pat<(sext_inreg (assertzexti32 (fp_to_uint FPR32:$rs1)), i32),
          (FCVT_WU_S $rs1, 0b001)>;

// FP->[u]int64
def : Pat<(fp_to_sint FPR32:$rs1), (FCVT_L_S $rs1, 0b001)>;
def : Pat<(fp_to_uint FPR32:$rs1), (FCVT_LU_S $rs1, 0b001)>;

// [u]int->fp. Match GCC and default to using dynamic rounding mode.
def : Pat<(sint_to_fp (sexti32 GPR:$rs1)), (FCVT_S_W $rs1, 0b111)>;
def : Pat<(uint_to_fp (zexti32 GPR:$rs1)), (FCVT_S_WU $rs1, 0b111)>;
def : Pat<(sint_to_fp GPR:$rs1), (FCVT_S_L $rs1, 0b111)>;
def : Pat<(uint_to_fp GPR:$rs1), (FCVT_S_LU $rs1, 0b111)>;
} // Predicates = [HasStdExtF, IsRV64]
