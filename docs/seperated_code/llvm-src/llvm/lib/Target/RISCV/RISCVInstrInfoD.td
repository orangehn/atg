//===-- RISCVInstrInfoD.td - RISC-V 'D' instructions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard 'D',
// Double-Precision Floating-Point instruction set extension.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// RISC-V specific DAG Nodes.
//===----------------------------------------------------------------------===//

def SDT_RISCVBuildPairF64 : SDTypeProfile<1, 2, [SDTCisVT<0, f64>,
                                                 SDTCisVT<1, i32>,
                                                 SDTCisSameAs<1, 2>]>;
def SDT_RISCVSplitF64     : SDTypeProfile<2, 1, [SDTCisVT<0, i32>,
                                                 SDTCisVT<1, i32>,
                                                 SDTCisVT<2, f64>]>;

def RISCVBuildPairF64 : SDNode<"RISCVISD::BuildPairF64", SDT_RISCVBuildPairF64>;
def RISCVSplitF64     : SDNode<"RISCVISD::SplitF64", SDT_RISCVSplitF64>;

//===----------------------------------------------------------------------===//
// Instruction Class Templates
//===----------------------------------------------------------------------===//

include "gen/RISCVInstrInfoD2.td"

// class FPFMADDynFrmAlias<FPFMAD_rrr_frm Inst, string OpcodeStr>
//     : InstAlias<OpcodeStr#" $rd, $rs1, $rs2, $rs3",
//                 (Inst FPR64:$rd, FPR64:$rs1, FPR64:$rs2, FPR64:$rs3, 0b111)>;

// class FPALUDDynFrmAlias<FPALUD_rr_frm Inst, string OpcodeStr>
//     : InstAlias<OpcodeStr#" $rd, $rs1, $rs2",
//                 (Inst FPR64:$rd, FPR64:$rs1, FPR64:$rs2, 0b111)>;

class FPFMADDynFrmAlias<ATGInst Inst, string OpcodeStr>
    : InstAlias<OpcodeStr#" $rd, $rs1, $rs2, $rs3",
                (Inst FPR64:$rd, FPR64:$rs1, FPR64:$rs2, FPR64:$rs3, 0b111)>;

class FPALUDDynFrmAlias<ATGInst Inst, string OpcodeStr>
    : InstAlias<OpcodeStr#" $rd, $rs1, $rs2",
                (Inst FPR64:$rd, FPR64:$rs1, FPR64:$rs2, 0b111)>;


//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtD] in {

def          : FPFMADDynFrmAlias<FMADD_D, "fmadd.d">;
def          : FPFMADDynFrmAlias<FMSUB_D, "fmsub.d">;
def          : FPFMADDynFrmAlias<FNMSUB_D, "fnmsub.d">;
def          : FPFMADDynFrmAlias<FNMADD_D, "fnmadd.d">;
def        : FPALUDDynFrmAlias<FADD_D, "fadd.d">;
def        : FPALUDDynFrmAlias<FSUB_D, "fsub.d">;
def        : FPALUDDynFrmAlias<FMUL_D, "fmul.d">;
def        : FPALUDDynFrmAlias<FDIV_D, "fdiv.d">;
def         : FPUnaryOpDynFrmAlias<FSQRT_D, "fsqrt.d", FPR64, FPR64>;

def          : FPUnaryOpDynFrmAlias<FCVT_S_D, "fcvt.s.d", FPR32, FPR64>;

def          : FPUnaryOpDynFrmAlias<FCVT_W_D, "fcvt.w.d", GPR, FPR64>;

def           : FPUnaryOpDynFrmAlias<FCVT_WU_D, "fcvt.wu.d", GPR, FPR64>;
} // Predicates = [HasStdExtD]

let Predicates = [HasStdExtD, IsRV64] in {
def          : FPUnaryOpDynFrmAlias<FCVT_L_D, "fcvt.l.d", GPR, FPR64>;
def           : FPUnaryOpDynFrmAlias<FCVT_LU_D, "fcvt.lu.d", GPR, FPR64>;
def          : FPUnaryOpDynFrmAlias<FCVT_D_L, "fcvt.d.l", FPR64, GPR>;
def           : FPUnaryOpDynFrmAlias<FCVT_D_LU, "fcvt.d.lu", FPR64, GPR>;
} // Predicates = [HasStdExtD, IsRV64]

//===----------------------------------------------------------------------===//
// Assembler Pseudo Instructions (User-Level ISA, Version 2.2, Chapter 20)
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtD] in {
def : InstAlias<"fld $rd, (${rs1})",  (FLD FPR64:$rd,  GPR:$rs1, 0), 0>;
def : InstAlias<"fsd $rs2, (${rs1})", (FSD FPR64:$rs2, GPR:$rs1, 0), 0>;

def : InstAlias<"fmv.d $rd, $rs",  (FSGNJ_D  FPR64:$rd, FPR64:$rs, FPR64:$rs)>;
def : InstAlias<"fabs.d $rd, $rs", (FSGNJX_D FPR64:$rd, FPR64:$rs, FPR64:$rs)>;
def : InstAlias<"fneg.d $rd, $rs", (FSGNJN_D FPR64:$rd, FPR64:$rs, FPR64:$rs)>;

// fgt.d/fge.d are recognised by the GNU assembler but the canonical
// flt.d/fle.d forms will always be printed. Therefore, set a zero weight.
def : InstAlias<"fgt.d $rd, $rs, $rt",
                (FLT_D GPR:$rd, FPR64:$rt, FPR64:$rs), 0>;
def : InstAlias<"fge.d $rd, $rs, $rt",
                (FLE_D GPR:$rd, FPR64:$rt, FPR64:$rs), 0>;

def PseudoFLD  : PseudoFloatLoad<"fld", FPR64>;
def PseudoFSD  : PseudoStore<"fsd", FPR64>;
} // Predicates = [HasStdExtD]

//===----------------------------------------------------------------------===//
// Pseudo-instructions and codegen patterns
//===----------------------------------------------------------------------===//

// class PatFpr64Fpr64<SDPatternOperator OpNode, RVInstR Inst>
//     : Pat<(OpNode FPR64:$rs1, FPR64:$rs2), (Inst $rs1, $rs2)>;

// class PatFpr64Fpr64DynFrm<SDPatternOperator OpNode, RVInstRFrm Inst>
//     : Pat<(OpNode FPR64:$rs1, FPR64:$rs2), (Inst $rs1, $rs2, 0b111)>;

class PatFpr64Fpr64<SDPatternOperator OpNode, ATGInst Inst>
    : Pat<(OpNode FPR64:$rs1, FPR64:$rs2), (Inst $rs1, $rs2)>;

class PatFpr64Fpr64DynFrm<SDPatternOperator OpNode, ATGInst Inst>
    : Pat<(OpNode FPR64:$rs1, FPR64:$rs2), (Inst $rs1, $rs2, 0b111)>;

let Predicates = [HasStdExtD] in {

/// Float conversion operations

// f64 -> f32, f32 -> f64
def : Pat<(fpround FPR64:$rs1), (FCVT_S_D FPR64:$rs1, 0b111)>;
def : Pat<(fpextend FPR32:$rs1), (FCVT_D_S FPR32:$rs1)>;

// [u]int<->double conversion patterns must be gated on IsRV32 or IsRV64, so
// are defined later.

/// Float arithmetic operations

def : PatFpr64Fpr64DynFrm<fadd, FADD_D>;
def : PatFpr64Fpr64DynFrm<fsub, FSUB_D>;
def : PatFpr64Fpr64DynFrm<fmul, FMUL_D>;
def : PatFpr64Fpr64DynFrm<fdiv, FDIV_D>;

def : Pat<(fsqrt FPR64:$rs1), (FSQRT_D FPR64:$rs1, 0b111)>;

def : Pat<(fneg FPR64:$rs1), (FSGNJN_D $rs1, $rs1)>;
def : Pat<(fabs FPR64:$rs1), (FSGNJX_D $rs1, $rs1)>;

def : PatFpr64Fpr64<fcopysign, FSGNJ_D>;
def : Pat<(fcopysign FPR64:$rs1, (fneg FPR64:$rs2)), (FSGNJN_D $rs1, $rs2)>;
def : Pat<(fcopysign FPR64:$rs1, FPR32:$rs2), (FSGNJ_D $rs1, (FCVT_D_S $rs2))>;
def : Pat<(fcopysign FPR32:$rs1, FPR64:$rs2), (FSGNJ_S $rs1, (FCVT_S_D $rs2,
                                                              0b111))>;

// fmadd: rs1 * rs2 + rs3
def : Pat<(fma FPR64:$rs1, FPR64:$rs2, FPR64:$rs3),
          (FMADD_D $rs1, $rs2, $rs3, 0b111)>;

// fmsub: rs1 * rs2 - rs3
def : Pat<(fma FPR64:$rs1, FPR64:$rs2, (fneg FPR64:$rs3)),
          (FMSUB_D FPR64:$rs1, FPR64:$rs2, FPR64:$rs3, 0b111)>;

// fnmsub: -rs1 * rs2 + rs3
def : Pat<(fma (fneg FPR64:$rs1), FPR64:$rs2, FPR64:$rs3),
          (FNMSUB_D FPR64:$rs1, FPR64:$rs2, FPR64:$rs3, 0b111)>;

// fnmadd: -rs1 * rs2 - rs3
def : Pat<(fma (fneg FPR64:$rs1), FPR64:$rs2, (fneg FPR64:$rs3)),
          (FNMADD_D FPR64:$rs1, FPR64:$rs2, FPR64:$rs3, 0b111)>;

// The RISC-V 2.2 user-level ISA spec defines fmin and fmax as returning the
// canonical NaN when giving a signaling NaN. This doesn't match the LLVM
// behaviour (see https://bugs.llvm.org/show_bug.cgi?id=27363). However, the
// draft 2.3 ISA spec changes the definition of fmin and fmax in a way that
// matches LLVM's fminnum and fmaxnum
// <https://github.com/riscv/riscv-isa-manual/commit/cd20cee7efd9bac7c5aa127ec3b451749d2b3cce>.
def : PatFpr64Fpr64<fminnum, FMIN_D>;
def : PatFpr64Fpr64<fmaxnum, FMAX_D>;

/// Setcc

def : PatFpr64Fpr64<seteq, FEQ_D>;
def : PatFpr64Fpr64<setoeq, FEQ_D>;
def : PatFpr64Fpr64<setlt, FLT_D>;
def : PatFpr64Fpr64<setolt, FLT_D>;
def : PatFpr64Fpr64<setle, FLE_D>;
def : PatFpr64Fpr64<setole, FLE_D>;

def Select_FPR64_Using_CC_GPR : SelectCC_rrirr<FPR64, GPR>;

/// Loads

defm : LdPat<load, FLD>;

/// Stores

defm : StPat<store, FSD, FPR64>;

/// Pseudo-instructions needed for the soft-float ABI with RV32D

// Moves two GPRs to an FPR.
let usesCustomInserter = 1 in
def BuildPairF64Pseudo
    : Pseudo<(outs FPR64:$dst), (ins GPR:$src1, GPR:$src2),
             [(set FPR64:$dst, (RISCVBuildPairF64 GPR:$src1, GPR:$src2))]>;

// Moves an FPR to two GPRs.
let usesCustomInserter = 1 in
def SplitF64Pseudo
    : Pseudo<(outs GPR:$dst1, GPR:$dst2), (ins FPR64:$src),
             [(set GPR:$dst1, GPR:$dst2, (RISCVSplitF64 FPR64:$src))]>;

} // Predicates = [HasStdExtD]

let Predicates = [HasStdExtD, IsRV32] in {

/// Float constants
def : Pat<(f64 (fpimm0)), (FCVT_D_W X0)>;

// double->[u]int. Round-to-zero must be used.
def : Pat<(fp_to_sint FPR64:$rs1), (FCVT_W_D FPR64:$rs1, 0b001)>;
def : Pat<(fp_to_uint FPR64:$rs1), (FCVT_WU_D FPR64:$rs1, 0b001)>;

// [u]int->double.
def : Pat<(sint_to_fp GPR:$rs1), (FCVT_D_W GPR:$rs1)>;
def : Pat<(uint_to_fp GPR:$rs1), (FCVT_D_WU GPR:$rs1)>;
} // Predicates = [HasStdExtD, IsRV32]

let Predicates = [HasStdExtD, IsRV64] in {

/// Float constants
def : Pat<(f64 (fpimm0)), (FMV_D_X X0)>;

// Moves (no conversion)
def : Pat<(bitconvert GPR:$rs1), (FMV_D_X GPR:$rs1)>;
def : Pat<(bitconvert FPR64:$rs1), (FMV_X_D FPR64:$rs1)>;

// FP->[u]int32 is mostly handled by the FP->[u]int64 patterns. This is safe
// because fpto[u|s]i produce poison if the value can't fit into the target.
// We match the single case below because fcvt.wu.d sign-extends its result so
// is cheaper than fcvt.lu.d+sext.w.
def : Pat<(sext_inreg (assertzexti32 (fp_to_uint FPR64:$rs1)), i32),
          (FCVT_WU_D $rs1, 0b001)>;

// [u]int32->fp
def : Pat<(sint_to_fp (sexti32 GPR:$rs1)), (FCVT_D_W $rs1)>;
def : Pat<(uint_to_fp (zexti32 GPR:$rs1)), (FCVT_D_WU $rs1)>;

def : Pat<(fp_to_sint FPR64:$rs1), (FCVT_L_D FPR64:$rs1, 0b001)>;
def : Pat<(fp_to_uint FPR64:$rs1), (FCVT_LU_D FPR64:$rs1, 0b001)>;

// [u]int64->fp. Match GCC and default to using dynamic rounding mode.
def : Pat<(sint_to_fp GPR:$rs1), (FCVT_D_L GPR:$rs1, 0b111)>;
def : Pat<(uint_to_fp GPR:$rs1), (FCVT_D_LU GPR:$rs1, 0b111)>;
} // Predicates = [HasStdExtD, IsRV64]
