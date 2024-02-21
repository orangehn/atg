//===-- RISCVInstrInfoFH.td - RISC-V 'FH' instructions -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard 'Zfh'
// half-precision floating-point extension, version 0.1.
// This version is still experimental as the 'Zfh' extension hasn't been
// ratified yet.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// RISC-V specific DAG Nodes.
//===----------------------------------------------------------------------===//

def SDT_RISCVFMV_H_X
    : SDTypeProfile<1, 1, [SDTCisVT<0, f16>, SDTCisVT<1, XLenVT>]>;
def SDT_RISCVFMV_X_ANYEXTH
    : SDTypeProfile<1, 1, [SDTCisVT<0, XLenVT>, SDTCisVT<1, f16>]>;

def riscv_fmv_h_x
    : SDNode<"RISCVISD::FMV_H_X", SDT_RISCVFMV_H_X>;
def riscv_fmv_x_anyexth
    : SDNode<"RISCVISD::FMV_X_ANYEXTH", SDT_RISCVFMV_X_ANYEXTH>;

//===----------------------------------------------------------------------===//
// Instruction class templates
//===----------------------------------------------------------------------===//
include "gen/RISCVInstrInfoZfh2.td"

// class FPFMAHDynFrmAlias<FPFMAH_rrr_frm Inst, string OpcodeStr>
//     : InstAlias<OpcodeStr#" $rd, $rs1, $rs2, $rs3",
//                 (Inst FPR16:$rd, FPR16:$rs1, FPR16:$rs2, FPR16:$rs3, 0b111)>;


// class FPALUHDynFrmAlias<FPALUH_rr_frm Inst, string OpcodeStr>
//     : InstAlias<OpcodeStr#" $rd, $rs1, $rs2",
//                 (Inst FPR16:$rd, FPR16:$rs1, FPR16:$rs2, 0b111)>;
class FPFMAHDynFrmAlias<ATGInst Inst, string OpcodeStr>
    : InstAlias<OpcodeStr#" $rd, $rs1, $rs2, $rs3",
                (Inst FPR16:$rd, FPR16:$rs1, FPR16:$rs2, FPR16:$rs3, 0b111)>;


class FPALUHDynFrmAlias<ATGInst Inst, string OpcodeStr>
    : InstAlias<OpcodeStr#" $rd, $rs1, $rs2",
                (Inst FPR16:$rd, FPR16:$rs1, FPR16:$rs2, 0b111)>;

let Predicates = [HasStdExtZfh] in {
def          : FPFMAHDynFrmAlias<FMADD_H, "fmadd.h">;
def          : FPFMAHDynFrmAlias<FMSUB_H, "fmsub.h">;
def          : FPFMAHDynFrmAlias<FNMSUB_H, "fnmsub.h">;
def          : FPFMAHDynFrmAlias<FNMADD_H, "fnmadd.h">;
def        : FPALUHDynFrmAlias<FADD_H, "fadd.h">;
def        : FPALUHDynFrmAlias<FSUB_H, "fsub.h">;
def        : FPALUHDynFrmAlias<FMUL_H, "fmul.h">;
def        : FPALUHDynFrmAlias<FDIV_H, "fdiv.h">;
def         : FPUnaryOpDynFrmAlias<FSQRT_H, "fsqrt.h", FPR16, FPR16>;
def          : FPUnaryOpDynFrmAlias<FCVT_W_H, "fcvt.w.h", GPR, FPR16>;
def           : FPUnaryOpDynFrmAlias<FCVT_WU_H, "fcvt.wu.h", GPR, FPR16>;
def          : FPUnaryOpDynFrmAlias<FCVT_H_W, "fcvt.h.w", FPR16, GPR>;
def           : FPUnaryOpDynFrmAlias<FCVT_H_WU, "fcvt.h.wu", FPR16, GPR>;
def          : FPUnaryOpDynFrmAlias<FCVT_H_S, "fcvt.h.s", FPR16, FPR32>;
} // Predicates = [HasStdExtZfh]        

let Predicates = [HasStdExtZfh, IsRV64] in {
def           : FPUnaryOpDynFrmAlias<FCVT_L_H, "fcvt.l.h", GPR, FPR16>;
def            : FPUnaryOpDynFrmAlias<FCVT_LU_H, "fcvt.lu.h", GPR, FPR16>;
def          : FPUnaryOpDynFrmAlias<FCVT_H_L, "fcvt.h.l", FPR16, GPR>;
def           : FPUnaryOpDynFrmAlias<FCVT_H_LU, "fcvt.h.lu", FPR16, GPR>;
} // Predicates = [HasStdExtZfh, IsRV64]



let Predicates = [HasStdExtZfh, HasStdExtD] in {
def          : FPUnaryOpDynFrmAlias<FCVT_H_D, "fcvt.h.d", FPR16, FPR64>;
} // Predicates = [HasStdExtZfh, HasStdExtD]



//===----------------------------------------------------------------------===//
// Assembler Pseudo Instructions (User-Level ISA, Version 2.2, Chapter 20)
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZfh] in {
def : InstAlias<"flh $rd, (${rs1})",  (FLH FPR16:$rd,  GPR:$rs1, 0), 0>;
def : InstAlias<"fsh $rs2, (${rs1})", (FSH FPR16:$rs2, GPR:$rs1, 0), 0>;

def : InstAlias<"fmv.h $rd, $rs",  (FSGNJ_H  FPR16:$rd, FPR16:$rs, FPR16:$rs)>;
def : InstAlias<"fabs.h $rd, $rs", (FSGNJX_H FPR16:$rd, FPR16:$rs, FPR16:$rs)>;
def : InstAlias<"fneg.h $rd, $rs", (FSGNJN_H FPR16:$rd, FPR16:$rs, FPR16:$rs)>;

// fgt.h/fge.h are recognised by the GNU assembler but the canonical
// flt.h/fle.h forms will always be printed. Therefore, set a zero weight.
def : InstAlias<"fgt.h $rd, $rs, $rt",
                (FLT_H GPR:$rd, FPR16:$rt, FPR16:$rs), 0>;
def : InstAlias<"fge.h $rd, $rs, $rt",
                (FLE_H GPR:$rd, FPR16:$rt, FPR16:$rs), 0>;

def PseudoFLH  : PseudoFloatLoad<"flh", FPR16>;
def PseudoFSH  : PseudoStore<"fsh", FPR16>;
} // Predicates = [HasStdExtZfh]

//===----------------------------------------------------------------------===//
// Pseudo-instructions and codegen patterns
//===----------------------------------------------------------------------===//

/// Generic pattern classes
// class PatFpr16Fpr16<SDPatternOperator OpNode, RVInstR Inst>
//     : Pat<(OpNode FPR16:$rs1, FPR16:$rs2), (Inst $rs1, $rs2)>;

// class PatFpr16Fpr16DynFrm<SDPatternOperator OpNode, RVInstRFrm Inst>
//     : Pat<(OpNode FPR16:$rs1, FPR16:$rs2), (Inst $rs1, $rs2, 0b111)>;

class PatFpr16Fpr16<SDPatternOperator OpNode, ATGInst Inst>
    : Pat<(OpNode FPR16:$rs1, FPR16:$rs2), (Inst $rs1, $rs2)>;

class PatFpr16Fpr16DynFrm<SDPatternOperator OpNode, ATGInst Inst>
    : Pat<(OpNode FPR16:$rs1, FPR16:$rs2), (Inst $rs1, $rs2, 0b111)>;

let Predicates = [HasStdExtZfh] in {

/// Float constants
def : Pat<(f16 (fpimm0)), (FMV_H_X X0)>;

/// Float conversion operations

// [u]int32<->float conversion patterns must be gated on IsRV32 or IsRV64, so
// are defined later.

/// Float arithmetic operations

def : PatFpr16Fpr16DynFrm<fadd, FADD_H>;
def : PatFpr16Fpr16DynFrm<fsub, FSUB_H>;
def : PatFpr16Fpr16DynFrm<fmul, FMUL_H>;
def : PatFpr16Fpr16DynFrm<fdiv, FDIV_H>;

def : Pat<(fsqrt FPR16:$rs1), (FSQRT_H FPR16:$rs1, 0b111)>;

def : Pat<(fneg FPR16:$rs1), (FSGNJN_H $rs1, $rs1)>;
def : Pat<(fabs FPR16:$rs1), (FSGNJX_H $rs1, $rs1)>;

def : PatFpr16Fpr16<fcopysign, FSGNJ_H>;
def : Pat<(fcopysign FPR16:$rs1, (fneg FPR16:$rs2)), (FSGNJN_H $rs1, $rs2)>;
def : Pat<(fcopysign FPR16:$rs1, FPR32:$rs2),
          (FSGNJ_H $rs1, (FCVT_H_S $rs2, 0b111))>;
def : Pat<(fcopysign FPR16:$rs1, FPR64:$rs2),
          (FSGNJ_H $rs1, (FCVT_H_D $rs2, 0b111))>;
def : Pat<(fcopysign FPR32:$rs1, FPR16:$rs2), (FSGNJ_S $rs1, (FCVT_S_H $rs2))>;
def : Pat<(fcopysign FPR64:$rs1, FPR16:$rs2), (FSGNJ_D $rs1, (FCVT_D_H $rs2))>;

// fmadd: rs1 * rs2 + rs3
def : Pat<(fma FPR16:$rs1, FPR16:$rs2, FPR16:$rs3),
          (FMADD_H $rs1, $rs2, $rs3, 0b111)>;

// fmsub: rs1 * rs2 - rs3
def : Pat<(fma FPR16:$rs1, FPR16:$rs2, (fneg FPR16:$rs3)),
          (FMSUB_H FPR16:$rs1, FPR16:$rs2, FPR16:$rs3, 0b111)>;

// fnmsub: -rs1 * rs2 + rs3
def : Pat<(fma (fneg FPR16:$rs1), FPR16:$rs2, FPR16:$rs3),
          (FNMSUB_H FPR16:$rs1, FPR16:$rs2, FPR16:$rs3, 0b111)>;

// fnmadd: -rs1 * rs2 - rs3
def : Pat<(fma (fneg FPR16:$rs1), FPR16:$rs2, (fneg FPR16:$rs3)),
          (FNMADD_H FPR16:$rs1, FPR16:$rs2, FPR16:$rs3, 0b111)>;

def : PatFpr16Fpr16<fminnum, FMIN_H>;
def : PatFpr16Fpr16<fmaxnum, FMAX_H>;

/// Setcc

def : PatFpr16Fpr16<seteq, FEQ_H>;
def : PatFpr16Fpr16<setoeq, FEQ_H>;
def : PatFpr16Fpr16<setlt, FLT_H>;
def : PatFpr16Fpr16<setolt, FLT_H>;
def : PatFpr16Fpr16<setle, FLE_H>;
def : PatFpr16Fpr16<setole, FLE_H>;

def Select_FPR16_Using_CC_GPR : SelectCC_rrirr<FPR16, GPR>;

/// Loads

defm : LdPat<load, FLH>;

/// Stores

defm : StPat<store, FSH, FPR16>;

/// Float conversion operations

// f32 -> f16, f16 -> f32
def : Pat<(fpround FPR32:$rs1), (FCVT_H_S FPR32:$rs1, 0b111)>;
def : Pat<(fpextend FPR16:$rs1), (FCVT_S_H FPR16:$rs1)>;

// Moves (no conversion)
def : Pat<(riscv_fmv_h_x GPR:$src), (FMV_H_X GPR:$src)>;
def : Pat<(riscv_fmv_x_anyexth FPR16:$src), (FMV_X_H FPR16:$src)>;
} // Predicates = [HasStdExtZfh]

let Predicates = [HasStdExtZfh, IsRV32] in {
// float->[u]int. Round-to-zero must be used.
def : Pat<(fp_to_sint FPR16:$rs1), (FCVT_W_H $rs1, 0b001)>;
def : Pat<(fp_to_uint FPR16:$rs1), (FCVT_WU_H $rs1, 0b001)>;

// [u]int->float. Match GCC and default to using dynamic rounding mode.
def : Pat<(sint_to_fp GPR:$rs1), (FCVT_H_W $rs1, 0b111)>;
def : Pat<(uint_to_fp GPR:$rs1), (FCVT_H_WU $rs1, 0b111)>;
} // Predicates = [HasStdExtZfh, IsRV32]

let Predicates = [HasStdExtZfh, IsRV64] in {
// FP->[u]int32 is mostly handled by the FP->[u]int64 patterns. This is safe
// because fpto[u|s]i produces poison if the value can't fit into the target.
// We match the single case below because fcvt.wu.s sign-extends its result so
// is cheaper than fcvt.lu.h+sext.w.
def : Pat<(sext_inreg (assertzexti32 (fp_to_uint FPR16:$rs1)), i32),
          (FCVT_WU_H $rs1, 0b001)>;

// FP->[u]int64
def : Pat<(fp_to_sint FPR16:$rs1), (FCVT_L_H $rs1, 0b001)>;
def : Pat<(fp_to_uint FPR16:$rs1), (FCVT_LU_H $rs1, 0b001)>;

// [u]int->fp. Match GCC and default to using dynamic rounding mode.
def : Pat<(sint_to_fp (sexti32 GPR:$rs1)), (FCVT_H_W $rs1, 0b111)>;
def : Pat<(uint_to_fp (zexti32 GPR:$rs1)), (FCVT_H_WU $rs1, 0b111)>;
def : Pat<(sint_to_fp GPR:$rs1), (FCVT_H_L $rs1, 0b111)>;
def : Pat<(uint_to_fp GPR:$rs1), (FCVT_H_LU $rs1, 0b111)>;
} // Predicates = [HasStdExtZfh, IsRV64]

let Predicates = [HasStdExtZfh, HasStdExtD] in {
/// Float conversion operations
// f64 -> f16, f16 -> f64
def : Pat<(fpround FPR64:$rs1), (FCVT_H_D FPR64:$rs1, 0b111)>;
def : Pat<(fpextend FPR16:$rs1), (FCVT_D_H FPR16:$rs1)>;
}
