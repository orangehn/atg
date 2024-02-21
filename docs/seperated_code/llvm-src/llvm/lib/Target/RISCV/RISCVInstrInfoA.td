//===-- RISCVInstrInfoA.td - RISC-V 'A' instructions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard 'A', Atomic
// Instructions extension.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operand and SDNode transformation definitions.
//===----------------------------------------------------------------------===//

// A parse method for (${gpr}) or 0(${gpr}), where the 0 is be silently ignored.
// Used for GNU as Compatibility.
def AtomicMemOpOperand : AsmOperandClass {
  let Name = "AtomicMemOpOperand";
  let RenderMethod = "addRegOperands";
  let PredicateMethod = "isGPR";
  let ParserMethod = "parseAtomicMemOp";
}

def GPRMemAtomic : RegisterOperand<GPR> {
  let ParserMatchClass = AtomicMemOpOperand;
  let PrintMethod = "printAtomicMemOp";
}

//===----------------------------------------------------------------------===//
// Instruction class
//===----------------------------------------------------------------------===//
include "gen/RISCVInstrInfoA2.td"

// multiclass AtomicStPat<PatFrag StoreOp, RVInst Inst, RegisterClass StTy> {
multiclass AtomicStPat<PatFrag StoreOp, ATGInst Inst, RegisterClass StTy> {
  def : Pat<(StoreOp GPR:$rs1, StTy:$rs2), (Inst StTy:$rs2, GPR:$rs1, 0)>;
  def : Pat<(StoreOp AddrFI:$rs1, StTy:$rs2), (Inst StTy:$rs2, AddrFI:$rs1, 0)>;
  def : Pat<(StoreOp (add GPR:$rs1, simm12:$imm12), StTy:$rs2),
            (Inst StTy:$rs2, GPR:$rs1, simm12:$imm12)>;
  def : Pat<(StoreOp (add AddrFI:$rs1, simm12:$imm12), StTy:$rs2),
            (Inst StTy:$rs2, AddrFI:$rs1, simm12:$imm12)>;
  def : Pat<(StoreOp (IsOrAdd AddrFI:$rs1, simm12:$imm12), StTy:$rs2),
            (Inst StTy:$rs2, AddrFI:$rs1, simm12:$imm12)>;
}

//===----------------------------------------------------------------------===//
// Pseudo-instructions and codegen patterns
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtA] in {

/// Atomic loads and stores

// Fences will be inserted for atomic load/stores according to the logic in
// RISCVTargetLowering::{emitLeadingFence,emitTrailingFence}.

defm : LdPat<atomic_load_8,  LB>;
defm : LdPat<atomic_load_16, LH>;
defm : LdPat<atomic_load_32, LW>;

defm : AtomicStPat<atomic_store_8,  SB, GPR>;
defm : AtomicStPat<atomic_store_16, SH, GPR>;
defm : AtomicStPat<atomic_store_32, SW, GPR>;

/// AMOs

multiclass AMOPat<string AtomicOp, string BaseInst> {
  // def : PatGprGpr<!cast<PatFrag>(AtomicOp#"_monotonic"),
  //                 !cast<RVInst>(BaseInst)>;
  // def : PatGprGpr<!cast<PatFrag>(AtomicOp#"_acquire"),
  //                 !cast<RVInst>(BaseInst#"_AQ")>;
  // def : PatGprGpr<!cast<PatFrag>(AtomicOp#"_release"),
  //                 !cast<RVInst>(BaseInst#"_RL")>;
  // def : PatGprGpr<!cast<PatFrag>(AtomicOp#"_acq_rel"),
  //                 !cast<RVInst>(BaseInst#"_AQ_RL")>;
  // def : PatGprGpr<!cast<PatFrag>(AtomicOp#"_seq_cst"),
  //                 !cast<RVInst>(BaseInst#"_AQ_RL")>;
    def : PatGprGpr<!cast<PatFrag>(AtomicOp#"_monotonic"),
                  !cast<ATGInst>(BaseInst)>;
  def : PatGprGpr<!cast<PatFrag>(AtomicOp#"_acquire"),
                  !cast<ATGInst>(BaseInst#"_AQ")>;
  def : PatGprGpr<!cast<PatFrag>(AtomicOp#"_release"),
                  !cast<ATGInst>(BaseInst#"_RL")>;
  def : PatGprGpr<!cast<PatFrag>(AtomicOp#"_acq_rel"),
                  !cast<ATGInst>(BaseInst#"_AQ_RL")>;
  def : PatGprGpr<!cast<PatFrag>(AtomicOp#"_seq_cst"),
                  !cast<ATGInst>(BaseInst#"_AQ_RL")>;
}

defm : AMOPat<"atomic_swap_32", "AMOSWAP_W">;
defm : AMOPat<"atomic_load_add_32", "AMOADD_W">;
defm : AMOPat<"atomic_load_and_32", "AMOAND_W">;
defm : AMOPat<"atomic_load_or_32", "AMOOR_W">;
defm : AMOPat<"atomic_load_xor_32", "AMOXOR_W">;
defm : AMOPat<"atomic_load_max_32", "AMOMAX_W">;
defm : AMOPat<"atomic_load_min_32", "AMOMIN_W">;
defm : AMOPat<"atomic_load_umax_32", "AMOMAXU_W">;
defm : AMOPat<"atomic_load_umin_32", "AMOMINU_W">;

def : Pat<(atomic_load_sub_32_monotonic GPR:$addr, GPR:$incr),
          (AMOADD_W GPR:$addr, (SUB X0, GPR:$incr))>;
def : Pat<(atomic_load_sub_32_acquire GPR:$addr, GPR:$incr),
          (AMOADD_W_AQ GPR:$addr, (SUB X0, GPR:$incr))>;
def : Pat<(atomic_load_sub_32_release GPR:$addr, GPR:$incr),
          (AMOADD_W_RL GPR:$addr, (SUB X0, GPR:$incr))>;
def : Pat<(atomic_load_sub_32_acq_rel GPR:$addr, GPR:$incr),
          (AMOADD_W_AQ_RL GPR:$addr, (SUB X0, GPR:$incr))>;
def : Pat<(atomic_load_sub_32_seq_cst GPR:$addr, GPR:$incr),
          (AMOADD_W_AQ_RL GPR:$addr, (SUB X0, GPR:$incr))>;

/// Pseudo AMOs

class PseudoAMO : Pseudo<(outs GPR:$res, GPR:$scratch),
                         (ins GPR:$addr, GPR:$incr, ixlenimm:$ordering), []> {
  let Constraints = "@earlyclobber $res,@earlyclobber $scratch";
  let mayLoad = 1;
  let mayStore = 1;
  let hasSideEffects = 0;
}

def PseudoAtomicLoadNand32 : PseudoAMO;
// Ordering constants must be kept in sync with the AtomicOrdering enum in
// AtomicOrdering.h.
def : Pat<(atomic_load_nand_32_monotonic GPR:$addr, GPR:$incr),
          (PseudoAtomicLoadNand32 GPR:$addr, GPR:$incr, 2)>;
def : Pat<(atomic_load_nand_32_acquire GPR:$addr, GPR:$incr),
          (PseudoAtomicLoadNand32 GPR:$addr, GPR:$incr, 4)>;
def : Pat<(atomic_load_nand_32_release GPR:$addr, GPR:$incr),
          (PseudoAtomicLoadNand32 GPR:$addr, GPR:$incr, 5)>;
def : Pat<(atomic_load_nand_32_acq_rel GPR:$addr, GPR:$incr),
          (PseudoAtomicLoadNand32 GPR:$addr, GPR:$incr, 6)>;
def : Pat<(atomic_load_nand_32_seq_cst GPR:$addr, GPR:$incr),
          (PseudoAtomicLoadNand32 GPR:$addr, GPR:$incr, 7)>;

class PseudoMaskedAMO
    : Pseudo<(outs GPR:$res, GPR:$scratch),
             (ins GPR:$addr, GPR:$incr, GPR:$mask, ixlenimm:$ordering), []> {
  let Constraints = "@earlyclobber $res,@earlyclobber $scratch";
  let mayLoad = 1;
  let mayStore = 1;
  let hasSideEffects = 0;
}

class PseudoMaskedAMOMinMax
    : Pseudo<(outs GPR:$res, GPR:$scratch1, GPR:$scratch2),
             (ins GPR:$addr, GPR:$incr, GPR:$mask, ixlenimm:$sextshamt,
              ixlenimm:$ordering), []> {
  let Constraints = "@earlyclobber $res,@earlyclobber $scratch1,"
                    "@earlyclobber $scratch2";
  let mayLoad = 1;
  let mayStore = 1;
  let hasSideEffects = 0;
}

class PseudoMaskedAMOUMinUMax
    : Pseudo<(outs GPR:$res, GPR:$scratch1, GPR:$scratch2),
             (ins GPR:$addr, GPR:$incr, GPR:$mask, ixlenimm:$ordering), []> {
  let Constraints = "@earlyclobber $res,@earlyclobber $scratch1,"
                    "@earlyclobber $scratch2";
  let mayLoad = 1;
  let mayStore = 1;
  let hasSideEffects = 0;
}

class PseudoMaskedAMOPat<Intrinsic intrin, Pseudo AMOInst>
    : Pat<(intrin GPR:$addr, GPR:$incr, GPR:$mask, timm:$ordering),
          (AMOInst GPR:$addr, GPR:$incr, GPR:$mask, timm:$ordering)>;

class PseudoMaskedAMOMinMaxPat<Intrinsic intrin, Pseudo AMOInst>
    : Pat<(intrin GPR:$addr, GPR:$incr, GPR:$mask, GPR:$shiftamt,
           timm:$ordering),
          (AMOInst GPR:$addr, GPR:$incr, GPR:$mask, GPR:$shiftamt,
           timm:$ordering)>;

def PseudoMaskedAtomicSwap32 : PseudoMaskedAMO;
def : PseudoMaskedAMOPat<int_riscv_masked_atomicrmw_xchg_i32,
                         PseudoMaskedAtomicSwap32>;
def PseudoMaskedAtomicLoadAdd32 : PseudoMaskedAMO;
def : PseudoMaskedAMOPat<int_riscv_masked_atomicrmw_add_i32,
                         PseudoMaskedAtomicLoadAdd32>;
def PseudoMaskedAtomicLoadSub32 : PseudoMaskedAMO;
def : PseudoMaskedAMOPat<int_riscv_masked_atomicrmw_sub_i32,
                         PseudoMaskedAtomicLoadSub32>;
def PseudoMaskedAtomicLoadNand32 : PseudoMaskedAMO;
def : PseudoMaskedAMOPat<int_riscv_masked_atomicrmw_nand_i32,
                         PseudoMaskedAtomicLoadNand32>;
def PseudoMaskedAtomicLoadMax32 : PseudoMaskedAMOMinMax;
def : PseudoMaskedAMOMinMaxPat<int_riscv_masked_atomicrmw_max_i32,
                               PseudoMaskedAtomicLoadMax32>;
def PseudoMaskedAtomicLoadMin32 : PseudoMaskedAMOMinMax;
def : PseudoMaskedAMOMinMaxPat<int_riscv_masked_atomicrmw_min_i32,
                               PseudoMaskedAtomicLoadMin32>;
def PseudoMaskedAtomicLoadUMax32 : PseudoMaskedAMOUMinUMax;
def : PseudoMaskedAMOPat<int_riscv_masked_atomicrmw_umax_i32,
                         PseudoMaskedAtomicLoadUMax32>;
def PseudoMaskedAtomicLoadUMin32 : PseudoMaskedAMOUMinUMax;
def : PseudoMaskedAMOPat<int_riscv_masked_atomicrmw_umin_i32,
                         PseudoMaskedAtomicLoadUMin32>;

/// Compare and exchange

class PseudoCmpXchg
    : Pseudo<(outs GPR:$res, GPR:$scratch),
             (ins GPR:$addr, GPR:$cmpval, GPR:$newval, ixlenimm:$ordering), []> {
  let Constraints = "@earlyclobber $res,@earlyclobber $scratch";
  let mayLoad = 1;
  let mayStore = 1;
  let hasSideEffects = 0;
}

// Ordering constants must be kept in sync with the AtomicOrdering enum in
// AtomicOrdering.h.
multiclass PseudoCmpXchgPat<string Op, Pseudo CmpXchgInst> {
  def : Pat<(!cast<PatFrag>(Op#"_monotonic") GPR:$addr, GPR:$cmp, GPR:$new),
            (CmpXchgInst GPR:$addr, GPR:$cmp, GPR:$new, 2)>;
  def : Pat<(!cast<PatFrag>(Op#"_acquire") GPR:$addr, GPR:$cmp, GPR:$new),
            (CmpXchgInst GPR:$addr, GPR:$cmp, GPR:$new, 4)>;
  def : Pat<(!cast<PatFrag>(Op#"_release") GPR:$addr, GPR:$cmp, GPR:$new),
            (CmpXchgInst GPR:$addr, GPR:$cmp, GPR:$new, 5)>;
  def : Pat<(!cast<PatFrag>(Op#"_acq_rel") GPR:$addr, GPR:$cmp, GPR:$new),
            (CmpXchgInst GPR:$addr, GPR:$cmp, GPR:$new, 6)>;
  def : Pat<(!cast<PatFrag>(Op#"_seq_cst") GPR:$addr, GPR:$cmp, GPR:$new),
            (CmpXchgInst GPR:$addr, GPR:$cmp, GPR:$new, 7)>;
}

def PseudoCmpXchg32 : PseudoCmpXchg;
defm : PseudoCmpXchgPat<"atomic_cmp_swap_32", PseudoCmpXchg32>;

def PseudoMaskedCmpXchg32
    : Pseudo<(outs GPR:$res, GPR:$scratch),
             (ins GPR:$addr, GPR:$cmpval, GPR:$newval, GPR:$mask,
              ixlenimm:$ordering), []> {
  let Constraints = "@earlyclobber $res,@earlyclobber $scratch";
  let mayLoad = 1;
  let mayStore = 1;
  let hasSideEffects = 0;
}

def : Pat<(int_riscv_masked_cmpxchg_i32
            GPR:$addr, GPR:$cmpval, GPR:$newval, GPR:$mask, timm:$ordering),
          (PseudoMaskedCmpXchg32
            GPR:$addr, GPR:$cmpval, GPR:$newval, GPR:$mask, timm:$ordering)>;

} // Predicates = [HasStdExtA]

let Predicates = [HasStdExtA, IsRV64] in {

/// 64-bit atomic loads and stores

// Fences will be inserted for atomic load/stores according to the logic in
// RISCVTargetLowering::{emitLeadingFence,emitTrailingFence}.
defm : LdPat<atomic_load_64, LD>;
defm : AtomicStPat<atomic_store_64, SD, GPR>;

defm : AMOPat<"atomic_swap_64", "AMOSWAP_D">;
defm : AMOPat<"atomic_load_add_64", "AMOADD_D">;
defm : AMOPat<"atomic_load_and_64", "AMOAND_D">;
defm : AMOPat<"atomic_load_or_64", "AMOOR_D">;
defm : AMOPat<"atomic_load_xor_64", "AMOXOR_D">;
defm : AMOPat<"atomic_load_max_64", "AMOMAX_D">;
defm : AMOPat<"atomic_load_min_64", "AMOMIN_D">;
defm : AMOPat<"atomic_load_umax_64", "AMOMAXU_D">;
defm : AMOPat<"atomic_load_umin_64", "AMOMINU_D">;

/// 64-bit AMOs

def : Pat<(atomic_load_sub_64_monotonic GPR:$addr, GPR:$incr),
          (AMOADD_D GPR:$addr, (SUB X0, GPR:$incr))>;
def : Pat<(atomic_load_sub_64_acquire GPR:$addr, GPR:$incr),
          (AMOADD_D_AQ GPR:$addr, (SUB X0, GPR:$incr))>;
def : Pat<(atomic_load_sub_64_release GPR:$addr, GPR:$incr),
          (AMOADD_D_RL GPR:$addr, (SUB X0, GPR:$incr))>;
def : Pat<(atomic_load_sub_64_acq_rel GPR:$addr, GPR:$incr),
          (AMOADD_D_AQ_RL GPR:$addr, (SUB X0, GPR:$incr))>;
def : Pat<(atomic_load_sub_64_seq_cst GPR:$addr, GPR:$incr),
          (AMOADD_D_AQ_RL GPR:$addr, (SUB X0, GPR:$incr))>;

/// 64-bit pseudo AMOs

def PseudoAtomicLoadNand64 : PseudoAMO;
// Ordering constants must be kept in sync with the AtomicOrdering enum in
// AtomicOrdering.h.
def : Pat<(atomic_load_nand_64_monotonic GPR:$addr, GPR:$incr),
          (PseudoAtomicLoadNand64 GPR:$addr, GPR:$incr, 2)>;
def : Pat<(atomic_load_nand_64_acquire GPR:$addr, GPR:$incr),
          (PseudoAtomicLoadNand64 GPR:$addr, GPR:$incr, 4)>;
def : Pat<(atomic_load_nand_64_release GPR:$addr, GPR:$incr),
          (PseudoAtomicLoadNand64 GPR:$addr, GPR:$incr, 5)>;
def : Pat<(atomic_load_nand_64_acq_rel GPR:$addr, GPR:$incr),
          (PseudoAtomicLoadNand64 GPR:$addr, GPR:$incr, 6)>;
def : Pat<(atomic_load_nand_64_seq_cst GPR:$addr, GPR:$incr),
          (PseudoAtomicLoadNand64 GPR:$addr, GPR:$incr, 7)>;

def : PseudoMaskedAMOPat<int_riscv_masked_atomicrmw_xchg_i64,
                         PseudoMaskedAtomicSwap32>;
def : PseudoMaskedAMOPat<int_riscv_masked_atomicrmw_add_i64,
                         PseudoMaskedAtomicLoadAdd32>;
def : PseudoMaskedAMOPat<int_riscv_masked_atomicrmw_sub_i64,
                         PseudoMaskedAtomicLoadSub32>;
def : PseudoMaskedAMOPat<int_riscv_masked_atomicrmw_nand_i64,
                         PseudoMaskedAtomicLoadNand32>;
def : PseudoMaskedAMOMinMaxPat<int_riscv_masked_atomicrmw_max_i64,
                               PseudoMaskedAtomicLoadMax32>;
def : PseudoMaskedAMOMinMaxPat<int_riscv_masked_atomicrmw_min_i64,
                               PseudoMaskedAtomicLoadMin32>;
def : PseudoMaskedAMOPat<int_riscv_masked_atomicrmw_umax_i64,
                         PseudoMaskedAtomicLoadUMax32>;
def : PseudoMaskedAMOPat<int_riscv_masked_atomicrmw_umin_i64,
                         PseudoMaskedAtomicLoadUMin32>;

/// 64-bit compare and exchange

def PseudoCmpXchg64 : PseudoCmpXchg;
defm : PseudoCmpXchgPat<"atomic_cmp_swap_64", PseudoCmpXchg64>;

def : Pat<(int_riscv_masked_cmpxchg_i64
            GPR:$addr, GPR:$cmpval, GPR:$newval, GPR:$mask, timm:$ordering),
          (PseudoMaskedCmpXchg32
            GPR:$addr, GPR:$cmpval, GPR:$newval, GPR:$mask, timm:$ordering)>;
} // Predicates = [HasStdExtA, IsRV64]
