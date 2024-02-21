//===- RISCVInstrInfoC.td - Compressed RISCV instructions -*- tblgen-*-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operand definitions.
//===----------------------------------------------------------------------===//

def UImmLog2XLenNonZeroAsmOperand : AsmOperandClass {
  let Name = "UImmLog2XLenNonZero";
  let RenderMethod = "addImmOperands";
  let DiagnosticType = "InvalidUImmLog2XLenNonZero";
}

def uimmlog2xlennonzero : Operand<XLenVT>, ImmLeaf<XLenVT, [{
  if (Subtarget->is64Bit())
    return isUInt<6>(Imm) && (Imm != 0);
  return isUInt<5>(Imm) && (Imm != 0);
}]> {
  let ParserMatchClass = UImmLog2XLenNonZeroAsmOperand;
  // TODO: should ensure invalid shamt is rejected when decoding.
  let DecoderMethod = "decodeUImmOperand<6>";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (!MCOp.evaluateAsConstantImm(Imm))
      return false;
    if (STI.getTargetTriple().isArch64Bit())
      return  isUInt<6>(Imm) && (Imm != 0);
    return isUInt<5>(Imm) && (Imm != 0);
  }];
}

def simm6 : Operand<XLenVT>, ImmLeaf<XLenVT, [{return isInt<6>(Imm);}]> {
  let ParserMatchClass = SImmAsmOperand<6>;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeSImmOperand<6>";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (MCOp.evaluateAsConstantImm(Imm))
      return isInt<6>(Imm);
    return MCOp.isBareSymbolRef();
  }];
}

def simm6nonzero : Operand<XLenVT>,
                   ImmLeaf<XLenVT, [{return (Imm != 0) && isInt<6>(Imm);}]> {
  let ParserMatchClass = SImmAsmOperand<6, "NonZero">;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeSImmOperand<6>";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (MCOp.evaluateAsConstantImm(Imm))
      return (Imm != 0) && isInt<6>(Imm);
    return MCOp.isBareSymbolRef();
  }];
}

def immzero : Operand<XLenVT>,
              ImmLeaf<XLenVT, [{return (Imm == 0);}]> {
  let ParserMatchClass = ImmZeroAsmOperand;
}

def CLUIImmAsmOperand : AsmOperandClass {
  let Name = "CLUIImm";
  let RenderMethod = "addImmOperands";
  let DiagnosticType = !strconcat("Invalid", Name);
}


// c_lui_imm checks the immediate range is in [1, 31] or [0xfffe0, 0xfffff].
// The RISC-V ISA describes the constraint as [1, 63], with that value being
// loaded in to bits 17-12 of the destination register and sign extended from
// bit 17. Therefore, this 6-bit immediate can represent values in the ranges
// [1, 31] and [0xfffe0, 0xfffff].
def c_lui_imm : Operand<XLenVT>,
                ImmLeaf<XLenVT, [{return (Imm != 0) &&
                                 (isUInt<5>(Imm) ||
                                  (Imm >= 0xfffe0 && Imm <= 0xfffff));}]> {
  let ParserMatchClass = CLUIImmAsmOperand;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeCLUIImmOperand";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (MCOp.evaluateAsConstantImm(Imm))
      return (Imm != 0) && (isUInt<5>(Imm) ||
             (Imm >= 0xfffe0 && Imm <= 0xfffff));
    return MCOp.isBareSymbolRef();
  }];
}

// A 7-bit unsigned immediate where the least significant two bits are zero.
def uimm7_lsb00 : Operand<XLenVT>,
                  ImmLeaf<XLenVT, [{return isShiftedUInt<5, 2>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<7, "Lsb00">;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeUImmOperand<7>";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (!MCOp.evaluateAsConstantImm(Imm))
      return false;
    return isShiftedUInt<5, 2>(Imm);
  }];
}

// A 8-bit unsigned immediate where the least significant two bits are zero.
def uimm8_lsb00 : Operand<XLenVT>,
                  ImmLeaf<XLenVT, [{return isShiftedUInt<6, 2>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<8, "Lsb00">;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeUImmOperand<8>";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (!MCOp.evaluateAsConstantImm(Imm))
      return false;
    return isShiftedUInt<6, 2>(Imm);
  }];
}

// A 8-bit unsigned immediate where the least significant three bits are zero.
def uimm8_lsb000 : Operand<XLenVT>,
                   ImmLeaf<XLenVT, [{return isShiftedUInt<5, 3>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<8, "Lsb000">;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeUImmOperand<8>";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (!MCOp.evaluateAsConstantImm(Imm))
      return false;
    return isShiftedUInt<5, 3>(Imm);
  }];
}

// A 9-bit signed immediate where the least significant bit is zero.
def simm9_lsb0 : Operand<OtherVT>,
                 ImmLeaf<XLenVT, [{return isShiftedInt<8, 1>(Imm);}]> {
  let ParserMatchClass = SImmAsmOperand<9, "Lsb0">;
  let PrintMethod = "printBranchOperand";
  let EncoderMethod = "getImmOpValueAsr1";
  let DecoderMethod = "decodeSImmOperandAndLsl1<9>";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (MCOp.evaluateAsConstantImm(Imm))
      return isShiftedInt<8, 1>(Imm);
    return MCOp.isBareSymbolRef();

  }];
  let OperandType = "OPERAND_PCREL";
}

// A 9-bit unsigned immediate where the least significant three bits are zero.
def uimm9_lsb000 : Operand<XLenVT>,
                   ImmLeaf<XLenVT, [{return isShiftedUInt<6, 3>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<9, "Lsb000">;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeUImmOperand<9>";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (!MCOp.evaluateAsConstantImm(Imm))
      return false;
    return isShiftedUInt<6, 3>(Imm);
  }];
}

// A 10-bit unsigned immediate where the least significant two bits are zero
// and the immediate can't be zero.
def uimm10_lsb00nonzero : Operand<XLenVT>,
                          ImmLeaf<XLenVT,
                          [{return isShiftedUInt<8, 2>(Imm) && (Imm != 0);}]> {
  let ParserMatchClass = UImmAsmOperand<10, "Lsb00NonZero">;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeUImmNonZeroOperand<10>";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (!MCOp.evaluateAsConstantImm(Imm))
      return false;
    return isShiftedUInt<8, 2>(Imm) && (Imm != 0);
  }];
}

// A 10-bit signed immediate where the least significant four bits are zero.
def simm10_lsb0000nonzero : Operand<XLenVT>,
                            ImmLeaf<XLenVT,
                            [{return (Imm != 0) && isShiftedInt<6, 4>(Imm);}]> {
  let ParserMatchClass = SImmAsmOperand<10, "Lsb0000NonZero">;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeSImmNonZeroOperand<10>";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (!MCOp.evaluateAsConstantImm(Imm))
      return false;
    return isShiftedInt<6, 4>(Imm) && (Imm != 0);
  }];
}

// A 12-bit signed immediate where the least significant bit is zero.
def simm12_lsb0 : Operand<XLenVT>,
                  ImmLeaf<XLenVT, [{return isShiftedInt<11, 1>(Imm);}]> {
  let ParserMatchClass = SImmAsmOperand<12, "Lsb0">;
  let PrintMethod = "printBranchOperand";
  let EncoderMethod = "getImmOpValueAsr1";
  let DecoderMethod = "decodeSImmOperandAndLsl1<12>";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (MCOp.evaluateAsConstantImm(Imm))
      return isShiftedInt<11, 1>(Imm);
    return MCOp.isBareSymbolRef();
  }];
  let OperandType = "OPERAND_PCREL";
}

//===----------------------------------------------------------------------===//
// Instruction Class
//===----------------------------------------------------------------------===//
include "gen/RISCVInstrInfoC2.td"

//===----------------------------------------------------------------------===//
// Assembler Pseudo Instructions
//===----------------------------------------------------------------------===//

let EmitPriority = 0 in {
let Predicates = [HasStdExtC, HasStdExtD] in
def : InstAlias<"c.fld $rd, (${rs1})", (C_FLD FPR64C:$rd, GPRC:$rs1, 0)>;

def : InstAlias<"c.lw $rd, (${rs1})", (C_LW GPRC:$rd, GPRC:$rs1, 0)>;

let Predicates = [HasStdExtC, HasStdExtF, IsRV32] in
def : InstAlias<"c.flw $rd, (${rs1})", (C_FLW FPR32C:$rd, GPRC:$rs1, 0)>;

let Predicates = [HasStdExtC, IsRV64] in
def : InstAlias<"c.ld $rd, (${rs1})", (C_LD GPRC:$rd, GPRC:$rs1, 0)>;

let Predicates = [HasStdExtC, HasStdExtD] in
def : InstAlias<"c.fsd $rs2, (${rs1})", (C_FSD FPR64C:$rs2, GPRC:$rs1, 0)>;

def : InstAlias<"c.sw $rs2, (${rs1})", (C_SW GPRC:$rs2, GPRC:$rs1, 0)>;

let Predicates = [HasStdExtC, HasStdExtF, IsRV32] in
def : InstAlias<"c.fsw $rs2, (${rs1})", (C_FSW FPR32C:$rs2, GPRC:$rs1, 0)>;

let Predicates = [HasStdExtC, IsRV64] in
def : InstAlias<"c.sd $rs2, (${rs1})", (C_SD GPRC:$rs2, GPRC:$rs1, 0)>;

let Predicates = [HasStdExtC, HasStdExtD] in
def : InstAlias<"c.fldsp $rd, (${rs1})", (C_FLDSP FPR64C:$rd, SP:$rs1, 0)>;

def : InstAlias<"c.lwsp $rd, (${rs1})", (C_LWSP GPRC:$rd, SP:$rs1, 0)>;

let Predicates = [HasStdExtC, HasStdExtF, IsRV32] in
def : InstAlias<"c.flwsp $rd, (${rs1})", (C_FLWSP FPR32C:$rd, SP:$rs1, 0)>;

let Predicates = [HasStdExtC, IsRV64] in
def : InstAlias<"c.ldsp $rd, (${rs1})", (C_LDSP GPRC:$rd, SP:$rs1, 0)>;

let Predicates = [HasStdExtC, HasStdExtD] in
def : InstAlias<"c.fsdsp $rs2, (${rs1})", (C_FSDSP FPR64C:$rs2, SP:$rs1, 0)>;

def : InstAlias<"c.swsp $rs2, (${rs1})", (C_SWSP GPRC:$rs2, SP:$rs1, 0)>;

let Predicates = [HasStdExtC, HasStdExtF, IsRV32] in
def : InstAlias<"c.fswsp $rs2, (${rs1})", (C_FSWSP FPR32C:$rs2, SP:$rs1, 0)>;

let Predicates = [HasStdExtC, IsRV64] in
def : InstAlias<"c.sdsp $rs2, (${rs1})", (C_SDSP GPRC:$rs2, SP:$rs1, 0)>;
}

//===----------------------------------------------------------------------===//
// Compress Instruction tablegen backend.
//===----------------------------------------------------------------------===//

class CompressPat<dag input, dag output> {
  dag Input  = input;
  dag Output    = output;
  list<Predicate> Predicates = [];
  bit isCompressOnly = false;
}

// Patterns are defined in the same order the compressed instructions appear
// on page 82 of the ISA manual.

// Quadrant 0
let Predicates = [HasStdExtC] in {
def : CompressPat<(ADDI GPRC:$rd, SP:$rs1, uimm10_lsb00nonzero:$imm),
                  (C_ADDI4SPN GPRC:$rd, SP:$rs1, uimm10_lsb00nonzero:$imm)>;
} // Predicates = [HasStdExtC]

let Predicates = [HasStdExtC, HasStdExtD] in {
def : CompressPat<(FLD FPR64C:$rd, GPRC:$rs1, uimm8_lsb000:$imm),
                  (C_FLD FPR64C:$rd, GPRC:$rs1, uimm8_lsb000:$imm)>;
} // Predicates = [HasStdExtC, HasStdExtD]

let Predicates = [HasStdExtC] in {
def : CompressPat<(LW GPRC:$rd, GPRC:$rs1, uimm7_lsb00:$imm),
                  (C_LW GPRC:$rd, GPRC:$rs1, uimm7_lsb00:$imm)>;
} // Predicates = [HasStdExtC]

let Predicates = [HasStdExtC, HasStdExtF, IsRV32] in {
def : CompressPat<(FLW FPR32C:$rd, GPRC:$rs1, uimm7_lsb00:$imm),
                  (C_FLW FPR32C:$rd, GPRC:$rs1, uimm7_lsb00:$imm)>;
} // Predicates = [HasStdExtC, HasStdExtF, IsRV32]

let Predicates = [HasStdExtC, IsRV64] in {
def : CompressPat<(LD GPRC:$rd, GPRC:$rs1, uimm8_lsb000:$imm),
                  (C_LD GPRC:$rd, GPRC:$rs1, uimm8_lsb000:$imm)>;
} // Predicates = [HasStdExtC, IsRV64]

let Predicates = [HasStdExtC, HasStdExtD] in {
def : CompressPat<(FSD FPR64C:$rs2, GPRC:$rs1, uimm8_lsb000:$imm),
                  (C_FSD FPR64C:$rs2, GPRC:$rs1, uimm8_lsb000:$imm)>;
} // Predicates = [HasStdExtC, HasStdExtD]

let Predicates = [HasStdExtC] in {
def : CompressPat<(SW GPRC:$rs2, GPRC:$rs1, uimm7_lsb00:$imm),
                  (C_SW GPRC:$rs2, GPRC:$rs1, uimm7_lsb00:$imm)>;
} // Predicates = [HasStdExtC]

let Predicates = [HasStdExtC, HasStdExtF, IsRV32] in {
def : CompressPat<(FSW FPR32C:$rs2, GPRC:$rs1,uimm7_lsb00:$imm),
                  (C_FSW FPR32C:$rs2, GPRC:$rs1, uimm7_lsb00:$imm)>;
} // Predicate = [HasStdExtC, HasStdExtF, IsRV32]

let Predicates = [HasStdExtC, IsRV64] in {
def : CompressPat<(SD GPRC:$rs2, GPRC:$rs1, uimm8_lsb000:$imm),
                  (C_SD GPRC:$rs2, GPRC:$rs1, uimm8_lsb000:$imm)>;
} // Predicates = [HasStdExtC, IsRV64]

// Quadrant 1
let Predicates = [HasStdExtC] in {
def : CompressPat<(ADDI X0, X0, 0), (C_NOP)>;
def : CompressPat<(ADDI GPRNoX0:$rs1, GPRNoX0:$rs1, simm6nonzero:$imm),
                  (C_ADDI GPRNoX0:$rs1, simm6nonzero:$imm)>;
} // Predicates = [HasStdExtC]

let Predicates = [HasStdExtC, IsRV32] in {
def : CompressPat<(JAL X1, simm12_lsb0:$offset),
                  (C_JAL simm12_lsb0:$offset)>;
} // Predicates = [HasStdExtC, IsRV32]

let Predicates = [HasStdExtC, IsRV64] in {
def : CompressPat<(ADDIW GPRNoX0:$rs1, GPRNoX0:$rs1, simm6:$imm),
                  (C_ADDIW GPRNoX0:$rs1, simm6:$imm)>;
} // Predicates = [HasStdExtC, IsRV64]

let Predicates = [HasStdExtC] in {
def : CompressPat<(ADDI GPRNoX0:$rd, X0, simm6:$imm),
                  (C_LI GPRNoX0:$rd, simm6:$imm)>;
def : CompressPat<(ADDI X2, X2, simm10_lsb0000nonzero:$imm),
                  (C_ADDI16SP X2, simm10_lsb0000nonzero:$imm)>;
def : CompressPat<(LUI GPRNoX0X2:$rd, c_lui_imm:$imm),
                  (C_LUI GPRNoX0X2:$rd, c_lui_imm:$imm)>;
def : CompressPat<(SRLI GPRC:$rs1, GPRC:$rs1, uimmlog2xlennonzero:$imm),
                  (C_SRLI GPRC:$rs1, uimmlog2xlennonzero:$imm)>;
def : CompressPat<(SRAI GPRC:$rs1, GPRC:$rs1, uimmlog2xlennonzero:$imm),
                  (C_SRAI GPRC:$rs1, uimmlog2xlennonzero:$imm)>;
def : CompressPat<(ANDI GPRC:$rs1, GPRC:$rs1, simm6:$imm),
                  (C_ANDI GPRC:$rs1, simm6:$imm)>;
def : CompressPat<(SUB GPRC:$rs1, GPRC:$rs1, GPRC:$rs2),
                  (C_SUB GPRC:$rs1, GPRC:$rs2)>;
def : CompressPat<(XOR GPRC:$rs1, GPRC:$rs1, GPRC:$rs2),
                  (C_XOR GPRC:$rs1, GPRC:$rs2)>;
let isCompressOnly = true in
def : CompressPat<(XOR GPRC:$rs1, GPRC:$rs2, GPRC:$rs1),
                  (C_XOR GPRC:$rs1, GPRC:$rs2)>;
def : CompressPat<(OR GPRC:$rs1, GPRC:$rs1, GPRC:$rs2),
                  (C_OR GPRC:$rs1, GPRC:$rs2)>;
let isCompressOnly = true in
def : CompressPat<(OR GPRC:$rs1, GPRC:$rs2, GPRC:$rs1),
                  (C_OR GPRC:$rs1, GPRC:$rs2)>;
def : CompressPat<(AND GPRC:$rs1, GPRC:$rs1, GPRC:$rs2),
                  (C_AND GPRC:$rs1, GPRC:$rs2)>;
let isCompressOnly = true in
def : CompressPat<(AND GPRC:$rs1, GPRC:$rs2, GPRC:$rs1),
                  (C_AND GPRC:$rs1, GPRC:$rs2)>;
} //  Predicates = [HasStdExtC]

let Predicates = [HasStdExtC, IsRV64] in {
let isCompressOnly = true in
def : CompressPat<(ADDIW GPRNoX0:$rd, X0, simm6:$imm),
                  (C_LI GPRNoX0:$rd, simm6:$imm)>;
def : CompressPat<(SUBW GPRC:$rs1, GPRC:$rs1, GPRC:$rs2),
                  (C_SUBW GPRC:$rs1, GPRC:$rs2)>;
def : CompressPat<(ADDW GPRC:$rs1, GPRC:$rs1, GPRC:$rs2),
                   (C_ADDW GPRC:$rs1, GPRC:$rs2)>;
let isCompressOnly = true in
def : CompressPat<(ADDW GPRC:$rs1, GPRC:$rs2, GPRC:$rs1),
                   (C_ADDW GPRC:$rs1, GPRC:$rs2)>;
} // Predicates = [HasStdExtC, IsRV64]

let Predicates = [HasStdExtC] in {
def : CompressPat<(JAL X0, simm12_lsb0:$offset),
                  (C_J simm12_lsb0:$offset)>;
def : CompressPat<(BEQ GPRC:$rs1, X0, simm9_lsb0:$imm),
                  (C_BEQZ GPRC:$rs1, simm9_lsb0:$imm)>;
def : CompressPat<(BNE GPRC:$rs1, X0, simm9_lsb0:$imm),
                  (C_BNEZ GPRC:$rs1, simm9_lsb0:$imm)>;
} //  Predicates = [HasStdExtC]

// Quadrant 2
let Predicates = [HasStdExtC] in {
def : CompressPat<(SLLI GPRNoX0:$rs1, GPRNoX0:$rs1, uimmlog2xlennonzero:$imm),
                  (C_SLLI GPRNoX0:$rs1, uimmlog2xlennonzero:$imm)>;
} //  Predicates = [HasStdExtC]

let Predicates = [HasStdExtC, HasStdExtD] in {
def : CompressPat<(FLD FPR64:$rd, SP:$rs1, uimm9_lsb000:$imm),
                  (C_FLDSP FPR64:$rd, SP:$rs1, uimm9_lsb000:$imm)>;
} // Predicates = [HasStdExtC, HasStdExtD]

let Predicates = [HasStdExtC] in {
def : CompressPat<(LW GPRNoX0:$rd, SP:$rs1,  uimm8_lsb00:$imm),
                  (C_LWSP GPRNoX0:$rd, SP:$rs1, uimm8_lsb00:$imm)>;
} // Predicates = [HasStdExtC]

let Predicates = [HasStdExtC, HasStdExtF, IsRV32] in {
def : CompressPat<(FLW FPR32:$rd, SP:$rs1, uimm8_lsb00:$imm),
                  (C_FLWSP FPR32:$rd, SP:$rs1, uimm8_lsb00:$imm)>;
} // Predicates = [HasStdExtC, HasStdExtF, IsRV32]

let Predicates = [HasStdExtC, IsRV64] in {
def : CompressPat<(LD GPRNoX0:$rd, SP:$rs1, uimm9_lsb000:$imm),
                  (C_LDSP GPRNoX0:$rd, SP:$rs1, uimm9_lsb000:$imm)>;
} // Predicates = [HasStdExtC, IsRV64]

let Predicates = [HasStdExtC] in {
def : CompressPat<(JALR X0, GPRNoX0:$rs1, 0),
                  (C_JR GPRNoX0:$rs1)>;
let isCompressOnly = true in {
def : CompressPat<(ADD GPRNoX0:$rs1, X0, GPRNoX0:$rs2),
                  (C_MV GPRNoX0:$rs1, GPRNoX0:$rs2)>;
def : CompressPat<(ADD GPRNoX0:$rs1, GPRNoX0:$rs2, X0),
                  (C_MV GPRNoX0:$rs1, GPRNoX0:$rs2)>;
}
def : CompressPat<(ADDI GPRNoX0:$rs1, GPRNoX0:$rs2, 0),
                  (C_MV GPRNoX0:$rs1, GPRNoX0:$rs2)>;
def : CompressPat<(EBREAK), (C_EBREAK)>;
def : CompressPat<(UNIMP), (C_UNIMP)>;
def : CompressPat<(JALR X1, GPRNoX0:$rs1, 0),
                  (C_JALR GPRNoX0:$rs1)>;
def : CompressPat<(ADD GPRNoX0:$rs1, GPRNoX0:$rs1, GPRNoX0:$rs2),
                  (C_ADD GPRNoX0:$rs1, GPRNoX0:$rs2)>;
let isCompressOnly = true in
def : CompressPat<(ADD GPRNoX0:$rs1, GPRNoX0:$rs2, GPRNoX0:$rs1),
                  (C_ADD GPRNoX0:$rs1, GPRNoX0:$rs2)>;
} // Predicates = [HasStdExtC]

let Predicates = [HasStdExtC, HasStdExtD] in {
def : CompressPat<(FSD FPR64:$rs2, SP:$rs1, uimm9_lsb000:$imm),
                  (C_FSDSP FPR64:$rs2, SP:$rs1, uimm9_lsb000:$imm)>;
} // Predicates = [HasStdExtC, HasStdExtD]

let Predicates = [HasStdExtC] in {
def : CompressPat<(SW GPR:$rs2, SP:$rs1, uimm8_lsb00:$imm),
                  (C_SWSP GPR:$rs2, SP:$rs1, uimm8_lsb00:$imm)>;
} // Predicates = [HasStdExtC]

let Predicates = [HasStdExtC, HasStdExtF, IsRV32] in {
def : CompressPat<(FSW FPR32:$rs2, SP:$rs1, uimm8_lsb00:$imm),
                  (C_FSWSP FPR32:$rs2, SP:$rs1, uimm8_lsb00:$imm)>;
} // Predicates = [HasStdExtC, HasStdExtF, IsRV32]

let Predicates = [HasStdExtC, IsRV64] in {
def : CompressPat<(SD GPR:$rs2, SP:$rs1, uimm9_lsb000:$imm),
                  (C_SDSP GPR:$rs2, SP:$rs1, uimm9_lsb000:$imm)>;
} //  Predicates = [HasStdExtC, IsRV64]
