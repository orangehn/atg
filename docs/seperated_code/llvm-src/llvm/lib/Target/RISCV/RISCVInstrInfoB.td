//===-- RISCVInstrInfoB.td - RISC-V 'B' instructions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard 'B' Bitmanip
// extension, version 0.93.
// This version is still experimental as the 'B' extension hasn't been
// ratified yet.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operand and SDNode transformation definitions.
//===----------------------------------------------------------------------===//

def riscv_rolw : SDNode<"RISCVISD::ROLW", SDTIntShiftOp>;
def riscv_rorw : SDNode<"RISCVISD::RORW", SDTIntShiftOp>;
def riscv_fslw : SDNode<"RISCVISD::FSLW", SDTIntShiftDOp>;
def riscv_fsrw : SDNode<"RISCVISD::FSRW", SDTIntShiftDOp>;

def UImmLog2XLenHalfAsmOperand : AsmOperandClass {
  let Name = "UImmLog2XLenHalf";
  let RenderMethod = "addImmOperands";
  let DiagnosticType = "InvalidUImmLog2XLenHalf";
}

def shfl_uimm : Operand<XLenVT>, ImmLeaf<XLenVT, [{
  if (Subtarget->is64Bit())
    return isUInt<5>(Imm);
  return isUInt<4>(Imm);
}]> {
  let ParserMatchClass = UImmLog2XLenHalfAsmOperand;
  let DecoderMethod = "decodeUImmOperand<5>";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (!MCOp.evaluateAsConstantImm(Imm))
      return false;
    if (STI.getTargetTriple().isArch64Bit())
      return  isUInt<5>(Imm);
    return isUInt<4>(Imm);
  }];
}

// Checks if this mask has a single 0 bit and cannot be used with ANDI.
def BCLRMask : ImmLeaf<XLenVT, [{
  if (Subtarget->is64Bit())
    return !isInt<12>(Imm) && isPowerOf2_64(~Imm);
  return !isInt<12>(Imm) && isPowerOf2_32(~Imm);
}]>;

// Checks if this mask has a single 1 bit and cannot be used with ORI/XORI.
def BSETINVMask : ImmLeaf<XLenVT, [{
  if (Subtarget->is64Bit())
    return !isInt<12>(Imm) && isPowerOf2_64(Imm);
  return !isInt<12>(Imm) && isPowerOf2_32(Imm);
}]>;

def BCLRXForm : SDNodeXForm<imm, [{
  // Find the lowest 0.
  return CurDAG->getTargetConstant(N->getAPIntValue().countTrailingOnes(),
                                   SDLoc(N), N->getValueType(0));
}]>;

def BSETINVXForm : SDNodeXForm<imm, [{
  // Find the lowest 1.
  return CurDAG->getTargetConstant(N->getAPIntValue().countTrailingZeros(),
                                   SDLoc(N), N->getValueType(0));
}]>;

// Similar to above, but makes sure the immediate has 33 sign bits. When used
// with an AND/OR/XOR where the other operand has at least 33 sign bits, the
// result will have 33 sign bits. This can match BCLRIW/BSETIW/BINVIW.
def BCLRWMask : ImmLeaf<i64, [{
  // After checking the sign bits, truncate to 32 bits for power of 2 check.
  return isInt<32>(Imm) && !isInt<12>(Imm) && isPowerOf2_32(~Imm);
}]>;

def BSETINVWMask : ImmLeaf<i64, [{
  return isInt<32>(Imm) && !isInt<12>(Imm) && isPowerOf2_32(Imm);
}]>;

//===----------------------------------------------------------------------===//
// Instruction class
//===----------------------------------------------------------------------===//
include "gen/RISCVInstrInfoB2.td"

//===----------------------------------------------------------------------===//
// Pseudo Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZba, IsRV64] in {
// NOTE: The 0.93 spec shows zext.w as an alias of pack/packw. It has been
// changed to add.uw in a draft after 0.94.
def : InstAlias<"zext.w $rd, $rs", (ADDUW GPR:$rd, GPR:$rs, X0)>;
}

let Predicates = [HasStdExtZbp] in {
def : InstAlias<"rev.p $rd, $rs",  (GREVI GPR:$rd, GPR:$rs, 0b00001)>;
def : InstAlias<"rev2.n $rd, $rs", (GREVI GPR:$rd, GPR:$rs, 0b00010)>;
def : InstAlias<"rev.n $rd, $rs",  (GREVI GPR:$rd, GPR:$rs, 0b00011)>;
def : InstAlias<"rev4.b $rd, $rs", (GREVI GPR:$rd, GPR:$rs, 0b00100)>;
def : InstAlias<"rev2.b $rd, $rs", (GREVI GPR:$rd, GPR:$rs, 0b00110)>;
def : InstAlias<"rev.b $rd, $rs",  (GREVI GPR:$rd, GPR:$rs, 0b00111)>;
def : InstAlias<"rev8.h $rd, $rs", (GREVI GPR:$rd, GPR:$rs, 0b01000)>;
def : InstAlias<"rev4.h $rd, $rs", (GREVI GPR:$rd, GPR:$rs, 0b01100)>;
def : InstAlias<"rev2.h $rd, $rs", (GREVI GPR:$rd, GPR:$rs, 0b01110)>;
def : InstAlias<"rev.h $rd, $rs",  (GREVI GPR:$rd, GPR:$rs, 0b01111)>;

def : InstAlias<"zip.n $rd, $rs",    (SHFLI   GPR:$rd, GPR:$rs, 0b0001)>;
def : InstAlias<"unzip.n $rd, $rs",  (UNSHFLI GPR:$rd, GPR:$rs, 0b0001)>;
def : InstAlias<"zip2.b $rd, $rs",   (SHFLI   GPR:$rd, GPR:$rs, 0b0010)>;
def : InstAlias<"unzip2.b $rd, $rs", (UNSHFLI GPR:$rd, GPR:$rs, 0b0010)>;
def : InstAlias<"zip.b $rd, $rs",    (SHFLI   GPR:$rd, GPR:$rs, 0b0011)>;
def : InstAlias<"unzip.b $rd, $rs",  (UNSHFLI GPR:$rd, GPR:$rs, 0b0011)>;
def : InstAlias<"zip4.h $rd, $rs",   (SHFLI   GPR:$rd, GPR:$rs, 0b0100)>;
def : InstAlias<"unzip4.h $rd, $rs", (UNSHFLI GPR:$rd, GPR:$rs, 0b0100)>;
def : InstAlias<"zip2.h $rd, $rs",   (SHFLI   GPR:$rd, GPR:$rs, 0b0110)>;
def : InstAlias<"unzip2.h $rd, $rs", (UNSHFLI GPR:$rd, GPR:$rs, 0b0110)>;
def : InstAlias<"zip.h $rd, $rs",    (SHFLI   GPR:$rd, GPR:$rs, 0b0111)>;
def : InstAlias<"unzip.h $rd, $rs",  (UNSHFLI GPR:$rd, GPR:$rs, 0b0111)>;

def : InstAlias<"orc.p $rd, $rs",  (GORCI GPR:$rd, GPR:$rs, 0b00001)>;
def : InstAlias<"orc2.n $rd, $rs", (GORCI GPR:$rd, GPR:$rs, 0b00010)>;
def : InstAlias<"orc.n $rd, $rs",  (GORCI GPR:$rd, GPR:$rs, 0b00011)>;
def : InstAlias<"orc4.b $rd, $rs", (GORCI GPR:$rd, GPR:$rs, 0b00100)>;
def : InstAlias<"orc2.b $rd, $rs", (GORCI GPR:$rd, GPR:$rs, 0b00110)>;
// orc.b is considered an instruction rather than an alias.
def : InstAlias<"orc8.h $rd, $rs", (GORCI GPR:$rd, GPR:$rs, 0b01000)>;
def : InstAlias<"orc4.h $rd, $rs", (GORCI GPR:$rd, GPR:$rs, 0b01100)>;
def : InstAlias<"orc2.h $rd, $rs", (GORCI GPR:$rd, GPR:$rs, 0b01110)>;
def : InstAlias<"orc.h $rd, $rs",  (GORCI GPR:$rd, GPR:$rs, 0b01111)>;
} // Predicates = [HasStdExtZbp]

let Predicates = [HasStdExtZbp, IsRV32] in {
def : InstAlias<"rev16 $rd, $rs", (GREVI GPR:$rd, GPR:$rs, 0b10000)>;
// rev8 is considered an instruction rather than an alias.
def : InstAlias<"rev4 $rd, $rs",  (GREVI GPR:$rd, GPR:$rs, 0b11100)>;
def : InstAlias<"rev2 $rd, $rs",  (GREVI GPR:$rd, GPR:$rs, 0b11110)>;
def : InstAlias<"rev $rd, $rs",   (GREVI GPR:$rd, GPR:$rs, 0b11111)>;

def : InstAlias<"zip8 $rd, $rs",   (SHFLI   GPR:$rd, GPR:$rs, 0b1000)>;
def : InstAlias<"unzip8 $rd, $rs", (UNSHFLI GPR:$rd, GPR:$rs, 0b1000)>;
def : InstAlias<"zip4 $rd, $rs",   (SHFLI   GPR:$rd, GPR:$rs, 0b1100)>;
def : InstAlias<"unzip4 $rd, $rs", (UNSHFLI GPR:$rd, GPR:$rs, 0b1100)>;
def : InstAlias<"zip2 $rd, $rs",   (SHFLI   GPR:$rd, GPR:$rs, 0b1110)>;
def : InstAlias<"unzip2 $rd, $rs", (UNSHFLI GPR:$rd, GPR:$rs, 0b1110)>;
def : InstAlias<"zip $rd, $rs",    (SHFLI   GPR:$rd, GPR:$rs, 0b1111)>;
def : InstAlias<"unzip $rd, $rs",  (UNSHFLI GPR:$rd, GPR:$rs, 0b1111)>;

def : InstAlias<"orc16 $rd, $rs", (GORCI GPR:$rd, GPR:$rs, 0b10000)>;
def : InstAlias<"orc8 $rd, $rs",  (GORCI GPR:$rd, GPR:$rs, 0b11000)>;
def : InstAlias<"orc4 $rd, $rs",  (GORCI GPR:$rd, GPR:$rs, 0b11100)>;
def : InstAlias<"orc2 $rd, $rs",  (GORCI GPR:$rd, GPR:$rs, 0b11110)>;
def : InstAlias<"orc $rd, $rs",   (GORCI GPR:$rd, GPR:$rs, 0b11111)>;
} // Predicates = [HasStdExtZbp, IsRV32]

let Predicates = [HasStdExtZbp, IsRV64] in {
def : InstAlias<"rev16.w $rd, $rs", (GREVI GPR:$rd, GPR:$rs, 0b010000)>;
def : InstAlias<"rev8.w $rd, $rs",  (GREVI GPR:$rd, GPR:$rs, 0b011000)>;
def : InstAlias<"rev4.w $rd, $rs",  (GREVI GPR:$rd, GPR:$rs, 0b011100)>;
def : InstAlias<"rev2.w $rd, $rs",  (GREVI GPR:$rd, GPR:$rs, 0b011110)>;
def : InstAlias<"rev.w $rd, $rs",   (GREVI GPR:$rd, GPR:$rs, 0b011111)>;
def : InstAlias<"rev32 $rd, $rs",   (GREVI GPR:$rd, GPR:$rs, 0b100000)>;
def : InstAlias<"rev16 $rd, $rs",   (GREVI GPR:$rd, GPR:$rs, 0b110000)>;
// rev8 is considered an instruction rather than an alias.
def : InstAlias<"rev4 $rd, $rs",    (GREVI GPR:$rd, GPR:$rs, 0b111100)>;
def : InstAlias<"rev2 $rd, $rs",    (GREVI GPR:$rd, GPR:$rs, 0b111110)>;
def : InstAlias<"rev $rd, $rs",     (GREVI GPR:$rd, GPR:$rs, 0b111111)>;

def : InstAlias<"zip8.w $rd, $rs",   (SHFLI   GPR:$rd, GPR:$rs, 0b01000)>;
def : InstAlias<"unzip8.w $rd, $rs", (UNSHFLI GPR:$rd, GPR:$rs, 0b01000)>;
def : InstAlias<"zip4.w $rd, $rs",   (SHFLI   GPR:$rd, GPR:$rs, 0b01100)>;
def : InstAlias<"unzip4.w $rd, $rs", (UNSHFLI GPR:$rd, GPR:$rs, 0b01100)>;
def : InstAlias<"zip2.w $rd, $rs",   (SHFLI   GPR:$rd, GPR:$rs, 0b01110)>;
def : InstAlias<"unzip2.w $rd, $rs", (UNSHFLI GPR:$rd, GPR:$rs, 0b01110)>;
def : InstAlias<"zip.w $rd, $rs",    (SHFLI   GPR:$rd, GPR:$rs, 0b01111)>;
def : InstAlias<"unzip.w $rd, $rs",  (UNSHFLI GPR:$rd, GPR:$rs, 0b01111)>;
def : InstAlias<"zip16 $rd, $rs",    (SHFLI   GPR:$rd, GPR:$rs, 0b10000)>;
def : InstAlias<"unzip16 $rd, $rs",  (UNSHFLI GPR:$rd, GPR:$rs, 0b10000)>;
def : InstAlias<"zip8 $rd, $rs",     (SHFLI   GPR:$rd, GPR:$rs, 0b11000)>;
def : InstAlias<"unzip8 $rd, $rs",   (UNSHFLI GPR:$rd, GPR:$rs, 0b11000)>;
def : InstAlias<"zip4 $rd, $rs",     (SHFLI   GPR:$rd, GPR:$rs, 0b11100)>;
def : InstAlias<"unzip4 $rd, $rs",   (UNSHFLI GPR:$rd, GPR:$rs, 0b11100)>;
def : InstAlias<"zip2 $rd, $rs",     (SHFLI   GPR:$rd, GPR:$rs, 0b11110)>;
def : InstAlias<"unzip2 $rd, $rs",   (UNSHFLI GPR:$rd, GPR:$rs, 0b11110)>;
def : InstAlias<"zip $rd, $rs",      (SHFLI   GPR:$rd, GPR:$rs, 0b11111)>;
def : InstAlias<"unzip $rd, $rs",    (UNSHFLI GPR:$rd, GPR:$rs, 0b11111)>;

def : InstAlias<"orc16.w $rd, $rs", (GORCI GPR:$rd, GPR:$rs, 0b010000)>;
def : InstAlias<"orc8.w $rd, $rs",  (GORCI GPR:$rd, GPR:$rs, 0b011000)>;
def : InstAlias<"orc4.w $rd, $rs",  (GORCI GPR:$rd, GPR:$rs, 0b011100)>;
def : InstAlias<"orc2.w $rd, $rs",  (GORCI GPR:$rd, GPR:$rs, 0b011110)>;
def : InstAlias<"orc.w $rd, $rs",   (GORCI GPR:$rd, GPR:$rs, 0b011111)>;
def : InstAlias<"orc32 $rd, $rs",   (GORCI GPR:$rd, GPR:$rs, 0b100000)>;
def : InstAlias<"orc16 $rd, $rs",   (GORCI GPR:$rd, GPR:$rs, 0b110000)>;
def : InstAlias<"orc8 $rd, $rs",    (GORCI GPR:$rd, GPR:$rs, 0b111000)>;
def : InstAlias<"orc4 $rd, $rs",    (GORCI GPR:$rd, GPR:$rs, 0b111100)>;
def : InstAlias<"orc2 $rd, $rs",    (GORCI GPR:$rd, GPR:$rs, 0b111110)>;
def : InstAlias<"orc $rd, $rs",     (GORCI GPR:$rd, GPR:$rs, 0b111111)>;
} // Predicates = [HasStdExtZbp, IsRV64]

//===----------------------------------------------------------------------===//
// Compressed Instruction patterns
//===----------------------------------------------------------------------===//
let Predicates = [HasStdExtZbproposedc, HasStdExtC] in {
def : CompressPat<(XORI GPRC:$rs1, GPRC:$rs1, -1),
                  (C_NOT GPRC:$rs1)>;
def : CompressPat<(SUB GPRC:$rs1, X0, GPRC:$rs1),
                  (C_NEG GPRC:$rs1)>;
} // Predicates = [HasStdExtZbproposedc, HasStdExtC]

let Predicates = [HasStdExtZbproposedc, HasStdExtZba, HasStdExtC, IsRV64] in {
def : CompressPat<(ADDUW GPRC:$rs1, GPRC:$rs1, X0),
                  (C_ZEXTW GPRC:$rs1)>;
} // Predicates = [HasStdExtZbproposedc, HasStdExtC, IsRV64]

//===----------------------------------------------------------------------===//
// Codegen patterns
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZbbOrZbp] in {
def : Pat<(and GPR:$rs1, (not GPR:$rs2)), (ANDN GPR:$rs1, GPR:$rs2)>;
def : Pat<(or  GPR:$rs1, (not GPR:$rs2)), (ORN  GPR:$rs1, GPR:$rs2)>;
def : Pat<(xor GPR:$rs1, (not GPR:$rs2)), (XNOR GPR:$rs1, GPR:$rs2)>;
} // Predicates = [HasStdExtZbbOrZbp]

let Predicates = [HasStdExtZbbOrZbp] in {
def : Pat<(rotl GPR:$rs1, GPR:$rs2), (ROL GPR:$rs1, GPR:$rs2)>;
def : Pat<(rotr GPR:$rs1, GPR:$rs2), (ROR GPR:$rs1, GPR:$rs2)>;
} // Predicates = [HasStdExtZbbOrZbp]

let Predicates = [HasStdExtZbs] in {
def : Pat<(and (not (shiftop<shl> 1, GPR:$rs2)), GPR:$rs1),
          (BCLR GPR:$rs1, GPR:$rs2)>;
def : Pat<(and (rotl -2, GPR:$rs2), GPR:$rs1), (BCLR GPR:$rs1, GPR:$rs2)>;
def : Pat<(or (shiftop<shl> 1, GPR:$rs2), GPR:$rs1),
          (BSET GPR:$rs1, GPR:$rs2)>;
def : Pat<(xor (shiftop<shl> 1, GPR:$rs2), GPR:$rs1),
          (BINV GPR:$rs1, GPR:$rs2)>;
def : Pat<(and (shiftop<srl> GPR:$rs1, GPR:$rs2), 1),
          (BEXT GPR:$rs1, GPR:$rs2)>;

def : Pat<(shiftop<shl> 1, GPR:$rs2),
          (BSET X0, GPR:$rs2)>;

def : Pat<(and GPR:$rs1, BCLRMask:$mask),
          (BCLRI GPR:$rs1, (BCLRXForm imm:$mask))>;
def : Pat<(or GPR:$rs1, BSETINVMask:$mask),
          (BSETI GPR:$rs1, (BSETINVXForm imm:$mask))>;
def : Pat<(xor GPR:$rs1, BSETINVMask:$mask),
          (BINVI GPR:$rs1, (BSETINVXForm imm:$mask))>;

def : Pat<(and (srl GPR:$rs1, uimmlog2xlen:$shamt), (XLenVT 1)),
          (BEXTI GPR:$rs1, uimmlog2xlen:$shamt)>;
}

// There's no encoding for roli in the the 'B' extension as it can be
// implemented with rori by negating the immediate.
let Predicates = [HasStdExtZbbOrZbp] in {
def : Pat<(rotr GPR:$rs1, uimmlog2xlen:$shamt),
          (RORI GPR:$rs1, uimmlog2xlen:$shamt)>;
def : Pat<(rotl GPR:$rs1, uimmlog2xlen:$shamt),
          (RORI GPR:$rs1, (ImmSubFromXLen uimmlog2xlen:$shamt))>;
}

def riscv_grevi    : SDNode<"RISCVISD::GREVI", SDTIntBinOp, []>;
def riscv_greviw   : SDNode<"RISCVISD::GREVIW", SDTIntBinOp, []>;
def riscv_gorci    : SDNode<"RISCVISD::GORCI", SDTIntBinOp, []>;
def riscv_gorciw   : SDNode<"RISCVISD::GORCIW", SDTIntBinOp, []>;

let Predicates = [HasStdExtZbp] in {
def : Pat<(riscv_grevi GPR:$rs1, timm:$shamt), (GREVI GPR:$rs1, timm:$shamt)>;
def : Pat<(riscv_gorci GPR:$rs1, timm:$shamt), (GORCI GPR:$rs1, timm:$shamt)>;

// We treat orc.b as a separate instruction, so match it directly.
def : Pat<(riscv_gorci GPR:$rs1, (XLenVT 7)), (ORCB GPR:$rs1)>;
} // Predicates = [HasStdExtZbp]

let Predicates = [HasStdExtZbp, IsRV32] in {
def : Pat<(rotr (riscv_grevi GPR:$rs1, (i32 24)), (i32 16)), (GREVI GPR:$rs1, 8)>;
def : Pat<(rotl (riscv_grevi GPR:$rs1, (i32 24)), (i32 16)), (GREVI GPR:$rs1, 8)>;

// We treat rev8 as a separate instruction, so match it directly.
def : Pat<(riscv_grevi GPR:$rs1, (i32 24)), (REV8_RV32 GPR:$rs1)>;
} // Predicates = [HasStdExtZbp, IsRV32]

let Predicates = [HasStdExtZbp, IsRV64] in {
// We treat rev8 as a separate instruction, so match it directly.
def : Pat<(riscv_grevi GPR:$rs1, (i64 56)), (REV8_RV64 GPR:$rs1)>;
} // Predicates = [HasStdExtZbp, IsRV64]

let Predicates = [HasStdExtZbt] in {
def : Pat<(or (and (not GPR:$rs2), GPR:$rs3), (and GPR:$rs2, GPR:$rs1)),
          (CMIX GPR:$rs1, GPR:$rs2, GPR:$rs3)>;

def : Pat<(select (XLenVT (setne GPR:$rs2, 0)), GPR:$rs1, GPR:$rs3),
          (CMOV GPR:$rs1, GPR:$rs2, GPR:$rs3)>;
def : Pat<(select (XLenVT (seteq GPR:$rs2, 0)), GPR:$rs3, GPR:$rs1),
          (CMOV GPR:$rs1, GPR:$rs2, GPR:$rs3)>;
def : Pat<(select (XLenVT (setne GPR:$x, simm12_plus1:$y)), GPR:$rs1, GPR:$rs3),
          (CMOV GPR:$rs1, (ADDI GPR:$x, (NegImm simm12_plus1:$y)), GPR:$rs3)>;
def : Pat<(select (XLenVT (seteq GPR:$x, simm12_plus1:$y)), GPR:$rs3, GPR:$rs1),
          (CMOV GPR:$rs1, (ADDI GPR:$x, (NegImm simm12_plus1:$y)), GPR:$rs3)>;
def : Pat<(select (XLenVT (setne GPR:$x, GPR:$y)), GPR:$rs1, GPR:$rs3),
          (CMOV GPR:$rs1, (XOR GPR:$x, GPR:$y), GPR:$rs3)>;
def : Pat<(select (XLenVT (seteq GPR:$x, GPR:$y)), GPR:$rs3, GPR:$rs1),
          (CMOV GPR:$rs1, (XOR GPR:$x, GPR:$y), GPR:$rs3)>;
def : Pat<(select (XLenVT (setuge GPR:$x, GPR:$y)), GPR:$rs3, GPR:$rs1),
          (CMOV GPR:$rs1, (SLTU GPR:$x, GPR:$y), GPR:$rs3)>;
def : Pat<(select (XLenVT (setule GPR:$y, GPR:$x)), GPR:$rs3, GPR:$rs1),
          (CMOV GPR:$rs1, (SLTU GPR:$x, GPR:$y), GPR:$rs3)>;
def : Pat<(select (XLenVT (setge GPR:$x, GPR:$y)), GPR:$rs3, GPR:$rs1),
          (CMOV GPR:$rs1, (SLT GPR:$x, GPR:$y), GPR:$rs3)>;
def : Pat<(select (XLenVT (setle GPR:$y, GPR:$x)), GPR:$rs3, GPR:$rs1),
          (CMOV GPR:$rs1, (SLT GPR:$x, GPR:$y), GPR:$rs3)>;
def : Pat<(select GPR:$rs2, GPR:$rs1, GPR:$rs3),
          (CMOV GPR:$rs1, GPR:$rs2, GPR:$rs3)>;
} // Predicates = [HasStdExtZbt]

// fshl and fshr concatenate their operands in the same order. fsr and fsl
// instruction use different orders. fshl will return its first operand for
// shift of zero, fshr will return its second operand. fsl and fsr both return
// $rs1 so the patterns need to have different operand orders.
//
// fshl and fshr only read the lower log2(xlen) bits of the shift amount, but
// fsl/fsr instructions read log2(xlen)+1 bits. DAG combine may have removed
// an AND mask on the shift amount that we need to add back to avoid a one in
// the extra bit.
// FIXME: If we can prove that the extra bit in the shift amount is zero, we
// don't need this mask.
let Predicates = [HasStdExtZbt, IsRV32] in {
def : Pat<(fshl GPR:$rs1, GPR:$rs3, GPR:$rs2),
          (FSL GPR:$rs1, (ANDI GPR:$rs2, 31), GPR:$rs3)>;
def : Pat<(fshr GPR:$rs3, GPR:$rs1, GPR:$rs2),
          (FSR GPR:$rs1, (ANDI GPR:$rs2, 31), GPR:$rs3)>;
}
let Predicates = [HasStdExtZbt, IsRV64] in {
def : Pat<(fshl GPR:$rs1, GPR:$rs3, GPR:$rs2),
          (FSL GPR:$rs1, (ANDI GPR:$rs2, 63), GPR:$rs3)>;
def : Pat<(fshr GPR:$rs3, GPR:$rs1, GPR:$rs2),
          (FSR GPR:$rs1, (ANDI GPR:$rs2, 63), GPR:$rs3)>;
}
let Predicates = [HasStdExtZbt] in {
def : Pat<(fshr GPR:$rs3, GPR:$rs1, uimmlog2xlen:$shamt),
          (FSRI GPR:$rs1, GPR:$rs3, uimmlog2xlen:$shamt)>;
// We can use FSRI for fshl by immediate if we subtract the immediate from
// XLen and swap the operands.
def : Pat<(fshl GPR:$rs3, GPR:$rs1, uimmlog2xlen:$shamt),
          (FSRI GPR:$rs1, GPR:$rs3, (ImmSubFromXLen uimmlog2xlen:$shamt))>;
} // Predicates = [HasStdExtZbt]

let Predicates = [HasStdExtZbb] in {
def : Pat<(ctlz GPR:$rs1), (CLZ GPR:$rs1)>;
def : Pat<(cttz GPR:$rs1), (CTZ GPR:$rs1)>;
def : Pat<(ctpop GPR:$rs1), (CPOP GPR:$rs1)>;
} // Predicates = [HasStdExtZbb]

let Predicates = [HasStdExtZbb] in {
def : Pat<(sext_inreg GPR:$rs1, i8), (SEXTB GPR:$rs1)>;
def : Pat<(sext_inreg GPR:$rs1, i16), (SEXTH GPR:$rs1)>;
}

let Predicates = [HasStdExtZbb] in {
def : Pat<(smin GPR:$rs1, GPR:$rs2), (MIN  GPR:$rs1, GPR:$rs2)>;
def : Pat<(smax GPR:$rs1, GPR:$rs2), (MAX  GPR:$rs1, GPR:$rs2)>;
def : Pat<(umin GPR:$rs1, GPR:$rs2), (MINU GPR:$rs1, GPR:$rs2)>;
def : Pat<(umax GPR:$rs1, GPR:$rs2), (MAXU GPR:$rs1, GPR:$rs2)>;
} // Predicates = [HasStdExtZbb]

let Predicates = [HasStdExtZbb, IsRV32] in {
def : Pat<(bswap GPR:$rs1), (REV8_RV32 GPR:$rs1)>;
} // Predicates = [HasStdExtZbb, IsRV32]

let Predicates = [HasStdExtZbb, IsRV64] in {
def : Pat<(bswap GPR:$rs1), (REV8_RV64 GPR:$rs1)>;
} // Predicates = [HasStdExtZbb, IsRV64]

let Predicates = [HasStdExtZbp, IsRV32] in
def : Pat<(or (and GPR:$rs1, 0x0000FFFF), (shl GPR:$rs2, (i32 16))),
          (PACK GPR:$rs1, GPR:$rs2)>;
let Predicates = [HasStdExtZbp, IsRV64] in
def : Pat<(or (and GPR:$rs1, 0x00000000FFFFFFFF), (shl GPR:$rs2, (i64 32))),
          (PACK GPR:$rs1, GPR:$rs2)>;
let Predicates = [HasStdExtZbp, IsRV32] in
def : Pat<(or (and GPR:$rs2, 0xFFFF0000), (srl GPR:$rs1, (i32 16))),
          (PACKU GPR:$rs1, GPR:$rs2)>;
let Predicates = [HasStdExtZbp, IsRV64] in
def : Pat<(or (and GPR:$rs2, 0xFFFFFFFF00000000), (srl GPR:$rs1, (i64 32))),
          (PACKU GPR:$rs1, GPR:$rs2)>;
let Predicates = [HasStdExtZbp] in
def : Pat<(or (and (shl GPR:$rs2, (XLenVT 8)), 0xFF00),
              (and GPR:$rs1, 0x00FF)),
          (PACKH GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasStdExtZbbOrZbp, IsRV32] in
def : Pat<(and GPR:$rs, 0x0000FFFF), (ZEXTH_RV32 GPR:$rs)>;
let Predicates = [HasStdExtZbbOrZbp, IsRV64] in {
def : Pat<(and GPR:$rs, 0x000000000000FFFF), (ZEXTH_RV64 GPR:$rs)>;
}

let Predicates = [HasStdExtZbp, IsRV32] in {
def : Pat<(or (or (and (shl GPR:$rs1, (i32 8)), (i32 0x00FF0000)),
                  (and GPR:$rs1, (i32 0xFF0000FF))),
              (and (srl GPR:$rs1, (i32 8)), (i32 0x0000FF00))),
          (SHFLI GPR:$rs1, (i32 8))>;
def : Pat<(or (or (and (shl GPR:$rs1, (i32 4)), (i32 0x0F000F00)),
                  (and GPR:$rs1, (i32 0xF00FF00F))),
              (and (srl GPR:$rs1, (i32 4)), (i32 0x00F000F0))),
          (SHFLI GPR:$rs1, (i32 4))>;
def : Pat<(or (or (and (shl GPR:$rs1, (i32 2)), (i32 0x30303030)),
                  (and GPR:$rs1, (i32 0xC3C3C3C3))),
              (and (srl GPR:$rs1, (i32 2)), (i32 0x0C0C0C0C))),
          (SHFLI GPR:$rs1, (i32 2))>;
def : Pat<(or (or (and (shl GPR:$rs1, (i32 1)), (i32 0x44444444)),
                  (and GPR:$rs1, (i32 0x99999999))),
              (and (srl GPR:$rs1, (i32 1)), (i32 0x22222222))),
          (SHFLI GPR:$rs1, (i32 1))>;
} // Predicates = [HasStdExtZbp, IsRV32]

let Predicates = [HasStdExtZbp, IsRV64] in {
def : Pat<(or (or (and (shl GPR:$rs1, (i64 16)), (i64 0x0000FFFF00000000)),
                  (and GPR:$rs1, (i64 0xFFFF00000000FFFF))),
              (and (srl GPR:$rs1, (i64 16)), (i64 0x00000000FFFF0000))),
          (SHFLI GPR:$rs1, (i64 16))>;
def : Pat<(or (or (and (shl GPR:$rs1, (i64 8)), (i64 0x00FF000000FF0000)),
                  (and GPR:$rs1, (i64 0xFF0000FFFF0000FF))),
              (and (srl GPR:$rs1, (i64 8)), (i64 0x0000FF000000FF00))),
          (SHFLI GPR:$rs1, (i64 8))>;
def : Pat<(or (or (and (shl GPR:$rs1, (i64 4)), (i64 0x0F000F000F000F00)),
                  (and GPR:$rs1, (i64 0xF00FF00FF00FF00F))),
              (and (srl GPR:$rs1, (i64 4)), (i64 0x00F000F000F000F0))),
          (SHFLI GPR:$rs1, (i64 4))>;
def : Pat<(or (or (and (shl GPR:$rs1, (i64 2)), (i64 0x3030303030303030)),
                  (and GPR:$rs1, (i64 0xC3C3C3C3C3C3C3C3))),
              (and (srl GPR:$rs1, (i64 2)), (i64 0x0C0C0C0C0C0C0C0C))),
          (SHFLI GPR:$rs1, (i64 2))>;
def : Pat<(or (or (and (shl GPR:$rs1, (i64 1)), (i64 0x4444444444444444)),
                  (and GPR:$rs1, (i64 0x9999999999999999))),
              (and (srl GPR:$rs1, (i64 1)), (i64 0x2222222222222222))),
          (SHFLI GPR:$rs1, (i64 1))>;
} // Predicates = [HasStdExtZbp, IsRV64]

let Predicates = [HasStdExtZba] in {
def : Pat<(add (shl GPR:$rs1, (XLenVT 1)), GPR:$rs2),
          (SH1ADD GPR:$rs1, GPR:$rs2)>;
def : Pat<(add (shl GPR:$rs1, (XLenVT 2)), GPR:$rs2),
          (SH2ADD GPR:$rs1, GPR:$rs2)>;
def : Pat<(add (shl GPR:$rs1, (XLenVT 3)), GPR:$rs2),
          (SH3ADD GPR:$rs1, GPR:$rs2)>;
} // Predicates = [HasStdExtZba]

let Predicates = [HasStdExtZba, IsRV64] in {
def : Pat<(SLLIUWPat GPR:$rs1, uimm5:$shamt),
          (SLLIUW GPR:$rs1, uimm5:$shamt)>;
def : Pat<(shl (and GPR:$rs1, 0xFFFFFFFF), uimm5:$shamt),
          (SLLIUW GPR:$rs1, uimm5:$shamt)>;
def : Pat<(add (and GPR:$rs1, (i64 0xFFFFFFFF)), GPR:$rs2),
          (ADDUW GPR:$rs1, GPR:$rs2)>;
def : Pat<(and GPR:$rs, 0x00000000FFFFFFFF), (ADDUW GPR:$rs, X0)>;

def : Pat<(add (shl (and GPR:$rs1, (i64 0xFFFFFFFF)), (XLenVT 1)), GPR:$rs2),
          (SH1ADDUW GPR:$rs1, GPR:$rs2)>;
def : Pat<(add (shl (and GPR:$rs1, (i64 0xFFFFFFFF)), (XLenVT 2)), GPR:$rs2),
          (SH2ADDUW GPR:$rs1, GPR:$rs2)>;
def : Pat<(add (shl (and GPR:$rs1, (i64 0xFFFFFFFF)), (XLenVT 3)), GPR:$rs2),
          (SH3ADDUW GPR:$rs1, GPR:$rs2)>;

def : Pat<(add (SLLIUWPat GPR:$rs1, (XLenVT 1)), GPR:$rs2),
          (SH1ADDUW GPR:$rs1, GPR:$rs2)>;
def : Pat<(add (SLLIUWPat GPR:$rs1, (XLenVT 2)), GPR:$rs2),
          (SH2ADDUW GPR:$rs1, GPR:$rs2)>;
def : Pat<(add (SLLIUWPat GPR:$rs1, (XLenVT 3)), GPR:$rs2),
          (SH3ADDUW GPR:$rs1, GPR:$rs2)>;
} // Predicates = [HasStdExtZba, IsRV64]

let Predicates = [HasStdExtZbbOrZbp, IsRV64] in {
def : Pat<(riscv_rolw GPR:$rs1, GPR:$rs2),
          (ROLW GPR:$rs1, GPR:$rs2)>;
def : Pat<(riscv_rorw GPR:$rs1, GPR:$rs2),
          (RORW GPR:$rs1, GPR:$rs2)>;
def : Pat<(riscv_rorw GPR:$rs1, uimm5:$rs2),
          (RORIW GPR:$rs1, uimm5:$rs2)>;
def : Pat<(riscv_rolw GPR:$rs1, uimm5:$rs2),
          (RORIW GPR:$rs1, (ImmSubFrom32 uimm5:$rs2))>;
} // Predicates = [HasStdExtZbbOrZbp, IsRV64]

let Predicates = [HasStdExtZbs, IsRV64] in {
def : Pat<(and (not (riscv_sllw 1, GPR:$rs2)), (assertsexti32 GPR:$rs1)),
          (BCLRW GPR:$rs1, GPR:$rs2)>;
def : Pat<(sext_inreg (and (not (riscv_sllw 1, GPR:$rs2)), GPR:$rs1), i32),
          (BCLRW GPR:$rs1, GPR:$rs2)>;
def : Pat<(or (riscv_sllw 1, GPR:$rs2), (assertsexti32 GPR:$rs1)),
          (BSETW GPR:$rs1, GPR:$rs2)>;
def : Pat<(sext_inreg (or (riscv_sllw 1, GPR:$rs2), GPR:$rs1), i32),
          (BSETW GPR:$rs1, GPR:$rs2)>;
def : Pat<(xor (riscv_sllw 1, GPR:$rs2), (assertsexti32 GPR:$rs1)),
          (BINVW GPR:$rs1, GPR:$rs2)>;
def : Pat<(sext_inreg (xor (riscv_sllw 1, GPR:$rs2), GPR:$rs1), i32),
          (BINVW GPR:$rs1, GPR:$rs2)>;
def : Pat<(and (riscv_srlw GPR:$rs1, GPR:$rs2), 1),
          (BEXTW GPR:$rs1, GPR:$rs2)>;

def : Pat<(riscv_sllw 1, GPR:$rs2),
          (BSETW X0, GPR:$rs2)>;

def : Pat<(and (assertsexti32 GPR:$rs1), BCLRWMask:$mask),
          (BCLRIW GPR:$rs1, (BCLRXForm imm:$mask))>;
def : Pat<(or (assertsexti32 GPR:$rs1), BSETINVWMask:$mask),
          (BSETIW GPR:$rs1, (BSETINVXForm imm:$mask))>;
def : Pat<(xor (assertsexti32 GPR:$rs1), BSETINVWMask:$mask),
          (BINVIW GPR:$rs1, (BSETINVXForm imm:$mask))>;

} // Predicates = [HasStdExtZbs, IsRV64]

let Predicates = [HasStdExtZbp, IsRV64] in {
def : Pat<(riscv_rorw (riscv_greviw GPR:$rs1, 24), (i64 16)), (GREVIW GPR:$rs1, 8)>;
def : Pat<(riscv_rolw (riscv_greviw GPR:$rs1, 24), (i64 16)), (GREVIW GPR:$rs1, 8)>;
def : Pat<(riscv_greviw GPR:$rs1, timm:$shamt), (GREVIW GPR:$rs1, timm:$shamt)>;
def : Pat<(riscv_gorciw GPR:$rs1, timm:$shamt), (GORCIW GPR:$rs1, timm:$shamt)>;
} // Predicates = [HasStdExtZbp, IsRV64]

let Predicates = [HasStdExtZbt, IsRV64] in {
def : Pat<(riscv_fslw GPR:$rs1, GPR:$rs3, GPR:$rs2),
          (FSLW GPR:$rs1, GPR:$rs2, GPR:$rs3)>;
def : Pat<(riscv_fsrw GPR:$rs3, GPR:$rs1, GPR:$rs2),
          (FSRW GPR:$rs1, GPR:$rs2, GPR:$rs3)>;
def : Pat<(riscv_fsrw GPR:$rs3, GPR:$rs1, uimm5:$shamt),
          (FSRIW GPR:$rs1, GPR:$rs3, uimm5:$shamt)>;
def : Pat<(riscv_fslw GPR:$rs3, GPR:$rs1, uimm5:$shamt),
          (FSRIW GPR:$rs1, GPR:$rs3, (ImmSubFrom32 uimm5:$shamt))>;
} // Predicates = [HasStdExtZbt, IsRV64]

let Predicates = [HasStdExtZbb, IsRV64] in {
def : Pat<(add (ctlz (and GPR:$rs1, (i64 0xFFFFFFFF))), (i64 -32)),
          (CLZW GPR:$rs1)>;
// computeKnownBits can't figure out that the and mask on the add result is
// unnecessary so we need to pattern match it away.
def : Pat<(and (add (ctlz (and GPR:$rs1, (i64 0xFFFFFFFF))), (i64 -32)),
               (i64 0xFFFFFFFF)),
          (CLZW GPR:$rs1)>;
def : Pat<(cttz (or GPR:$rs1, (i64 0x100000000))),
          (CTZW GPR:$rs1)>;
def : Pat<(ctpop (and GPR:$rs1, (i64 0xFFFFFFFF))), (CPOPW GPR:$rs1)>;
} // Predicates = [HasStdExtZbb, IsRV64]

let Predicates = [HasStdExtZbp, IsRV64] in {
def : Pat<(sext_inreg (or (shl GPR:$rs2, (i64 16)),
                          (and GPR:$rs1, 0x000000000000FFFF)),
                      i32),
          (PACKW GPR:$rs1, GPR:$rs2)>;
def : Pat<(or (and (assertsexti32 GPR:$rs2), 0xFFFFFFFFFFFF0000),
              (SRLIWPat GPR:$rs1, (i64 16))),
          (PACKUW GPR:$rs1, GPR:$rs2)>;
} // Predicates = [HasStdExtZbp, IsRV64]
