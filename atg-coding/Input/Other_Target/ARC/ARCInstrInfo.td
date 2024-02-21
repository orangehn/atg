include "ARCInstrFormats.td"

// Generic 3 operand binary instructions (i.e., add r0, r1, r2).
multiclass ArcBinaryInst<bits<5> major, bits<6> mincode,
                       string opasm> {
  // 3 register variant.
  def _rrr : F32_DOP_RR<major, mincode, 0, (outs GPR32:$A),
                        (ins GPR32:$B, GPR32:$C),
                        !strconcat(opasm, "\t$A, $B, $C"),
                        []>;
  def _f_rrr : F32_DOP_RR<major, mincode, 1, (outs GPR32:$A),
                          (ins GPR32:$B, GPR32:$C),
                          !strconcat(opasm, ".f\t$A, $B, $C"),
                          []>
  { let Defs = [STATUS32]; }

  // 2 register with unsigned 6-bit immediate variant.
  def _rru6 : F32_DOP_RU6<major, mincode, 0, (outs GPR32:$A),
                          (ins GPR32:$B, immU6:$U6),
                          !strconcat(opasm, "\t$A, $B, $U6"),
                          []>;
  def _f_rru6 : F32_DOP_RU6<major, mincode, 1, (outs GPR32:$A),
                            (ins GPR32:$B, immU6:$U6),
                            !strconcat(opasm, ".f\t$A, $B, $U6"),
                            []>
  { let Defs = [STATUS32]; }

  // 2 register with 32-bit immediate variant.
  def _rrlimm : F32_DOP_RLIMM<major, mincode, 0,
                              (outs GPR32:$A),
                              (ins GPR32:$B, i32imm:$LImm),
                              !strconcat(opasm, "\t$A, $B, $LImm"),
                              []>;
  def _f_rrlimm : F32_DOP_RLIMM<major, mincode, 1,
                                (outs GPR32:$A),
                                (ins GPR32:$B, i32imm:$LImm),
                                !strconcat(opasm, ".f\t$A, $B, $LImm"),
                                []>
  { let Defs = [STATUS32]; }

  // 2 matched-register with signed 12-bit immediate variant (add r0, r0, -1).
  def _rrs12 : F32_DOP_RS12<major, mincode, 0,
                            (outs GPR32:$B),
                            (ins GPR32:$in, immS<12>:$S12),
                            !strconcat(opasm, "\t$B, $in, $S12"),
                            []>
  { let Constraints = "$B = $in"; }
  def _f_rrs12 : F32_DOP_RS12<major, mincode, 1,
                              (outs GPR32:$B),
                              (ins GPR32:$in, immS<12>:$S12),
                              !strconcat(opasm, ".f\t$B, $in, $S12"),
                              []>
  { let Constraints = "$B = $in"; let Defs = [STATUS32]; }
}

// Special multivariant GEN4 DOP format instruction that take 2 registers.
// This is the class that is used for various comparison instructions.
multiclass ArcSpecialDOPInst<bits<6> subop, string opasm, bit F> {
  def _rr : F32_DOP_RR<0b00100, subop, F, (outs), (ins GPR32:$B, GPR32:$C),
               !strconcat(opasm, "\t$B, $C"),
               []>;

  def _ru6 : F32_DOP_RU6<0b00100, subop, F, (outs), (ins GPR32:$B, i32imm:$U6),
               !strconcat(opasm, "\t$B, $U6"),
               []>;

  def _rlimm : F32_DOP_RLIMM<0b00100, subop, F, (outs),
               (ins GPR32:$B, i32imm:$LImm),
               !strconcat(opasm, "\t$B, $LImm"),
               []>;
}

// Generic 2-operand unary instructions.
multiclass ArcUnaryInst<bits<5> major, bits<6> subop,
                        string opasm> {
  def _rr : F32_SOP_RR<major, subop, 0, (outs GPR32:$B), (ins GPR32:$C),
                       !strconcat(opasm, "\t$B, $C"), []>;

  def _f_rr : F32_SOP_RR<major, subop, 1, (outs GPR32:$B), (ins GPR32:$C),
                       !strconcat(opasm, ".f\t$B, $C"), []>
  { let Defs = [STATUS32]; }
}


//multiclass ArcBinaryGEN4Inst<bits<6> mincode, string opasm> :
//  ArcBinaryInst<0b00100, mincode, opasm>;
//multiclass ArcBinaryEXT5Inst<bits<6> mincode, string opasm> :
 // ArcBinaryInst<0b00101, mincode, opasm>;

//multiclass ArcUnaryGEN4Inst<bits<6> mincode, string opasm> :
 // ArcUnaryInst<0b00100, mincode, opasm>;

defm ADD : ArcBinaryInst<0b00100, 0b000000, "add">;
defm SUB : ArcBinaryInst<0b00100, 0b000010, "sub">;
defm SUB1 :  ArcBinaryInst<0b00100, 0b010111, "sub1">;
defm SUB2 :  ArcBinaryInst<0b00100, 0b011000, "sub2">;
defm SUB3 :  ArcBinaryInst<0b00100, 0b011001, "sub3">;
defm OR  :  ArcBinaryInst<0b00100, 0b000101, "or">;
defm AND :  ArcBinaryInst<0b00100, 0b000100, "and">;
defm XOR :  ArcBinaryInst<0b00100, 0b000111, "xor">;
defm MAX :  ArcBinaryInst<0b00100, 0b001000, "max">;
defm MIN :  ArcBinaryInst<0b00100, 0b001001, "min">;
defm ASL :  ArcBinaryInst<0b00101, 0b000000, "asl">;
defm LSR :  ArcBinaryInst<0b00101, 0b000001, "lsr">;
defm ASR :  ArcBinaryInst<0b00101, 0b000010, "asr">;
defm ROR :  ArcBinaryInst<0b00101, 0b000011, "ror">;
defm MPY  : ArcBinaryInst<0b00100, 0b011010, "mpy">;
defm MPYM : ArcBinaryInst<0b00100, 0b011011, "mpym">;
defm MPYMU : ArcBinaryInst<0b00100, 0b011100, "mpymu">;
defm SETEQ : ArcBinaryInst<0b00100, 0b111000, "seteq">;

//defm ADD : ArcBinaryGEN4Inst<0b000000, "add">;
//defm SUB : ArcBinaryGEN4Inst<0b000010, "sub">;
//defm SUB1 : ArcBinaryGEN4Inst<0b010111, "sub1">;
//defm SUB2 : ArcBinaryGEN4Inst<0b011000, "sub2">;
//defm SUB3 : ArcBinaryGEN4Inst<0b011001, "sub3">;
//defm OR  : ArcBinaryGEN4Inst<0b000101, "or">;
//defm AND : ArcBinaryGEN4Inst<0b000100, "and">;
//defm XOR : ArcBinaryGEN4Inst<0b000111, "xor">;
//defm MAX : ArcBinaryGEN4Inst<0b001000, "max">;
//defm MIN : ArcBinaryGEN4Inst<0b001001, "min">;
//defm ASL : ArcBinaryEXT5Inst<0b000000, "asl">;
//defm LSR : ArcBinaryEXT5Inst<0b000001, "lsr">;
//defm ASR : ArcBinaryEXT5Inst<0b000010, "asr">;
//defm ROR : ArcBinaryEXT5Inst<0b000011, "ror">;
//defm MPY  : ArcBinaryGEN4Inst<0b011010, "mpy">;
//defm MPYM : ArcBinaryGEN4Inst<0b011011, "mpym">;
//defm MPYMU : ArcBinaryGEN4Inst<0b011100, "mpymu">;
//defm SETEQ : ArcBinaryGEN4Inst<0b111000, "seteq">;

// ---------------------------------------------------------------------------
// Unary Instruction definitions.
// ---------------------------------------------------------------------------
// General unary instruction definitions.
defm SEXB : ArcUnaryInst<0b00100, 0b000101, "sexb">;
defm SEXH : ArcUnaryInst<0b00100, 0b000110, "sexh">;

// Comparison instruction definition
let isCompare = 1, Defs = [STATUS32] in {
defm CMP : ArcSpecialDOPInst<0b001100, "cmp", 1>;
}


// ---------------------------------------------------------------------------
// MOV instruction and variants (conditional mov).
// ---------------------------------------------------------------------------
let isReMaterializable = 1, isAsCheapAsAMove = 1, isMoveImm = 1 in {
def MOV_rs12 : F32_DOP_RS12<0b00100, 0b001010, 0,
                 (outs GPR32:$B), (ins immS<12>:$S12),
                 "mov\t$B, $S12",
                 [(set GPR32:$B, immS<12>:$S12)]>;
}
def MOV_rr : F32_DOP_RR<0b00100, 0b001010, 0,
                (outs GPR32:$B), (ins GPR32:$C),
                "mov\t$B, $C", []>;

def MOV_rlimm : F32_DOP_RLIMM<0b00100, 0b001010, 0,
                      (outs GPR32:$B), (ins i32imm:$LImm),
                      "mov\t$B, $LImm", []>;

def MOV_ru6 : F32_DOP_RU6<0b00100, 0b001010, 0,
                          (outs GPR32:$B), (ins immU6:$U6),
                          "mov\t$B, $U6", []>;

let Uses = [STATUS32] in {
def MOVcc : F32_DOP_CC_RR<0b00100, 0b001010, 0,
               (outs GPR32:$B),
               (ins GPR32:$C, GPR32:$fval, cmovpred:$cc),
               !strconcat("mov.", "$cc\t$B, $C"),
               [(set GPR32:$B, (cmov i32:$C, i32:$fval, cmovpred:$cc))]> {
  let Constraints = "$B = $fval";
}
}


// ---------------------------------------------------------------------------
// Control flow instructions (branch, return, calls, etc).
// ---------------------------------------------------------------------------

// Branch instructions
let isBranch = 1, isTerminator = 1 in {

  // Unconditional branch.
  let isBarrier = 1 in
  def BR : F32_BR0_UCOND_FAR<(outs), (ins btargetS25:$S25),
                             "b\t$S25", [(br bb:$S25)]>;

  let Uses=[STATUS32] in
  // Conditional branch.
  def Bcc : F32_BR0_COND<(outs), (ins btargetS21:$S21, ccond:$cc),
                         "b$cc\t$S21", []>;

  // Compare and branch (limited range).
  def BRcc_rr  : F32_BR1_BCC<(outs),
                             (ins btargetS9:$S9, GPR32:$B, GPR32:$C, brccond:$cc),
                             "br$cc\t$B, $C, $S9", 0, []>;
  def BRcc_ru6 : F32_BR1_BCC<(outs),
                             (ins btargetS9:$S9, GPR32:$B, immU6:$C, brccond:$cc),
                             "br$cc\t$B, $C, $S9", 1, []>;
} 

// Unconditional Jump.
let isBranch = 1, isTerminator = 1, isBarrier = 1 in {
  // Indirect.
  let isIndirectBranch = 1 in
  def J :  F32_DOP_RR<0b00100, 0b100000, 0,
                      (outs), (ins GPR32:$C),
                      "j\t[$C]", [(brind i32:$C)]>;

  // Direct.
  def J_LImm : F32_DOP_RLIMM<0b00100, 0b100000, 0,
                             (outs), (ins i32imm:$LImm),
                             "j\t$LImm", []>;
}

let isCall = 1, isBarrier = 1, Defs = [BLINK], Uses = [SP] in {
  def BL : F32_BR1_BL_UCOND_FAR<(outs), (ins calltargetS25:$S25),
                      "bl\t$S25", [(ARCBranchLink tglobaladdr:$S25)]>;

  let isIndirectBranch = 1 in
  def JL : F32_DOP_RR<0b00100, 0b100010, 0, (outs), (ins GPR32:$C),
                     "jl\t[$C]", [(ARCJumpLink i32:$C)]>;

  // Direct unconditional call.
  def JL_LImm : F32_DOP_RLIMM<0b00100, 0b100010, 0, (outs), (ins i32imm:$LImm),
                              "jl\t$LImm", []>;
} 

