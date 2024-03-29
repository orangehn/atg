//===- Mips64InstrInfo.td - Mips64 Instruction Information -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes Mips64 instructions.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Mips Operand, Complex Patterns and Transformations Definitions.
//===----------------------------------------------------------------------===//




//===----------------------------------------------------------------------===//
// Instruction definition
//===----------------------------------------------------------------------===//
let DecoderNamespace = "Mips64" in {
/// Arithmetic Instructions (ALU Immediate)
def DADDi   : ArithLogicI<"daddi", simm16_64, GPR64Opnd, II_DADDI>,
              ADDI_FM<0x18>, ISA_MIPS3_NOT_32R6_64R6;
let AdditionalPredicates = [NotInMicroMips] in {
  def DADDiu : ArithLogicI<"daddiu", simm16_64, GPR64Opnd, II_DADDIU,
                           immSExt16, add>,
               ADDI_FM<0x19>, IsAsCheapAsAMove, ISA_MIPS3;
}

let isCodeGenOnly = 1 in {
def SLTi64  : SetCC_I<"slti", setlt, simm16_64, immSExt16, GPR64Opnd>,
              SLTI_FM<0xa>, GPR_64;
def SLTiu64 : SetCC_I<"sltiu", setult, simm16_64, immSExt16, GPR64Opnd>,
              SLTI_FM<0xb>, GPR_64;
def ANDi64 : ArithLogicI<"andi", uimm16_64, GPR64Opnd, II_AND, immZExt16, and>,
             ADDI_FM<0xc>, GPR_64;
def ORi64   : ArithLogicI<"ori", uimm16_64, GPR64Opnd, II_OR, immZExt16, or>,
              ADDI_FM<0xd>, GPR_64;
def XORi64  : ArithLogicI<"xori", uimm16_64, GPR64Opnd, II_XOR, immZExt16, xor>,
              ADDI_FM<0xe>, GPR_64;
def LUi64   : LoadUpper<"lui", GPR64Opnd, uimm16_64_relaxed>, LUI_FM, GPR_64;
}

/// Arithmetic Instructions (3-Operand, R-Type)
let AdditionalPredicates = [NotInMicroMips] in {
  def DADD   : ArithLogicR<"dadd", GPR64Opnd, 1, II_DADD>, ADD_FM<0, 0x2c>,
               ISA_MIPS3;
  def DADDu  : ArithLogicR<"daddu", GPR64Opnd, 1, II_DADDU, add>,
               ADD_FM<0, 0x2d>, ISA_MIPS3;
  def DSUBu  : ArithLogicR<"dsubu", GPR64Opnd, 0, II_DSUBU, sub>,
               ADD_FM<0, 0x2f>, ISA_MIPS3;
  def DSUB   : ArithLogicR<"dsub", GPR64Opnd, 0, II_DSUB>, ADD_FM<0, 0x2e>,
               ISA_MIPS3;
}

let isCodeGenOnly = 1 in {
def SLT64  : SetCC_R<"slt", setlt, GPR64Opnd>, ADD_FM<0, 0x2a>, GPR_64;
def SLTu64 : SetCC_R<"sltu", setult, GPR64Opnd>, ADD_FM<0, 0x2b>, GPR_64;
def AND64  : ArithLogicR<"and", GPR64Opnd, 1, II_AND, and>, ADD_FM<0, 0x24>,
             GPR_64;
def OR64   : ArithLogicR<"or", GPR64Opnd, 1, II_OR, or>, ADD_FM<0, 0x25>,
             GPR_64;
def XOR64  : ArithLogicR<"xor", GPR64Opnd, 1, II_XOR, xor>, ADD_FM<0, 0x26>,
             GPR_64;
def NOR64  : LogicNOR<"nor", GPR64Opnd>, ADD_FM<0, 0x27>, GPR_64;
}

/// Shift Instructions
let AdditionalPredicates = [NotInMicroMips] in {
  def DSLL : shift_rotate_imm<"dsll", uimm6, GPR64Opnd, II_DSLL, shl,
                              immZExt6>,
             SRA_FM<0x38, 0>, ISA_MIPS3;
  def DSRL : shift_rotate_imm<"dsrl", uimm6, GPR64Opnd, II_DSRL, srl,
                              immZExt6>,
             SRA_FM<0x3a, 0>, ISA_MIPS3;
  def DSRA : shift_rotate_imm<"dsra", uimm6, GPR64Opnd, II_DSRA, sra,
                              immZExt6>,
             SRA_FM<0x3b, 0>, ISA_MIPS3;
  def DSLLV  : shift_rotate_reg<"dsllv", GPR64Opnd, II_DSLLV, shl>,
               SRLV_FM<0x14, 0>, ISA_MIPS3;
  def DSRAV  : shift_rotate_reg<"dsrav", GPR64Opnd, II_DSRAV, sra>,
               SRLV_FM<0x17, 0>, ISA_MIPS3;
  def DSRLV  : shift_rotate_reg<"dsrlv", GPR64Opnd, II_DSRLV, srl>,
               SRLV_FM<0x16, 0>, ISA_MIPS3;
  def DSLL32 : shift_rotate_imm<"dsll32", uimm5, GPR64Opnd, II_DSLL32>,
               SRA_FM<0x3c, 0>, ISA_MIPS3;
  def DSRL32 : shift_rotate_imm<"dsrl32", uimm5, GPR64Opnd, II_DSRL32>,
               SRA_FM<0x3e, 0>, ISA_MIPS3;
  def DSRA32 : shift_rotate_imm<"dsra32", uimm5, GPR64Opnd, II_DSRA32>,
               SRA_FM<0x3f, 0>, ISA_MIPS3;

// Rotate Instructions
  def DROTR  : shift_rotate_imm<"drotr", uimm6, GPR64Opnd, II_DROTR, rotr,
                                immZExt6>,
               SRA_FM<0x3a, 1>, ISA_MIPS64R2;
  def DROTRV : shift_rotate_reg<"drotrv", GPR64Opnd, II_DROTRV, rotr>,
               SRLV_FM<0x16, 1>, ISA_MIPS64R2;
  def DROTR32 : shift_rotate_imm<"drotr32", uimm5, GPR64Opnd, II_DROTR32>,
                SRA_FM<0x3e, 1>, ISA_MIPS64R2;
}

/// Load and Store Instructions
///  aligned
let isCodeGenOnly = 1 in {
def LB64  : Load<"lb", GPR64Opnd, sextloadi8, II_LB>, LW_FM<0x20>, GPR_64;
def LBu64 : Load<"lbu", GPR64Opnd, zextloadi8, II_LBU>, LW_FM<0x24>, GPR_64;
def LH64  : Load<"lh", GPR64Opnd, sextloadi16, II_LH>, LW_FM<0x21>, GPR_64;
def LHu64 : Load<"lhu", GPR64Opnd, zextloadi16, II_LHU>, LW_FM<0x25>, GPR_64;
def LW64  : Load<"lw", GPR64Opnd, sextloadi32, II_LW>, LW_FM<0x23>, GPR_64;
def SB64  : Store<"sb", GPR64Opnd, truncstorei8, II_SB>, LW_FM<0x28>, GPR_64;
def SH64  : Store<"sh", GPR64Opnd, truncstorei16, II_SH>, LW_FM<0x29>,
            GPR_64;
def SW64  : Store<"sw", GPR64Opnd, truncstorei32, II_SW>, LW_FM<0x2b>,
            GPR_64;
}

let AdditionalPredicates = [NotInMicroMips] in {
  def LWu : MMRel, Load<"lwu", GPR64Opnd, zextloadi32, II_LWU>,
            LW_FM<0x27>, ISA_MIPS3;
  def LD  : LoadMemory<"ld", GPR64Opnd, mem_simmptr, load, II_LD>,
            LW_FM<0x37>, ISA_MIPS3;
  def SD  : StoreMemory<"sd", GPR64Opnd, mem_simmptr, store, II_SD>,
            LW_FM<0x3f>, ISA_MIPS3;
}



/// load/store left/right
let isCodeGenOnly = 1 in {
def LWL64 : LoadLeftRight<"lwl", MipsLWL, GPR64Opnd, II_LWL>, LW_FM<0x22>,
            GPR_64;
def LWR64 : LoadLeftRight<"lwr", MipsLWR, GPR64Opnd, II_LWR>, LW_FM<0x26>,
            GPR_64;
def SWL64 : StoreLeftRight<"swl", MipsSWL, GPR64Opnd, II_SWL>, LW_FM<0x2a>,
            GPR_64;
def SWR64 : StoreLeftRight<"swr", MipsSWR, GPR64Opnd, II_SWR>, LW_FM<0x2e>,
            GPR_64;
}

def LDL   : LoadLeftRight<"ldl", MipsLDL, GPR64Opnd, II_LDL>, LW_FM<0x1a>,
            ISA_MIPS3_NOT_32R6_64R6;
def LDR   : LoadLeftRight<"ldr", MipsLDR, GPR64Opnd, II_LDR>, LW_FM<0x1b>,
            ISA_MIPS3_NOT_32R6_64R6;
def SDL   : StoreLeftRight<"sdl", MipsSDL, GPR64Opnd, II_SDL>, LW_FM<0x2c>,
            ISA_MIPS3_NOT_32R6_64R6;
def SDR   : StoreLeftRight<"sdr", MipsSDR, GPR64Opnd, II_SDR>, LW_FM<0x2d>,
            ISA_MIPS3_NOT_32R6_64R6;

/// Load-linked, Store-conditional
let AdditionalPredicates = [NotInMicroMips] in {
  def LLD : LLBase<"lld", GPR64Opnd, mem_simmptr>, LW_FM<0x34>,
            ISA_MIPS3_NOT_32R6_64R6;
}
def SCD : SCBase<"scd", GPR64Opnd>, LW_FM<0x3c>, ISA_MIPS3_NOT_32R6_64R6;

let AdditionalPredicates = [NotInMicroMips],
    DecoderNamespace = "Mips32_64_PTR64" in {
def LL64 : LLBase<"ll", GPR32Opnd>, LW_FM<0x30>, PTR_64,
           ISA_MIPS2_NOT_32R6_64R6;
def SC64 : SCBase<"sc", GPR32Opnd>, LW_FM<0x38>, PTR_64,
           ISA_MIPS2_NOT_32R6_64R6;
def JR64   : IndirectBranch<"jr", GPR64Opnd>, MTLO_FM<8>, PTR_64;
}

def JALR64 : JumpLinkReg<"jalr", GPR64Opnd>, JALR_FM, PTR_64;

/// Jump and Branch Instructions
let isCodeGenOnly = 1 in {
  def BEQ64  : CBranch<"beq", brtarget, seteq, GPR64Opnd>, BEQ_FM<4>,
               GPR_64;
  def BNE64  : CBranch<"bne", brtarget, setne, GPR64Opnd>, BEQ_FM<5>,
               GPR_64;
  def BGEZ64 : CBranchZero<"bgez", brtarget, setge, GPR64Opnd>, BGEZ_FM<1, 1>,
               GPR_64;
  def BGTZ64 : CBranchZero<"bgtz", brtarget, setgt, GPR64Opnd>, BGEZ_FM<7, 0>,
               GPR_64;
  def BLEZ64 : CBranchZero<"blez", brtarget, setle, GPR64Opnd>, BGEZ_FM<6, 0>,
               GPR_64;
  def BLTZ64 : CBranchZero<"bltz", brtarget, setlt, GPR64Opnd>, BGEZ_FM<1, 0>,
               GPR_64;
}
let AdditionalPredicates = [NotInMicroMips],
    DecoderNamespace = "Mips64" in {
  def JR_HB64 : JR_HB_DESC<GPR64Opnd>, JR_HB_ENC, ISA_MIPS64_NOT_64R6;
  def JALR_HB64 : JALR_HB_DESC<GPR64Opnd>, JALR_HB_ENC, ISA_MIPS64R2;
}


/// Multiply and Divide Instructions.
let AdditionalPredicates = [NotInMicroMips] in {
  def DMULT  : Mult<"dmult", II_DMULT, GPR64Opnd, [HI0_64, LO0_64]>,
               MULT_FM<0, 0x1c>, ISA_MIPS3_NOT_32R6_64R6;
  def DMULTu : Mult<"dmultu", II_DMULTU, GPR64Opnd, [HI0_64, LO0_64]>,
               MULT_FM<0, 0x1d>, ISA_MIPS3_NOT_32R6_64R6;
}

let AdditionalPredicates = [NotInMicroMips] in {
  def DSDIV : Div<"ddiv", II_DDIV, GPR64Opnd, [HI0_64, LO0_64]>,
              MULT_FM<0, 0x1e>, ISA_MIPS3_NOT_32R6_64R6;
  def DUDIV : Div<"ddivu", II_DDIVU, GPR64Opnd, [HI0_64, LO0_64]>,
              MULT_FM<0, 0x1f>, ISA_MIPS3_NOT_32R6_64R6;
}

let isCodeGenOnly = 1 in {
def MTHI64 : MoveToLOHI<"mthi", GPR64Opnd, [HI0_64]>, MTLO_FM<0x11>,
             ISA_MIPS3_NOT_32R6_64R6;
def MTLO64 : MoveToLOHI<"mtlo", GPR64Opnd, [LO0_64]>, MTLO_FM<0x13>,
             ISA_MIPS3_NOT_32R6_64R6;
def MFHI64 : MoveFromLOHI<"mfhi", GPR64Opnd, AC0_64>, MFLO_FM<0x10>,
             ISA_MIPS3_NOT_32R6_64R6;
def MFLO64 : MoveFromLOHI<"mflo", GPR64Opnd, AC0_64>, MFLO_FM<0x12>,
             ISA_MIPS3_NOT_32R6_64R6;
/// Sign Ext In Register Instructions.
def SEB64 : SignExtInReg<"seb", i8, GPR64Opnd, II_SEB>, SEB_FM<0x10, 0x20>,
            ISA_MIPS32R2, GPR_64;
def SEH64 : SignExtInReg<"seh", i16, GPR64Opnd, II_SEH>, SEB_FM<0x18, 0x20>,
            ISA_MIPS32R2, GPR_64;
}

/// Count Leading
let AdditionalPredicates = [NotInMicroMips] in {
  def DCLZ : CountLeading0<"dclz", GPR64Opnd, II_DCLZ>, CLO_FM<0x24>,
             ISA_MIPS64_NOT_64R6, GPR_64;
  def DCLO : CountLeading1<"dclo", GPR64Opnd, II_DCLO>, CLO_FM<0x25>,
             ISA_MIPS64_NOT_64R6, GPR_64;

/// Double Word Swap Bytes/HalfWords
  def DSBH : SubwordSwap<"dsbh", GPR64Opnd, II_DSBH>, SEB_FM<2, 0x24>,
             ISA_MIPS64R2;
  def DSHD : SubwordSwap<"dshd", GPR64Opnd, II_DSHD>, SEB_FM<5, 0x24>,
             ISA_MIPS64R2;

  def LEA_ADDiu64 : EffectiveAddress<"daddiu", GPR64Opnd>, LW_FM<0x19>,
                    GPR_64;
}

let isCodeGenOnly = 1 in
def RDHWR64 : ReadHardware<GPR64Opnd, HWRegsOpnd>, RDHWR_FM, GPR_64;

let AdditionalPredicates = [NotInMicroMips] in {
  // The 'pos + size' constraints for code generation are enforced by the
  // code that lowers into MipsISD::Ext.
  // For assembly parsing, we alias dextu and dextm to dext, and match by
  // operand were possible then check the 'pos + size' in MipsAsmParser.
  // We override the generated decoder to enforce that dext always comes out
  // for dextm and dextu like binutils.
  let DecoderMethod = "DecodeDEXT" in {
    def DEXT : ExtBase<"dext", GPR64Opnd, uimm5_report_uimm6,
                       uimm5_plus1_report_uimm6, immZExt5, immZExt5Plus1,
                       MipsExt>, EXT_FM<3>, ISA_MIPS64R2;
    def DEXTM : ExtBase<"dextm", GPR64Opnd, uimm5, uimm5_plus33, immZExt5,
                        immZExt5Plus33, MipsExt>, EXT_FM<1>, ISA_MIPS64R2;
    def DEXTU : ExtBase<"dextu", GPR64Opnd, uimm5_plus32, uimm5_plus1,
                        immZExt5Plus32, immZExt5Plus1, MipsExt>, EXT_FM<2>,
                        ISA_MIPS64R2;
  }
  // The 'pos + size' constraints for code generation are enforced by the
  // code that lowers into MipsISD::Ins.
  // For assembly parsing, we alias dinsu and dinsm to dins, and match by
  // operand were possible then check the 'pos + size' in MipsAsmParser.
  // We override the generated decoder to enforce that dins always comes out
  // for dinsm and dinsu like binutils.
  let DecoderMethod = "DecodeDINS" in {
    def DINS  : InsBase<"dins", GPR64Opnd, uimm6, uimm5_inssize_plus1,
                        immZExt5, immZExt5Plus1>, EXT_FM<7>,
                ISA_MIPS64R2;
    def DINSU : InsBase<"dinsu", GPR64Opnd, uimm5_plus32, uimm5_inssize_plus1,
                        immZExt5Plus32, immZExt5Plus1>,
                EXT_FM<6>, ISA_MIPS64R2;
    def DINSM : InsBase<"dinsm", GPR64Opnd, uimm5, uimm_range_2_64,
                        immZExt5, immZExtRange2To64>,
                EXT_FM<5>, ISA_MIPS64R2;
  }
}

let isCodeGenOnly = 1, AdditionalPredicates = [NotInMicroMips] in {
  def DEXT64_32 : InstSE<(outs GPR64Opnd:$rt),
                         (ins GPR32Opnd:$rs, uimm5_report_uimm6:$pos,
                              uimm5_plus1:$size),
                         "dext $rt, $rs, $pos, $size", [], II_EXT, FrmR, "dext">,
                  EXT_FM<3>, ISA_MIPS64R2;
}

let isCodeGenOnly = 1, rs = 0, shamt = 0 in {
  def DSLL64_32 : FR<0x00, 0x3c, (outs GPR64:$rd), (ins GPR32:$rt),
                     "dsll\t$rd, $rt, 32", [], II_DSLL>, GPR_64;
  let isMoveReg = 1 in {
    def SLL64_32 : FR<0x0, 0x00, (outs GPR64:$rd), (ins GPR32:$rt),
                      "sll\t$rd, $rt, 0", [], II_SLL>, GPR_64;
    def SLL64_64 : FR<0x0, 0x00, (outs GPR64:$rd), (ins GPR64:$rt),
                      "sll\t$rd, $rt, 0", [], II_SLL>, GPR_64;
  }
}

// We need the following pseudo instruction to avoid offset calculation for
// long branches.  See the comment in file MipsLongBranch.cpp for detailed
// explanation.



// Cavium Octeon cnMIPS instructions
 let DecoderNamespace = "CnMips",
     // FIXME: The lack of HasStdEnc is probably a bug
     EncodingPredicates = []<Predicate> in {
 
 class Count1s<string opstr, RegisterOperand RO>:
   InstSE<(outs RO:$rd), (ins RO:$rs), !strconcat(opstr, "\t$rd, $rs"),
          [(set RO:$rd, (ctpop RO:$rs))], II_POP, FrmR, opstr> {
   let TwoOperandAliasConstraint = "$rd = $rs";
 }
class ExtsCins<string opstr, InstrItinClass itin, RegisterOperand RO,
               PatFrag PosImm, SDPatternOperator Op = null_frag>:
  InstSE<(outs RO:$rt), (ins RO:$rs, uimm5:$pos, uimm5:$lenm1),
         !strconcat(opstr, "\t$rt, $rs, $pos, $lenm1"),
         [(set RO:$rt, (Op RO:$rs, PosImm:$pos, imm:$lenm1))],
         itin, FrmR, opstr> {
  let TwoOperandAliasConstraint = "$rt = $rs";
}

class SetCC64_R<string opstr, PatFrag cond_op> :
  InstSE<(outs GPR64Opnd:$rd), (ins GPR64Opnd:$rs, GPR64Opnd:$rt),
         !strconcat(opstr, "\t$rd, $rs, $rt"),
         [(set GPR64Opnd:$rd, (zext (cond_op GPR64Opnd:$rs,
                                             GPR64Opnd:$rt)))],
         II_SEQ_SNE, FrmR, opstr> {
  let TwoOperandAliasConstraint = "$rd = $rs";
}

class SetCC64_I<string opstr, PatFrag cond_op>:
  InstSE<(outs GPR64Opnd:$rt), (ins GPR64Opnd:$rs, simm10_64:$imm10),
         !strconcat(opstr, "\t$rt, $rs, $imm10"),
         [(set GPR64Opnd:$rt, (zext (cond_op GPR64Opnd:$rs,
                                             immSExt10_64:$imm10)))],
         II_SEQI_SNEI, FrmI, opstr> {
  let TwoOperandAliasConstraint = "$rt = $rs";
}

class CBranchBitNum<string opstr, DAGOperand opnd, PatFrag cond_op,
                    RegisterOperand RO, Operand ImmOp, bits<64> shift = 1> :
  InstSE<(outs), (ins RO:$rs, ImmOp:$p, opnd:$offset),
         !strconcat(opstr, "\t$rs, $p, $offset"),
         [(brcond (i32 (cond_op (and RO:$rs, (shl shift, immZExt5_64:$p)), 0)),
                  bb:$offset)], II_BBIT, FrmI, opstr> {
  let isBranch = 1;
  let isTerminator = 1;
  let hasDelaySlot = 1;
  let Defs = [AT];
}

class MFC2OP<string asmstr, RegisterOperand RO, InstrItinClass itin> :
  InstSE<(outs RO:$rt, uimm16:$imm16), (ins),
         !strconcat(asmstr, "\t$rt, $imm16"), [], itin, FrmFR>;

// Unsigned Byte Add
def BADDu  : ArithLogicR<"baddu", GPR64Opnd, 1, II_BADDU>,
             ADD_FM<0x1c, 0x28>, ASE_CNMIPS {
  let Pattern = [(set GPR64Opnd:$rd,
                      (and (add GPR64Opnd:$rs, GPR64Opnd:$rt), 255))];
}

// Branch on Bit Clear /+32
def BBIT0  : CBranchBitNum<"bbit0", brtarget, seteq, GPR64Opnd,
                           uimm5_64_report_uimm6>, BBIT_FM<0x32>, ASE_CNMIPS;
def BBIT032: CBranchBitNum<"bbit032", brtarget, seteq, GPR64Opnd, uimm5_64,
                           0x100000000>, BBIT_FM<0x36>, ASE_CNMIPS;

// Branch on Bit Set /+32
def BBIT1  : CBranchBitNum<"bbit1", brtarget, setne, GPR64Opnd,
                           uimm5_64_report_uimm6>, BBIT_FM<0x3a>, ASE_CNMIPS;
def BBIT132: CBranchBitNum<"bbit132", brtarget, setne, GPR64Opnd, uimm5_64,
                           0x100000000>, BBIT_FM<0x3e>, ASE_CNMIPS;

// Multiply Doubleword to GPR
def DMUL  : ArithLogicR<"dmul", GPR64Opnd, 1, II_DMUL, mul>,
            ADD_FM<0x1c, 0x03>, ASE_CNMIPS {
  let Defs = [HI0, LO0, P0, P1, P2];
}

let AdditionalPredicates = [NotInMicroMips] in {
  // Extract a signed bit field /+32
  def EXTS  : ExtsCins<"exts", II_EXT, GPR64Opnd, immZExt5>, EXTS_FM<0x3a>,
              ASE_MIPS64_CNMIPS;
  def EXTS32: ExtsCins<"exts32", II_EXT, GPR64Opnd, immZExt5Plus32>,
              EXTS_FM<0x3b>, ASE_MIPS64_CNMIPS;

  // Clear and insert a bit field /+32
  def CINS  : ExtsCins<"cins", II_INS, GPR64Opnd, immZExt5, MipsCIns>,
              EXTS_FM<0x32>, ASE_MIPS64_CNMIPS;
  def CINS32: ExtsCins<"cins32", II_INS, GPR64Opnd, immZExt5Plus32, MipsCIns>,
              EXTS_FM<0x33>, ASE_MIPS64_CNMIPS;
  let isCodeGenOnly = 1 in {
    def CINS_i32 : ExtsCins<"cins", II_INS, GPR32Opnd, immZExt5, MipsCIns>,
                   EXTS_FM<0x32>, ASE_MIPS64_CNMIPS;
    def CINS64_32 :InstSE<(outs GPR64Opnd:$rt),
                          (ins GPR32Opnd:$rs, uimm5:$pos, uimm5:$lenm1),
                          "cins\t$rt, $rs, $pos, $lenm1", [], II_INS, FrmR,
                          "cins">,
                   EXTS_FM<0x32>, ASE_MIPS64_CNMIPS;
  }
}

// Move to multiplier/product register
def MTM0   : MoveToLOHI<"mtm0", GPR64Opnd, [MPL0, P0, P1, P2]>, MTMR_FM<0x08>,
             ASE_CNMIPS;
def MTM1   : MoveToLOHI<"mtm1", GPR64Opnd, [MPL1, P0, P1, P2]>, MTMR_FM<0x0c>,
             ASE_CNMIPS;
def MTM2   : MoveToLOHI<"mtm2", GPR64Opnd, [MPL2, P0, P1, P2]>, MTMR_FM<0x0d>,
             ASE_CNMIPS;
def MTP0   : MoveToLOHI<"mtp0", GPR64Opnd, [P0]>, MTMR_FM<0x09>, ASE_CNMIPS;
def MTP1   : MoveToLOHI<"mtp1", GPR64Opnd, [P1]>, MTMR_FM<0x0a>, ASE_CNMIPS;
def MTP2   : MoveToLOHI<"mtp2", GPR64Opnd, [P2]>, MTMR_FM<0x0b>, ASE_CNMIPS;

// Count Ones in a Word/Doubleword
 def POP   : Count1s<"pop", GPR32Opnd>, POP_FM<0x2c>, ASE_CNMIPS;
 def DPOP  : Count1s<"dpop", GPR64Opnd>, POP_FM<0x2d>, ASE_CNMIPS;

// Set on equal/not equal
def SEQ   : SetCC64_R<"seq", seteq>, SEQ_FM<0x2a>, ASE_CNMIPS;
def SEQi  : SetCC64_I<"seqi", seteq>, SEQI_FM<0x2e>, ASE_CNMIPS;
def SNE   : SetCC64_R<"sne", setne>, SEQ_FM<0x2b>, ASE_CNMIPS;
def SNEi  : SetCC64_I<"snei", setne>, SEQI_FM<0x2f>, ASE_CNMIPS;

// 192-bit x 64-bit Unsigned Multiply and Add
def V3MULU: ArithLogicR<"v3mulu", GPR64Opnd, 0, II_DMUL>, ADD_FM<0x1c, 0x11>,
            ASE_CNMIPS {
  let Defs = [P0, P1, P2];
}

// 64-bit Unsigned Multiply and Add Move
def VMM0  : ArithLogicR<"vmm0", GPR64Opnd, 0, II_DMUL>, ADD_FM<0x1c, 0x10>,
            ASE_CNMIPS {
  let Defs = [MPL0, P0, P1, P2];
}

// 64-bit Unsigned Multiply and Add
def VMULU : ArithLogicR<"vmulu", GPR64Opnd, 0, II_DMUL>, ADD_FM<0x1c, 0x0f>,
            ASE_CNMIPS {
  let Defs = [MPL1, MPL2, P0, P1, P2];
}

// Move between CPU and coprocessor registers
def DMFC2_OCTEON : MFC2OP<"dmfc2", GPR64Opnd, II_DMFC2>, MFC2OP_FM<0x12, 1>,
                   ASE_CNMIPS;
def DMTC2_OCTEON : MFC2OP<"dmtc2", GPR64Opnd, II_DMTC2>, MFC2OP_FM<0x12, 5>,
                   ASE_CNMIPS;
}

}

/// Move between CPU and coprocessor registers
let DecoderNamespace = "Mips64" in {
def DMFC0 : MFC3OP<"dmfc0", GPR64Opnd, COP0Opnd, II_DMFC0>,
            MFC3OP_FM<0x10, 1, 0>, ISA_MIPS3, GPR_64;
def DMTC0 : MTC3OP<"dmtc0", COP0Opnd, GPR64Opnd, II_DMTC0>,
            MFC3OP_FM<0x10, 5, 0>, ISA_MIPS3, GPR_64;
def DMFC2 : MFC3OP<"dmfc2", GPR64Opnd, COP2Opnd, II_DMFC2>,
            MFC3OP_FM<0x12, 1, 0>, ISA_MIPS3, GPR_64;
def DMTC2 : MTC3OP<"dmtc2", COP2Opnd, GPR64Opnd, II_DMTC2>,
            MFC3OP_FM<0x12, 5, 0>, ISA_MIPS3, GPR_64;
}

/// Move between CPU and guest coprocessor registers (Virtualization ASE)
let DecoderNamespace = "Mips64" in {
  def DMFGC0 : MFC3OP<"dmfgc0", GPR64Opnd, COP0Opnd, II_DMFGC0>,
               MFC3OP_FM<0x10, 3, 1>, ISA_MIPS64R5, ASE_VIRT;
  def DMTGC0 : MTC3OP<"dmtgc0", COP0Opnd, GPR64Opnd, II_DMTGC0>,
               MFC3OP_FM<0x10, 3, 3>, ISA_MIPS64R5, ASE_VIRT;
}
