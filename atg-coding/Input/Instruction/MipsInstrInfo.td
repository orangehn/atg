//===- MipsInstrInfo.td - Target Description for Mips Target -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Mips implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// Mips profiles and nodes
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Mips Instruction Predicate Definitions.
//===----------------------------------------------------------------------===//
//  def HasMips2     :    Predicate<"Subtarget->hasMips2()">,
//                        AssemblerPredicate<"FeatureMips2">;
//  def HasMips3_32  :    Predicate<"Subtarget->hasMips3_32()">,
//                        AssemblerPredicate<"FeatureMips3_32">;
//  def HasMips3_32r2 :   Predicate<"Subtarget->hasMips3_32r2()">,
//                        AssemblerPredicate<"FeatureMips3_32r2">;
//  def HasMips3     :    Predicate<"Subtarget->hasMips3()">,
//                        AssemblerPredicate<"FeatureMips3">;
//  def NotMips3     :    Predicate<"!Subtarget->hasMips3()">,
//                        AssemblerPredicate<"!FeatureMips3">;
//  def HasMips4_32  :    Predicate<"Subtarget->hasMips4_32()">,
//                        AssemblerPredicate<"FeatureMips4_32">;
//  def NotMips4_32  :    Predicate<"!Subtarget->hasMips4_32()">,
//                        AssemblerPredicate<"!FeatureMips4_32">;
//  def HasMips4_32r2 :   Predicate<"Subtarget->hasMips4_32r2()">,
//                        AssemblerPredicate<"FeatureMips4_32r2">;
//  def HasMips5_32r2 :   Predicate<"Subtarget->hasMips5_32r2()">,
//                        AssemblerPredicate<"FeatureMips5_32r2">;
//  def HasMips32    :    Predicate<"Subtarget->hasMips32()">,
//                        AssemblerPredicate<"FeatureMips32">;
//  def HasMips32r2  :    Predicate<"Subtarget->hasMips32r2()">,
//                        AssemblerPredicate<"FeatureMips32r2">;
//  def HasMips32r5  :    Predicate<"Subtarget->hasMips32r5()">,
//                        AssemblerPredicate<"FeatureMips32r5">;
//  def HasMips32r6  :    Predicate<"Subtarget->hasMips32r6()">,
//                        AssemblerPredicate<"FeatureMips32r6">;
//  def NotMips32r6  :    Predicate<"!Subtarget->hasMips32r6()">,
//                        AssemblerPredicate<"!FeatureMips32r6">;
//  def IsGP64bit    :    Predicate<"Subtarget->isGP64bit()">,
//                        AssemblerPredicate<"FeatureGP64Bit">;
//  def IsGP32bit    :    Predicate<"!Subtarget->isGP64bit()">,
//                        AssemblerPredicate<"!FeatureGP64Bit">;
//  def IsPTR64bit    :   Predicate<"Subtarget->isABI_N64()">,
//                        AssemblerPredicate<"FeaturePTR64Bit">;
//  def IsPTR32bit    :   Predicate<"!Subtarget->isABI_N64()">,
//                        AssemblerPredicate<"!FeaturePTR64Bit">;
//  def HasMips64    :    Predicate<"Subtarget->hasMips64()">,
//                        AssemblerPredicate<"FeatureMips64">;
//  def NotMips64    :    Predicate<"!Subtarget->hasMips64()">,
//                        AssemblerPredicate<"!FeatureMips64">;
//  def HasMips64r2  :    Predicate<"Subtarget->hasMips64r2()">,
//                        AssemblerPredicate<"FeatureMips64r2">;
//  def HasMips64r5  :    Predicate<"Subtarget->hasMips64r5()">,
//                        AssemblerPredicate<"FeatureMips64r5">;
//  def HasMips64r6  :    Predicate<"Subtarget->hasMips64r6()">,
//                        AssemblerPredicate<"FeatureMips64r6">;
//  def NotMips64r6  :    Predicate<"!Subtarget->hasMips64r6()">,
//                        AssemblerPredicate<"!FeatureMips64r6">;
//  def InMips16Mode :    Predicate<"Subtarget->inMips16Mode()">,
//                        AssemblerPredicate<"FeatureMips16">;
//  def NotInMips16Mode : Predicate<"!Subtarget->inMips16Mode()">,
//                        AssemblerPredicate<"!FeatureMips16">;
//  def HasCnMips    :    Predicate<"Subtarget->hasCnMips()">,
//                        AssemblerPredicate<"FeatureCnMips">;
//  def NotCnMips    :    Predicate<"!Subtarget->hasCnMips()">,
//                        AssemblerPredicate<"!FeatureCnMips">;
//  def IsSym32     :     Predicate<"Subtarget->HasSym32()">,
//                        AssemblerPredicate<"FeatureSym32">;
//  def IsSym64     :     Predicate<"!Subtarget->HasSym32()">,
//                        AssemblerPredicate<"!FeatureSym32">;
//  def IsN64       :     Predicate<"Subtarget->isABI_N64()">;
//  def IsNotN64    :     Predicate<"!Subtarget->isABI_N64()">;
//  def RelocNotPIC :     Predicate<"!TM.isPositionIndependent()">;
//  def RelocPIC    :     Predicate<"TM.isPositionIndependent()">;
//  def NoNaNsFPMath :    Predicate<"TM.Options.NoNaNsFPMath">;
//  def UseAbs :          Predicate<"Subtarget->inAbs2008Mode() ||"
//                                  "TM.Options.NoNaNsFPMath">;
//  def HasStdEnc :       Predicate<"Subtarget->hasStandardEncoding()">,
//                        AssemblerPredicate<"!FeatureMips16">;
//  def NotDSP :          Predicate<"!Subtarget->hasDSP()">;
//  def InMicroMips    :  Predicate<"Subtarget->inMicroMipsMode()">,
//                        AssemblerPredicate<"FeatureMicroMips">;
//  def NotInMicroMips :  Predicate<"!Subtarget->inMicroMipsMode()">,
//                        AssemblerPredicate<"!FeatureMicroMips">;
//  def IsLE           :  Predicate<"Subtarget->isLittle()">;
//  def IsBE           :  Predicate<"!Subtarget->isLittle()">;
//  def IsNotNaCl    :    Predicate<"!Subtarget->isTargetNaCl()">;
//  def UseTCCInDIV    :  AssemblerPredicate<"FeatureUseTCCInDIV">;
//  def HasEVA       :    Predicate<"Subtarget->hasEVA()">,
//                        AssemblerPredicate<"FeatureEVA">;
//  def HasMSA : Predicate<"Subtarget->hasMSA()">,
//               AssemblerPredicate<"FeatureMSA">;
//  def HasMadd4 : Predicate<"!Subtarget->disableMadd4()">,
//                 AssemblerPredicate<"!FeatureMadd4">;
//  def HasMT  : Predicate<"Subtarget->hasMT()">,
//               AssemblerPredicate<"FeatureMT">;
//  def UseIndirectJumpsHazard : Predicate<"Subtarget->useIndirectJumpsHazard()">,
//                              AssemblerPredicate<"FeatureUseIndirectJumpsHazard">;
//  def NoIndirectJumpGuards : Predicate<"!Subtarget->useIndirectJumpsHazard()">,
//                             AssemblerPredicate<"!FeatureUseIndirectJumpsHazard">;
//  def HasCRC   : Predicate<"Subtarget->hasCRC()">,
//                 AssemblerPredicate<"FeatureCRC">;
//  def HasVirt  : Predicate<"Subtarget->hasVirt()">,
//                 AssemblerPredicate<"FeatureVirt">;
//  def HasGINV  : Predicate<"Subtarget->hasGINV()">,
//                 AssemblerPredicate<"FeatureGINV">;
//  // TODO: Add support for FPOpFusion::Standard
//  def AllowFPOpFusion : Predicate<"TM.Options.AllowFPOpFusion =="
//                                  " FPOpFusion::Fast">;
//===----------------------------------------------------------------------===//
// Mips GPR size adjectives.
// They are mutually exclusive.
//===----------------------------------------------------------------------===//

class GPR_32 { list<Predicate> GPRPredicates = [IsGP32bit]; }
class GPR_64 { list<Predicate> GPRPredicates = [IsGP64bit]; }

class PTR_32 { list<Predicate> PTRPredicates = [IsPTR32bit]; }
class PTR_64 { list<Predicate> PTRPredicates = [IsPTR64bit]; }

//===----------------------------------------------------------------------===//
// Mips Symbol size adjectives.
// They are mutally exculsive.
//===----------------------------------------------------------------------===//

class SYM_32 { list<Predicate> SYMPredicates = [IsSym32]; }
class SYM_64 { list<Predicate> SYMPredicates = [IsSym64]; }

//===----------------------------------------------------------------------===//
// Mips ISA/ASE membership and instruction group membership adjectives.
// They are mutually exclusive.
//===----------------------------------------------------------------------===//

// FIXME: I'd prefer to use additive predicates to build the instruction sets
//        but we are short on assembler feature bits at the moment. Using a
//        subtractive predicate will hopefully keep us under the 32 predicate
//        limit long enough to develop an alternative way to handle P1||P2
//        predicates.
class ISA_MIPS1 {
  list<Predicate> EncodingPredicates = [HasStdEnc];
}
class ISA_MIPS1_NOT_MIPS3 {
  list<Predicate> InsnPredicates = [NotMips3];
  list<Predicate> EncodingPredicates = [HasStdEnc];
}
class ISA_MIPS1_NOT_4_32 {
  list<Predicate> InsnPredicates = [NotMips4_32];
  list<Predicate> EncodingPredicates = [HasStdEnc];
}
class ISA_MIPS1_NOT_32R6_64R6 {
  list<Predicate> InsnPredicates = [NotMips32r6, NotMips64r6];
  list<Predicate> EncodingPredicates = [HasStdEnc];
}
class ISA_MIPS2 {
  list<Predicate> InsnPredicates = [HasMips2];
  list<Predicate> EncodingPredicates = [HasStdEnc];
}
class ISA_MIPS2_NOT_32R6_64R6 {
  list<Predicate> InsnPredicates = [HasMips2, NotMips32r6, NotMips64r6];
  list<Predicate> EncodingPredicates = [HasStdEnc];
}
class ISA_MIPS3 {
  list<Predicate> InsnPredicates = [HasMips3];
  list<Predicate> EncodingPredicates = [HasStdEnc];
}
class ISA_MIPS3_NOT_32R6_64R6 {
  list<Predicate> InsnPredicates = [HasMips3, NotMips32r6, NotMips64r6];
  list<Predicate> EncodingPredicates = [HasStdEnc];
}
class ISA_MIPS32 {
  list<Predicate> InsnPredicates = [HasMips32];
  list<Predicate> EncodingPredicates = [HasStdEnc];
}
class ISA_MIPS32_NOT_32R6_64R6 {
  list<Predicate> InsnPredicates = [HasMips32, NotMips32r6, NotMips64r6];
  list<Predicate> EncodingPredicates = [HasStdEnc];
}
class ISA_MIPS32R2 {
  list<Predicate> InsnPredicates = [HasMips32r2];
  list<Predicate> EncodingPredicates = [HasStdEnc];
}
class ISA_MIPS32R2_NOT_32R6_64R6 {
  list<Predicate> InsnPredicates = [HasMips32r2, NotMips32r6, NotMips64r6];
  list<Predicate> EncodingPredicates = [HasStdEnc];
}
class ISA_MIPS32R5 {
  list<Predicate> InsnPredicates = [HasMips32r5];
  list<Predicate> EncodingPredicates = [HasStdEnc];
}
class ISA_MIPS64 {
  list<Predicate> InsnPredicates = [HasMips64];
  list<Predicate> EncodingPredicates = [HasStdEnc];
}
class ISA_MIPS64_NOT_64R6 {
  list<Predicate> InsnPredicates = [HasMips64, NotMips64r6];
  list<Predicate> EncodingPredicates = [HasStdEnc];
}
class ISA_MIPS64R2 {
  list<Predicate> InsnPredicates = [HasMips64r2];
  list<Predicate> EncodingPredicates = [HasStdEnc];
}
class ISA_MIPS64R5 {
  list<Predicate> InsnPredicates = [HasMips64r5];
  list<Predicate> EncodingPredicates = [HasStdEnc];
}
class ISA_MIPS32R6 {
  list<Predicate> InsnPredicates = [HasMips32r6];
  list<Predicate> EncodingPredicates = [HasStdEnc];
}
class ISA_MIPS64R6 {
  list<Predicate> InsnPredicates = [HasMips64r6];
  list<Predicate> EncodingPredicates = [HasStdEnc];
}
class ISA_MICROMIPS {
  list<Predicate> EncodingPredicates = [InMicroMips];
}
class ISA_MICROMIPS32R5 {
  list<Predicate> InsnPredicates = [HasMips32r5];
  list<Predicate> EncodingPredicates = [InMicroMips];
}
class ISA_MICROMIPS32R6 {
  list<Predicate> InsnPredicates = [HasMips32r6];
  list<Predicate> EncodingPredicates = [InMicroMips];
}
class ISA_MICROMIPS64R6 {
  list<Predicate> InsnPredicates = [HasMips64r6];
  list<Predicate> EncodingPredicates = [InMicroMips];
}
class ISA_MICROMIPS32_NOT_MIPS32R6 {
  list<Predicate> InsnPredicates = [NotMips32r6];
  list<Predicate> EncodingPredicates = [InMicroMips];
}
class ASE_EVA { list<Predicate> ASEPredicate = [HasEVA]; }

// The portions of MIPS-III that were also added to MIPS32
class INSN_MIPS3_32 {
  list<Predicate> InsnPredicates = [HasMips3_32];
  list<Predicate> EncodingPredicates = [HasStdEnc];
}

// The portions of MIPS-III that were also added to MIPS32 but were removed in
// MIPS32r6 and MIPS64r6.
class INSN_MIPS3_32_NOT_32R6_64R6 {
  list<Predicate> InsnPredicates = [HasMips3_32, NotMips32r6, NotMips64r6];
  list<Predicate> EncodingPredicates = [HasStdEnc];
}

// The portions of MIPS-III that were also added to MIPS32
class INSN_MIPS3_32R2 {
  list<Predicate> InsnPredicates = [HasMips3_32r2];
  list<Predicate> EncodingPredicates = [HasStdEnc];
}

// The portions of MIPS-IV that were also added to MIPS32.
class INSN_MIPS4_32 {
  list <Predicate> InsnPredicates = [HasMips4_32];
  list<Predicate> EncodingPredicates = [HasStdEnc];
}

// The portions of MIPS-IV that were also added to MIPS32 but were removed in
// MIPS32r6 and MIPS64r6.
class INSN_MIPS4_32_NOT_32R6_64R6 {
  list<Predicate> InsnPredicates = [HasMips4_32, NotMips32r6, NotMips64r6];
  list<Predicate> EncodingPredicates = [HasStdEnc];
}

// The portions of MIPS-IV that were also added to MIPS32r2 but were removed in
// MIPS32r6 and MIPS64r6.
class INSN_MIPS4_32R2_NOT_32R6_64R6 {
  list<Predicate> InsnPredicates = [HasMips4_32r2, NotMips32r6, NotMips64r6];
  list<Predicate> EncodingPredicates = [HasStdEnc];
}

// The portions of MIPS-IV that were also added to MIPS32r2.
class INSN_MIPS4_32R2 {
  list<Predicate> InsnPredicates = [HasMips4_32r2];
  list<Predicate> EncodingPredicates = [HasStdEnc];
}

// The portions of MIPS-V that were also added to MIPS32r2 but were removed in
// MIPS32r6 and MIPS64r6.
class INSN_MIPS5_32R2_NOT_32R6_64R6 {
  list<Predicate> InsnPredicates = [HasMips5_32r2, NotMips32r6, NotMips64r6];
  list<Predicate> EncodingPredicates = [HasStdEnc];
}

class ASE_CNMIPS {
  list<Predicate> ASEPredicate = [HasCnMips];
  list<Predicate> EncodingPredicates = [];
}

class NOT_ASE_CNMIPS {
  list<Predicate> ASEPredicate = [NotCnMips];
  list<Predicate> EncodingPredicates = [];
}

class ASE_MIPS64_CNMIPS {
  list<Predicate> ASEPredicate = [HasMips64, HasCnMips];
  list<Predicate> EncodingPredicates = [];
}

class ASE_MSA {
  list<Predicate> ASEPredicate = [HasMSA];
  list<Predicate> EncodingPredicates = [];
}

class ASE_MSA_NOT_MSA64 {
  list<Predicate> ASEPredicate = [HasMSA, NotMips64];
  list<Predicate> EncodingPredicates = [];
}

class ASE_MSA64 {
  list<Predicate> ASEPredicate = [HasMSA, HasMips64];
  list<Predicate> EncodingPredicates = [];
}

class ASE_MT {
  list <Predicate> ASEPredicate = [HasMT];
  list<Predicate> EncodingPredicates = [];
}

class ASE_CRC {
  list <Predicate> ASEPredicate = [HasCRC];
  list<Predicate> EncodingPredicates = [];
}

class ASE_VIRT {
  list <Predicate> ASEPredicate = [HasVirt];
  list<Predicate> EncodingPredicates = [];
}

class ASE_GINV {
  list <Predicate> ASEPredicate = [HasGINV];
  list<Predicate> EncodingPredicates = [];
}

// Class used for separating microMIPSr6 and microMIPS (r3) instruction.
// It can be used only on instructions that doesn't inherit PredicateControl.
class ISA_MICROMIPS_NOT_32R6 : PredicateControl {
  let InsnPredicates = [NotMips32r6];
  let EncodingPredicates = [InMicroMips];
}

class ASE_NOT_DSP {
  list<Predicate> ASEPredicate = [NotDSP];
  list<Predicate> EncodingPredicates = [];
}

class MADD4 {
  list<Predicate> AdditionalPredicates = [HasMadd4];
  list<Predicate> EncodingPredicates = [];
}

// Classses used for separating expansions that differ based on the ABI in
// use.
class ABI_N64 {
  list<Predicate> AdditionalPredicates = [IsN64];
  list<Predicate> EncodingPredicates = [];
}

class ABI_NOT_N64 {
  list<Predicate> AdditionalPredicates = [IsNotN64];
  list<Predicate> EncodingPredicates = [];
}

class FPOP_FUSION_FAST {
  list <Predicate> AdditionalPredicates = [AllowFPOpFusion];
  list<Predicate> EncodingPredicates = [];
}

//===----------------------------------------------------------------------===//
class IsCommutable {
  bit isCommutable = 1;
}

class IsBranch {
  bit isBranch = 1;
  bit isCTI = 1;
}

class IsReturn {
  bit isReturn = 1;
  bit isCTI = 1;
}

class IsCall {
  bit isCall = 1;
  bit isCTI = 1;
}

class IsTailCall {
  bit isCall = 1;
  bit isTerminator = 1;
  bit isReturn = 1;
  bit isBarrier = 1;
  bit hasExtraSrcRegAllocReq = 1;
  bit isCodeGenOnly = 1;
  bit isCTI = 1;
}

class IsAsCheapAsAMove {
  bit isAsCheapAsAMove = 1;
}

class NeverHasSideEffects {
  bit hasSideEffects = 0;
}

//===----------------------------------------------------------------------===//
// Instruction format superclass
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// Instructions specific format
//===----------------------------------------------------------------------===//

// Arithmetic and logical instructions with 3 register operands.
class ArithLogicR<string opstr, RegisterOperand RO, bit isComm = 0,
                  InstrItinClass Itin = NoItinerary,
                  SDPatternOperator OpNode = null_frag>:
  InstSE<(outs RO:$rd), (ins RO:$rs, RO:$rt),
         !strconcat(opstr, "\t$rd, $rs, $rt"),
         [(set RO:$rd, (OpNode RO:$rs, RO:$rt))], Itin, FrmR, opstr> {
  let isCommutable = isComm;
  let isReMaterializable = 1;
  let TwoOperandAliasConstraint = "$rd = $rs";
}

// Arithmetic and logical instructions with 2 register operands.
class ArithLogicI<string opstr, Operand Od, RegisterOperand RO,
                  InstrItinClass Itin = NoItinerary,
                  SDPatternOperator imm_type = null_frag,
                  SDPatternOperator OpNode = null_frag> :
  InstSE<(outs RO:$rt), (ins RO:$rs, Od:$imm16),
         !strconcat(opstr, "\t$rt, $rs, $imm16"),
         [(set RO:$rt, (OpNode RO:$rs, imm_type:$imm16))],
         Itin, FrmI, opstr> {
  let isReMaterializable = 1;
  let TwoOperandAliasConstraint = "$rs = $rt";
}

// Arithmetic Multiply ADD/SUB
class MArithR<string opstr, InstrItinClass itin, bit isComm = 0> :
  InstSE<(outs), (ins GPR32Opnd:$rs, GPR32Opnd:$rt),
         !strconcat(opstr, "\t$rs, $rt"), [], itin, FrmR, opstr> {
  let Defs = [HI0, LO0];
  let Uses = [HI0, LO0];
  let isCommutable = isComm;
}

//  Logical
class LogicNOR<string opstr, RegisterOperand RO>:
  InstSE<(outs RO:$rd), (ins RO:$rs, RO:$rt),
         !strconcat(opstr, "\t$rd, $rs, $rt"),
         [(set RO:$rd, (not (or RO:$rs, RO:$rt)))], II_NOR, FrmR, opstr> {
  let isCommutable = 1;
}

// Shifts
class shift_rotate_imm<string opstr, Operand ImmOpnd,
                       RegisterOperand RO, InstrItinClass itin,
                       SDPatternOperator OpNode = null_frag,
                       SDPatternOperator PF = null_frag> :
  InstSE<(outs RO:$rd), (ins RO:$rt, ImmOpnd:$shamt),
         !strconcat(opstr, "\t$rd, $rt, $shamt"),
         [(set RO:$rd, (OpNode RO:$rt, PF:$shamt))], itin, FrmR, opstr> {
  let TwoOperandAliasConstraint = "$rt = $rd";
}

class shift_rotate_reg<string opstr, RegisterOperand RO, InstrItinClass itin,
                       SDPatternOperator OpNode = null_frag>:
  InstSE<(outs RO:$rd), (ins RO:$rt, GPR32Opnd:$rs),
         !strconcat(opstr, "\t$rd, $rt, $rs"),
         [(set RO:$rd, (OpNode RO:$rt, GPR32Opnd:$rs))], itin, FrmR,
         opstr>;

// Load Upper Immediate
class LoadUpper<string opstr, RegisterOperand RO, Operand Imm>:
  InstSE<(outs RO:$rt), (ins Imm:$imm16), !strconcat(opstr, "\t$rt, $imm16"),
         [], II_LUI, FrmI, opstr>, IsAsCheapAsAMove {
  let hasSideEffects = 0;
  let isReMaterializable = 1;
}

// Memory Load/Store
class LoadMemory<string opstr, DAGOperand RO, DAGOperand MO,
                 SDPatternOperator OpNode = null_frag,
                 InstrItinClass Itin = NoItinerary,
                 ComplexPattern Addr = addr> :
  InstSE<(outs RO:$rt), (ins MO:$addr), !strconcat(opstr, "\t$rt, $addr"),
         [(set RO:$rt, (OpNode Addr:$addr))], Itin, FrmI, opstr> {
  let DecoderMethod = "DecodeMem";
  let canFoldAsLoad = 1;
  string BaseOpcode = opstr;
  let mayLoad = 1;
}

class Load<string opstr, DAGOperand RO, SDPatternOperator OpNode = null_frag,
           InstrItinClass Itin = NoItinerary, ComplexPattern Addr = addr> :
  LoadMemory<opstr, RO, mem, OpNode, Itin, Addr>;

class StoreMemory<string opstr, DAGOperand RO, DAGOperand MO,
            SDPatternOperator OpNode = null_frag,
            InstrItinClass Itin = NoItinerary, ComplexPattern Addr = addr> :
  InstSE<(outs), (ins RO:$rt, MO:$addr), !strconcat(opstr, "\t$rt, $addr"),
         [(OpNode RO:$rt, Addr:$addr)], Itin, FrmI, opstr> {
  let DecoderMethod = "DecodeMem";
  string BaseOpcode = opstr;
  let mayStore = 1;
}

class Store<string opstr, DAGOperand RO, SDPatternOperator OpNode = null_frag,
            InstrItinClass Itin = NoItinerary, ComplexPattern Addr = addr,
            DAGOperand MO = mem> :
  StoreMemory<opstr, RO, MO, OpNode, Itin, Addr>;

// Load/Store Left/Right
let canFoldAsLoad = 1 in
class LoadLeftRight<string opstr, SDNode OpNode, RegisterOperand RO,
                    InstrItinClass Itin> :
  InstSE<(outs RO:$rt), (ins mem:$addr, RO:$src),
         !strconcat(opstr, "\t$rt, $addr"),
         [(set RO:$rt, (OpNode addr:$addr, RO:$src))], Itin, FrmI> {
  let DecoderMethod = "DecodeMem";
  string Constraints = "$src = $rt";
  let BaseOpcode = opstr;
}

class StoreLeftRight<string opstr, SDNode OpNode, RegisterOperand RO,
                     InstrItinClass Itin> :
  InstSE<(outs), (ins RO:$rt, mem:$addr), !strconcat(opstr, "\t$rt, $addr"),
         [(OpNode RO:$rt, addr:$addr)], Itin, FrmI> {
  let DecoderMethod = "DecodeMem";
  let BaseOpcode = opstr;
}

// COP2 Load/Store
class LW_FT2<string opstr, RegisterOperand RC, InstrItinClass Itin,
             SDPatternOperator OpNode= null_frag> :
  InstSE<(outs RC:$rt), (ins mem_simm16:$addr),
         !strconcat(opstr, "\t$rt, $addr"),
         [(set RC:$rt, (OpNode addrDefault:$addr))], Itin, FrmFI, opstr> {
  let DecoderMethod = "DecodeFMem2";
  let mayLoad = 1;
}

class SW_FT2<string opstr, RegisterOperand RC, InstrItinClass Itin,
             SDPatternOperator OpNode= null_frag> :
  InstSE<(outs), (ins RC:$rt, mem_simm16:$addr),
         !strconcat(opstr, "\t$rt, $addr"),
         [(OpNode RC:$rt, addrDefault:$addr)], Itin, FrmFI, opstr> {
  let DecoderMethod = "DecodeFMem2";
  let mayStore = 1;
}

// COP3 Load/Store
class LW_FT3<string opstr, RegisterOperand RC, InstrItinClass Itin,
             SDPatternOperator OpNode= null_frag> :
  InstSE<(outs RC:$rt), (ins mem:$addr), !strconcat(opstr, "\t$rt, $addr"),
         [(set RC:$rt, (OpNode addrDefault:$addr))], Itin, FrmFI, opstr> {
  let DecoderMethod = "DecodeFMem3";
  let mayLoad = 1;
}

class SW_FT3<string opstr, RegisterOperand RC, InstrItinClass Itin,
             SDPatternOperator OpNode= null_frag> :
  InstSE<(outs), (ins RC:$rt, mem:$addr), !strconcat(opstr, "\t$rt, $addr"),
         [(OpNode RC:$rt, addrDefault:$addr)], Itin, FrmFI, opstr> {
  let DecoderMethod = "DecodeFMem3";
  let mayStore = 1;
}

// Conditional Branch
class CBranch<string opstr, DAGOperand opnd, PatFrag cond_op,
              RegisterOperand RO> :
  InstSE<(outs), (ins RO:$rs, RO:$rt, opnd:$offset),
         !strconcat(opstr, "\t$rs, $rt, $offset"),
         [(brcond (i32 (cond_op RO:$rs, RO:$rt)), bb:$offset)], II_BCC,
         FrmI, opstr> {
  let isBranch = 1;
  let isTerminator = 1;
  let hasDelaySlot = 1;
  let Defs = [AT];
  bit isCTI = 1;
}

class CBranchLikely<string opstr, DAGOperand opnd, RegisterOperand RO> :
  InstSE<(outs), (ins RO:$rs, RO:$rt, opnd:$offset),
         !strconcat(opstr, "\t$rs, $rt, $offset"), [], II_BCC, FrmI, opstr> {
  let isBranch = 1;
  let isTerminator = 1;
  let hasDelaySlot = 1;
  let Defs = [AT];
  bit isCTI = 1;
}

class CBranchZero<string opstr, DAGOperand opnd, PatFrag cond_op,
                  RegisterOperand RO> :
  InstSE<(outs), (ins RO:$rs, opnd:$offset),
         !strconcat(opstr, "\t$rs, $offset"),
         [(brcond (i32 (cond_op RO:$rs, 0)), bb:$offset)], II_BCCZ,
         FrmI, opstr> {
  let isBranch = 1;
  let isTerminator = 1;
  let hasDelaySlot = 1;
  let Defs = [AT];
  bit isCTI = 1;
}

class CBranchZeroLikely<string opstr, DAGOperand opnd, RegisterOperand RO> :
  InstSE<(outs), (ins RO:$rs, opnd:$offset),
         !strconcat(opstr, "\t$rs, $offset"), [], II_BCCZ, FrmI, opstr> {
  let isBranch = 1;
  let isTerminator = 1;
  let hasDelaySlot = 1;
  let Defs = [AT];
  bit isCTI = 1;
}

// SetCC
class SetCC_R<string opstr, PatFrag cond_op, RegisterOperand RO> :
  InstSE<(outs GPR32Opnd:$rd), (ins RO:$rs, RO:$rt),
         !strconcat(opstr, "\t$rd, $rs, $rt"),
         [(set GPR32Opnd:$rd, (cond_op RO:$rs, RO:$rt))],
         II_SLT_SLTU, FrmR, opstr>;

class SetCC_I<string opstr, PatFrag cond_op, Operand Od, PatLeaf imm_type,
              RegisterOperand RO>:
  InstSE<(outs GPR32Opnd:$rt), (ins RO:$rs, Od:$imm16),
         !strconcat(opstr, "\t$rt, $rs, $imm16"),
         [(set GPR32Opnd:$rt, (cond_op RO:$rs, imm_type:$imm16))],
         II_SLTI_SLTIU, FrmI, opstr>;

// Jump
class JumpFJ<DAGOperand opnd, string opstr, SDPatternOperator operator,
             SDPatternOperator targetoperator, string bopstr> :
  InstSE<(outs), (ins opnd:$target), !strconcat(opstr, "\t$target"),
         [(operator targetoperator:$target)], II_J, FrmJ, bopstr> {
  let isTerminator=1;
  let isBarrier=1;
  let hasDelaySlot = 1;
  let DecoderMethod = "DecodeJumpTarget";
  let Defs = [AT];
  bit isCTI = 1;
}


// Base class for indirect branch and return instruction classes.
let isTerminator=1, isBarrier=1, hasDelaySlot = 1, isCTI = 1 in
class JumpFR<string opstr, RegisterOperand RO,
             SDPatternOperator operator = null_frag>:
  InstSE<(outs), (ins RO:$rs), "jr\t$rs", [(operator RO:$rs)], II_JR,
         FrmR, opstr>;

// Indirect branch
class IndirectBranch<string opstr, RegisterOperand RO> : JumpFR<opstr, RO> {
  let isBranch = 1;
  let isIndirectBranch = 1;
}

// Jump and Link (Call)
let isCall=1, hasDelaySlot=1, isCTI=1, Defs = [RA] in {
  class JumpLink<string opstr, DAGOperand opnd> :
    InstSE<(outs), (ins opnd:$target), !strconcat(opstr, "\t$target"),
           [(MipsJmpLink tglobaladdr:$target)], II_JAL, FrmJ, opstr> {
    let DecoderMethod = "DecodeJumpTarget";
  }

  class JumpLinkReg<string opstr, RegisterOperand RO>:
    InstSE<(outs RO:$rd), (ins RO:$rs), !strconcat(opstr, "\t$rd, $rs"),
           [], II_JALR, FrmR, opstr> {
    let hasPostISelHook = 1;
  }

  class BGEZAL_FT<string opstr, DAGOperand opnd,
                  RegisterOperand RO> :
    InstSE<(outs), (ins RO:$rs, opnd:$offset),
           !strconcat(opstr, "\t$rs, $offset"), [], II_BCCZAL, FrmI, opstr> {
    let hasDelaySlot = 1;
  }

}


let isCTI = 1 in {
// Syscall
class SYS_FT<string opstr, Operand ImmOp, InstrItinClass itin = NoItinerary> :
  InstSE<(outs), (ins ImmOp:$code_),
         !strconcat(opstr, "\t$code_"), [], itin, FrmI, opstr>;
// Break
class BRK_FT<string opstr> :
  InstSE<(outs), (ins uimm10:$code_1, uimm10:$code_2),
         !strconcat(opstr, "\t$code_1, $code_2"), [], II_BREAK,
         FrmOther, opstr>;

// (D)Eret
class ER_FT<string opstr, InstrItinClass itin = NoItinerary> :
  InstSE<(outs), (ins),
         opstr, [], itin, FrmOther, opstr>{
  let isReturn = 1;
}

// Wait
class WAIT_FT<string opstr> :
  InstSE<(outs), (ins), opstr, [], II_WAIT, FrmOther, opstr>;
}

// Interrupts
class DEI_FT<string opstr, RegisterOperand RO,
             InstrItinClass itin = NoItinerary> :
  InstSE<(outs RO:$rt), (ins),
         !strconcat(opstr, "\t$rt"), [], itin, FrmOther, opstr>;

// Sync
let hasSideEffects = 1 in
class SYNC_FT<string opstr> :
  InstSE<(outs), (ins uimm5:$stype), "sync $stype",
         [(MipsSync immZExt5:$stype)], II_SYNC, FrmOther, opstr>;

class SYNCI_FT<string opstr, DAGOperand MO> :
  InstSE<(outs), (ins MO:$addr), !strconcat(opstr, "\t$addr"), [],
         II_SYNCI, FrmOther, opstr> {
  let hasSideEffects = 1;
  let DecoderMethod = "DecodeSyncI";
}

let hasSideEffects = 1, isCTI = 1 in {
class TEQ_FT<string opstr, RegisterOperand RO, Operand ImmOp,
             InstrItinClass itin = NoItinerary> :
  InstSE<(outs), (ins RO:$rs, RO:$rt, ImmOp:$code_),
         !strconcat(opstr, "\t$rs, $rt, $code_"), [], itin, FrmI, opstr>;

class TEQI_FT<string opstr, RegisterOperand RO,
              InstrItinClass itin = NoItinerary> :
  InstSE<(outs), (ins RO:$rs, simm16:$imm16),
         !strconcat(opstr, "\t$rs, $imm16"), [], itin, FrmOther, opstr>;
}

// Mul, Div
class Mult<string opstr, InstrItinClass itin, RegisterOperand RO,
           list<Register> DefRegs> :
  InstSE<(outs), (ins RO:$rs, RO:$rt), !strconcat(opstr, "\t$rs, $rt"), [],
         itin, FrmR, opstr> {
  let isCommutable = 1;
  let Defs = DefRegs;
  let hasSideEffects = 0;
}




class Div<string opstr, InstrItinClass itin, RegisterOperand RO,
          list<Register> DefRegs> :
  InstSE<(outs), (ins RO:$rs, RO:$rt), !strconcat(opstr, "\t$$zero, $rs, $rt"),
         [], itin, FrmR, opstr> {
  let Defs = DefRegs;
}



class MoveFromLOHI<string opstr, RegisterOperand RO, Register UseReg>:
  InstSE<(outs RO:$rd), (ins), !strconcat(opstr, "\t$rd"), [], II_MFHI_MFLO,
         FrmR, opstr> {
  let Uses = [UseReg];
  let hasSideEffects = 0;
  let isMoveReg = 1;
}


class MoveToLOHI<string opstr, RegisterOperand RO, list<Register> DefRegs>:
  InstSE<(outs), (ins RO:$rs), !strconcat(opstr, "\t$rs"), [], II_MTHI_MTLO,
  FrmR, opstr> {
  let Defs = DefRegs;
  let hasSideEffects = 0;
  let isMoveReg = 1;
}

class EffectiveAddress<string opstr, RegisterOperand RO> :
  InstSE<(outs RO:$rt), (ins mem_ea:$addr), !strconcat(opstr, "\t$rt, $addr"),
         [(set RO:$rt, addr:$addr)], II_ADDIU, FrmI,
         !strconcat(opstr, "_lea")> {
  let isCodeGenOnly = 1;
  let hasNoSchedulingInfo = 1;
  let DecoderMethod = "DecodeMem";
}

// Count Leading Ones/Zeros in Word
class CountLeading0<string opstr, RegisterOperand RO,
                  InstrItinClass itin = NoItinerary>:
  InstSE<(outs RO:$rd), (ins RO:$rs), !strconcat(opstr, "\t$rd, $rs"),
         [(set RO:$rd, (ctlz RO:$rs))], itin, FrmR, opstr>;

class CountLeading1<string opstr, RegisterOperand RO,
                  InstrItinClass itin = NoItinerary>:
  InstSE<(outs RO:$rd), (ins RO:$rs), !strconcat(opstr, "\t$rd, $rs"),
         [(set RO:$rd, (ctlz (not RO:$rs)))], itin, FrmR, opstr>;

// Sign Extend in Register.
class SignExtInReg<string opstr, ValueType vt, RegisterOperand RO,
                   InstrItinClass itin> :
  InstSE<(outs RO:$rd), (ins RO:$rt), !strconcat(opstr, "\t$rd, $rt"),
         [(set RO:$rd, (sext_inreg RO:$rt, vt))], itin, FrmR, opstr>;

// Subword Swap
class SubwordSwap<string opstr, RegisterOperand RO,
                  InstrItinClass itin = NoItinerary>:
  InstSE<(outs RO:$rd), (ins RO:$rt), !strconcat(opstr, "\t$rd, $rt"), [], itin,
         FrmR, opstr> {
  let hasSideEffects = 0;
}

// Read Hardware
class ReadHardware<RegisterOperand CPURegOperand, RegisterOperand RO> :
  InstSE<(outs CPURegOperand:$rt), (ins RO:$rd, uimm8:$sel),
         "rdhwr\t$rt, $rd, $sel", [], II_RDHWR, FrmR, "rdhwr">;

// Ext and Ins
class ExtBase<string opstr, RegisterOperand RO, Operand PosOpnd,
              Operand SizeOpnd, PatFrag PosImm, PatFrag SizeImm,
              SDPatternOperator Op = null_frag> :
  InstSE<(outs RO:$rt), (ins RO:$rs, PosOpnd:$pos, SizeOpnd:$size),
         !strconcat(opstr, "\t$rt, $rs, $pos, $size"),
         [(set RO:$rt, (Op RO:$rs, PosImm:$pos, SizeImm:$size))], II_EXT,
         FrmR, opstr>;

// 'ins' and its' 64 bit variants are matched by C++ code.
class InsBase<string opstr, RegisterOperand RO, Operand PosOpnd,
              Operand SizeOpnd, PatFrag PosImm, PatFrag SizeImm>:
  InstSE<(outs RO:$rt), (ins RO:$rs, PosOpnd:$pos, SizeOpnd:$size, RO:$src),
         !strconcat(opstr, "\t$rt, $rs, $pos, $size"),
         [(set RO:$rt, (null_frag RO:$rs, PosImm:$pos, SizeImm:$size,
                                  RO:$src))],
         II_INS, FrmR, opstr> {
  let Constraints = "$src = $rt";
}

class LLBase<string opstr, RegisterOperand RO, DAGOperand MO = mem> :
  InstSE<(outs RO:$rt), (ins MO:$addr), !strconcat(opstr, "\t$rt, $addr"),
         [], II_LL, FrmI, opstr> {
  let DecoderMethod = "DecodeMem";
  let mayLoad = 1;
}

class SCBase<string opstr, RegisterOperand RO> :
  InstSE<(outs RO:$dst), (ins RO:$rt, mem:$addr),
         !strconcat(opstr, "\t$rt, $addr"), [], II_SC, FrmI> {
  let DecoderMethod = "DecodeMem";
  let mayStore = 1;
  let Constraints = "$rt = $dst";
}

class MFC3OP<string asmstr, RegisterOperand RO, RegisterOperand RD,
             InstrItinClass itin> :
  InstSE<(outs RO:$rt), (ins RD:$rd, uimm3:$sel),
         !strconcat(asmstr, "\t$rt, $rd, $sel"), [], itin, FrmFR> {
  let BaseOpcode = asmstr;
}

class MTC3OP<string asmstr, RegisterOperand RO, RegisterOperand RD,
             InstrItinClass itin> :
  InstSE<(outs RO:$rd), (ins RD:$rt, uimm3:$sel),
         !strconcat(asmstr, "\t$rt, $rd, $sel"), [], itin, FrmFR> {
  let BaseOpcode = asmstr;
}



//===----------------------------------------------------------------------===//
// Instruction definition
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// MipsI Instructions
//===----------------------------------------------------------------------===//

/// Arithmetic Instructions (ALU Immediate)
let AdditionalPredicates = [NotInMicroMips] in {
  def ADDiu : MMRel, StdMMR6Rel, ArithLogicI<"addiu", simm16_relaxed, GPR32Opnd,
                                             II_ADDIU, immSExt16, add>,
              ADDI_FM<0x9>, IsAsCheapAsAMove, ISA_MIPS1;

  def ANDi : MMRel, StdMMR6Rel,
             ArithLogicI<"andi", uimm16, GPR32Opnd, II_ANDI, immZExt16, and>,
             ADDI_FM<0xc>, ISA_MIPS1;
  def ORi  : MMRel, StdMMR6Rel,
             ArithLogicI<"ori", uimm16, GPR32Opnd, II_ORI, immZExt16, or>,
             ADDI_FM<0xd>, ISA_MIPS1;
  def XORi : MMRel, StdMMR6Rel,
             ArithLogicI<"xori", uimm16, GPR32Opnd, II_XORI, immZExt16, xor>,
             ADDI_FM<0xe>, ISA_MIPS1;
  def ADDi  : MMRel, ArithLogicI<"addi", simm16_relaxed, GPR32Opnd, II_ADDI>,
              ADDI_FM<0x8>, ISA_MIPS1_NOT_32R6_64R6;
  def SLTi  : MMRel, SetCC_I<"slti", setlt, simm16, immSExt16, GPR32Opnd>,
              SLTI_FM<0xa>, ISA_MIPS1;
  def SLTiu : MMRel, SetCC_I<"sltiu", setult, simm16, immSExt16, GPR32Opnd>,
              SLTI_FM<0xb>, ISA_MIPS1;

  def LUi   : MMRel, LoadUpper<"lui", GPR32Opnd, uimm16_relaxed>, LUI_FM,
              ISA_MIPS1;

  /// Arithmetic Instructions (3-Operand, R-Type)
  def ADDu  : MMRel, StdMMR6Rel, ArithLogicR<"addu", GPR32Opnd, 1, II_ADDU, add>,
              ADD_FM<0, 0x21>, ISA_MIPS1;
  def SUBu  : MMRel, StdMMR6Rel, ArithLogicR<"subu", GPR32Opnd, 0, II_SUBU, sub>,
              ADD_FM<0, 0x23>, ISA_MIPS1;

  let Defs = [HI0, LO0] in
    def MUL   : MMRel, ArithLogicR<"mul", GPR32Opnd, 1, II_MUL, mul>,
                ADD_FM<0x1c, 2>, ISA_MIPS32_NOT_32R6_64R6;

  def ADD   : MMRel, StdMMR6Rel, ArithLogicR<"add", GPR32Opnd, 1, II_ADD>,
              ADD_FM<0, 0x20>, ISA_MIPS1;
  def SUB   : MMRel, StdMMR6Rel, ArithLogicR<"sub", GPR32Opnd, 0, II_SUB>,
              ADD_FM<0, 0x22>, ISA_MIPS1;

  def SLT   : MMRel, SetCC_R<"slt", setlt, GPR32Opnd>, ADD_FM<0, 0x2a>,
              ISA_MIPS1;
  def SLTu  : MMRel, SetCC_R<"sltu", setult, GPR32Opnd>, ADD_FM<0, 0x2b>,
              ISA_MIPS1;
  def AND   : MMRel, StdMMR6Rel, ArithLogicR<"and", GPR32Opnd, 1, II_AND, and>,
              ADD_FM<0, 0x24>, ISA_MIPS1;
  def OR    : MMRel, StdMMR6Rel, ArithLogicR<"or", GPR32Opnd, 1, II_OR, or>,
              ADD_FM<0, 0x25>, ISA_MIPS1;
  def XOR   : MMRel, StdMMR6Rel, ArithLogicR<"xor", GPR32Opnd, 1, II_XOR, xor>,
              ADD_FM<0, 0x26>, ISA_MIPS1;
  def NOR   : MMRel, StdMMR6Rel, LogicNOR<"nor", GPR32Opnd>, ADD_FM<0, 0x27>,
              ISA_MIPS1;
}

let AdditionalPredicates = [NotInMicroMips] in {
  /// Shift Instructions
  def SLL  : MMRel, shift_rotate_imm<"sll", uimm5, GPR32Opnd, II_SLL, shl,
                                     immZExt5>, SRA_FM<0, 0>, ISA_MIPS1;
  def SRL  : MMRel, shift_rotate_imm<"srl", uimm5, GPR32Opnd, II_SRL, srl,
                                     immZExt5>, SRA_FM<2, 0>, ISA_MIPS1;
  def SRA  : MMRel, shift_rotate_imm<"sra", uimm5, GPR32Opnd, II_SRA, sra,
                                     immZExt5>, SRA_FM<3, 0>, ISA_MIPS1;
  def SLLV : MMRel, shift_rotate_reg<"sllv", GPR32Opnd, II_SLLV, shl>,
             SRLV_FM<4, 0>, ISA_MIPS1;
  def SRLV : MMRel, shift_rotate_reg<"srlv", GPR32Opnd, II_SRLV, srl>,
             SRLV_FM<6, 0>, ISA_MIPS1;
  def SRAV : MMRel, shift_rotate_reg<"srav", GPR32Opnd, II_SRAV, sra>,
             SRLV_FM<7, 0>, ISA_MIPS1;

  // Rotate Instructions
  def ROTR  : MMRel, shift_rotate_imm<"rotr", uimm5, GPR32Opnd, II_ROTR, rotr,
                                      immZExt5>,
              SRA_FM<2, 1>, ISA_MIPS32R2;
  def ROTRV : MMRel, shift_rotate_reg<"rotrv", GPR32Opnd, II_ROTRV, rotr>,
              SRLV_FM<6, 1>, ISA_MIPS32R2;
}

/// Load and Store Instructions
///  aligned
let AdditionalPredicates = [NotInMicroMips] in {
  def LB  : LoadMemory<"lb", GPR32Opnd, mem_simmptr, sextloadi8, II_LB>, MMRel,
            LW_FM<0x20>, ISA_MIPS1;
  def LBu : LoadMemory<"lbu", GPR32Opnd, mem_simmptr, zextloadi8, II_LBU,
                       addrDefault>, MMRel, LW_FM<0x24>, ISA_MIPS1;
  def LH  : LoadMemory<"lh", GPR32Opnd, mem_simmptr, sextloadi16, II_LH,
                       addrDefault>, MMRel, LW_FM<0x21>, ISA_MIPS1;
  def LHu : LoadMemory<"lhu", GPR32Opnd, mem_simmptr, zextloadi16, II_LHU>,
            MMRel, LW_FM<0x25>, ISA_MIPS1;
  def LW  : StdMMR6Rel, Load<"lw", GPR32Opnd, load, II_LW, addrDefault>, MMRel,
            LW_FM<0x23>, ISA_MIPS1;
  def SB  : StdMMR6Rel, Store<"sb", GPR32Opnd, truncstorei8, II_SB>, MMRel,
            LW_FM<0x28>, ISA_MIPS1;
  def SH  : Store<"sh", GPR32Opnd, truncstorei16, II_SH>, MMRel, LW_FM<0x29>,
            ISA_MIPS1;
  def SW  : StdMMR6Rel, Store<"sw", GPR32Opnd, store, II_SW>, MMRel, LW_FM<0x2b>, ISA_MIPS1;
}

/// load/store left/right
let AdditionalPredicates = [NotInMicroMips] in {
def LWL : MMRel, LoadLeftRight<"lwl", MipsLWL, GPR32Opnd, II_LWL>, LW_FM<0x22>,
          ISA_MIPS1_NOT_32R6_64R6;
def LWR : MMRel, LoadLeftRight<"lwr", MipsLWR, GPR32Opnd, II_LWR>, LW_FM<0x26>,
          ISA_MIPS1_NOT_32R6_64R6;
def SWL : MMRel, StoreLeftRight<"swl", MipsSWL, GPR32Opnd, II_SWL>, LW_FM<0x2a>,
          ISA_MIPS1_NOT_32R6_64R6;
def SWR : MMRel, StoreLeftRight<"swr", MipsSWR, GPR32Opnd, II_SWR>, LW_FM<0x2e>,
          ISA_MIPS1_NOT_32R6_64R6;

// COP2 Memory Instructions
def LWC2 : StdMMR6Rel, LW_FT2<"lwc2", COP2Opnd, II_LWC2, load>, LW_FM<0x32>,
           ISA_MIPS1_NOT_32R6_64R6;
def SWC2 : StdMMR6Rel, SW_FT2<"swc2", COP2Opnd, II_SWC2, store>,
           LW_FM<0x3a>, ISA_MIPS1_NOT_32R6_64R6;
def LDC2 : StdMMR6Rel, LW_FT2<"ldc2", COP2Opnd, II_LDC2, load>, LW_FM<0x36>,
           ISA_MIPS2_NOT_32R6_64R6;
def SDC2 : StdMMR6Rel, SW_FT2<"sdc2", COP2Opnd, II_SDC2, store>,
           LW_FM<0x3e>, ISA_MIPS2_NOT_32R6_64R6;

// COP3 Memory Instructions
let DecoderNamespace = "COP3_" in {
  def LWC3 : LW_FT3<"lwc3", COP3Opnd, II_LWC3, load>, LW_FM<0x33>,
             ISA_MIPS1_NOT_32R6_64R6, NOT_ASE_CNMIPS;
  def SWC3 : SW_FT3<"swc3", COP3Opnd, II_SWC3, store>, LW_FM<0x3b>,
             ISA_MIPS1_NOT_32R6_64R6, NOT_ASE_CNMIPS;
  def LDC3 : LW_FT3<"ldc3", COP3Opnd, II_LDC3, load>, LW_FM<0x37>,
             ISA_MIPS2, NOT_ASE_CNMIPS;
  def SDC3 : SW_FT3<"sdc3", COP3Opnd, II_SDC3, store>, LW_FM<0x3f>,
             ISA_MIPS2, NOT_ASE_CNMIPS;
}

  def SYNC : MMRel, StdMMR6Rel, SYNC_FT<"sync">, SYNC_FM, ISA_MIPS2;
  def SYNCI : MMRel, StdMMR6Rel, SYNCI_FT<"synci", mem_simm16>, SYNCI_FM,
              ISA_MIPS32R2;
}

let AdditionalPredicates = [NotInMicroMips] in {
  def TEQ : MMRel, TEQ_FT<"teq", GPR32Opnd, uimm10, II_TEQ>, TEQ_FM<0x34>,
            ISA_MIPS2;
  def TGE : MMRel, TEQ_FT<"tge", GPR32Opnd, uimm10, II_TGE>, TEQ_FM<0x30>,
            ISA_MIPS2;
  def TGEU : MMRel, TEQ_FT<"tgeu", GPR32Opnd, uimm10, II_TGEU>, TEQ_FM<0x31>,
             ISA_MIPS2;
  def TLT : MMRel, TEQ_FT<"tlt", GPR32Opnd, uimm10, II_TLT>, TEQ_FM<0x32>,
            ISA_MIPS2;
  def TLTU : MMRel, TEQ_FT<"tltu", GPR32Opnd, uimm10, II_TLTU>, TEQ_FM<0x33>,
            ISA_MIPS2;
  def TNE : MMRel, TEQ_FT<"tne", GPR32Opnd, uimm10, II_TNE>, TEQ_FM<0x36>,
            ISA_MIPS2;

  def TEQI : MMRel, TEQI_FT<"teqi", GPR32Opnd, II_TEQI>, TEQI_FM<0xc>,
             ISA_MIPS2_NOT_32R6_64R6;
  def TGEI : MMRel, TEQI_FT<"tgei", GPR32Opnd, II_TGEI>, TEQI_FM<0x8>,
             ISA_MIPS2_NOT_32R6_64R6;
  def TGEIU : MMRel, TEQI_FT<"tgeiu", GPR32Opnd, II_TGEIU>, TEQI_FM<0x9>,
              ISA_MIPS2_NOT_32R6_64R6;
  def TLTI : MMRel, TEQI_FT<"tlti", GPR32Opnd, II_TLTI>, TEQI_FM<0xa>,
             ISA_MIPS2_NOT_32R6_64R6;
  def TTLTIU : MMRel, TEQI_FT<"tltiu", GPR32Opnd, II_TTLTIU>, TEQI_FM<0xb>,
               ISA_MIPS2_NOT_32R6_64R6;
  def TNEI : MMRel, TEQI_FT<"tnei", GPR32Opnd, II_TNEI>, TEQI_FM<0xe>,
             ISA_MIPS2_NOT_32R6_64R6;
}

let AdditionalPredicates = [NotInMicroMips] in {
  def BREAK : MMRel, StdMMR6Rel, BRK_FT<"break">, BRK_FM<0xd>, ISA_MIPS1;
  def SYSCALL : MMRel, SYS_FT<"syscall", uimm20, II_SYSCALL>, SYS_FM<0xc>,
                ISA_MIPS1;
  def SDBBP : MMRel, SYS_FT<"sdbbp", uimm20, II_SDBBP>, SDBBP_FM,
              ISA_MIPS32_NOT_32R6_64R6;

  def ERET : MMRel, ER_FT<"eret", II_ERET>, ER_FM<0x18, 0x0>, INSN_MIPS3_32;
  def ERETNC : MMRel, ER_FT<"eretnc", II_ERETNC>, ER_FM<0x18, 0x1>,
               ISA_MIPS32R5;
  def DERET : MMRel, ER_FT<"deret", II_DERET>, ER_FM<0x1f, 0x0>, ISA_MIPS32;

  def EI : MMRel, StdMMR6Rel, DEI_FT<"ei", GPR32Opnd, II_EI>, EI_FM<1>,
           ISA_MIPS32R2;
  def DI : MMRel, StdMMR6Rel, DEI_FT<"di", GPR32Opnd, II_DI>, EI_FM<0>,
           ISA_MIPS32R2;

  def WAIT : MMRel, StdMMR6Rel, WAIT_FT<"wait">, WAIT_FM, INSN_MIPS3_32;
}

let AdditionalPredicates = [NotInMicroMips] in {
/// Load-linked, Store-conditional
def LL : LLBase<"ll", GPR32Opnd>, LW_FM<0x30>, PTR_32, ISA_MIPS2_NOT_32R6_64R6;
def SC : SCBase<"sc", GPR32Opnd>, LW_FM<0x38>, PTR_32, ISA_MIPS2_NOT_32R6_64R6;
}
/// Jump and Branch Instructions
let AdditionalPredicates = [NotInMicroMips, RelocNotPIC] in
def J       : MMRel, JumpFJ<jmptarget, "j", br, bb, "j">, FJ<2>,
              IsBranch, ISA_MIPS1;

let AdditionalPredicates = [NotInMicroMips] in {
def JR      : MMRel, IndirectBranch<"jr", GPR32Opnd>, MTLO_FM<8>, ISA_MIPS1_NOT_32R6_64R6;
def BEQ     : MMRel, CBranch<"beq", brtarget, seteq, GPR32Opnd>, BEQ_FM<4>,
              ISA_MIPS1;
def BEQL    : MMRel, CBranchLikely<"beql", brtarget, GPR32Opnd>,
              BEQ_FM<20>, ISA_MIPS2_NOT_32R6_64R6;
def BNE     : MMRel, CBranch<"bne", brtarget, setne, GPR32Opnd>, BEQ_FM<5>,
              ISA_MIPS1;
def BNEL    : MMRel, CBranchLikely<"bnel", brtarget, GPR32Opnd>,
              BEQ_FM<21>, ISA_MIPS2_NOT_32R6_64R6;
def BGEZ    : MMRel, CBranchZero<"bgez", brtarget, setge, GPR32Opnd>,
              BGEZ_FM<1, 1>, ISA_MIPS1;
def BGEZL   : MMRel, CBranchZeroLikely<"bgezl", brtarget, GPR32Opnd>,
              BGEZ_FM<1, 3>, ISA_MIPS2_NOT_32R6_64R6;
def BGTZ    : MMRel, CBranchZero<"bgtz", brtarget, setgt, GPR32Opnd>,
              BGEZ_FM<7, 0>, ISA_MIPS1;
def BGTZL   : MMRel, CBranchZeroLikely<"bgtzl", brtarget, GPR32Opnd>,
              BGEZ_FM<23, 0>, ISA_MIPS2_NOT_32R6_64R6;
def BLEZ    : MMRel, CBranchZero<"blez", brtarget, setle, GPR32Opnd>,
              BGEZ_FM<6, 0>, ISA_MIPS1;
def BLEZL   : MMRel, CBranchZeroLikely<"blezl", brtarget, GPR32Opnd>,
              BGEZ_FM<22, 0>, ISA_MIPS2_NOT_32R6_64R6;
def BLTZ    : MMRel, CBranchZero<"bltz", brtarget, setlt, GPR32Opnd>,
              BGEZ_FM<1, 0>, ISA_MIPS1;
def BLTZL   : MMRel, CBranchZeroLikely<"bltzl", brtarget, GPR32Opnd>,
              BGEZ_FM<1, 2>, ISA_MIPS2_NOT_32R6_64R6;


def JAL  : MMRel, JumpLink<"jal", calltarget>, FJ<3>, ISA_MIPS1;

}

let AdditionalPredicates = [NotInMicroMips, NoIndirectJumpGuards] in {
  def JALR : JumpLinkReg<"jalr", GPR32Opnd>, JALR_FM, ISA_MIPS1;

}

let AdditionalPredicates = [NotInMicroMips] in {
  def JALX : MMRel, JumpLink<"jalx", calltarget>, FJ<0x1D>,
             ISA_MIPS32_NOT_32R6_64R6;
  def BGEZAL : MMRel, BGEZAL_FT<"bgezal", brtarget, GPR32Opnd>, BGEZAL_FM<0x11>,
               ISA_MIPS1_NOT_32R6_64R6;
  def BGEZALL : MMRel, BGEZAL_FT<"bgezall", brtarget, GPR32Opnd>,
                BGEZAL_FM<0x13>, ISA_MIPS2_NOT_32R6_64R6;
  def BLTZAL : MMRel, BGEZAL_FT<"bltzal", brtarget, GPR32Opnd>, BGEZAL_FM<0x10>,
               ISA_MIPS1_NOT_32R6_64R6;
  def BLTZALL : MMRel, BGEZAL_FT<"bltzall", brtarget, GPR32Opnd>,
                BGEZAL_FM<0x12>, ISA_MIPS2_NOT_32R6_64R6;
}

/// Multiply and Divide Instructions.
let AdditionalPredicates = [NotInMicroMips] in {
  def MULT  : MMRel, Mult<"mult", II_MULT, GPR32Opnd, [HI0, LO0]>,
              MULT_FM<0, 0x18>, ISA_MIPS1_NOT_32R6_64R6;
  def MULTu : MMRel, Mult<"multu", II_MULTU, GPR32Opnd, [HI0, LO0]>,
              MULT_FM<0, 0x19>, ISA_MIPS1_NOT_32R6_64R6;
  def SDIV  : MMRel, Div<"div", II_DIV, GPR32Opnd, [HI0, LO0]>,
              MULT_FM<0, 0x1a>, ISA_MIPS1_NOT_32R6_64R6;
  def UDIV  : MMRel, Div<"divu", II_DIVU, GPR32Opnd, [HI0, LO0]>,
              MULT_FM<0, 0x1b>, ISA_MIPS1_NOT_32R6_64R6;
  def MTHI : MMRel, MoveToLOHI<"mthi", GPR32Opnd, [HI0]>, MTLO_FM<0x11>,
             ISA_MIPS1_NOT_32R6_64R6;
  def MTLO : MMRel, MoveToLOHI<"mtlo", GPR32Opnd, [LO0]>, MTLO_FM<0x13>,
             ISA_MIPS1_NOT_32R6_64R6;
  def MFHI : MMRel, MoveFromLOHI<"mfhi", GPR32Opnd, AC0>, MFLO_FM<0x10>,
             ISA_MIPS1_NOT_32R6_64R6;
  def MFLO : MMRel, MoveFromLOHI<"mflo", GPR32Opnd, AC0>, MFLO_FM<0x12>,
             ISA_MIPS1_NOT_32R6_64R6;

  /// Sign Ext In Register Instructions.
  def SEB : MMRel, StdMMR6Rel, SignExtInReg<"seb", i8, GPR32Opnd, II_SEB>,
            SEB_FM<0x10, 0x20>, ISA_MIPS32R2;
  def SEH : MMRel, StdMMR6Rel, SignExtInReg<"seh", i16, GPR32Opnd, II_SEH>,
            SEB_FM<0x18, 0x20>, ISA_MIPS32R2;

  /// Count Leading
  def CLZ : MMRel, CountLeading0<"clz", GPR32Opnd, II_CLZ>, CLO_FM<0x20>,
            ISA_MIPS32_NOT_32R6_64R6;
  def CLO : MMRel, CountLeading1<"clo", GPR32Opnd, II_CLO>, CLO_FM<0x21>,
            ISA_MIPS32_NOT_32R6_64R6;

  /// Word Swap Bytes Within Halfwords
  def WSBH : MMRel, SubwordSwap<"wsbh", GPR32Opnd, II_WSBH>, SEB_FM<2, 0x20>,
             ISA_MIPS32R2;



  // FrameIndexes are legalized when they are operands from load/store
  // instructions. The same not happens for stack address copies, so an
  // add op with mem ComplexPattern is used and the stack address copy
  // can be matched. It's similar to Sparc LEA_ADDRi
  let AdditionalPredicates = [NotInMicroMips] in
    def LEA_ADDiu : MMRel, EffectiveAddress<"addiu", GPR32Opnd>, LW_FM<9>, ISA_MIPS1;

  // MADD*/MSUB*
  def MADD  : MMRel, MArithR<"madd", II_MADD, 1>, MULT_FM<0x1c, 0>,
              ISA_MIPS32_NOT_32R6_64R6;
  def MADDU : MMRel, MArithR<"maddu", II_MADDU, 1>, MULT_FM<0x1c, 1>,
              ISA_MIPS32_NOT_32R6_64R6;
  def MSUB  : MMRel, MArithR<"msub", II_MSUB>, MULT_FM<0x1c, 4>,
              ISA_MIPS32_NOT_32R6_64R6;
  def MSUBU : MMRel, MArithR<"msubu", II_MSUBU>, MULT_FM<0x1c, 5>,
              ISA_MIPS32_NOT_32R6_64R6;
}


let AdditionalPredicates = [NotInMicroMips] in {
  def RDHWR : MMRel, ReadHardware<GPR32Opnd, HWRegsOpnd>, RDHWR_FM, ISA_MIPS1;
  // TODO: Add '0 < pos+size <= 32' constraint check to ext instruction
  def EXT : MMRel, StdMMR6Rel, ExtBase<"ext", GPR32Opnd, uimm5, uimm5_plus1,
                                       immZExt5, immZExt5Plus1, MipsExt>,
            EXT_FM<0>, ISA_MIPS32R2;
  def INS : MMRel, StdMMR6Rel, InsBase<"ins", GPR32Opnd, uimm5,
                                       uimm5_inssize_plus1, immZExt5,
                                       immZExt5Plus1>,
            EXT_FM<4>, ISA_MIPS32R2;
}
/// Move Control Registers From/To CPU Registers
let AdditionalPredicates = [NotInMicroMips] in {
  def MTC0 : MTC3OP<"mtc0", COP0Opnd, GPR32Opnd, II_MTC0>,
             MFC3OP_FM<0x10, 4, 0>, ISA_MIPS1;
  def MFC0 : MFC3OP<"mfc0", GPR32Opnd, COP0Opnd, II_MFC0>,
             MFC3OP_FM<0x10, 0, 0>, ISA_MIPS1;
  def MFC2 : MFC3OP<"mfc2", GPR32Opnd, COP2Opnd, II_MFC2>,
             MFC3OP_FM<0x12, 0, 0>, ISA_MIPS1;
  def MTC2 : MTC3OP<"mtc2", COP2Opnd, GPR32Opnd, II_MTC2>,
             MFC3OP_FM<0x12, 4, 0>, ISA_MIPS1;
}

class Barrier<string asmstr, InstrItinClass itin = NoItinerary> :
  InstSE<(outs), (ins), asmstr, [], itin, FrmOther, asmstr>;
let AdditionalPredicates = [NotInMicroMips] in {
  def SSNOP : MMRel, StdMMR6Rel, Barrier<"ssnop", II_SSNOP>, BARRIER_FM<1>,
              ISA_MIPS1;
  def EHB : MMRel, Barrier<"ehb", II_EHB>, BARRIER_FM<3>, ISA_MIPS1;

  let isCTI = 1 in
  def PAUSE : MMRel, StdMMR6Rel, Barrier<"pause", II_PAUSE>, BARRIER_FM<5>,
              ISA_MIPS32R2;
}

// JR_HB and JALR_HB are defined here using the new style naming
// scheme because some of this code is shared with Mips32r6InstrInfo.td
// and because of that it doesn't follow the naming convention of the
// rest of the file. To avoid a mixture of old vs new style, the new
// style was chosen.
class JR_HB_DESC_BASE<string instr_asm, RegisterOperand GPROpnd> {
  dag OutOperandList = (outs);
  dag InOperandList = (ins GPROpnd:$rs);
  string AsmString = !strconcat(instr_asm, "\t$rs");
  list<dag> Pattern = [];
}

class JALR_HB_DESC_BASE<string instr_asm, RegisterOperand GPROpnd> {
  dag OutOperandList = (outs GPROpnd:$rd);
  dag InOperandList = (ins GPROpnd:$rs);
  string AsmString = !strconcat(instr_asm, "\t$rd, $rs");
  list<dag> Pattern = [];
}

class JR_HB_DESC<RegisterOperand RO> :
  InstSE<(outs), (ins), "", [], II_JR_HB, FrmJ>, JR_HB_DESC_BASE<"jr.hb", RO> {
  let isBranch=1;
  let isIndirectBranch=1;
  let hasDelaySlot=1;
  let isTerminator=1;
  let isBarrier=1;
  bit isCTI = 1;
}

class JALR_HB_DESC<RegisterOperand RO> :
  InstSE<(outs), (ins), "", [], II_JALR_HB, FrmJ>, JALR_HB_DESC_BASE<"jalr.hb",
                                                                     RO> {
  let isIndirectBranch=1;
  let hasDelaySlot=1;
  bit isCTI = 1;
}

class JR_HB_ENC : JR_HB_FM<8>;
class JALR_HB_ENC : JALR_HB_FM<9>;

def JR_HB : JR_HB_DESC<GPR32Opnd>, JR_HB_ENC, ISA_MIPS32R2_NOT_32R6_64R6;
def JALR_HB : JALR_HB_DESC<GPR32Opnd>, JALR_HB_ENC, ISA_MIPS32;


class TLB<string asmstr, InstrItinClass itin = NoItinerary> :
  InstSE<(outs), (ins), asmstr, [], itin, FrmOther, asmstr>;
let AdditionalPredicates = [NotInMicroMips] in {
  def TLBP : MMRel, TLB<"tlbp", II_TLBP>, COP0_TLB_FM<0x08>, ISA_MIPS1;
  def TLBR : MMRel, TLB<"tlbr", II_TLBR>, COP0_TLB_FM<0x01>, ISA_MIPS1;
  def TLBWI : MMRel, TLB<"tlbwi", II_TLBWI>, COP0_TLB_FM<0x02>, ISA_MIPS1;
  def TLBWR : MMRel, TLB<"tlbwr", II_TLBWR>, COP0_TLB_FM<0x06>, ISA_MIPS1;
}
class CacheOp<string instr_asm, Operand MemOpnd,
              InstrItinClass itin = NoItinerary> :
    InstSE<(outs), (ins  MemOpnd:$addr, uimm5:$hint),
           !strconcat(instr_asm, "\t$hint, $addr"), [], itin, FrmOther,
           instr_asm> {
  let DecoderMethod = "DecodeCacheOp";
}

let AdditionalPredicates = [NotInMicroMips] in {
  def CACHE : MMRel, CacheOp<"cache", mem, II_CACHE>, CACHEOP_FM<0b101111>,
              INSN_MIPS3_32_NOT_32R6_64R6;
  def PREF :  MMRel, CacheOp<"pref", mem, II_PREF>, CACHEOP_FM<0b110011>,
              INSN_MIPS3_32_NOT_32R6_64R6;
}

// Virtualization ASE
class HYPCALL_FT<string opstr> :
  InstSE<(outs), (ins uimm10:$code_),
         !strconcat(opstr, "\t$code_"), [], II_HYPCALL, FrmOther, opstr> {
  let BaseOpcode = opstr;
}

let AdditionalPredicates = [NotInMicroMips] in {
  def MFGC0    : MMRel, MFC3OP<"mfgc0", GPR32Opnd, COP0Opnd, II_MFGC0>,
                 MFC3OP_FM<0x10, 3, 0>, ISA_MIPS32R5, ASE_VIRT;
  def MTGC0    : MMRel, MTC3OP<"mtgc0", COP0Opnd, GPR32Opnd, II_MTGC0>,
                 MFC3OP_FM<0x10, 3, 2>, ISA_MIPS32R5, ASE_VIRT;
  def MFHGC0   : MMRel, MFC3OP<"mfhgc0", GPR32Opnd, COP0Opnd, II_MFHGC0>,
                 MFC3OP_FM<0x10, 3, 4>, ISA_MIPS32R5, ASE_VIRT;
  def MTHGC0   : MMRel, MTC3OP<"mthgc0", COP0Opnd, GPR32Opnd, II_MTHGC0>,
                 MFC3OP_FM<0x10, 3, 6>, ISA_MIPS32R5, ASE_VIRT;
  def TLBGINV  : MMRel, TLB<"tlbginv", II_TLBGINV>, COP0_TLB_FM<0b001011>,
                 ISA_MIPS32R5, ASE_VIRT;
  def TLBGINVF : MMRel, TLB<"tlbginvf", II_TLBGINVF>, COP0_TLB_FM<0b001100>,
                 ISA_MIPS32R5, ASE_VIRT;
  def TLBGP    : MMRel, TLB<"tlbgp", II_TLBGP>, COP0_TLB_FM<0b010000>,
                 ISA_MIPS32R5, ASE_VIRT;
  def TLBGR    : MMRel, TLB<"tlbgr", II_TLBGR>, COP0_TLB_FM<0b001001>,
                 ISA_MIPS32R5, ASE_VIRT;
  def TLBGWI   : MMRel, TLB<"tlbgwi", II_TLBGWI>, COP0_TLB_FM<0b001010>,
                 ISA_MIPS32R5, ASE_VIRT;
  def TLBGWR   : MMRel, TLB<"tlbgwr", II_TLBGWR>, COP0_TLB_FM<0b001110>,
                 ISA_MIPS32R5, ASE_VIRT;
  def HYPCALL  : MMRel, HYPCALL_FT<"hypcall">,
                 HYPCALL_FM<0b101000>, ISA_MIPS32R5, ASE_VIRT;
}









//===----------------------------------------------------------------------===//
// Instruction aliases
//===----------------------------------------------------------------------===//
