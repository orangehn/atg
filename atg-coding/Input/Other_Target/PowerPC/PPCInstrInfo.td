//===-- PPCInstrInfo.td - The PowerPC Instruction Set ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the subset of the 32-bit PowerPC instruction set, as used
// by the PowerPC instruction selector.
//
//===----------------------------------------------------------------------===//

include "PPCInstrFormats.td"

//===----------------------------------------------------------------------===//
// PowerPC Flag Definitions.

class isPPC64 { bit PPC64 = 1; }
class isDOT   { bit RC = 1; }

class RegConstraint<string C> {
  string Constraints = C;
}
class NoEncode<string E> {
  string DisableEncoding = E;
}



//===----------------------------------------------------------------------===//
// PowerPC Instruction Predicate Definitions.
def In32BitMode  : Predicate<"!PPCSubTarget->isPPC64()">;
def IsBookE  : Predicate<"PPCSubTarget->isBookE()">;
def HasOnlyMSYNC : Predicate<"PPCSubTarget->hasOnlyMSYNC()">;
def IsPPC4xx  : Predicate<"PPCSubTarget->isPPC4xx()">;
def IsPPC6xx  : Predicate<"PPCSubTarget->isPPC6xx()">;
def IsE500  : Predicate<"PPCSubTarget->isE500()">;
def HasSPE  : Predicate<"PPCSubTarget->hasSPE()">;
def HasICBT : Predicate<"PPCSubTarget->hasICBT()">;
def HasPartwordAtomics : Predicate<"PPCSubTarget->hasPartwordAtomics()">;
def NoNaNsFPMath : Predicate<"TM.Options.NoNaNsFPMath">;
def NaNsFPMath   : Predicate<"!TM.Options.NoNaNsFPMath">;
def HasBPERMD : Predicate<"PPCSubTarget->hasBPERMD()">;
def HasExtDiv : Predicate<"PPCSubTarget->hasExtDiv()">;
def IsISA3_0 : Predicate<"PPCSubTarget->isISA3_0()">;
def HasFPU : Predicate<"PPCSubTarget->hasFPU()">;

//===----------------------------------------------------------------------===//
// PowerPC Multiclass Definitions.

multiclass XForm_6r<bits<6> opcode, bits<10> xo, dag OOL, dag IOL,
                    string asmbase, string asmstr, InstrItinClass itin,
                    list<dag> pattern> {
  let BaseName = asmbase in {
    def NAME : XForm_6<opcode, xo, OOL, IOL,
                       !strconcat(asmbase, !strconcat(" ", asmstr)), itin,
                       pattern>, RecFormRel;
    let Defs = [CR0] in
    def o    : XForm_6<opcode, xo, OOL, IOL,
                       !strconcat(asmbase, !strconcat(". ", asmstr)), itin,
                       []>, isDOT, RecFormRel;
  }
}

multiclass XForm_6rc<bits<6> opcode, bits<10> xo, dag OOL, dag IOL,
                     string asmbase, string asmstr, InstrItinClass itin,
                     list<dag> pattern> {
  let BaseName = asmbase in {
    let Defs = [CARRY] in
    def NAME : XForm_6<opcode, xo, OOL, IOL,
                       !strconcat(asmbase, !strconcat(" ", asmstr)), itin,
                       pattern>, RecFormRel;
    let Defs = [CARRY, CR0] in
    def o    : XForm_6<opcode, xo, OOL, IOL,
                       !strconcat(asmbase, !strconcat(". ", asmstr)), itin,
                       []>, isDOT, RecFormRel;
  }
}

multiclass XForm_10rc<bits<6> opcode, bits<10> xo, dag OOL, dag IOL,
                      string asmbase, string asmstr, InstrItinClass itin,
                      list<dag> pattern> {
  let BaseName = asmbase in {
    let Defs = [CARRY] in
    def NAME : XForm_10<opcode, xo, OOL, IOL,
                       !strconcat(asmbase, !strconcat(" ", asmstr)), itin,
                       pattern>, RecFormRel;
    let Defs = [CARRY, CR0] in
    def o    : XForm_10<opcode, xo, OOL, IOL,
                       !strconcat(asmbase, !strconcat(". ", asmstr)), itin,
                       []>, isDOT, RecFormRel;
  }
}

multiclass XForm_11r<bits<6> opcode, bits<10> xo, dag OOL, dag IOL,
                    string asmbase, string asmstr, InstrItinClass itin,
                    list<dag> pattern> {
  let BaseName = asmbase in {
    def NAME : XForm_11<opcode, xo, OOL, IOL,
                       !strconcat(asmbase, !strconcat(" ", asmstr)), itin,
                       pattern>, RecFormRel;
    let Defs = [CR0] in
    def o    : XForm_11<opcode, xo, OOL, IOL,
                       !strconcat(asmbase, !strconcat(". ", asmstr)), itin,
                       []>, isDOT, RecFormRel;
  }
}

multiclass XOForm_1r<bits<6> opcode, bits<9> xo, bit oe, dag OOL, dag IOL,
                    string asmbase, string asmstr, InstrItinClass itin,
                    list<dag> pattern> {
  let BaseName = asmbase in {
    def NAME : XOForm_1<opcode, xo, oe, OOL, IOL,
                       !strconcat(asmbase, !strconcat(" ", asmstr)), itin,
                       pattern>, RecFormRel;
    let Defs = [CR0] in
    def o    : XOForm_1<opcode, xo, oe, OOL, IOL,
                       !strconcat(asmbase, !strconcat(". ", asmstr)), itin,
                       []>, isDOT, RecFormRel;
  }
}

// Multiclass for instructions for which the non record form is not cracked
// and the record form is cracked (i.e. divw, mullw, etc.)
multiclass XOForm_1rcr<bits<6> opcode, bits<9> xo, bit oe, dag OOL, dag IOL,
                      string asmbase, string asmstr, InstrItinClass itin,
                      list<dag> pattern> {
  let BaseName = asmbase in {
    def NAME : XOForm_1<opcode, xo, oe, OOL, IOL,
                       !strconcat(asmbase, !strconcat(" ", asmstr)), itin,
                       pattern>, RecFormRel;
    let Defs = [CR0] in
    def o    : XOForm_1<opcode, xo, oe, OOL, IOL,
                       !strconcat(asmbase, !strconcat(". ", asmstr)), itin,
                       []>, isDOT, RecFormRel, PPC970_DGroup_First,
                       PPC970_DGroup_Cracked;
  }
}

multiclass XOForm_1rc<bits<6> opcode, bits<9> xo, bit oe, dag OOL, dag IOL,
                      string asmbase, string asmstr, InstrItinClass itin,
                      list<dag> pattern> {
  let BaseName = asmbase in {
    let Defs = [CARRY] in
    def NAME : XOForm_1<opcode, xo, oe, OOL, IOL,
                       !strconcat(asmbase, !strconcat(" ", asmstr)), itin,
                       pattern>, RecFormRel;
    let Defs = [CARRY, CR0] in
    def o    : XOForm_1<opcode, xo, oe, OOL, IOL,
                       !strconcat(asmbase, !strconcat(". ", asmstr)), itin,
                       []>, isDOT, RecFormRel;
  }
}

multiclass XOForm_3r<bits<6> opcode, bits<9> xo, bit oe, dag OOL, dag IOL,
                    string asmbase, string asmstr, InstrItinClass itin,
                    list<dag> pattern> {
  let BaseName = asmbase in {
    def NAME : XOForm_3<opcode, xo, oe, OOL, IOL,
                       !strconcat(asmbase, !strconcat(" ", asmstr)), itin,
                       pattern>, RecFormRel;
    let Defs = [CR0] in
    def o    : XOForm_3<opcode, xo, oe, OOL, IOL,
                       !strconcat(asmbase, !strconcat(". ", asmstr)), itin,
                       []>, isDOT, RecFormRel;
  }
}

multiclass XOForm_3rc<bits<6> opcode, bits<9> xo, bit oe, dag OOL, dag IOL,
                      string asmbase, string asmstr, InstrItinClass itin,
                      list<dag> pattern> {
  let BaseName = asmbase in {
    let Defs = [CARRY] in
    def NAME : XOForm_3<opcode, xo, oe, OOL, IOL,
                       !strconcat(asmbase, !strconcat(" ", asmstr)), itin,
                       pattern>, RecFormRel;
    let Defs = [CARRY, CR0] in
    def o    : XOForm_3<opcode, xo, oe, OOL, IOL,
                       !strconcat(asmbase, !strconcat(". ", asmstr)), itin,
                       []>, isDOT, RecFormRel;
  }
}

multiclass MForm_2r<bits<6> opcode, dag OOL, dag IOL,
                    string asmbase, string asmstr, InstrItinClass itin,
                    list<dag> pattern> {
  let BaseName = asmbase in {
    def NAME : MForm_2<opcode, OOL, IOL,
                       !strconcat(asmbase, !strconcat(" ", asmstr)), itin,
                       pattern>, RecFormRel;
    let Defs = [CR0] in
    def o    : MForm_2<opcode, OOL, IOL,
                       !strconcat(asmbase, !strconcat(". ", asmstr)), itin,
                       []>, isDOT, RecFormRel;
  }
}

multiclass MDForm_1r<bits<6> opcode, bits<3> xo, dag OOL, dag IOL,
                    string asmbase, string asmstr, InstrItinClass itin,
                    list<dag> pattern> {
  let BaseName = asmbase in {
    def NAME : MDForm_1<opcode, xo, OOL, IOL,
                       !strconcat(asmbase, !strconcat(" ", asmstr)), itin,
                       pattern>, RecFormRel;
    let Defs = [CR0] in
    def o    : MDForm_1<opcode, xo, OOL, IOL,
                       !strconcat(asmbase, !strconcat(". ", asmstr)), itin,
                       []>, isDOT, RecFormRel;
  }
}

multiclass MDSForm_1r<bits<6> opcode, bits<4> xo, dag OOL, dag IOL,
                     string asmbase, string asmstr, InstrItinClass itin,
                     list<dag> pattern> {
  let BaseName = asmbase in {
    def NAME : MDSForm_1<opcode, xo, OOL, IOL,
                        !strconcat(asmbase, !strconcat(" ", asmstr)), itin,
                        pattern>, RecFormRel;
    let Defs = [CR0] in
    def o    : MDSForm_1<opcode, xo, OOL, IOL,
                        !strconcat(asmbase, !strconcat(". ", asmstr)), itin,
                        []>, isDOT, RecFormRel;
  }
}

multiclass XSForm_1rc<bits<6> opcode, bits<9> xo, dag OOL, dag IOL,
                      string asmbase, string asmstr, InstrItinClass itin,
                      list<dag> pattern> {
  let BaseName = asmbase in {
    let Defs = [CARRY] in
    def NAME : XSForm_1<opcode, xo, OOL, IOL,
                       !strconcat(asmbase, !strconcat(" ", asmstr)), itin,
                       pattern>, RecFormRel;
    let Defs = [CARRY, CR0] in
    def o    : XSForm_1<opcode, xo, OOL, IOL,
                       !strconcat(asmbase, !strconcat(". ", asmstr)), itin,
                       []>, isDOT, RecFormRel;
  }
}

multiclass XSForm_1r<bits<6> opcode, bits<9> xo, dag OOL, dag IOL,
                    string asmbase, string asmstr, InstrItinClass itin,
                    list<dag> pattern> {
  let BaseName = asmbase in {
    def NAME : XSForm_1<opcode, xo, OOL, IOL,
                       !strconcat(asmbase, !strconcat(" ", asmstr)), itin,
                       pattern>, RecFormRel;
    let Defs = [CR0] in
    def o    : XSForm_1<opcode, xo, OOL, IOL,
                       !strconcat(asmbase, !strconcat(". ", asmstr)), itin,
                       []>, isDOT, RecFormRel;
  }
}

multiclass XForm_26r<bits<6> opcode, bits<10> xo, dag OOL, dag IOL,
                    string asmbase, string asmstr, InstrItinClass itin,
                    list<dag> pattern> {
  let BaseName = asmbase in {
    def NAME : XForm_26<opcode, xo, OOL, IOL,
                       !strconcat(asmbase, !strconcat(" ", asmstr)), itin,
                       pattern>, RecFormRel;
    let Defs = [CR1] in
    def o    : XForm_26<opcode, xo, OOL, IOL,
                       !strconcat(asmbase, !strconcat(". ", asmstr)), itin,
                       []>, isDOT, RecFormRel;
  }
}

multiclass XForm_28r<bits<6> opcode, bits<10> xo, dag OOL, dag IOL,
                    string asmbase, string asmstr, InstrItinClass itin,
                    list<dag> pattern> {
  let BaseName = asmbase in {
    def NAME : XForm_28<opcode, xo, OOL, IOL,
                       !strconcat(asmbase, !strconcat(" ", asmstr)), itin,
                       pattern>, RecFormRel;
    let Defs = [CR1] in
    def o    : XForm_28<opcode, xo, OOL, IOL,
                       !strconcat(asmbase, !strconcat(". ", asmstr)), itin,
                       []>, isDOT, RecFormRel;
  }
}

multiclass AForm_1r<bits<6> opcode, bits<5> xo, dag OOL, dag IOL,
                    string asmbase, string asmstr, InstrItinClass itin,
                    list<dag> pattern> {
  let BaseName = asmbase in {
    def NAME : AForm_1<opcode, xo, OOL, IOL,
                       !strconcat(asmbase, !strconcat(" ", asmstr)), itin,
                       pattern>, RecFormRel;
    let Defs = [CR1] in
    def o    : AForm_1<opcode, xo, OOL, IOL,
                       !strconcat(asmbase, !strconcat(". ", asmstr)), itin,
                       []>, isDOT, RecFormRel;
  }
}

multiclass AForm_2r<bits<6> opcode, bits<5> xo, dag OOL, dag IOL,
                    string asmbase, string asmstr, InstrItinClass itin,
                    list<dag> pattern> {
  let BaseName = asmbase in {
    def NAME : AForm_2<opcode, xo, OOL, IOL,
                       !strconcat(asmbase, !strconcat(" ", asmstr)), itin,
                       pattern>, RecFormRel;
    let Defs = [CR1] in
    def o    : AForm_2<opcode, xo, OOL, IOL,
                       !strconcat(asmbase, !strconcat(". ", asmstr)), itin,
                       []>, isDOT, RecFormRel;
  }
}

multiclass AForm_3r<bits<6> opcode, bits<5> xo, dag OOL, dag IOL,
                    string asmbase, string asmstr, InstrItinClass itin,
                    list<dag> pattern> {
  let BaseName = asmbase in {
    def NAME : AForm_3<opcode, xo, OOL, IOL,
                       !strconcat(asmbase, !strconcat(" ", asmstr)), itin,
                       pattern>, RecFormRel;
    let Defs = [CR1] in
    def o    : AForm_3<opcode, xo, OOL, IOL,
                       !strconcat(asmbase, !strconcat(". ", asmstr)), itin,
                       []>, isDOT, RecFormRel;
  }
}

//===----------------------------------------------------------------------===//
// PowerPC Instruction Definitions.

let isTerminator = 1, isBarrier = 1, PPC970_Unit = 7 in {
  let isReturn = 1, Uses = [LR, RM] in
    def BLR : XLForm_2_ext<19, 16, 20, 0, 0, (outs), (ins), "blr", IIC_BrB,
                           [(retflag)]>, Requires<[In32BitMode]>;
  let isBranch = 1, isIndirectBranch = 1, Uses = [CTR] in {
    def BCTR : XLForm_2_ext<19, 528, 20, 0, 0, (outs), (ins), "bctr", IIC_BrB,
                            []>;

    let isCodeGenOnly = 1 in {
      def BCCCTR : XLForm_2_br<19, 528, 0, (outs), (ins pred:$cond),
                               "b${cond:cc}ctr${cond:pm} ${cond:reg}", IIC_BrB,
                               []>;

      def BCCTR :  XLForm_2_br2<19, 528, 12, 0, (outs), (ins crbitrc:$bi),
                                "bcctr 12, $bi, 0", IIC_BrB, []>;
      def BCCTRn : XLForm_2_br2<19, 528, 4, 0, (outs), (ins crbitrc:$bi),
                                "bcctr 4, $bi, 0", IIC_BrB, []>;
    }
  }
}

let isBranch = 1, isTerminator = 1, hasCtrlDep = 1, PPC970_Unit = 7 in {
  let isBarrier = 1 in {
  def B   : IForm<18, 0, 0, (outs), (ins directbrtarget:$dst),
                  "b $dst", IIC_BrB,
                  [(br bb:$dst)]>;
  def BA  : IForm<18, 1, 0, (outs), (ins absdirectbrtarget:$dst),
                  "ba $dst", IIC_BrB, []>;
  }

  // BCC represents an arbitrary conditional branch on a predicate.
  // FIXME: should be able to write a pattern for PPCcondbranch, but can't use
  // a two-value operand where a dag node expects two operands. :(
  let isCodeGenOnly = 1 in {
    class BCC_class : BForm<16, 0, 0, (outs), (ins pred:$cond, condbrtarget:$dst),
                            "b${cond:cc}${cond:pm} ${cond:reg}, $dst"
                            /*[(PPCcondbranch crrc:$crS, imm:$opc, bb:$dst)]*/>;
    def BCC : BCC_class;

    // The same as BCC, except that it's not a terminator. Used for introducing
    // control flow dependency without creating new blocks.
    let isTerminator = 0 in def CTRL_DEP : BCC_class;

    def BCCA : BForm<16, 1, 0, (outs), (ins pred:$cond, abscondbrtarget:$dst),
                     "b${cond:cc}a${cond:pm} ${cond:reg}, $dst">;

    let isReturn = 1, Uses = [LR, RM] in
    def BCCLR : XLForm_2_br<19, 16, 0, (outs), (ins pred:$cond),
                           "b${cond:cc}lr${cond:pm} ${cond:reg}", IIC_BrB, []>;
  }

  let isCodeGenOnly = 1 in {
    let Pattern = [(brcond i1:$bi, bb:$dst)] in
    def BC  : BForm_4<16, 12, 0, 0, (outs), (ins crbitrc:$bi, condbrtarget:$dst),
             "bc 12, $bi, $dst">;

    let Pattern = [(brcond (not i1:$bi), bb:$dst)] in
    def BCn : BForm_4<16, 4, 0, 0, (outs), (ins crbitrc:$bi, condbrtarget:$dst),
             "bc 4, $bi, $dst">;

    let isReturn = 1, Uses = [LR, RM] in
    def BCLR  : XLForm_2_br2<19, 16, 12, 0, (outs), (ins crbitrc:$bi),
                             "bclr 12, $bi, 0", IIC_BrB, []>;
    def BCLRn : XLForm_2_br2<19, 16, 4, 0, (outs), (ins crbitrc:$bi),
                             "bclr 4, $bi, 0", IIC_BrB, []>;
  }

  let isReturn = 1, Defs = [CTR], Uses = [CTR, LR, RM] in {
   def BDZLR  : XLForm_2_ext<19, 16, 18, 0, 0, (outs), (ins),
                             "bdzlr", IIC_BrB, []>;
   def BDNZLR : XLForm_2_ext<19, 16, 16, 0, 0, (outs), (ins),
                             "bdnzlr", IIC_BrB, []>;
   def BDZLRp : XLForm_2_ext<19, 16, 27, 0, 0, (outs), (ins),
                             "bdzlr+", IIC_BrB, []>;
   def BDNZLRp: XLForm_2_ext<19, 16, 25, 0, 0, (outs), (ins),
                             "bdnzlr+", IIC_BrB, []>;
   def BDZLRm : XLForm_2_ext<19, 16, 26, 0, 0, (outs), (ins),
                             "bdzlr-", IIC_BrB, []>;
   def BDNZLRm: XLForm_2_ext<19, 16, 24, 0, 0, (outs), (ins),
                             "bdnzlr-", IIC_BrB, []>;
  }

  let Defs = [CTR], Uses = [CTR] in {
    def BDZ  : BForm_1<16, 18, 0, 0, (outs), (ins condbrtarget:$dst),
                       "bdz $dst">;
    def BDNZ : BForm_1<16, 16, 0, 0, (outs), (ins condbrtarget:$dst),
                       "bdnz $dst">;
    def BDZA  : BForm_1<16, 18, 1, 0, (outs), (ins abscondbrtarget:$dst),
                        "bdza $dst">;
    def BDNZA : BForm_1<16, 16, 1, 0, (outs), (ins abscondbrtarget:$dst),
                        "bdnza $dst">;
    def BDZp : BForm_1<16, 27, 0, 0, (outs), (ins condbrtarget:$dst),
                       "bdz+ $dst">;
    def BDNZp: BForm_1<16, 25, 0, 0, (outs), (ins condbrtarget:$dst),
                       "bdnz+ $dst">;
    def BDZAp : BForm_1<16, 27, 1, 0, (outs), (ins abscondbrtarget:$dst),
                        "bdza+ $dst">;
    def BDNZAp: BForm_1<16, 25, 1, 0, (outs), (ins abscondbrtarget:$dst),
                        "bdnza+ $dst">;
    def BDZm : BForm_1<16, 26, 0, 0, (outs), (ins condbrtarget:$dst),
                       "bdz- $dst">;
    def BDNZm: BForm_1<16, 24, 0, 0, (outs), (ins condbrtarget:$dst),
                       "bdnz- $dst">;
    def BDZAm : BForm_1<16, 26, 1, 0, (outs), (ins abscondbrtarget:$dst),
                        "bdza- $dst">;
    def BDNZAm: BForm_1<16, 24, 1, 0, (outs), (ins abscondbrtarget:$dst),
                        "bdnza- $dst">;
  }
}

// The unconditional BCL used by the SjLj setjmp code.
let isCall = 1, hasCtrlDep = 1, isCodeGenOnly = 1, PPC970_Unit = 7 in {
  let Defs = [LR], Uses = [RM] in {
    def BCLalways  : BForm_2<16, 20, 31, 0, 1, (outs), (ins condbrtarget:$dst),
                            "bcl 20, 31, $dst">;
  }
}

let isCall = 1, PPC970_Unit = 7, Defs = [LR] in {
  // Convenient aliases for call instructions
  let Uses = [RM] in {
    def BL  : IForm<18, 0, 1, (outs), (ins calltarget:$func),
                    "bl $func", IIC_BrB, []>;  // See Pat patterns below.
    def BLA : IForm<18, 1, 1, (outs), (ins abscalltarget:$func),
                    "bla $func", IIC_BrB, [(PPCcall (i32 imm:$func))]>;

    let isCodeGenOnly = 1 in {
      def BL_TLS  : IForm<18, 0, 1, (outs), (ins tlscall32:$func),
                          "bl $func", IIC_BrB, []>;
      def BCCL : BForm<16, 0, 1, (outs), (ins pred:$cond, condbrtarget:$dst),
                       "b${cond:cc}l${cond:pm} ${cond:reg}, $dst">;
      def BCCLA : BForm<16, 1, 1, (outs), (ins pred:$cond, abscondbrtarget:$dst),
                        "b${cond:cc}la${cond:pm} ${cond:reg}, $dst">;

      def BCL  : BForm_4<16, 12, 0, 1, (outs),
                         (ins crbitrc:$bi, condbrtarget:$dst),
                         "bcl 12, $bi, $dst">;
      def BCLn : BForm_4<16, 4, 0, 1, (outs),
                         (ins crbitrc:$bi, condbrtarget:$dst),
                         "bcl 4, $bi, $dst">;
      def BL_NOP  : IForm_and_DForm_4_zero<18, 0, 1, 24,
                                           (outs), (ins calltarget:$func),
                                           "bl $func\n\tnop", IIC_BrB, []>;
    }
  }
  let Uses = [CTR, RM] in {
    def BCTRL : XLForm_2_ext<19, 528, 20, 0, 1, (outs), (ins),
                             "bctrl", IIC_BrB, [(PPCbctrl)]>,
                Requires<[In32BitMode]>;

    let isCodeGenOnly = 1 in {
      def BCCCTRL : XLForm_2_br<19, 528, 1, (outs), (ins pred:$cond),
                                "b${cond:cc}ctrl${cond:pm} ${cond:reg}", IIC_BrB,
                                []>;

      def BCCTRL  : XLForm_2_br2<19, 528, 12, 1, (outs), (ins crbitrc:$bi),
                                 "bcctrl 12, $bi, 0", IIC_BrB, []>;
      def BCCTRLn : XLForm_2_br2<19, 528, 4, 1, (outs), (ins crbitrc:$bi),
                                 "bcctrl 4, $bi, 0", IIC_BrB, []>;
    }
  }
  let Uses = [LR, RM] in {
    def BLRL : XLForm_2_ext<19, 16, 20, 0, 1, (outs), (ins),
                            "blrl", IIC_BrB, []>;

    let isCodeGenOnly = 1 in {
      def BCCLRL : XLForm_2_br<19, 16, 1, (outs), (ins pred:$cond),
                              "b${cond:cc}lrl${cond:pm} ${cond:reg}", IIC_BrB,
                              []>;

      def BCLRL  : XLForm_2_br2<19, 16, 12, 1, (outs), (ins crbitrc:$bi),
                                "bclrl 12, $bi, 0", IIC_BrB, []>;
      def BCLRLn : XLForm_2_br2<19, 16, 4, 1, (outs), (ins crbitrc:$bi),
                                "bclrl 4, $bi, 0", IIC_BrB, []>;
    }
  }
  let Defs = [CTR], Uses = [CTR, RM] in {
    def BDZL  : BForm_1<16, 18, 0, 1, (outs), (ins condbrtarget:$dst),
                        "bdzl $dst">;
    def BDNZL : BForm_1<16, 16, 0, 1, (outs), (ins condbrtarget:$dst),
                        "bdnzl $dst">;
    def BDZLA  : BForm_1<16, 18, 1, 1, (outs), (ins abscondbrtarget:$dst),
                         "bdzla $dst">;
    def BDNZLA : BForm_1<16, 16, 1, 1, (outs), (ins abscondbrtarget:$dst),
                         "bdnzla $dst">;
    def BDZLp : BForm_1<16, 27, 0, 1, (outs), (ins condbrtarget:$dst),
                        "bdzl+ $dst">;
    def BDNZLp: BForm_1<16, 25, 0, 1, (outs), (ins condbrtarget:$dst),
                        "bdnzl+ $dst">;
    def BDZLAp : BForm_1<16, 27, 1, 1, (outs), (ins abscondbrtarget:$dst),
                         "bdzla+ $dst">;
    def BDNZLAp: BForm_1<16, 25, 1, 1, (outs), (ins abscondbrtarget:$dst),
                         "bdnzla+ $dst">;
    def BDZLm : BForm_1<16, 26, 0, 1, (outs), (ins condbrtarget:$dst),
                        "bdzl- $dst">;
    def BDNZLm: BForm_1<16, 24, 0, 1, (outs), (ins condbrtarget:$dst),
                        "bdnzl- $dst">;
    def BDZLAm : BForm_1<16, 26, 1, 1, (outs), (ins abscondbrtarget:$dst),
                         "bdzla- $dst">;
    def BDNZLAm: BForm_1<16, 24, 1, 1, (outs), (ins abscondbrtarget:$dst),
                         "bdnzla- $dst">;
  }
  let Defs = [CTR], Uses = [CTR, LR, RM] in {
    def BDZLRL  : XLForm_2_ext<19, 16, 18, 0, 1, (outs), (ins),
                               "bdzlrl", IIC_BrB, []>;
    def BDNZLRL : XLForm_2_ext<19, 16, 16, 0, 1, (outs), (ins),
                               "bdnzlrl", IIC_BrB, []>;
    def BDZLRLp : XLForm_2_ext<19, 16, 27, 0, 1, (outs), (ins),
                               "bdzlrl+", IIC_BrB, []>;
    def BDNZLRLp: XLForm_2_ext<19, 16, 25, 0, 1, (outs), (ins),
                               "bdnzlrl+", IIC_BrB, []>;
    def BDZLRLm : XLForm_2_ext<19, 16, 26, 0, 1, (outs), (ins),
                               "bdzlrl-", IIC_BrB, []>;
    def BDNZLRLm: XLForm_2_ext<19, 16, 24, 0, 1, (outs), (ins),
                               "bdnzlrl-", IIC_BrB, []>;
  }
}


let Defs = [CR0], mayStore = 1, mayLoad = 0, hasSideEffects = 0 in {
def STBCX : XForm_1_memOp<31, 694, (outs), (ins gprc:$rS, memrr:$dst),
                    "stbcx. $rS, $dst", IIC_LdStSTWCX, []>,
                    isDOT, Requires<[HasPartwordAtomics]>;

def STHCX : XForm_1_memOp<31, 726, (outs), (ins gprc:$rS, memrr:$dst),
                    "sthcx. $rS, $dst", IIC_LdStSTWCX, []>,
                    isDOT, Requires<[HasPartwordAtomics]>;

def STWCX : XForm_1_memOp<31, 150, (outs), (ins gprc:$rS, memrr:$dst),
                    "stwcx. $rS, $dst", IIC_LdStSTWCX, []>, isDOT;
}

let mayStore = 1, mayLoad = 0, hasSideEffects = 0 in
def STWAT : X_RD5_RS5_IM5<31, 710, (outs), (ins gprc:$rS, gprc:$rA, u5imm:$FC),
                          "stwat $rS, $rA, $FC", IIC_LdStStore>,
            Requires<[IsISA3_0]>;


//===----------------------------------------------------------------------===//
// PPC32 Load Instructions.
//

// Unindexed (r+i) Loads. 
let PPC970_Unit = 2 in {
def LBZ : DForm_1<34, (outs gprc:$rD), (ins memri:$src),
                  "lbz $rD, $src", IIC_LdStLoad,
                  [(set i32:$rD, (zextloadi8 iaddr:$src))]>;
def LHA : DForm_1<42, (outs gprc:$rD), (ins memri:$src),
                  "lha $rD, $src", IIC_LdStLHA,
                  [(set i32:$rD, (sextloadi16 iaddr:$src))]>,
                  PPC970_DGroup_Cracked;
def LHZ : DForm_1<40, (outs gprc:$rD), (ins memri:$src),
                  "lhz $rD, $src", IIC_LdStLoad,
                  [(set i32:$rD, (zextloadi16 iaddr:$src))]>;
def LWZ : DForm_1<32, (outs gprc:$rD), (ins memri:$src),
                  "lwz $rD, $src", IIC_LdStLoad,
                  [(set i32:$rD, (load iaddr:$src))]>;

let Predicates = [HasFPU] in {
def LFS : DForm_1<48, (outs f4rc:$rD), (ins memri:$src),
                  "lfs $rD, $src", IIC_LdStLFD,
                  [(set f32:$rD, (load iaddr:$src))]>;
def LFD : DForm_1<50, (outs f8rc:$rD), (ins memri:$src),
                  "lfd $rD, $src", IIC_LdStLFD,
                  [(set f64:$rD, (load iaddr:$src))]>;
}


// Unindexed (r+i) Loads with Update (preinc).
let mayLoad = 1, mayStore = 0, hasSideEffects = 0 in {
def LBZU : DForm_1<35, (outs gprc:$rD, ptr_rc_nor0:$ea_result), (ins memri:$addr),
                   "lbzu $rD, $addr", IIC_LdStLoadUpd,
                   []>, RegConstraint<"$addr.reg = $ea_result">,
                   NoEncode<"$ea_result">;

def LHAU : DForm_1<43, (outs gprc:$rD, ptr_rc_nor0:$ea_result), (ins memri:$addr),
                   "lhau $rD, $addr", IIC_LdStLHAU,
                   []>, RegConstraint<"$addr.reg = $ea_result">,
                   NoEncode<"$ea_result">;

def LHZU : DForm_1<41, (outs gprc:$rD, ptr_rc_nor0:$ea_result), (ins memri:$addr),
                   "lhzu $rD, $addr", IIC_LdStLoadUpd,
                   []>, RegConstraint<"$addr.reg = $ea_result">,
                   NoEncode<"$ea_result">;

def LWZU : DForm_1<33, (outs gprc:$rD, ptr_rc_nor0:$ea_result), (ins memri:$addr),
                   "lwzu $rD, $addr", IIC_LdStLoadUpd,
                   []>, RegConstraint<"$addr.reg = $ea_result">,
                   NoEncode<"$ea_result">;

let Predicates = [HasFPU] in {
def LFSU : DForm_1<49, (outs f4rc:$rD, ptr_rc_nor0:$ea_result), (ins memri:$addr),
                  "lfsu $rD, $addr", IIC_LdStLFDU,
                  []>, RegConstraint<"$addr.reg = $ea_result">,
                   NoEncode<"$ea_result">;

def LFDU : DForm_1<51, (outs f8rc:$rD, ptr_rc_nor0:$ea_result), (ins memri:$addr),
                  "lfdu $rD, $addr", IIC_LdStLFDU,
                  []>, RegConstraint<"$addr.reg = $ea_result">,
                   NoEncode<"$ea_result">;
}


// Indexed (r+r) Loads with Update (preinc).
def LBZUX : XForm_1_memOp<31, 119, (outs gprc:$rD, ptr_rc_nor0:$ea_result),
                   (ins memrr:$addr),
                   "lbzux $rD, $addr", IIC_LdStLoadUpdX,
                   []>, RegConstraint<"$addr.ptrreg = $ea_result">,
                   NoEncode<"$ea_result">;

def LHAUX : XForm_1_memOp<31, 375, (outs gprc:$rD, ptr_rc_nor0:$ea_result),
                   (ins memrr:$addr),
                   "lhaux $rD, $addr", IIC_LdStLHAUX,
                   []>, RegConstraint<"$addr.ptrreg = $ea_result">,
                   NoEncode<"$ea_result">;

def LHZUX : XForm_1_memOp<31, 311, (outs gprc:$rD, ptr_rc_nor0:$ea_result),
                   (ins memrr:$addr),
                   "lhzux $rD, $addr", IIC_LdStLoadUpdX,
                   []>, RegConstraint<"$addr.ptrreg = $ea_result">,
                   NoEncode<"$ea_result">;

def LWZUX : XForm_1_memOp<31, 55, (outs gprc:$rD, ptr_rc_nor0:$ea_result),
                   (ins memrr:$addr),
                   "lwzux $rD, $addr", IIC_LdStLoadUpdX,
                   []>, RegConstraint<"$addr.ptrreg = $ea_result">,
                   NoEncode<"$ea_result">;

let Predicates = [HasFPU] in {
def LFSUX : XForm_1_memOp<31, 567, (outs f4rc:$rD, ptr_rc_nor0:$ea_result),
                   (ins memrr:$addr),
                   "lfsux $rD, $addr", IIC_LdStLFDUX,
                   []>, RegConstraint<"$addr.ptrreg = $ea_result">,
                   NoEncode<"$ea_result">;

def LFDUX : XForm_1_memOp<31, 631, (outs f8rc:$rD, ptr_rc_nor0:$ea_result),
                   (ins memrr:$addr),
                   "lfdux $rD, $addr", IIC_LdStLFDUX,
                   []>, RegConstraint<"$addr.ptrreg = $ea_result">,
                   NoEncode<"$ea_result">;
}
}
}

// Indexed (r+r) Loads.
//
let PPC970_Unit = 2, mayLoad = 1, mayStore = 0 in {
def LBZX : XForm_1_memOp<31,  87, (outs gprc:$rD), (ins memrr:$src),
                   "lbzx $rD, $src", IIC_LdStLoad,
                   [(set i32:$rD, (zextloadi8 xaddr:$src))]>;
def LHAX : XForm_1_memOp<31, 343, (outs gprc:$rD), (ins memrr:$src),
                   "lhax $rD, $src", IIC_LdStLHA,
                   [(set i32:$rD, (sextloadi16 xaddr:$src))]>,
                   PPC970_DGroup_Cracked;
def LHZX : XForm_1_memOp<31, 279, (outs gprc:$rD), (ins memrr:$src),
                   "lhzx $rD, $src", IIC_LdStLoad,
                   [(set i32:$rD, (zextloadi16 xaddr:$src))]>;
def LWZX : XForm_1_memOp<31,  23, (outs gprc:$rD), (ins memrr:$src),
                   "lwzx $rD, $src", IIC_LdStLoad,
                   [(set i32:$rD, (load xaddr:$src))]>;
def LHBRX : XForm_1_memOp<31, 790, (outs gprc:$rD), (ins memrr:$src),
                   "lhbrx $rD, $src", IIC_LdStLoad,
                   [(set i32:$rD, (PPClbrx xoaddr:$src, i16))]>;
def LWBRX : XForm_1_memOp<31,  534, (outs gprc:$rD), (ins memrr:$src),
                   "lwbrx $rD, $src", IIC_LdStLoad,
                   [(set i32:$rD, (PPClbrx xoaddr:$src, i32))]>;

let Predicates = [HasFPU] in {
def LFSX   : XForm_25_memOp<31, 535, (outs f4rc:$frD), (ins memrr:$src),
                      "lfsx $frD, $src", IIC_LdStLFD,
                      [(set f32:$frD, (load xaddr:$src))]>;
def LFDX   : XForm_25_memOp<31, 599, (outs f8rc:$frD), (ins memrr:$src),
                      "lfdx $frD, $src", IIC_LdStLFD,
                      [(set f64:$frD, (load xaddr:$src))]>;

def LFIWAX : XForm_25_memOp<31, 855, (outs f8rc:$frD), (ins memrr:$src),
                      "lfiwax $frD, $src", IIC_LdStLFD,
                      [(set f64:$frD, (PPClfiwax xoaddr:$src))]>;
def LFIWZX : XForm_25_memOp<31, 887, (outs f8rc:$frD), (ins memrr:$src),
                      "lfiwzx $frD, $src", IIC_LdStLFD,
                      [(set f64:$frD, (PPClfiwzx xoaddr:$src))]>;
}
}

// Load Multiple
def LMW : DForm_1<46, (outs gprc:$rD), (ins memri:$src),
                  "lmw $rD, $src", IIC_LdStLMW, []>;

//===----------------------------------------------------------------------===//
// PPC32 Store Instructions.
//

// Unindexed (r+i) Stores.
let PPC970_Unit = 2, mayStore = 1, mayLoad = 0 in {
def STB  : DForm_1<38, (outs), (ins gprc:$rS, memri:$dst),
                   "stb $rS, $dst", IIC_LdStStore,
                   [(truncstorei8 i32:$rS, iaddr:$dst)]>;
def STH  : DForm_1<44, (outs), (ins gprc:$rS, memri:$dst),
                   "sth $rS, $dst", IIC_LdStStore,
                   [(truncstorei16 i32:$rS, iaddr:$dst)]>;
def STW  : DForm_1<36, (outs), (ins gprc:$rS, memri:$dst),
                   "stw $rS, $dst", IIC_LdStStore,
                   [(store i32:$rS, iaddr:$dst)]>;
let Predicates = [HasFPU] in {
def STFS : DForm_1<52, (outs), (ins f4rc:$rS, memri:$dst),
                   "stfs $rS, $dst", IIC_LdStSTFD,
                   [(store f32:$rS, iaddr:$dst)]>;
def STFD : DForm_1<54, (outs), (ins f8rc:$rS, memri:$dst),
                   "stfd $rS, $dst", IIC_LdStSTFD,
                   [(store f64:$rS, iaddr:$dst)]>;
}
}

// Unindexed (r+i) Stores with Update (preinc).
let PPC970_Unit = 2, mayStore = 1, mayLoad = 0 in {
def STBU  : DForm_1<39, (outs ptr_rc_nor0:$ea_res), (ins gprc:$rS, memri:$dst),
                    "stbu $rS, $dst", IIC_LdStSTU, []>,
                    RegConstraint<"$dst.reg = $ea_res">, NoEncode<"$ea_res">;
def STHU  : DForm_1<45, (outs ptr_rc_nor0:$ea_res), (ins gprc:$rS, memri:$dst),
                    "sthu $rS, $dst", IIC_LdStSTU, []>,
                    RegConstraint<"$dst.reg = $ea_res">, NoEncode<"$ea_res">;
def STWU  : DForm_1<37, (outs ptr_rc_nor0:$ea_res), (ins gprc:$rS, memri:$dst),
                    "stwu $rS, $dst", IIC_LdStSTU, []>,
                    RegConstraint<"$dst.reg = $ea_res">, NoEncode<"$ea_res">;
let Predicates = [HasFPU] in {
def STFSU : DForm_1<53, (outs ptr_rc_nor0:$ea_res), (ins f4rc:$rS, memri:$dst),
                    "stfsu $rS, $dst", IIC_LdStSTFDU, []>,
                    RegConstraint<"$dst.reg = $ea_res">, NoEncode<"$ea_res">;
def STFDU : DForm_1<55, (outs ptr_rc_nor0:$ea_res), (ins f8rc:$rS, memri:$dst),
                    "stfdu $rS, $dst", IIC_LdStSTFDU, []>,
                    RegConstraint<"$dst.reg = $ea_res">, NoEncode<"$ea_res">;
}
}


// Indexed (r+r) Stores.
let PPC970_Unit = 2 in {
def STBX  : XForm_8_memOp<31, 215, (outs), (ins gprc:$rS, memrr:$dst),
                   "stbx $rS, $dst", IIC_LdStStore,
                   [(truncstorei8 i32:$rS, xaddr:$dst)]>,
                   PPC970_DGroup_Cracked;
def STHX  : XForm_8_memOp<31, 407, (outs), (ins gprc:$rS, memrr:$dst),
                   "sthx $rS, $dst", IIC_LdStStore,
                   [(truncstorei16 i32:$rS, xaddr:$dst)]>,
                   PPC970_DGroup_Cracked;
def STWX  : XForm_8_memOp<31, 151, (outs), (ins gprc:$rS, memrr:$dst),
                   "stwx $rS, $dst", IIC_LdStStore,
                   [(store i32:$rS, xaddr:$dst)]>,
                   PPC970_DGroup_Cracked;

def STHBRX: XForm_8_memOp<31, 918, (outs), (ins gprc:$rS, memrr:$dst),
                   "sthbrx $rS, $dst", IIC_LdStStore,
                   [(PPCstbrx i32:$rS, xoaddr:$dst, i16)]>,
                   PPC970_DGroup_Cracked;
def STWBRX: XForm_8_memOp<31, 662, (outs), (ins gprc:$rS, memrr:$dst),
                   "stwbrx $rS, $dst", IIC_LdStStore,
                   [(PPCstbrx i32:$rS, xoaddr:$dst, i32)]>,
                   PPC970_DGroup_Cracked;

let Predicates = [HasFPU] in {
def STFIWX: XForm_28_memOp<31, 983, (outs), (ins f8rc:$frS, memrr:$dst),
                     "stfiwx $frS, $dst", IIC_LdStSTFD,
                     [(PPCstfiwx f64:$frS, xoaddr:$dst)]>;

def STFSX : XForm_28_memOp<31, 663, (outs), (ins f4rc:$frS, memrr:$dst),
                     "stfsx $frS, $dst", IIC_LdStSTFD,
                     [(store f32:$frS, xaddr:$dst)]>;
def STFDX : XForm_28_memOp<31, 727, (outs), (ins f8rc:$frS, memrr:$dst),
                     "stfdx $frS, $dst", IIC_LdStSTFD,
                     [(store f64:$frS, xaddr:$dst)]>;
}
}

// Indexed (r+r) Stores with Update (preinc).
let PPC970_Unit = 2, mayStore = 1, mayLoad = 0 in {
def STBUX : XForm_8_memOp<31, 247, (outs ptr_rc_nor0:$ea_res),
                          (ins gprc:$rS, memrr:$dst),
                          "stbux $rS, $dst", IIC_LdStSTUX, []>,
                          RegConstraint<"$dst.ptrreg = $ea_res">,
                          NoEncode<"$ea_res">,
                          PPC970_DGroup_Cracked;
def STHUX : XForm_8_memOp<31, 439, (outs ptr_rc_nor0:$ea_res),
                          (ins gprc:$rS, memrr:$dst),
                          "sthux $rS, $dst", IIC_LdStSTUX, []>,
                          RegConstraint<"$dst.ptrreg = $ea_res">,
                          NoEncode<"$ea_res">,
                          PPC970_DGroup_Cracked;
def STWUX : XForm_8_memOp<31, 183, (outs ptr_rc_nor0:$ea_res),
                          (ins gprc:$rS, memrr:$dst),
                          "stwux $rS, $dst", IIC_LdStSTUX, []>,
                          RegConstraint<"$dst.ptrreg = $ea_res">,
                          NoEncode<"$ea_res">,
                          PPC970_DGroup_Cracked;
let Predicates = [HasFPU] in {
def STFSUX: XForm_8_memOp<31, 695, (outs ptr_rc_nor0:$ea_res),
                          (ins f4rc:$rS, memrr:$dst),
                          "stfsux $rS, $dst", IIC_LdStSTFDU, []>,
                          RegConstraint<"$dst.ptrreg = $ea_res">,
                          NoEncode<"$ea_res">,
                          PPC970_DGroup_Cracked;
def STFDUX: XForm_8_memOp<31, 759, (outs ptr_rc_nor0:$ea_res),
                          (ins f8rc:$rS, memrr:$dst),
                          "stfdux $rS, $dst", IIC_LdStSTFDU, []>,
                          RegConstraint<"$dst.ptrreg = $ea_res">,
                          NoEncode<"$ea_res">,
                          PPC970_DGroup_Cracked;
}
}


// Store Multiple
def STMW : DForm_1<47, (outs), (ins gprc:$rS, memri:$dst),
                   "stmw $rS, $dst", IIC_LdStLMW, []>;



//===----------------------------------------------------------------------===//
// PPC32 Arithmetic Instructions.
//

let PPC970_Unit = 1 in {  // FXU Operations.
def ADDI   : DForm_2<14, (outs gprc:$rD), (ins gprc_nor0:$rA, s16imm:$imm),
                     "addi $rD, $rA, $imm", IIC_IntSimple,
                     [(set i32:$rD, (add i32:$rA, imm32SExt16:$imm))]>;
let BaseName = "addic" in {
let Defs = [CARRY] in
def ADDIC  : DForm_2<12, (outs gprc:$rD), (ins gprc:$rA, s16imm:$imm),
                     "addic $rD, $rA, $imm", IIC_IntGeneral,
                     [(set i32:$rD, (addc i32:$rA, imm32SExt16:$imm))]>,
                     RecFormRel, PPC970_DGroup_Cracked;
let Defs = [CARRY, CR0] in
def ADDICo : DForm_2<13, (outs gprc:$rD), (ins gprc:$rA, s16imm:$imm),
                     "addic. $rD, $rA, $imm", IIC_IntGeneral,
                     []>, isDOT, RecFormRel;
}
def ADDIS  : DForm_2<15, (outs gprc:$rD), (ins gprc_nor0:$rA, s17imm:$imm),
                     "addis $rD, $rA, $imm", IIC_IntSimple,
                     [(set i32:$rD, (add i32:$rA, imm16ShiftedSExt:$imm))]>;
def MULLI  : DForm_2< 7, (outs gprc:$rD), (ins gprc:$rA, s16imm:$imm),
                     "mulli $rD, $rA, $imm", IIC_IntMulLI,
                     [(set i32:$rD, (mul i32:$rA, imm32SExt16:$imm))]>;
let Defs = [CARRY] in
def SUBFIC : DForm_2< 8, (outs gprc:$rD), (ins gprc:$rA, s16imm:$imm),
                     "subfic $rD, $rA, $imm", IIC_IntGeneral,
                     [(set i32:$rD, (subc imm32SExt16:$imm, i32:$rA))]>;

let isReMaterializable = 1, isAsCheapAsAMove = 1, isMoveImm = 1 in {
  def LI  : DForm_2_r0<14, (outs gprc:$rD), (ins s16imm:$imm),
                       "li $rD, $imm", IIC_IntSimple,
                       [(set i32:$rD, imm32SExt16:$imm)]>;
  def LIS : DForm_2_r0<15, (outs gprc:$rD), (ins s17imm:$imm),
                       "lis $rD, $imm", IIC_IntSimple,
                       [(set i32:$rD, imm16ShiftedSExt:$imm)]>;
}
}

let PPC970_Unit = 1 in {  // FXU Operations.
let Defs = [CR0] in {
def ANDIo : DForm_4<28, (outs gprc:$dst), (ins gprc:$src1, u16imm:$src2),
                    "andi. $dst, $src1, $src2", IIC_IntGeneral,
                    [(set i32:$dst, (and i32:$src1, immZExt16:$src2))]>,
                    isDOT;
def ANDISo : DForm_4<29, (outs gprc:$dst), (ins gprc:$src1, u16imm:$src2),
                    "andis. $dst, $src1, $src2", IIC_IntGeneral,
                    [(set i32:$dst, (and i32:$src1, imm16ShiftedZExt:$src2))]>,
                    isDOT;
}
def ORI   : DForm_4<24, (outs gprc:$dst), (ins gprc:$src1, u16imm:$src2),
                    "ori $dst, $src1, $src2", IIC_IntSimple,
                    [(set i32:$dst, (or i32:$src1, immZExt16:$src2))]>;
def ORIS  : DForm_4<25, (outs gprc:$dst), (ins gprc:$src1, u16imm:$src2),
                    "oris $dst, $src1, $src2", IIC_IntSimple,
                    [(set i32:$dst, (or i32:$src1, imm16ShiftedZExt:$src2))]>;
def XORI  : DForm_4<26, (outs gprc:$dst), (ins gprc:$src1, u16imm:$src2),
                    "xori $dst, $src1, $src2", IIC_IntSimple,
                    [(set i32:$dst, (xor i32:$src1, immZExt16:$src2))]>;
def XORIS : DForm_4<27, (outs gprc:$dst), (ins gprc:$src1, u16imm:$src2),
                    "xoris $dst, $src1, $src2", IIC_IntSimple,
                    [(set i32:$dst, (xor i32:$src1, imm16ShiftedZExt:$src2))]>;

def NOP   : DForm_4_zero<24, (outs), (ins), "nop", IIC_IntSimple,
                         []>;

let isCompare = 1, hasSideEffects = 0 in {
  def CMPWI : DForm_5_ext<11, (outs crrc:$crD), (ins gprc:$rA, s16imm:$imm),
                          "cmpwi $crD, $rA, $imm", IIC_IntCompare>;
  def CMPLWI : DForm_6_ext<10, (outs crrc:$dst), (ins gprc:$src1, u16imm:$src2),
                           "cmplwi $dst, $src1, $src2", IIC_IntCompare>;
  def CMPRB  : X_BF3_L1_RS5_RS5<31, 192, (outs crbitrc:$BF),
                                (ins u1imm:$L, g8rc:$rA, g8rc:$rB),
                                "cmprb $BF, $L, $rA, $rB", IIC_IntCompare, []>,
               Requires<[IsISA3_0]>;
}
}

let PPC970_Unit = 1, hasSideEffects = 0 in {  // FXU Operations.
let isCommutable = 1 in {
defm NAND : XForm_6r<31, 476, (outs gprc:$rA), (ins gprc:$rS, gprc:$rB),
                     "nand", "$rA, $rS, $rB", IIC_IntSimple,
                     [(set i32:$rA, (not (and i32:$rS, i32:$rB)))]>;
defm AND  : XForm_6r<31,  28, (outs gprc:$rA), (ins gprc:$rS, gprc:$rB),
                     "and", "$rA, $rS, $rB", IIC_IntSimple,
                     [(set i32:$rA, (and i32:$rS, i32:$rB))]>;
} // isCommutable
defm ANDC : XForm_6r<31,  60, (outs gprc:$rA), (ins gprc:$rS, gprc:$rB),
                     "andc", "$rA, $rS, $rB", IIC_IntSimple,
                     [(set i32:$rA, (and i32:$rS, (not i32:$rB)))]>;
let isCommutable = 1 in {
defm OR   : XForm_6r<31, 444, (outs gprc:$rA), (ins gprc:$rS, gprc:$rB),
                     "or", "$rA, $rS, $rB", IIC_IntSimple,
                     [(set i32:$rA, (or i32:$rS, i32:$rB))]>;
defm NOR  : XForm_6r<31, 124, (outs gprc:$rA), (ins gprc:$rS, gprc:$rB),
                     "nor", "$rA, $rS, $rB", IIC_IntSimple,
                     [(set i32:$rA, (not (or i32:$rS, i32:$rB)))]>;
} // isCommutable
defm ORC  : XForm_6r<31, 412, (outs gprc:$rA), (ins gprc:$rS, gprc:$rB),
                     "orc", "$rA, $rS, $rB", IIC_IntSimple,
                     [(set i32:$rA, (or i32:$rS, (not i32:$rB)))]>;
let isCommutable = 1 in {
defm EQV  : XForm_6r<31, 284, (outs gprc:$rA), (ins gprc:$rS, gprc:$rB),
                     "eqv", "$rA, $rS, $rB", IIC_IntSimple,
                     [(set i32:$rA, (not (xor i32:$rS, i32:$rB)))]>;
defm XOR  : XForm_6r<31, 316, (outs gprc:$rA), (ins gprc:$rS, gprc:$rB),
                     "xor", "$rA, $rS, $rB", IIC_IntSimple,
                     [(set i32:$rA, (xor i32:$rS, i32:$rB))]>;
} // isCommutable
defm SLW  : XForm_6r<31,  24, (outs gprc:$rA), (ins gprc:$rS, gprc:$rB),
                     "slw", "$rA, $rS, $rB", IIC_IntGeneral,
                     [(set i32:$rA, (PPCshl i32:$rS, i32:$rB))]>;
defm SRW  : XForm_6r<31, 536, (outs gprc:$rA), (ins gprc:$rS, gprc:$rB),
                     "srw", "$rA, $rS, $rB", IIC_IntGeneral,
                     [(set i32:$rA, (PPCsrl i32:$rS, i32:$rB))]>;
defm SRAW : XForm_6rc<31, 792, (outs gprc:$rA), (ins gprc:$rS, gprc:$rB),
                      "sraw", "$rA, $rS, $rB", IIC_IntShift,
                      [(set i32:$rA, (PPCsra i32:$rS, i32:$rB))]>;
}

let PPC970_Unit = 1 in {  // FXU Operations.
let hasSideEffects = 0 in {
defm SRAWI : XForm_10rc<31, 824, (outs gprc:$rA), (ins gprc:$rS, u5imm:$SH),
                        "srawi", "$rA, $rS, $SH", IIC_IntShift,
                        [(set i32:$rA, (sra i32:$rS, (i32 imm:$SH)))]>;
defm CNTLZW : XForm_11r<31,  26, (outs gprc:$rA), (ins gprc:$rS),
                        "cntlzw", "$rA, $rS", IIC_IntGeneral,
                        [(set i32:$rA, (ctlz i32:$rS))]>;
defm CNTTZW : XForm_11r<31, 538, (outs gprc:$rA), (ins gprc:$rS),
                        "cnttzw", "$rA, $rS", IIC_IntGeneral,
                        [(set i32:$rA, (cttz i32:$rS))]>, Requires<[IsISA3_0]>;
defm EXTSB  : XForm_11r<31, 954, (outs gprc:$rA), (ins gprc:$rS),
                        "extsb", "$rA, $rS", IIC_IntSimple,
                        [(set i32:$rA, (sext_inreg i32:$rS, i8))]>;
defm EXTSH  : XForm_11r<31, 922, (outs gprc:$rA), (ins gprc:$rS),
                        "extsh", "$rA, $rS", IIC_IntSimple,
                        [(set i32:$rA, (sext_inreg i32:$rS, i16))]>;

let isCommutable = 1 in
def CMPB : XForm_6<31, 508, (outs gprc:$rA), (ins gprc:$rS, gprc:$rB),
                   "cmpb $rA, $rS, $rB", IIC_IntGeneral,
                   [(set i32:$rA, (PPCcmpb i32:$rS, i32:$rB))]>;
}
let isCompare = 1, hasSideEffects = 0 in {
  def CMPW   : XForm_16_ext<31, 0, (outs crrc:$crD), (ins gprc:$rA, gprc:$rB),
                            "cmpw $crD, $rA, $rB", IIC_IntCompare>;
  def CMPLW  : XForm_16_ext<31, 32, (outs crrc:$crD), (ins gprc:$rA, gprc:$rB),
                            "cmplw $crD, $rA, $rB", IIC_IntCompare>;
}
}



let Predicates = [IsISA3_0] in {
def MODSW : XForm_8<31, 779, (outs gprc:$rT), (ins gprc:$rA, gprc:$rB),
                        "modsw $rT, $rA, $rB", IIC_IntDivW,
                        [(set i32:$rT, (srem i32:$rA, i32:$rB))]>;
def MODUW : XForm_8<31, 267, (outs gprc:$rT), (ins gprc:$rA, gprc:$rB),
                        "moduw $rT, $rA, $rB", IIC_IntDivW,
                        [(set i32:$rT, (urem i32:$rA, i32:$rB))]>;
}

let PPC970_Unit = 1, hasSideEffects = 0 in {  // FXU Operations.
// XO-Form instructions.  Arithmetic instructions that can set overflow bit
let isCommutable = 1 in
defm ADD4  : XOForm_1r<31, 266, 0, (outs gprc:$rT), (ins gprc:$rA, gprc:$rB),
                       "add", "$rT, $rA, $rB", IIC_IntSimple,
                       [(set i32:$rT, (add i32:$rA, i32:$rB))]>;
let isCodeGenOnly = 1 in
def ADD4TLS  : XOForm_1<31, 266, 0, (outs gprc:$rT), (ins gprc:$rA, tlsreg32:$rB),
                       "add $rT, $rA, $rB", IIC_IntSimple,
                       [(set i32:$rT, (add i32:$rA, tglobaltlsaddr:$rB))]>;
let isCommutable = 1 in
defm ADDC  : XOForm_1rc<31, 10, 0, (outs gprc:$rT), (ins gprc:$rA, gprc:$rB),
                        "addc", "$rT, $rA, $rB", IIC_IntGeneral,
                        [(set i32:$rT, (addc i32:$rA, i32:$rB))]>,
                        PPC970_DGroup_Cracked;

defm DIVW  : XOForm_1rcr<31, 491, 0, (outs gprc:$rT), (ins gprc:$rA, gprc:$rB),
                          "divw", "$rT, $rA, $rB", IIC_IntDivW,
                          [(set i32:$rT, (sdiv i32:$rA, i32:$rB))]>;
defm DIVWU : XOForm_1rcr<31, 459, 0, (outs gprc:$rT), (ins gprc:$rA, gprc:$rB),
                          "divwu", "$rT, $rA, $rB", IIC_IntDivW,
                          [(set i32:$rT, (udiv i32:$rA, i32:$rB))]>;
def DIVWE : XOForm_1<31, 427, 0, (outs gprc:$rT), (ins gprc:$rA, gprc:$rB),
                     "divwe $rT, $rA, $rB", IIC_IntDivW,
                     [(set i32:$rT, (int_ppc_divwe gprc:$rA, gprc:$rB))]>,
                     Requires<[HasExtDiv]>;
let Defs = [CR0] in
def DIVWEo : XOForm_1<31, 427, 0, (outs gprc:$rT), (ins gprc:$rA, gprc:$rB),
                      "divwe. $rT, $rA, $rB", IIC_IntDivW,
                      []>, isDOT, PPC970_DGroup_Cracked, PPC970_DGroup_First,
                      Requires<[HasExtDiv]>;
def DIVWEU : XOForm_1<31, 395, 0, (outs gprc:$rT), (ins gprc:$rA, gprc:$rB),
                      "divweu $rT, $rA, $rB", IIC_IntDivW,
                      [(set i32:$rT, (int_ppc_divweu gprc:$rA, gprc:$rB))]>,
                      Requires<[HasExtDiv]>;
let Defs = [CR0] in
def DIVWEUo : XOForm_1<31, 395, 0, (outs gprc:$rT), (ins gprc:$rA, gprc:$rB),
                       "divweu. $rT, $rA, $rB", IIC_IntDivW,
                       []>, isDOT, PPC970_DGroup_Cracked, PPC970_DGroup_First,
                       Requires<[HasExtDiv]>;
let isCommutable = 1 in {
defm MULHW : XOForm_1r<31, 75, 0, (outs gprc:$rT), (ins gprc:$rA, gprc:$rB),
                       "mulhw", "$rT, $rA, $rB", IIC_IntMulHW,
                       [(set i32:$rT, (mulhs i32:$rA, i32:$rB))]>;
defm MULHWU : XOForm_1r<31, 11, 0, (outs gprc:$rT), (ins gprc:$rA, gprc:$rB),
                       "mulhwu", "$rT, $rA, $rB", IIC_IntMulHWU,
                       [(set i32:$rT, (mulhu i32:$rA, i32:$rB))]>;
defm MULLW : XOForm_1r<31, 235, 0, (outs gprc:$rT), (ins gprc:$rA, gprc:$rB),
                       "mullw", "$rT, $rA, $rB", IIC_IntMulHW,
                       [(set i32:$rT, (mul i32:$rA, i32:$rB))]>;
} // isCommutable
defm SUBF  : XOForm_1r<31, 40, 0, (outs gprc:$rT), (ins gprc:$rA, gprc:$rB),
                       "subf", "$rT, $rA, $rB", IIC_IntGeneral,
                       [(set i32:$rT, (sub i32:$rB, i32:$rA))]>;
defm SUBFC : XOForm_1rc<31, 8, 0, (outs gprc:$rT), (ins gprc:$rA, gprc:$rB),
                        "subfc", "$rT, $rA, $rB", IIC_IntGeneral,
                        [(set i32:$rT, (subc i32:$rB, i32:$rA))]>,
                        PPC970_DGroup_Cracked;
defm NEG    : XOForm_3r<31, 104, 0, (outs gprc:$rT), (ins gprc:$rA),
                        "neg", "$rT, $rA", IIC_IntSimple,
                        [(set i32:$rT, (ineg i32:$rA))]>;
let Uses = [CARRY] in {
let isCommutable = 1 in
defm ADDE  : XOForm_1rc<31, 138, 0, (outs gprc:$rT), (ins gprc:$rA, gprc:$rB),
                        "adde", "$rT, $rA, $rB", IIC_IntGeneral,
                        [(set i32:$rT, (adde i32:$rA, i32:$rB))]>;
defm ADDME  : XOForm_3rc<31, 234, 0, (outs gprc:$rT), (ins gprc:$rA),
                         "addme", "$rT, $rA", IIC_IntGeneral,
                         [(set i32:$rT, (adde i32:$rA, -1))]>;
defm ADDZE  : XOForm_3rc<31, 202, 0, (outs gprc:$rT), (ins gprc:$rA),
                         "addze", "$rT, $rA", IIC_IntGeneral,
                         [(set i32:$rT, (adde i32:$rA, 0))]>;
defm SUBFE : XOForm_1rc<31, 136, 0, (outs gprc:$rT), (ins gprc:$rA, gprc:$rB),
                        "subfe", "$rT, $rA, $rB", IIC_IntGeneral,
                        [(set i32:$rT, (sube i32:$rB, i32:$rA))]>;
defm SUBFME : XOForm_3rc<31, 232, 0, (outs gprc:$rT), (ins gprc:$rA),
                         "subfme", "$rT, $rA", IIC_IntGeneral,
                         [(set i32:$rT, (sube -1, i32:$rA))]>;
defm SUBFZE : XOForm_3rc<31, 200, 0, (outs gprc:$rT), (ins gprc:$rA),
                         "subfze", "$rT, $rA", IIC_IntGeneral,
                         [(set i32:$rT, (sube 0, i32:$rA))]>;
}
}


let PPC970_Unit = 1 in {  // FXU Operations.
// M-Form instructions.  rotate and mask instructions.
//
let isCommutable = 1 in {
// RLWIMI can be commuted if the rotate amount is zero.
defm RLWIMI : MForm_2r<20, (outs gprc:$rA),
                       (ins gprc:$rSi, gprc:$rS, u5imm:$SH, u5imm:$MB,
                       u5imm:$ME), "rlwimi", "$rA, $rS, $SH, $MB, $ME",
                       IIC_IntRotate, []>, PPC970_DGroup_Cracked,
                       RegConstraint<"$rSi = $rA">, NoEncode<"$rSi">;
}
let BaseName = "rlwinm" in {
def RLWINM : MForm_2<21,
                     (outs gprc:$rA), (ins gprc:$rS, u5imm:$SH, u5imm:$MB, u5imm:$ME),
                     "rlwinm $rA, $rS, $SH, $MB, $ME", IIC_IntGeneral,
                     []>, RecFormRel;
let Defs = [CR0] in
def RLWINMo : MForm_2<21,
                      (outs gprc:$rA), (ins gprc:$rS, u5imm:$SH, u5imm:$MB, u5imm:$ME),
                      "rlwinm. $rA, $rS, $SH, $MB, $ME", IIC_IntGeneral,
                      []>, isDOT, RecFormRel, PPC970_DGroup_Cracked;
}
defm RLWNM  : MForm_2r<23, (outs gprc:$rA),
                       (ins gprc:$rS, gprc:$rB, u5imm:$MB, u5imm:$ME),
                       "rlwnm", "$rA, $rS, $rB, $MB, $ME", IIC_IntGeneral,
                       []>;
}
