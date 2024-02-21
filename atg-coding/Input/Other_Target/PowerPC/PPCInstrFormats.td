//===- PowerPCInstrFormats.td - PowerPC Instruction Formats --*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Classes used for relation maps.
//===----------------------------------------------------------------------===//
// RecFormRel - Filter class used to relate non-record-form instructions with
// their record-form variants.
class RecFormRel;

// AltVSXFMARel - Filter class used to relate the primary addend-killing VSX
// FMA instruction forms with their corresponding factor-killing forms.
class AltVSXFMARel {
  bit IsVSXFMAAlt = 0;
}
//===----------------------------------------------------------------------===//
//
// PowerPC instruction formats

class I<bits<6> opcode, dag OOL, dag IOL, string asmstr, InstrItinClass itin>
        : Instruction {
  field bits<32> Inst;
  field bits<32> SoftFail = 0;
  let Size = 4;

  bit PPC64 = 0;  // Default value, override with isPPC64

  let Namespace = "PPC";
  let Inst{0-5} = opcode;
  let OutOperandList = OOL;
  let InOperandList = IOL;
  let AsmString = asmstr;
  let Itinerary = itin;

  bits<1> PPC970_First = 0;
  bits<1> PPC970_Single = 0;
  bits<1> PPC970_Cracked = 0;
  bits<3> PPC970_Unit = 0;

  /// These fields correspond to the fields in PPCInstrInfo.h.  Any changes to
  /// these must be reflected there!  See comments there for what these are.
  let TSFlags{0}   = PPC970_First;
  let TSFlags{1}   = PPC970_Single;
  let TSFlags{2}   = PPC970_Cracked;
  let TSFlags{5-3} = PPC970_Unit;

  // Indicate that this instruction is of type X-Form Load or Store
  bits<1> XFormMemOp = 0;
  let TSFlags{7}  = XFormMemOp;

  // Fields used for relation models.
  string BaseName = "";

  // For cases where multiple instruction definitions really represent the
  // same underlying instruction but with one definition for 64-bit arguments
  // and one for 32-bit arguments, this bit breaks the degeneracy between
  // the two forms and allows TableGen to generate mapping tables.
  bit Interpretation64Bit = 0;
}

class PPC970_DGroup_First   { bits<1> PPC970_First = 1;  }
class PPC970_DGroup_Single  { bits<1> PPC970_Single = 1; }
class PPC970_DGroup_Cracked { bits<1> PPC970_Cracked = 1; }
class PPC970_MicroCode;

class PPC970_Unit_FPU      { bits<3> PPC970_Unit = 3;   }
class PPC970_Unit_CRU      { bits<3> PPC970_Unit = 4;   }

// Two joined instructions; used to emit two adjacent instructions as one.
// The itinerary from the first instruction is used for scheduling and
// classification.
class I2<bits<6> opcode1, bits<6> opcode2, dag OOL, dag IOL, string asmstr,
         InstrItinClass itin>
        : Instruction {
  field bits<64> Inst;
  field bits<64> SoftFail = 0;
  let Size = 8;

  bit PPC64 = 0;  // Default value, override with isPPC64

  let Namespace = "PPC";
  let Inst{0-5} = opcode1;
  let Inst{32-37} = opcode2;
  let OutOperandList = OOL;
  let InOperandList = IOL;
  let AsmString = asmstr;
  let Itinerary = itin;

  bits<1> PPC970_First = 0;
  bits<1> PPC970_Single = 0;
  bits<1> PPC970_Cracked = 0;
  bits<3> PPC970_Unit = 0;

  /// These fields correspond to the fields in PPCInstrInfo.h.  Any changes to
  /// these must be reflected there!  See comments there for what these are.
  let TSFlags{0}   = PPC970_First;
  let TSFlags{1}   = PPC970_Single;
  let TSFlags{2}   = PPC970_Cracked;
  let TSFlags{5-3} = PPC970_Unit;

  // Fields used for relation models.
  string BaseName = "";
  bit Interpretation64Bit = 0;
}

// Base class for all X-Form memory instructions
class IXFormMemOp<bits<6> opcode, dag OOL, dag IOL, string asmstr,
                  InstrItinClass itin>
        :I<opcode, OOL, IOL, asmstr, itin>;

// 1.7.1 I-Form
class IForm<bits<6> opcode, bit aa, bit lk, dag OOL, dag IOL, string asmstr,
            InstrItinClass itin, list<dag> pattern>
         : I<opcode, OOL, IOL, asmstr, itin> {
  let Pattern = pattern;
  bits<24> LI;

  let Inst{6-29}  = LI;
  let Inst{30}    = aa;
  let Inst{31}    = lk;
}

// 1.7.2 B-Form
class BForm<bits<6> opcode, bit aa, bit lk, dag OOL, dag IOL, string asmstr>
  : I<opcode, OOL, IOL, asmstr, IIC_BrB> {
  bits<7> BIBO;  // 2 bits of BI and 5 bits of BO.
  bits<3>  CR;
  bits<14> BD;

  bits<5> BI;
  //let BI{0-1} = BIBO{5-6};
  //let BI{2-4} = CR{0-2};

  let Inst{6-10}  = BIBO{4-0};
  let Inst{11-15} = BI;
  let Inst{16-29} = BD;
  let Inst{30}    = aa;
  let Inst{31}    = lk;
}

class BForm_1<bits<6> opcode, bits<5> bo, bit aa, bit lk, dag OOL, dag IOL,
             string asmstr>
  : BForm<opcode, aa, lk, OOL, IOL, asmstr> {
  //let BIBO{4-0} = bo;
//let BIBO{6-5} = 0;
  let CR = 0;
}

class BForm_2<bits<6> opcode, bits<5> bo, bits<5> bi, bit aa, bit lk,
              dag OOL, dag IOL, string asmstr>
  : I<opcode, OOL, IOL, asmstr, IIC_BrB> {
  bits<14> BD;

  let Inst{6-10}  = bo;
  let Inst{11-15} = bi;
  let Inst{16-29} = BD;
  let Inst{30}    = aa;
  let Inst{31}    = lk;
}

class BForm_4<bits<6> opcode, bits<5> bo, bit aa, bit lk,
              dag OOL, dag IOL, string asmstr>
  : I<opcode, OOL, IOL, asmstr, IIC_BrB> {
  bits<5> BI;
  bits<14> BD;

  let Inst{6-10}  = bo;
  let Inst{11-15} = BI;
  let Inst{16-29} = BD;
  let Inst{30}    = aa;
  let Inst{31}    = lk;
}

// 1.7.3 SC-Form
class SCForm<bits<6> opcode, bits<1> xo,
                     dag OOL, dag IOL, string asmstr, InstrItinClass itin,
                     list<dag> pattern>
  : I<opcode, OOL, IOL, asmstr, itin> {
  bits<7>  LEV;

  let Pattern = pattern;

  let Inst{20-26} = LEV;
  let Inst{30}    = xo;
}

// 1.7.4 D-Form
class DForm_base<bits<6> opcode, dag OOL, dag IOL, string asmstr,
                 InstrItinClass itin, list<dag> pattern> 
  : I<opcode, OOL, IOL, asmstr, itin> {
  bits<5>  A;
  bits<5>  B;
  bits<16> C;

  let Pattern = pattern;
  
  let Inst{6-10}  = A;
  let Inst{11-15} = B;
  let Inst{16-31} = C;
}

class DForm_1<bits<6> opcode, dag OOL, dag IOL, string asmstr,
              InstrItinClass itin, list<dag> pattern>
  : I<opcode, OOL, IOL, asmstr, itin> {
  bits<5>  A;
  bits<21> Addr;

  let Pattern = pattern;
  
  let Inst{6-10}  = A;
  let Inst{11-15} = Addr{20-16}; // Base Reg
  let Inst{16-31} = Addr{15-0};  // Displacement
}

class DForm_2<bits<6> opcode, dag OOL, dag IOL, string asmstr,
              InstrItinClass itin, list<dag> pattern>
  : DForm_base<opcode, OOL, IOL, asmstr, itin, pattern> {

  // Even though ADDICo does not really have an RC bit, provide
  // the declaration of one here so that isDOT has something to set.
  bit RC = 0;
}

class DForm_2_r0<bits<6> opcode, dag OOL, dag IOL, string asmstr,
                 InstrItinClass itin, list<dag> pattern>
  : I<opcode, OOL, IOL, asmstr, itin> {
  bits<5>  A;
  bits<16> B;
  
  let Pattern = pattern;
  
  let Inst{6-10}  = A;
  let Inst{11-15} = 0;
  let Inst{16-31} = B;
}

class DForm_4<bits<6> opcode, dag OOL, dag IOL, string asmstr,
              InstrItinClass itin, list<dag> pattern>
  : I<opcode, OOL, IOL, asmstr, itin> {
  bits<5>  B;
  bits<5>  A;
  bits<16> C;
  
  let Pattern = pattern;
  
  let Inst{6-10}  = A;
  let Inst{11-15} = B;
  let Inst{16-31} = C;
}
              
class DForm_4_zero<bits<6> opcode, dag OOL, dag IOL, string asmstr,
                   InstrItinClass itin, list<dag> pattern>
  : DForm_1<opcode, OOL, IOL, asmstr, itin, pattern> {
  let A = 0;
  let Addr = 0;
}

class IForm_and_DForm_1<bits<6> opcode1, bit aa, bit lk, bits<6> opcode2,
            dag OOL, dag IOL, string asmstr,
            InstrItinClass itin, list<dag> pattern>
         : I2<opcode1, opcode2, OOL, IOL, asmstr, itin> {
  bits<5>  A;
  bits<21> Addr;

  let Pattern = pattern;
  bits<24> LI;

  let Inst{6-29}  = LI;
  let Inst{30}    = aa;
  let Inst{31}    = lk;

  let Inst{38-42}  = A;
  let Inst{43-47} = Addr{20-16}; // Base Reg
  let Inst{48-63} = Addr{15-0};  // Displacement
}

class DForm_4_fixedreg_zero<bits<6> opcode, bits<5> R, dag OOL, dag IOL,
                            string asmstr, InstrItinClass itin,
                            list<dag> pattern>
  : DForm_4<opcode, OOL, IOL, asmstr, itin, pattern> {
  let A = R;
  let B = R;
  let C = 0; 
}

// This is used to emit BL8+NOP.
class IForm_and_DForm_4_zero<bits<6> opcode1, bit aa, bit lk, bits<6> opcode2,
            dag OOL, dag IOL, string asmstr,
            InstrItinClass itin, list<dag> pattern>
         :  IForm_and_DForm_1<opcode1, aa, lk, opcode2,
                              OOL, IOL, asmstr, itin, pattern> {
  let A = 0;
  let Addr = 0;
}


class DForm_5<bits<6> opcode, dag OOL, dag IOL, string asmstr,
              InstrItinClass itin>
  : I<opcode, OOL, IOL, asmstr, itin> {
  bits<3>  BF;
  bits<1>  L;
  bits<5>  RA;
  bits<16> I;

  let Inst{6-8}   = BF;
  let Inst{9}     = 0;
  let Inst{10}    = L;
  let Inst{11-15} = RA;
  let Inst{16-31} = I;
}

class DForm_5_ext<bits<6> opcode, dag OOL, dag IOL, string asmstr,
                  InstrItinClass itin>
  : DForm_5<opcode, OOL, IOL, asmstr, itin> {
  let L = PPC64;
}

class DForm_6<bits<6> opcode, dag OOL, dag IOL, string asmstr,
              InstrItinClass itin> 
  : DForm_5<opcode, OOL, IOL, asmstr, itin>;

class DForm_6_ext<bits<6> opcode, dag OOL, dag IOL, string asmstr,
                  InstrItinClass itin>
  : DForm_6<opcode, OOL, IOL, asmstr, itin> {
  let L = PPC64;
}


// 1.7.6 X-Form
class XForm_base_r3xo<bits<6> opcode, bits<10> xo, dag OOL, dag IOL, string asmstr, 
                      InstrItinClass itin, list<dag> pattern>
  : I<opcode, OOL, IOL, asmstr, itin> {
  bits<5> RST;
  bits<5> A;
  bits<5> B;

  let Pattern = pattern;

  bit RC = 0;    // set by isDOT

  let Inst{6-10}  = RST;
  let Inst{11-15} = A;
  let Inst{16-20} = B;
  let Inst{21-30} = xo;
  let Inst{31}    = RC;
}

class XForm_base_r3xo_memOp<bits<6> opcode, bits<10> xo, dag OOL, dag IOL,
                            string asmstr, InstrItinClass itin,
                            list<dag> pattern>
  : XForm_base_r3xo<opcode, xo, OOL, IOL, asmstr, itin, pattern>;

class XForm_tlb<bits<10> xo, dag OOL, dag IOL, string asmstr,
                InstrItinClass itin> : XForm_base_r3xo<31, xo, OOL, IOL, asmstr, itin, []> {
  let RST = 0;
}

class XForm_attn<bits<6> opcode, bits<10> xo, dag OOL, dag IOL, string asmstr,
                 InstrItinClass itin>
  : I<opcode, OOL, IOL, asmstr, itin> {
  let Inst{21-30} = xo;
}

// This is the same as XForm_base_r3xo, but the first two operands are swapped
// when code is emitted.
class XForm_base_r3xo_swapped
        <bits<6> opcode, bits<10> xo, dag OOL, dag IOL, string asmstr,
        InstrItinClass itin> 
  : I<opcode, OOL, IOL, asmstr, itin> {
  bits<5> A;
  bits<5> RST;
  bits<5> B;

  bit RC = 0;    // set by isDOT

  let Inst{6-10}  = RST;
  let Inst{11-15} = A;
  let Inst{16-20} = B;
  let Inst{21-30} = xo;
  let Inst{31}    = RC;
}


class XForm_1<bits<6> opcode, bits<10> xo, dag OOL, dag IOL, string asmstr,
              InstrItinClass itin, list<dag> pattern>
  : XForm_base_r3xo<opcode, xo, OOL, IOL, asmstr, itin, pattern>;

class XForm_1_memOp<bits<6> opcode, bits<10> xo, dag OOL, dag IOL, string asmstr,
              InstrItinClass itin, list<dag> pattern>
  : XForm_base_r3xo_memOp<opcode, xo, OOL, IOL, asmstr, itin, pattern>;

class XForm_6<bits<6> opcode, bits<10> xo, dag OOL, dag IOL, string asmstr,
              InstrItinClass itin, list<dag> pattern> 
  : XForm_base_r3xo_swapped<opcode, xo, OOL, IOL, asmstr, itin> {
  let Pattern = pattern;
}

class XForm_8<bits<6> opcode, bits<10> xo, dag OOL, dag IOL, string asmstr,
              InstrItinClass itin, list<dag> pattern> 
  : XForm_base_r3xo<opcode, xo, OOL, IOL, asmstr, itin, pattern>;

class XForm_8_memOp<bits<6> opcode, bits<10> xo, dag OOL, dag IOL, string asmstr,
              InstrItinClass itin, list<dag> pattern> 
  : XForm_base_r3xo_memOp<opcode, xo, OOL, IOL, asmstr, itin, pattern>;

class XForm_10<bits<6> opcode, bits<10> xo, dag OOL, dag IOL, string asmstr,
               InstrItinClass itin, list<dag> pattern> 
  : XForm_base_r3xo_swapped<opcode, xo, OOL, IOL, asmstr, itin> {
    let Pattern = pattern;
}

class XForm_11<bits<6> opcode, bits<10> xo, dag OOL, dag IOL, string asmstr,
               InstrItinClass itin, list<dag> pattern> 
  : XForm_base_r3xo_swapped<opcode, xo, OOL, IOL, asmstr, itin> {
  let B = 0;
  let Pattern = pattern;
}

class XForm_16<bits<6> opcode, bits<10> xo, dag OOL, dag IOL, string asmstr,
               InstrItinClass itin>
         : I<opcode, OOL, IOL, asmstr, itin> {
  bits<3> BF;
  bits<1> L; 
  bits<5> RA;
  bits<5> RB;
  
  let Inst{6-8}   = BF;
  let Inst{9}     = 0;
  let Inst{10}    = L;
  let Inst{11-15} = RA;
  let Inst{16-20} = RB;
  let Inst{21-30} = xo;
  let Inst{31}    = 0;
}

class XForm_icbt<bits<6> opcode, bits<10> xo, dag OOL, dag IOL, string asmstr,
                 InstrItinClass itin>
         : I<opcode, OOL, IOL, asmstr, itin> {
  bits<4> CT;
  bits<5> RA;
  bits<5> RB;

  let Inst{6} = 0;
  let Inst{7-10} = CT;
  let Inst{11-15} = RA;
  let Inst{16-20} = RB;
  let Inst{21-30} = xo;
  let Inst{31} = 0;
}


class XForm_16_ext<bits<6> opcode, bits<10> xo, dag OOL, dag IOL, string asmstr,
                   InstrItinClass itin>
  : XForm_16<opcode, xo, OOL, IOL, asmstr, itin> {
  let L = PPC64;
}

class XForm_17<bits<6> opcode, bits<10> xo, dag OOL, dag IOL, string asmstr,
               InstrItinClass itin>
         : I<opcode, OOL, IOL, asmstr, itin> {
  bits<3> BF;
  bits<5> FRA;
  bits<5> FRB;
  
  let Inst{6-8}   = BF;
  let Inst{9-10}  = 0;
  let Inst{11-15} = FRA;
  let Inst{16-20} = FRB;
  let Inst{21-30} = xo;
  let Inst{31}    = 0;
}

class XForm_24<bits<6> opcode, bits<10> xo, dag OOL, dag IOL, string asmstr,
               InstrItinClass itin, list<dag> pattern> 
  : I<opcode, OOL, IOL, asmstr, itin> {
  let Pattern = pattern;
  let Inst{6-10}  = 31;
  let Inst{11-15} = 0;
  let Inst{16-20} = 0;
  let Inst{21-30} = xo;
  let Inst{31}    = 0;
}

class XForm_24_sync<bits<6> opcode, bits<10> xo, dag OOL, dag IOL,
               string asmstr, InstrItinClass itin, list<dag> pattern> 
  : I<opcode, OOL, IOL, asmstr, itin> {
  bits<2> L;

  let Pattern = pattern;
  let Inst{6-8}   = 0;
  let Inst{9-10}  = L;
  let Inst{11-15} = 0;
  let Inst{16-20} = 0;
  let Inst{21-30} = xo;
  let Inst{31}    = 0;
}


class XForm_25_memOp<bits<6> opcode, bits<10> xo, dag OOL, dag IOL,
                    string asmstr, InstrItinClass itin, list<dag> pattern>
  : XForm_base_r3xo_memOp<opcode, xo, OOL, IOL, asmstr, itin, pattern> {
}

// [PO RT /// RB XO RC]
class XForm_26<bits<6> opcode, bits<10> xo, dag OOL, dag IOL, string asmstr,
               InstrItinClass itin, list<dag> pattern>
  : XForm_base_r3xo<opcode, xo, OOL, IOL, asmstr, itin, pattern> {
  let A = 0;
}

class XForm_28_memOp<bits<6> opcode, bits<10> xo, dag OOL, dag IOL,
                    string asmstr, InstrItinClass itin, list<dag> pattern>
  : XForm_base_r3xo_memOp<opcode, xo, OOL, IOL, asmstr, itin, pattern> {
}

class XForm_28<bits<6> opcode, bits<10> xo, dag OOL, dag IOL, string asmstr,
               InstrItinClass itin, list<dag> pattern>
  : XForm_base_r3xo<opcode, xo, OOL, IOL, asmstr, itin, pattern> {
}

// This is used for MFFS, MTFSB0, MTFSB1.  42 is arbitrary; this series of
// numbers presumably relates to some document, but I haven't found it.
class XForm_42<bits<6> opcode, bits<10> xo, dag OOL, dag IOL, string asmstr,
              InstrItinClass itin, list<dag> pattern>
  : XForm_base_r3xo<opcode, xo, OOL, IOL, asmstr, itin, pattern> {
  let Pattern = pattern;

  bit RC = 0;    // set by isDOT

  let Inst{6-10}  = RST;
  let Inst{11-20} = 0;
  let Inst{21-30} = xo;
  let Inst{31}    = RC;
}
class XForm_43<bits<6> opcode, bits<10> xo, dag OOL, dag IOL, string asmstr,
              InstrItinClass itin, list<dag> pattern>
  : XForm_base_r3xo<opcode, xo, OOL, IOL, asmstr, itin, pattern> {
  let Pattern = pattern;
  bits<5> FM;

  bit RC = 0;    // set by isDOT

  let Inst{6-10}  = FM;
  let Inst{11-20} = 0;
  let Inst{21-30} = xo;
  let Inst{31}    = RC;
}

class X_FRT5_XO2_XO3_XO10<bits<6> opcode, bits<2> xo1, bits<3> xo2, bits<10> xo,
                         dag OOL, dag IOL, string asmstr, InstrItinClass itin,
                         list<dag> pattern>
  : XForm_base_r3xo<opcode, xo, OOL, IOL, asmstr, itin, pattern> {
  let Pattern = pattern;

  let Inst{6-10}  = RST;
  let Inst{11-12} = xo1;
  let Inst{13-15} = xo2;
  let Inst{16-20} = 0;
  let Inst{21-30} = xo;
  let Inst{31}    = 0;
}

class X_FRT5_XO2_XO3_FRB5_XO10<bits<6> opcode, bits<2> xo1, bits<3> xo2,
                              bits<10> xo, dag OOL, dag IOL, string asmstr,
                              InstrItinClass itin, list<dag> pattern>
  : XForm_base_r3xo<opcode, xo, OOL, IOL, asmstr, itin, pattern> {
  let Pattern = pattern;
  bits<5> FRB;

  let Inst{6-10}  = RST;
  let Inst{11-12} = xo1;
  let Inst{13-15} = xo2;
  let Inst{16-20} = FRB;
  let Inst{21-30} = xo;
  let Inst{31}    = 0;
}

class X_FRT5_XO2_XO3_DRM3_XO10<bits<6> opcode, bits<2> xo1, bits<3> xo2,
                              bits<10> xo, dag OOL, dag IOL, string asmstr,
                              InstrItinClass itin, list<dag> pattern>
  : XForm_base_r3xo<opcode, xo, OOL, IOL, asmstr, itin, pattern> {
  let Pattern = pattern;
  bits<3> DRM;

  let Inst{6-10}  = RST;
  let Inst{11-12} = xo1;
  let Inst{13-15} = xo2;
  let Inst{16-17} = 0;
  let Inst{18-20} = DRM;
  let Inst{21-30} = xo;
  let Inst{31}    = 0;
}

class X_FRT5_XO2_XO3_RM2_X10<bits<6> opcode, bits<2> xo1, bits<3> xo2,
                            bits<10> xo, dag OOL, dag IOL, string asmstr,
                            InstrItinClass itin, list<dag> pattern>
  : XForm_base_r3xo<opcode, xo, OOL, IOL, asmstr, itin, pattern> {
  let Pattern = pattern;
  bits<2> RM;

  let Inst{6-10}  = RST;
  let Inst{11-12} = xo1;
  let Inst{13-15} = xo2;
  let Inst{16-18} = 0;
  let Inst{19-20} = RM;
  let Inst{21-30} = xo;
  let Inst{31}    = 0;
}

class XForm_0<bits<6> opcode, bits<10> xo, dag OOL, dag IOL, string asmstr,
              InstrItinClass itin, list<dag> pattern>
  : XForm_base_r3xo<opcode, xo, OOL, IOL, asmstr, itin, pattern> {
  let RST = 0;
  let A = 0;
  let B = 0;
}

// [PO RT RA RB XO /]
class X_BF3_L1_RS5_RS5<bits<6> opcode, bits<10> xo, dag OOL, dag IOL,
                       string asmstr, InstrItinClass itin, list<dag> pattern>
  : I<opcode, OOL, IOL, asmstr, itin> {
  bits<3> BF;
  bits<1> L;
  bits<5> RA;
  bits<5> RB;

  let Pattern = pattern;

  let Inst{6-8}   = BF;
  let Inst{9}     = 0;
  let Inst{10}    = L;
  let Inst{11-15} = RA;
  let Inst{16-20} = RB;
  let Inst{21-30} = xo;
  let Inst{31}    = 0;
}


// XForm_base_r3xo for instructions such as P9 atomics where we don't want
// to specify an SDAG pattern for matching.
class X_RD5_RS5_IM5<bits<6> opcode, bits<10> xo, dag OOL, dag IOL,
                    string asmstr, InstrItinClass itin>
  : XForm_base_r3xo_memOp<opcode, xo, OOL, IOL, asmstr, itin, []> {
}

class X_BF3<bits<6> opcode, bits<10> xo, dag OOL, dag IOL, string asmstr,
            InstrItinClass itin>
  : XForm_17<opcode, xo, OOL, IOL, asmstr, itin> {
  let FRA = 0;
  let FRB = 0;
}


// 1.7.7 XL-Form
class XLForm_1<bits<6> opcode, bits<10> xo, dag OOL, dag IOL, string asmstr,
               InstrItinClass itin, list<dag> pattern>
    : I<opcode, OOL, IOL, asmstr, itin> {
  bits<5> CRD;
  bits<5> CRA;
  bits<5> CRB;
  
  let Pattern = pattern;
  
  let Inst{6-10}  = CRD;
  let Inst{11-15} = CRA;
  let Inst{16-20} = CRB;
  let Inst{21-30} = xo;
  let Inst{31}    = 0;
}

class XLForm_1_np<bits<6> opcode, bits<10> xo, dag OOL, dag IOL, string asmstr,
                  InstrItinClass itin, list<dag> pattern>
  : XLForm_1<opcode, xo, OOL, IOL, asmstr, itin, pattern> {
  let CRD = 0;
  let CRA = 0;
  let CRB = 0;
}

class XLForm_1_gen<bits<6> opcode, bits<10> xo, dag OOL, dag IOL, string asmstr,
                   InstrItinClass itin, list<dag> pattern>
  : XLForm_1<opcode, xo, OOL, IOL, asmstr, itin, pattern> {
  bits<5> RT;
  bits<5> RB;

  let CRD = RT;
  let CRA = 0;
  let CRB = RB;
}

class XLForm_1_ext<bits<6> opcode, bits<10> xo, dag OOL, dag IOL, string asmstr,
               InstrItinClass itin, list<dag> pattern>
    : I<opcode, OOL, IOL, asmstr, itin> {
  bits<5> CRD;
  
  let Pattern = pattern;
  
  let Inst{6-10}  = CRD;
  let Inst{11-15} = CRD;
  let Inst{16-20} = CRD;
  let Inst{21-30} = xo;
  let Inst{31}    = 0;
}

class XLForm_2<bits<6> opcode, bits<10> xo, bit lk, dag OOL, dag IOL, string asmstr, 
               InstrItinClass itin, list<dag> pattern>
    : I<opcode, OOL, IOL, asmstr, itin> {
  bits<5> BO;
  bits<5> BI;
  bits<2> BH;
  
  let Pattern = pattern;
  
  let Inst{6-10}  = BO;
  let Inst{11-15} = BI;
  let Inst{16-18} = 0;
  let Inst{19-20} = BH;
  let Inst{21-30} = xo;
  let Inst{31}    = lk;
}

class XLForm_2_br<bits<6> opcode, bits<10> xo, bit lk,
                  dag OOL, dag IOL, string asmstr, InstrItinClass itin, list<dag> pattern>
  : XLForm_2<opcode, xo, lk, OOL, IOL, asmstr, itin, pattern> {
  bits<7> BIBO;  // 2 bits of BI and 5 bits of BO.
  bits<3>  CR;
  
  let BO = BIBO{4-0};
  //let BI{0-1} = BIBO{5-6};
  //let BI{2-4} = CR{0-2};
  let BH = 0;
}

class XLForm_2_br2<bits<6> opcode, bits<10> xo, bits<5> bo, bit lk,
                   dag OOL, dag IOL, string asmstr, InstrItinClass itin, list<dag> pattern>
  : XLForm_2<opcode, xo, lk, OOL, IOL, asmstr, itin, pattern> {
  let BO = bo;
  let BH = 0;
}

class XLForm_2_ext<bits<6> opcode, bits<10> xo, bits<5> bo,  bits<5> bi, bit lk,
                  dag OOL, dag IOL, string asmstr, InstrItinClass itin, list<dag> pattern>
  : XLForm_2<opcode, xo, lk, OOL, IOL, asmstr, itin, pattern> {
  let BO = bo;
  let BI = bi;
  let BH = 0;
}

class XLForm_3<bits<6> opcode, bits<10> xo, dag OOL, dag IOL, string asmstr,
               InstrItinClass itin>
         : I<opcode, OOL, IOL, asmstr, itin> {
  bits<3> BF;
  bits<3> BFA;
  
  let Inst{6-8}   = BF;
  let Inst{9-10}  = 0;
  let Inst{11-13} = BFA;
  let Inst{14-15} = 0;
  let Inst{16-20} = 0;
  let Inst{21-30} = xo;
  let Inst{31}    = 0;
}

class XLForm_4<bits<6> opcode, bits<10> xo, dag OOL, dag IOL, string asmstr,
               InstrItinClass itin>
         : I<opcode, OOL, IOL, asmstr, itin> {
  bits<3> BF;
  bit W;
  bits<4> U;
  
  bit RC = 0;
  
  let Inst{6-8}   = BF;
  let Inst{9-10}  = 0;
  let Inst{11-14} = 0;
  let Inst{15}    = W;
  let Inst{16-19} = U;
  let Inst{20}    = 0;
  let Inst{21-30} = xo;
  let Inst{31}    = RC;
}

class XLForm_S<bits<6> opcode, bits<10> xo, dag OOL, dag IOL, string asmstr,
               InstrItinClass itin, list<dag> pattern>
    : I<opcode, OOL, IOL, asmstr, itin> {
  bits<1> S;
  
  let Pattern = pattern;
  
  let Inst{6-19}  = 0;
  let Inst{20}    = S;
  let Inst{21-30} = xo;
  let Inst{31}    = 0;
}

class XLForm_2_and_DSForm_1<bits<6> opcode1, bits<10> xo1, bit lk,
                            bits<6> opcode2, bits<2> xo2,
                            dag OOL, dag IOL, string asmstr,
                            InstrItinClass itin, list<dag> pattern>
        : I2<opcode1, opcode2, OOL, IOL, asmstr, itin> {
  bits<5> BO;
  bits<5> BI;
  bits<2> BH;

  bits<5>  RST;
  bits<19> DS_RA;

  let Pattern = pattern;

  let Inst{6-10}  = BO;
  let Inst{11-15} = BI;
  let Inst{16-18} = 0;
  let Inst{19-20} = BH;
  let Inst{21-30} = xo1;
  let Inst{31}    = lk;

  let Inst{38-42} = RST;
  let Inst{43-47} = DS_RA{18-14};  // Register #
  let Inst{48-61} = DS_RA{13-0};   // Displacement.
  let Inst{62-63} = xo2;
}

class XLForm_2_ext_and_DSForm_1<bits<6> opcode1, bits<10> xo1,
                                bits<5> bo, bits<5> bi, bit lk,
                                bits<6> opcode2, bits<2> xo2,
                                dag OOL, dag IOL, string asmstr,
                                InstrItinClass itin, list<dag> pattern>
  : XLForm_2_and_DSForm_1<opcode1, xo1, lk, opcode2, xo2,
                          OOL, IOL, asmstr, itin, pattern> {
  let BO = bo;
  let BI = bi;
  let BH = 0;
}

// 1.7.10 XS-Form - SRADI.
class XSForm_1<bits<6> opcode, bits<9> xo, dag OOL, dag IOL, string asmstr,
               InstrItinClass itin, list<dag> pattern>
         : I<opcode, OOL, IOL, asmstr, itin> {
  bits<5> A;
  bits<5> RS;
  bits<6> SH;

  bit RC = 0;    // set by isDOT
  let Pattern = pattern;

  let Inst{6-10}  = RS;
  let Inst{11-15} = A;
  let Inst{16-20} = SH{4,3,2,1,0};
  let Inst{21-29} = xo;
  let Inst{30}    = SH{5};
  let Inst{31}    = RC;
}

// 1.7.11 XO-Form
class XOForm_1<bits<6> opcode, bits<9> xo, bit oe, dag OOL, dag IOL, string asmstr,
               InstrItinClass itin, list<dag> pattern>
         : I<opcode, OOL, IOL, asmstr, itin> {
  bits<5> RT;
  bits<5> RA;
  bits<5> RB;

  let Pattern = pattern;

  bit RC = 0;    // set by isDOT

  let Inst{6-10}  = RT;
  let Inst{11-15} = RA;
  let Inst{16-20} = RB;
  let Inst{21}    = oe;
  let Inst{22-30} = xo;
  let Inst{31}    = RC;  
}

class XOForm_3<bits<6> opcode, bits<9> xo, bit oe, 
               dag OOL, dag IOL, string asmstr, InstrItinClass itin, list<dag> pattern>
  : XOForm_1<opcode, xo, oe, OOL, IOL, asmstr, itin, pattern> {
  let RB = 0;
}

// 1.7.12 A-Form
class AForm_1<bits<6> opcode, bits<5> xo, dag OOL, dag IOL, string asmstr, 
              InstrItinClass itin, list<dag> pattern>
         : I<opcode, OOL, IOL, asmstr, itin> {
  bits<5> FRT;
  bits<5> FRA;
  bits<5> FRC;
  bits<5> FRB;

  let Pattern = pattern;

  bit RC = 0;    // set by isDOT

  let Inst{6-10}  = FRT;
  let Inst{11-15} = FRA;
  let Inst{16-20} = FRB;
  let Inst{21-25} = FRC;
  let Inst{26-30} = xo;
  let Inst{31}    = RC;
}

class AForm_2<bits<6> opcode, bits<5> xo, dag OOL, dag IOL, string asmstr,
              InstrItinClass itin, list<dag> pattern>
  : AForm_1<opcode, xo, OOL, IOL, asmstr, itin, pattern> {
  let FRC = 0;
}

class AForm_3<bits<6> opcode, bits<5> xo, dag OOL, dag IOL, string asmstr,
              InstrItinClass itin, list<dag> pattern> 
  : AForm_1<opcode, xo, OOL, IOL, asmstr, itin, pattern> {
  let FRB = 0;
}

class AForm_4<bits<6> opcode, bits<5> xo, dag OOL, dag IOL, string asmstr, 
              InstrItinClass itin, list<dag> pattern>
         : I<opcode, OOL, IOL, asmstr, itin> {
  bits<5> RT;
  bits<5> RA;
  bits<5> RB;
  bits<5> COND;

  let Pattern = pattern;

  let Inst{6-10}  = RT;
  let Inst{11-15} = RA;
  let Inst{16-20} = RB;
  let Inst{21-25} = COND;
  let Inst{26-30} = xo;
  let Inst{31}    = 0;
}

// Used for QPX
class AForm_4a<bits<6> opcode, bits<5> xo, dag OOL, dag IOL, string asmstr,
              InstrItinClass itin, list<dag> pattern>
  : AForm_1<opcode, xo, OOL, IOL, asmstr, itin, pattern> {
  let FRA = 0;
  let FRC = 0;
}

// 1.7.13 M-Form
class MForm_1<bits<6> opcode, dag OOL, dag IOL, string asmstr,
              InstrItinClass itin, list<dag> pattern>
    : I<opcode, OOL, IOL, asmstr, itin> {
  bits<5> RA;
  bits<5> RS;
  bits<5> RB;
  bits<5> MB;
  bits<5> ME;

  let Pattern = pattern;

  bit RC = 0;    // set by isDOT

  let Inst{6-10}  = RS;
  let Inst{11-15} = RA;
  let Inst{16-20} = RB;
  let Inst{21-25} = MB;
  let Inst{26-30} = ME;
  let Inst{31}    = RC;
}

class MForm_2<bits<6> opcode, dag OOL, dag IOL, string asmstr,
              InstrItinClass itin, list<dag> pattern>
  : MForm_1<opcode, OOL, IOL, asmstr, itin, pattern> {
}

// 1.7.14 MD-Form
class MDForm_1<bits<6> opcode, bits<3> xo, dag OOL, dag IOL, string asmstr,
               InstrItinClass itin, list<dag> pattern>
    : I<opcode, OOL, IOL, asmstr, itin> {
  bits<5> RA;
  bits<5> RS;
  bits<6> SH;
  bits<6> MBE;

  let Pattern = pattern;

  bit RC = 0;    // set by isDOT

  let Inst{6-10}  = RS;
  let Inst{11-15} = RA;
  let Inst{16-20} = SH{4,3,2,1,0};
  let Inst{21-26} = MBE{4,3,2,1,0,5};
  let Inst{27-29} = xo;
  let Inst{30}    = SH{5};
  let Inst{31}    = RC;
}

class MDSForm_1<bits<6> opcode, bits<4> xo, dag OOL, dag IOL, string asmstr,
                InstrItinClass itin, list<dag> pattern>
    : I<opcode, OOL, IOL, asmstr, itin> {
  bits<5> RA;
  bits<5> RS;
  bits<5> RB;
  bits<6> MBE;

  let Pattern = pattern;

  bit RC = 0;    // set by isDOT

  let Inst{6-10}  = RS;
  let Inst{11-15} = RA;
  let Inst{16-20} = RB;
  let Inst{21-26} = MBE{4,3,2,1,0,5};
  let Inst{27-30} = xo;
  let Inst{31}    = RC;
}
