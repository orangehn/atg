//===- ARCInstrFormats.td - ARC Instruction Formats --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction format superclass
//===----------------------------------------------------------------------===//

class Encoding64 {
  field bits<64> Inst;
  field bits<64> SoftFail = 0;
}


class InstARC<int sz, dag outs, dag ins, string asmstr, list<dag> pattern>
    : Instruction, Encoding64 {

  let Namespace = "ARC";
  dag OutOperandList = outs;
  dag InOperandList = ins;
  let AsmString = asmstr;
  let Pattern = pattern;
  let Size = sz;

  // Load/Store instruction properties
  //DataSizeMode ZZ = WordSM;
  //ExtMode X = NoEM;
  //AddrMode AA = NoAM;
  //CacheMode DI = NoCC;

  // Field used for relation models
  string BaseOpcode = "";

  //TSFlags
  let TSFlags{1-0} = AA.Value;
}

// Branch Instructions.
class F32_BR<bits<5> major, dag outs, dag ins, bit b16, string asmstr,
             list<dag> pattern> :
  InstARC<4, outs, ins, asmstr, pattern> {
  bit N;

  let Inst{31-27} = major;
  let Inst{16} = b16;
  let Inst{5} = N;
}

class F32_BR_COND<bits<5> major, dag outs, dag ins, bit b16, string asmstr,
                  list<dag> pattern> :
  F32_BR<major, outs, ins, b16, asmstr, pattern> {
  bits<21> S21; // 2-byte aligned 21-bit byte-offset.
  bits<5> cc; 
  let Inst{26-18} = S21{10-2};
  let Inst{15-6} = S21{20-11};
  let Inst{4-0} = cc;
}

class F32_BR_UCOND_FAR<bits<5> major, dag outs, dag ins, bit b16, string asmstr,
                       list<dag> pattern> :
  F32_BR<major, outs, ins, b16, asmstr, pattern> {
  bits<25> S25; // 2-byte aligned 25-bit byte-offset.
  let Inst{26-18} = S25{10-2};
  let Inst{15-6} = S25{20-11};
  let Inst{4} = 0;
  let Inst{3-0} = S25{24-21};
}

class F32_BR0_COND<dag outs, dag ins, string asmstr, list<dag> pat> :
  F32_BR_COND<0b00000, outs, ins, 0, asmstr, pat> {
  let Inst{17} = S21{1};
}

// Branch targets are 2-byte aligned, so S25[0] is implied 0.
// |26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10|9|8|7|6|5|4|3|2|1|0   |
// |S25[10-1]                    | 1|S25[20-11]               |N|0|S25[24-21]|
class F32_BR0_UCOND_FAR<dag outs, dag ins, string asmstr, list<dag> pat> :
  F32_BR_UCOND_FAR<0b00000, outs, ins, 1, asmstr, pat> {
  let Inst{17} = S25{1};
}

class F32_BR1_BL_UCOND_FAR<dag outs, dag ins, string asmstr, list<dag> pat> :
  F32_BR_UCOND_FAR<0b00001, outs, ins, 0, asmstr, pat> {
  let Inst{17} = 1;
}

class F32_BR1_BCC<dag outs, dag ins, string asmstr, bit IsU6,
                  list<dag> pattern> :
  InstARC<4, outs, ins, asmstr, pattern> {

  bits<3> cc;
  bits<6> B;
  bits<6> C;
  bit N;
  bits<9> S9; // 2-byte aligned 9-bit byte-offset.

  let Inst{31-27} = 0b00001;
  let Inst{26-24} = B{2-0};
  let Inst{23-17} = S9{7-1};
  let Inst{16} = 1;
  let Inst{15} = S9{8};
  let Inst{14-12} = B{5-3};
  let Inst{11-6} = C;
  let Inst{5} = N;
  let Inst{4} = IsU6;
  let Inst{3} = 0;
  let Inst{2-0} = cc;
}

class F32_SOP_RR<bits<5> major, bits<6> subop, bit F, dag outs, dag ins,
                 string asmstr, list<dag> pattern> :
  InstARC<4, outs, ins, asmstr, pattern> {

  bits<6> C;
  bits<6> B;

  let Inst{31-27} = major;
  let Inst{26-24} = B{2-0};
  let Inst{23-22} = 0b00;
  let Inst{21-16} = 0b101111;
  let Inst{15} = F;
  let Inst{14-12} = B{5-3};
  let Inst{11-6} = C;
  let Inst{5-0} = subop;
}

class F32_DOP_RR<bits<5> major, bits<6> subop, bit F, dag outs, dag ins,
                 string asmstr, list<dag> pattern> :
  InstARC<4, outs, ins, asmstr, pattern> {
  bits<6> C;
  bits<6> B;
  bits<6> A;

  let Inst{31-27} = major;
  let Inst{26-24} = B{2-0};
  let Inst{23-22} = 0b00;
  let Inst{21-16} = subop;
  let Inst{15} = F;
  let Inst{14-12} = B{5-3};
  let Inst{11-6} = C;
  let Inst{5-0} = A;
}

class F32_DOP_CC_RR<bits<5> major, bits<6> subop, bit F, dag outs, dag ins,
                    string asmstr, list<dag> pattern> :
  InstARC<4, outs, ins, asmstr, pattern> {
  bits<5> cc;
  bits<6> C;
  bits<6> B;

  let Inst{31-27} = major;
  let Inst{26-24} = B{2-0};
  let Inst{23-22} = 0b11;
  let Inst{21-16} = subop;
  let Inst{15} = F;
  let Inst{14-12} = B{5-3};
  let Inst{11-6} = C;
  let Inst{5} = 0;
  let Inst{4-0} = cc;
}

class F32_DOP_RU6<bits<5> major, bits<6> subop, bit F, dag outs, dag ins,
                  string asmstr, list<dag> pattern> :
  InstARC<4, outs, ins, asmstr, pattern> {
  bits<6> U6;
  bits<6> B;
  bits<6> A;

  let Inst{31-27} = major;
  let Inst{26-24} = B{2-0};
  let Inst{23-22} = 0b01;
  let Inst{21-16} = subop;
  let Inst{15} = F;
  let Inst{14-12} = B{5-3};
  let Inst{11-6} = U6;
  let Inst{5-0} = A;
}

class F32_DOP_RS12<bits<5> major, bits<6> subop, bit F, dag outs, dag ins,
                   string asmstr, list<dag> pattern> :
  InstARC<4, outs, ins, asmstr, pattern> {
  bits<6> B;
  bits<12> S12;
  
  let Inst{31-27} = major;
  let Inst{26-24} = B{2-0};
  let Inst{23-22} = 0b10;
  let Inst{21-16} = subop;
  let Inst{15} = F;
  let Inst{14-12} = B{5-3};
  let Inst{11-6} = S12{5-0};
  let Inst{5-0} = S12{11-6};
}

class F32_DOP_RLIMM<bits<5> major, bits<6> subop, bit F, dag outs, dag ins,
                    string asmstr, list<dag> pattern> :
  InstARC<8, outs, ins, asmstr, pattern> {
  bits<6> B;
  bits<6> A;
  bits<32> LImm;

  let Inst{63-32} = LImm;
  let Inst{31-27} = major;
  let Inst{26-24} = B{2-0};
  let Inst{23-22} = 0b00;
  let Inst{21-16} = subop;
  let Inst{15} = F;
  let Inst{14-12} = B{5-3};
  let Inst{11-6} = 0b111110;
  let Inst{5-0} = A;
}


class F32_LD_ADDR<bit x, bits<2> aa, bit di, bits<2> zz, dag outs, dag ins,
                  string asmstr, list<dag> pattern> :
  F32_LD_RS9<x, aa, di, zz, outs, ins, asmstr, pattern> {
  bits<15> addr;

  let B = addr{14-9};
  let S9 = addr{8-0};

  let BaseOpcode = "ld_rs9";
}

class F32_LD_RS9<bit x, bits<2> aa, bit di, bits<2> zz, dag outs, dag ins,
                 string asmstr, list<dag> pattern> :
  InstARC<4, outs, ins, asmstr, pattern> {
  bits<6> B;
  bits<6> A;
  bits<9> S9;

  let Inst{31-27} = 0b00010;
  let Inst{26-24} = B{2-0};
  let Inst{23-16} = S9{7-0};
  let Inst{15} = S9{8};
  let Inst{14-12} = B{5-3};
  let Inst{11} = di;
  let Inst{10-9} = aa;
  let Inst{8-7} = zz;
  let Inst{6} = x;
  let Inst{5-0} = A;

  let BaseOpcode = "ld_rs9";
}

////class F32_LD_ADDR<bit x, bits<2> aa, bit di, bits<2> zz, dag outs, dag ins,
////                  string asmstr, list<dag> pattern> :
////  F32_LD_RS9<x, aa, di, zz, outs, ins, asmstr, pattern> {
//// bits<15> addr;
//
// // let B = addr{14-9};
// // let S9 = addr{8-0};
//
////  let BaseOpcode = "ld_rs9";
////}


// LImm Load.  The 32-bit immediate address is in Inst[63-32].
// |26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10|9|8|7|6|5|4|3|2|1|0|
// | 1| 1| 0| 0                        | 1| 1| 1|di| 0|0|zz |x|A          |
class F32_LD_LIMM<bit x, bit di, bits<2> zz, dag outs, dag ins,
                  string asmstr, list<dag> pattern> :
  InstARC<8, outs, ins, asmstr, pattern> {
  bits<6> LImmReg = 0b111110;
  bits<6> A;
  bits<32> LImm;

  let Inst{63-32} = LImm;
  let Inst{31-27} = 0b00010;
  let Inst{26-24} = LImmReg{2-0};
  let Inst{23-15} = 0;
  let Inst{14-12} = LImmReg{5-3};
  let Inst{11} = di;
  let Inst{10-9} = 0;
  let Inst{8-7} = zz;
  let Inst{6} = x;
  let Inst{5-0} = A;
  let DecoderMethod = "DecodeLdLImmInstruction";

  let BaseOpcode = "ld_limm";
}

// Register + LImm load.  The 32-bit immediate address is in Inst[63-32].
// |26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10|9|8|7|6|5|4|3|2|1|0|
// |B[2-0]  |aa   | 1| 1| 0|zz   | x|di|B[5-3]  | 1| 1|1|1|1|0|A          |
class F32_LD_RLIMM<bit x, bits<2> aa, bit di, bits<2> zz, dag outs, dag ins,
                   string asmstr, list<dag> pattern> :
  InstARC<8, outs, ins, asmstr, pattern> {
  bits<6> LImmReg = 0b111110;
  bits<32> LImm;
  bits<6> B;
  bits<6> A;
  bits<38> addr;
  let B = addr{37-32};
  let LImm = addr{31-0};

  let Inst{63-32} = LImm;
  let Inst{31-27} = 0b00100;
  let Inst{26-24} = B{2-0};
  let Inst{23-22} = aa;
  let Inst{21-19} = 0b110;
  let Inst{18-17} = zz;
  let Inst{16} = x;
  let Inst{15} = di;
  let Inst{14-12} = B{5-3};
  let Inst{11-6} = LImmReg;
  let Inst{5-0} = A;
  let DecoderMethod = "DecodeLdRLImmInstruction";

  let BaseOpcode = "ld_rlimm";
}

// Register + S9 Store. (B + S9)
// |26|25|24|23|22|21|20|19|18|17|16|15   |14|13|12|11|10|9|8|7|6|5 |4|3|2|1|0|
// |B[2-0]  |S9[7-0]                |S9[8]|B[5-3]  |C            |di|aa |zz |0|
class F32_ST_RS9<bits<2> aa, bit di, bits<2> zz, dag outs, dag ins,
                 string asmstr, list<dag> pattern> :
  InstARC<4, outs, ins, asmstr, pattern> {
  bits<6> B;
  bits<6> C;
  bits<9> S9;

  let Inst{31-27} = 0b00011;
  let Inst{26-24} = B{2-0};
  let Inst{23-16} = S9{7-0};
  let Inst{15} = S9{8};
  let Inst{14-12} = B{5-3};
  let Inst{11-6} = C;
  let Inst{5} = di;
  let Inst{4-3} = aa;
  let Inst{2-1} = zz;
  let Inst{0} = 0;

  let BaseOpcode = "st_rs9";
}

class F32_ST_ADDR<bits<2> aa, bit di, bits<2> zz, dag outs, dag ins,
                  string asmstr, list<dag> pattern> :
  F32_ST_RS9<aa, di, zz, outs, ins, asmstr, pattern> {
  bits<15> addr;

  let B = addr{14-9};
  let S9 = addr{8-0};

  let BaseOpcode = "st_rs9";
}


// LImm Store.
// |26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10|9|8|7|6|5 |4|3|2|1|0|
// | 1| 1| 0| 0                        | 1| 1| 1|C            |di|0|0|zz |0|
class F32_ST_LIMM<bit di, bits<2> zz, dag outs, dag ins,
                  string asmstr, list<dag> pattern> :
  InstARC<8, outs, ins, asmstr, pattern> {
  bits<6> LImmReg = 0b111110;
  bits<6> C;
  bits<32> LImm;

  let Inst{63-32} = LImm;
  let Inst{31-27} = 0b00011;
  let Inst{26-24} = LImmReg{2-0};
  let Inst{23-15} = 0;
  let Inst{14-12} = LImmReg{5-3};
  let Inst{11-6} = C;
  let Inst{5} = di;
  let Inst{4-3} = 0;
  let Inst{2-1} = zz;
  let Inst{0} = 0;
  let DecoderMethod = "DecodeStLImmInstruction";

  let BaseOpcode = "st_limm";
}

