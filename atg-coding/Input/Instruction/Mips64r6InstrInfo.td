//=- Mips64r6InstrInfo.td - Mips64r6 Instruction Information -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes Mips64r6 instructions.
//
//===----------------------------------------------------------------------===//

// Notes about removals/changes from MIPS32r6:
// Reencoded: dclo, dclz

//===----------------------------------------------------------------------===//
//
// Instruction Encodings
//
//===----------------------------------------------------------------------===//

class DALIGN_ENC  : SPECIAL3_DALIGN_FM<OPCODE6_DALIGN>;
class DAUI_ENC    : DAUI_FM;
class DAHI_ENC    : REGIMM_FM<OPCODE5_DAHI>;
class DATI_ENC    : REGIMM_FM<OPCODE5_DATI>;
class DBITSWAP_ENC : SPECIAL3_2R_FM<OPCODE6_DBITSWAP>;
class DCLO_R6_ENC : SPECIAL_2R_FM<OPCODE6_DCLO>;
class DCLZ_R6_ENC : SPECIAL_2R_FM<OPCODE6_DCLZ>;
class DDIV_ENC    : SPECIAL_3R_FM<0b00010, 0b011110>;
class DDIVU_ENC   : SPECIAL_3R_FM<0b00010, 0b011111>;
class DLSA_R6_ENC : SPECIAL_LSA_FM<OPCODE6_DLSA>;
class DMOD_ENC    : SPECIAL_3R_FM<0b00011, 0b011110>;
class DMODU_ENC   : SPECIAL_3R_FM<0b00011, 0b011111>;
class DMUH_ENC    : SPECIAL_3R_FM<0b00011, 0b011100>;
class DMUHU_ENC   : SPECIAL_3R_FM<0b00011, 0b011101>;
class DMUL_R6_ENC : SPECIAL_3R_FM<0b00010, 0b011100>;
class DMULU_ENC   : SPECIAL_3R_FM<0b00010, 0b011101>;
class LDPC_ENC    : PCREL18_FM<OPCODE3_LDPC>;
class LWUPC_ENC   : PCREL19_FM<OPCODE2_LWUPC>;
class LLD_R6_ENC : SPECIAL3_LL_SC_FM<OPCODE6_LLD>;
class SCD_R6_ENC : SPECIAL3_LL_SC_FM<OPCODE6_SCD>;
class CRC32D_ENC  : SPECIAL3_2R_SZ_CRC<3,0>;
class CRC32CD_ENC : SPECIAL3_2R_SZ_CRC<3,1>;

//===----------------------------------------------------------------------===//
//
// Instruction Descriptions
//
//===----------------------------------------------------------------------===//

class AHI_ATI_DESC_BASE<string instr_asm, RegisterOperand GPROpnd, InstrItinClass itin> {
  dag OutOperandList = (outs GPROpnd:$rs);
  dag InOperandList = (ins GPROpnd:$rt, uimm16_altrelaxed:$imm);
  string AsmString = !strconcat(instr_asm, "\t$rs, $rt, $imm");
  string Constraints = "$rs = $rt";
  InstrItinClass Itinerary = itin;
}

class DALIGN_DESC  : ALIGN_DESC_BASE<"dalign", GPR64Opnd, uimm3, II_DALIGN>;
class DAHI_DESC    : AHI_ATI_DESC_BASE<"dahi", GPR64Opnd, II_DAHI>;
class DATI_DESC    : AHI_ATI_DESC_BASE<"dati", GPR64Opnd, II_DATI>;
class DAUI_DESC    : AUI_DESC_BASE<"daui", GPR64Opnd, II_DAUI>;
class DBITSWAP_DESC : BITSWAP_DESC_BASE<"dbitswap", GPR64Opnd, II_DBITSWAP>;
class DCLO_R6_DESC : CLO_R6_DESC_BASE<"dclo", GPR64Opnd, II_DCLO>;
class DCLZ_R6_DESC : CLZ_R6_DESC_BASE<"dclz", GPR64Opnd, II_DCLZ>;
class DDIV_DESC    : DIVMOD_DESC_BASE<"ddiv", GPR64Opnd, II_DDIV, sdiv>;
class DDIVU_DESC   : DIVMOD_DESC_BASE<"ddivu", GPR64Opnd, II_DDIVU, udiv>;
class DLSA_R6_DESC : LSA_R6_DESC_BASE<"dlsa", GPR64Opnd, uimm2_plus1, II_DLSA>;
class DMOD_DESC    : DIVMOD_DESC_BASE<"dmod", GPR64Opnd, II_DMOD, srem>;
class DMODU_DESC   : DIVMOD_DESC_BASE<"dmodu", GPR64Opnd, II_DMODU, urem>;
class DMUH_DESC    : MUL_R6_DESC_BASE<"dmuh", GPR64Opnd, II_DMUH, mulhs>;
class DMUHU_DESC   : MUL_R6_DESC_BASE<"dmuhu", GPR64Opnd, II_DMUHU, mulhu>;
class DMUL_R6_DESC : MUL_R6_DESC_BASE<"dmul", GPR64Opnd, II_DMUL, mul>;
class DMULU_DESC   : MUL_R6_DESC_BASE<"dmulu", GPR64Opnd, II_DMUL>;
class LDPC_DESC    : PCREL_DESC_BASE<"ldpc", GPR64Opnd, simm18_lsl3, II_LDPC>;
class LWUPC_DESC   : PCREL_DESC_BASE<"lwupc", GPR32Opnd, simm19_lsl2, II_LWUPC>;
class LLD_R6_DESC   : LL_R6_DESC_BASE<"lld", GPR64Opnd, mem_simmptr, II_LLD>;
class SCD_R6_DESC   : SC_R6_DESC_BASE<"scd", GPR64Opnd, II_SCD>;
class SELEQZ64_DESC : SELEQNE_Z_DESC_BASE<"seleqz", GPR64Opnd>;
class SELNEZ64_DESC : SELEQNE_Z_DESC_BASE<"selnez", GPR64Opnd>;

class BGEC64_DESC : CMP_BC_DESC_BASE<"bgec", brtarget, GPR64Opnd>;
class BGEUC64_DESC : CMP_BC_DESC_BASE<"bgeuc", brtarget, GPR64Opnd>;
class BEQC64_DESC : CMP_BC_DESC_BASE<"beqc", brtarget, GPR64Opnd>;
class BNEC64_DESC : CMP_BC_DESC_BASE<"bnec", brtarget, GPR64Opnd>;
class BLTC64_DESC : CMP_BC_DESC_BASE<"bltc", brtarget, GPR64Opnd>;
class BLTUC64_DESC : CMP_BC_DESC_BASE<"bltuc", brtarget, GPR64Opnd>;
class BLTZC64_DESC : CMP_CBR_RT_Z_DESC_BASE<"bltzc", brtarget, GPR64Opnd>;
class BGEZC64_DESC : CMP_CBR_RT_Z_DESC_BASE<"bgezc", brtarget, GPR64Opnd>;
class BLEZC64_DESC : CMP_CBR_RT_Z_DESC_BASE<"blezc", brtarget, GPR64Opnd>;
class BGTZC64_DESC : CMP_CBR_RT_Z_DESC_BASE<"bgtzc", brtarget, GPR64Opnd>;
class BEQZC64_DESC : CMP_CBR_EQNE_Z_DESC_BASE<"beqzc", brtarget21, GPR64Opnd>;
class BNEZC64_DESC : CMP_CBR_EQNE_Z_DESC_BASE<"bnezc", brtarget21, GPR64Opnd>;

class JIALC64_DESC : JMP_IDX_COMPACT_DESC_BASE<"jialc", calloffset16,
                                               GPR64Opnd, II_JIALC> {
  bit isCall = 1;
  list<Register> Defs = [RA];
}

class JIC64_DESC : JMP_IDX_COMPACT_DESC_BASE<"jic", jmpoffset16, GPR64Opnd,
                                             II_JIC> {
  bit isBarrier = 1;
  bit isTerminator = 1;
  list<Register> Defs = [AT];
}

class LL64_R6_DESC : LL_R6_DESC_BASE<"ll", GPR32Opnd, mem_simm9, II_LL>;
class SC64_R6_DESC : SC_R6_DESC_BASE<"sc", GPR32Opnd, II_SC>;

class JR_HB64_R6_DESC : JR_HB_DESC_BASE<"jr.hb", GPR64Opnd> {
  bit isBranch = 1;
  bit isIndirectBranch = 1;
  bit hasDelaySlot = 1;
  bit isTerminator=1;
  bit isBarrier=1;
  bit isCTI = 1;
  InstrItinClass Itinerary = II_JR_HB;
}

class CRC32D_DESC  : CRC_DESC_BASE<"crc32d", GPR32Opnd, II_CRC32D>;
class CRC32CD_DESC : CRC_DESC_BASE<"crc32cd", GPR32Opnd, II_CRC32CD>;

//===----------------------------------------------------------------------===//
//
// Instruction Definitions
//
//===----------------------------------------------------------------------===//

let AdditionalPredicates = [NotInMicroMips] in {
  let DecoderMethod = "DecodeDAHIDATI" in {
    def DATI : DATI_ENC, DATI_DESC, ISA_MIPS64R6;
    def DAHI : DAHI_ENC, DAHI_DESC, ISA_MIPS64R6;
  }
  def DAUI : DAUI_ENC, DAUI_DESC, ISA_MIPS64R6;
  def DALIGN : DALIGN_ENC, DALIGN_DESC, ISA_MIPS64R6;
  def DBITSWAP : DBITSWAP_ENC, DBITSWAP_DESC, ISA_MIPS64R6;
  def DCLO_R6 : DCLO_R6_ENC, DCLO_R6_DESC, ISA_MIPS64R6;
  def DCLZ_R6 : DCLZ_R6_ENC, DCLZ_R6_DESC, ISA_MIPS64R6;
  def DDIV : DDIV_ENC, DDIV_DESC, ISA_MIPS64R6;
  def DDIVU : DDIVU_ENC, DDIVU_DESC, ISA_MIPS64R6;
  def DMOD : DMOD_ENC, DMOD_DESC, ISA_MIPS64R6;
  def DMODU : DMODU_ENC, DMODU_DESC, ISA_MIPS64R6;
  def DLSA_R6 : DLSA_R6_ENC, DLSA_R6_DESC, ISA_MIPS64R6;
  def DMUH: DMUH_ENC, DMUH_DESC, ISA_MIPS64R6;
  def DMUHU: DMUHU_ENC, DMUHU_DESC, ISA_MIPS64R6;
  def DMUL_R6: DMUL_R6_ENC, DMUL_R6_DESC, ISA_MIPS64R6;
  def DMULU: DMULU_ENC, DMULU_DESC, ISA_MIPS64R6;
  def LLD_R6 : LLD_R6_ENC, LLD_R6_DESC, ISA_MIPS64R6;
}
def LDPC: LDPC_ENC, LDPC_DESC, ISA_MIPS64R6;
def LWUPC : LWUPC_ENC, LWUPC_DESC, ISA_MIPS64R6;
def SCD_R6 : SCD_R6_ENC, SCD_R6_DESC, ISA_MIPS32R6;
let DecoderNamespace = "Mips32r6_64r6_GP64" in {
  def SELEQZ64 : SELEQZ_ENC, SELEQZ64_DESC, ISA_MIPS32R6, GPR_64;
  def SELNEZ64 : SELNEZ_ENC, SELNEZ64_DESC, ISA_MIPS32R6, GPR_64;
  def JR_HB64_R6 : JR_HB_R6_ENC, JR_HB64_R6_DESC, ISA_MIPS32R6;
}
let AdditionalPredicates = [NotInMicroMips],
    DecoderNamespace = "Mips32r6_64r6_PTR64" in {
  def LL64_R6 : LL_R6_ENC, LL64_R6_DESC, PTR_64, ISA_MIPS64R6;
  def SC64_R6 : SC_R6_ENC, SC64_R6_DESC, PTR_64, ISA_MIPS64R6;
}

let DecoderNamespace = "Mips32r6_64r6_GP64" in {
// Jump and Branch Instructions
def JIALC64 : JIALC_ENC, JIALC64_DESC, ISA_MIPS64R6, GPR_64;
def JIC64 : JIC_ENC, JIC64_DESC, ISA_MIPS64R6, GPR_64;

def BEQC64 : BEQC_ENC, BEQC64_DESC, ISA_MIPS64R6, GPR_64;
def BEQZC64 : BEQZC_ENC, BEQZC64_DESC, ISA_MIPS64R6, GPR_64;
def BGEC64 : BGEC_ENC, BGEC64_DESC, ISA_MIPS64R6, GPR_64;
def BGEUC64 : BGEUC_ENC, BGEUC64_DESC, ISA_MIPS64R6, GPR_64;
def BGTZC64 : BGTZC_ENC, BGTZC64_DESC, ISA_MIPS64R6, GPR_64;
def BLEZC64 : BLEZC_ENC, BLEZC64_DESC, ISA_MIPS64R6, GPR_64;
def BLTC64 : BLTC_ENC, BLTC64_DESC, ISA_MIPS64R6, GPR_64;
def BLTUC64 : BLTUC_ENC, BLTUC64_DESC, ISA_MIPS64R6, GPR_64;
def BNEC64 : BNEC_ENC, BNEC64_DESC, ISA_MIPS64R6, GPR_64;
def BNEZC64 : BNEZC_ENC, BNEZC64_DESC, ISA_MIPS64R6, GPR_64;
}
let DecoderNamespace = "Mips32r6_64r6_BranchZero" in {
def BLTZC64 : BLTZC_ENC, BLTZC64_DESC, ISA_MIPS64R6, GPR_64;
def BGEZC64 : BGEZC_ENC, BGEZC64_DESC, ISA_MIPS64R6, GPR_64;
}
let AdditionalPredicates = [NotInMicroMips] in {
  def CRC32D : R6MMR6Rel, CRC32D_ENC, CRC32D_DESC, ISA_MIPS64R6, ASE_CRC;
  def CRC32CD : R6MMR6Rel, CRC32CD_ENC, CRC32CD_DESC, ISA_MIPS64R6, ASE_CRC;
}

//===----------------------------------------------------------------------===//
//
// Instruction Aliases
//
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
//
// Patterns and Pseudo Instructions
//
//===----------------------------------------------------------------------===//

