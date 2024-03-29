//===-- MipsSchedule.td - Mips Scheduling Definitions ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Functional units across Mips chips sets. Based on GCC/Mips backend files.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction Itinerary classes used for Mips
//===----------------------------------------------------------------------===//
// IIM16Alu is a placeholder class for most MIPS16 instructions.
def IIM16Alu           : InstrItinClass;
def IIPseudo           : InstrItinClass;

def II_ABS              : InstrItinClass;
def II_ADDI             : InstrItinClass;
def II_ADDIU            : InstrItinClass;
def II_ADDIUPC          : InstrItinClass;
def II_ADD              : InstrItinClass;
def II_ADDU             : InstrItinClass;
def II_ADD_D            : InstrItinClass;
def II_ADD_S            : InstrItinClass;
def II_ALIGN            : InstrItinClass;
def II_AND              : InstrItinClass;
def II_ANDI             : InstrItinClass;
def II_ALUIPC           : InstrItinClass;
def II_AUI              : InstrItinClass;
def II_AUIPC            : InstrItinClass;
def II_B                : InstrItinClass;
def II_BADDU            : InstrItinClass;
def II_BBIT             : InstrItinClass; // bbit[01], bbit[01]32
def II_BALC             : InstrItinClass;
def II_BC               : InstrItinClass;
def II_BC1F             : InstrItinClass;
def II_BC1FL            : InstrItinClass;
def II_BC1T             : InstrItinClass;
def II_BC1TL            : InstrItinClass;
def II_BC1CCZ           : InstrItinClass;
def II_BC2CCZ           : InstrItinClass;
def II_BCC              : InstrItinClass; // beq and bne
def II_BCCZ             : InstrItinClass; // b[gl][et]z
def II_BCCC             : InstrItinClass; // b<cc>c
def II_BCCZAL           : InstrItinClass; // bgezal and bltzal
def II_BCCZALS          : InstrItinClass; // bgezals and bltzals
def II_BCCZC            : InstrItinClass; // beqzc, bnezc
def II_BITSWAP          : InstrItinClass;
def II_CEIL             : InstrItinClass;
def II_CFC1             : InstrItinClass;
def II_CFC2             : InstrItinClass;
def II_CLO              : InstrItinClass;
def II_CLZ              : InstrItinClass;
def II_CRC32B           : InstrItinClass;
def II_CRC32CB          : InstrItinClass;
def II_CRC32CD          : InstrItinClass;
def II_CRC32CH          : InstrItinClass;
def II_CRC32CW          : InstrItinClass;
def II_CRC32D           : InstrItinClass;
def II_CRC32H           : InstrItinClass;
def II_CRC32W           : InstrItinClass;
def II_CTC1             : InstrItinClass;
def II_CTC2             : InstrItinClass;
def II_CVT              : InstrItinClass;
def II_C_CC_D           : InstrItinClass; // Any c.<cc>.d instruction
def II_C_CC_S           : InstrItinClass; // Any c.<cc>.s instruction
def II_CMP_CC_D         : InstrItinClass; // Any cmp.<cc>.d instruction
def II_CMP_CC_S         : InstrItinClass; // Any cmp.<cc>.s instruction
def II_CLASS_D          : InstrItinClass;
def II_CLASS_S          : InstrItinClass;
def II_DADDIU           : InstrItinClass;
def II_DADDU            : InstrItinClass;
def II_DADDI            : InstrItinClass;
def II_DADD             : InstrItinClass;
def II_DAHI             : InstrItinClass;
def II_DATI             : InstrItinClass;
def II_DAUI             : InstrItinClass;
def II_DALIGN           : InstrItinClass;
def II_DBITSWAP         : InstrItinClass;
def II_DCLO             : InstrItinClass;
def II_DCLZ             : InstrItinClass;
def II_DDIV             : InstrItinClass;
def II_DDIVU            : InstrItinClass;
def II_DIV              : InstrItinClass;
def II_DIVU             : InstrItinClass;
def II_DIV_D            : InstrItinClass;
def II_DIV_S            : InstrItinClass;
def II_DMFC0            : InstrItinClass;
def II_DMFGC0           : InstrItinClass;
def II_DMT              : InstrItinClass;
def II_DMTC0            : InstrItinClass;
def II_DMTGC0           : InstrItinClass;
def II_DMFC1            : InstrItinClass;
def II_DMTC1            : InstrItinClass;
def II_DMOD             : InstrItinClass;
def II_DMODU            : InstrItinClass;
def II_DMUH             : InstrItinClass;
def II_DMUHU            : InstrItinClass;
def II_DMFC2            : InstrItinClass;
def II_DMTC2            : InstrItinClass;
def II_DMUL             : InstrItinClass;
def II_DMULU            : InstrItinClass;
def II_DMULT            : InstrItinClass;
def II_DMULTU           : InstrItinClass;
def II_DROTR            : InstrItinClass;
def II_DROTR32          : InstrItinClass;
def II_DROTRV           : InstrItinClass;
def II_DSLL             : InstrItinClass;
def II_DSLL32           : InstrItinClass;
def II_DSLLV            : InstrItinClass;
def II_DSRA             : InstrItinClass;
def II_DSRA32           : InstrItinClass;
def II_DSRAV            : InstrItinClass;
def II_DSRL             : InstrItinClass;
def II_DSRL32           : InstrItinClass;
def II_DSRLV            : InstrItinClass;
def II_DSBH             : InstrItinClass;
def II_DSHD             : InstrItinClass;
def II_DSUBU            : InstrItinClass;
def II_DSUB             : InstrItinClass;
def II_DVPE             : InstrItinClass;
def II_EMT              : InstrItinClass;
def II_EVPE             : InstrItinClass;
def II_EXT              : InstrItinClass; // Any EXT instruction
def II_FLOOR            : InstrItinClass;
def II_FORK             : InstrItinClass;
def II_GINVI            : InstrItinClass;
def II_GINVT            : InstrItinClass;
def II_HYPCALL          : InstrItinClass;
def II_INS              : InstrItinClass; // Any INS instruction
def II_IndirectBranchPseudo : InstrItinClass; // Indirect branch pseudo.
def II_J                : InstrItinClass;
def II_JAL              : InstrItinClass;
def II_JALR             : InstrItinClass;
def II_JALR_HB          : InstrItinClass;
def II_JALRC            : InstrItinClass;
def II_JALRS            : InstrItinClass;
def II_JALS             : InstrItinClass;
def II_JIC              : InstrItinClass;
def II_JIALC            : InstrItinClass;
def II_JR               : InstrItinClass;
def II_JR_HB            : InstrItinClass;
def II_JRADDIUSP        : InstrItinClass;
def II_JRC              : InstrItinClass;
def II_ReturnPseudo     : InstrItinClass; // Return pseudo.
def II_ERET             : InstrItinClass;
def II_DERET            : InstrItinClass;
def II_ERETNC           : InstrItinClass;
def II_EHB              : InstrItinClass;
def II_SDBBP            : InstrItinClass;
def II_SIGRIE           : InstrItinClass;
def II_SSNOP            : InstrItinClass;
def II_SYSCALL          : InstrItinClass;
def II_PAUSE            : InstrItinClass;
def II_WAIT             : InstrItinClass;
def II_EI               : InstrItinClass;
def II_DI               : InstrItinClass;
def II_TEQ              : InstrItinClass;
def II_TEQI             : InstrItinClass;
def II_TGE              : InstrItinClass;
def II_TGEI             : InstrItinClass;
def II_TGEIU            : InstrItinClass;
def II_TGEU             : InstrItinClass;
def II_TNE              : InstrItinClass;
def II_TNEI             : InstrItinClass;
def II_TLT              : InstrItinClass;
def II_TLTI             : InstrItinClass;
def II_TLTU             : InstrItinClass;
def II_TTLTIU           : InstrItinClass;
def II_TLBP             : InstrItinClass;
def II_TLBR             : InstrItinClass;
def II_TLBWI            : InstrItinClass;
def II_TLBWR            : InstrItinClass;
def II_TRAP             : InstrItinClass;
def II_BREAK            : InstrItinClass;
def II_SYNC             : InstrItinClass;
def II_SYNCI            : InstrItinClass;
def II_LB               : InstrItinClass;
def II_LBE              : InstrItinClass;
def II_LBU              : InstrItinClass;
def II_LBUE             : InstrItinClass;
def II_LD               : InstrItinClass;
def II_LDC1             : InstrItinClass;
def II_LDC2             : InstrItinClass;
def II_LDC3             : InstrItinClass;
def II_LDL              : InstrItinClass;
def II_LDR              : InstrItinClass;
def II_LDPC             : InstrItinClass;
def II_LDXC1            : InstrItinClass;
def II_LH               : InstrItinClass;
def II_LHE              : InstrItinClass;
def II_LHU              : InstrItinClass;
def II_LHUE             : InstrItinClass;
def II_LL               : InstrItinClass;
def II_LI               : InstrItinClass;
def II_LLD              : InstrItinClass;
def II_LUI              : InstrItinClass;
def II_LUXC1            : InstrItinClass;
def II_LW               : InstrItinClass;
def II_LWE              : InstrItinClass;
def II_LWC1             : InstrItinClass;
def II_LWC2             : InstrItinClass;
def II_LWC3             : InstrItinClass;
def II_LWM              : InstrItinClass;
def II_LWL              : InstrItinClass;
def II_LWLE             : InstrItinClass;
def II_LWPC             : InstrItinClass;
def II_LWP              : InstrItinClass;
def II_LWR              : InstrItinClass;
def II_LWRE             : InstrItinClass;
def II_LWU              : InstrItinClass;
def II_LWUPC            : InstrItinClass;
def II_LWXC1            : InstrItinClass;
def II_LWXS             : InstrItinClass;
def II_LSA              : InstrItinClass;
def II_DLSA             : InstrItinClass;
def II_MADD             : InstrItinClass;
def II_MADDU            : InstrItinClass;
def II_MADD_D           : InstrItinClass;
def II_MADD_S           : InstrItinClass;
def II_MADDF_D          : InstrItinClass;
def II_MADDF_S          : InstrItinClass;
def II_MAX_D            : InstrItinClass;
def II_MAX_S            : InstrItinClass;
def II_MAXA_D           : InstrItinClass;
def II_MAXA_S           : InstrItinClass;
def II_MIN_D            : InstrItinClass;
def II_MIN_S            : InstrItinClass;
def II_MINA_D           : InstrItinClass;
def II_MINA_S           : InstrItinClass;
def II_MFC0             : InstrItinClass;
def II_MFHC0            : InstrItinClass;
def II_MFC1             : InstrItinClass;
def II_MFHC1            : InstrItinClass;
def II_MFC2             : InstrItinClass;
def II_MFGC0            : InstrItinClass;
def II_MFHGC0           : InstrItinClass;
def II_MFHI_MFLO        : InstrItinClass; // mfhi and mflo
def II_MFTR             : InstrItinClass;
def II_MOD              : InstrItinClass;
def II_MODU             : InstrItinClass;
def II_MOVE             : InstrItinClass;
def II_MOVF             : InstrItinClass;
def II_MOVF_D           : InstrItinClass;
def II_MOVF_S           : InstrItinClass;
def II_MOVN             : InstrItinClass;
def II_MOVN_D           : InstrItinClass;
def II_MOVN_S           : InstrItinClass;
def II_MOVT             : InstrItinClass;
def II_MOVT_D           : InstrItinClass;
def II_MOVT_S           : InstrItinClass;
def II_MOVZ             : InstrItinClass;
def II_MOVZ_D           : InstrItinClass;
def II_MOVZ_S           : InstrItinClass;
def II_MOV_D            : InstrItinClass;
def II_MOV_S            : InstrItinClass;
def II_MSUB             : InstrItinClass;
def II_MSUBU            : InstrItinClass;
def II_MSUB_D           : InstrItinClass;
def II_MSUB_S           : InstrItinClass;
def II_MSUBF_D          : InstrItinClass;
def II_MSUBF_S          : InstrItinClass;
def II_MTC0             : InstrItinClass;
def II_MTHC0            : InstrItinClass;
def II_MTC1             : InstrItinClass;
def II_MTHC1            : InstrItinClass;
def II_MTC2             : InstrItinClass;
def II_MTGC0            : InstrItinClass;
def II_MTHGC0           : InstrItinClass;
def II_MTHI_MTLO        : InstrItinClass; // mthi and mtlo
def II_MTTR             : InstrItinClass;
def II_MUL              : InstrItinClass;
def II_MUH              : InstrItinClass;
def II_MUHU             : InstrItinClass;
def II_MULU             : InstrItinClass;
def II_MULT             : InstrItinClass;
def II_MULTU            : InstrItinClass;
def II_MUL_D            : InstrItinClass;
def II_MUL_S            : InstrItinClass;
def II_NEG              : InstrItinClass;
def II_NMADD_D          : InstrItinClass;
def II_NMADD_S          : InstrItinClass;
def II_NMSUB_D          : InstrItinClass;
def II_NMSUB_S          : InstrItinClass;
def II_NOR              : InstrItinClass;
def II_NOT              : InstrItinClass;
def II_OR               : InstrItinClass;
def II_ORI              : InstrItinClass;
def II_POP              : InstrItinClass;
def II_RDHWR            : InstrItinClass;
def II_RESTORE          : InstrItinClass;
def II_RECIP_S          : InstrItinClass;
def II_RECIP_D          : InstrItinClass;
def II_RINT_S           : InstrItinClass;
def II_RINT_D           : InstrItinClass;
def II_ROTR             : InstrItinClass;
def II_ROTRV            : InstrItinClass;
def II_ROUND            : InstrItinClass;
def II_RSQRT_S          : InstrItinClass;
def II_RSQRT_D          : InstrItinClass;
def II_SAVE             : InstrItinClass;
def II_SC               : InstrItinClass;
def II_SCD              : InstrItinClass;
def II_SB               : InstrItinClass;
def II_SBE              : InstrItinClass;
def II_SD               : InstrItinClass;
def II_SDC1             : InstrItinClass;
def II_SDC2             : InstrItinClass;
def II_SDC3             : InstrItinClass;
def II_SDL              : InstrItinClass;
def II_SDR              : InstrItinClass;
def II_SDXC1            : InstrItinClass;
def II_SEB              : InstrItinClass;
def II_SEH              : InstrItinClass;
def II_SELCCZ           : InstrItinClass;
def II_SELCCZ_D         : InstrItinClass;
def II_SELCCZ_S         : InstrItinClass;
def II_SEQ_SNE          : InstrItinClass; // seq and sne
def II_SEQI_SNEI        : InstrItinClass; // seqi and snei
def II_SH               : InstrItinClass;
def II_SHE              : InstrItinClass;
def II_SLL              : InstrItinClass;
def II_SLLV             : InstrItinClass;
def II_SLTI_SLTIU       : InstrItinClass; // slti and sltiu
def II_SLT_SLTU         : InstrItinClass; // slt and sltu
def II_SQRT_D           : InstrItinClass;
def II_SQRT_S           : InstrItinClass;
def II_SEL_D            : InstrItinClass;
def II_SEL_S            : InstrItinClass;
def II_SRA              : InstrItinClass;
def II_SRAV             : InstrItinClass;
def II_SRL              : InstrItinClass;
def II_SRLV             : InstrItinClass;
def II_SUB              : InstrItinClass;
def II_SUBU             : InstrItinClass;
def II_SUB_D            : InstrItinClass;
def II_SUB_S            : InstrItinClass;
def II_SUXC1            : InstrItinClass;
def II_SW               : InstrItinClass;
def II_SWE              : InstrItinClass;
def II_SWC1             : InstrItinClass;
def II_SWC2             : InstrItinClass;
def II_SWC3             : InstrItinClass;
def II_SWL              : InstrItinClass;
def II_SWLE             : InstrItinClass;
def II_SWM              : InstrItinClass;
def II_SWP              : InstrItinClass;
def II_SWR              : InstrItinClass;
def II_SWRE             : InstrItinClass;
def II_SWXC1            : InstrItinClass;
def II_TRUNC            : InstrItinClass;
def II_WSBH             : InstrItinClass;
def II_XOR              : InstrItinClass;
def II_XORI             : InstrItinClass;
def II_CACHE            : InstrItinClass;
def II_PREF             : InstrItinClass;
def II_CACHEE           : InstrItinClass;
def II_PREFE            : InstrItinClass;
def II_LLE              : InstrItinClass;
def II_SCE              : InstrItinClass;
def II_TLBGINV          : InstrItinClass;
def II_TLBGINVF         : InstrItinClass;
def II_TLBGP            : InstrItinClass;
def II_TLBGR            : InstrItinClass;
def II_TLBGWI           : InstrItinClass;
def II_TLBGWR           : InstrItinClass;
def II_TLBINV           : InstrItinClass;
def II_TLBINVF          : InstrItinClass;
def II_WRPGPR           : InstrItinClass;
def II_RDPGPR           : InstrItinClass;
def II_DVP              : InstrItinClass;
def II_EVP              : InstrItinClass;
def II_YIELD            : InstrItinClass;

//===----------------------------------------------------------------------===//
// Mips Generic instruction itineraries.
//===----------------------------------------------------------------------===//
