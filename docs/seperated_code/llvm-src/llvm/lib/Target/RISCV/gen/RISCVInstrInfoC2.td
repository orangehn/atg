include "RISCVInstrFormatsC.td"
include "gen/c_inst.td"


// not generated

let Predicates = [HasStdExtC, HasRVCHints], hasSideEffects = 0, mayLoad = 0,
    mayStore = 0 in
{
  
let rd = 0 in
def C_NOP_HINT : RVInst16CI<0b000, 0b01, (outs), (ins simm6nonzero:$imm),
                            "c.nop", "$imm">, Sched<[WriteNop]> {
  let Inst{6-2} = imm{4-0};
  let DecoderMethod = "decodeRVCInstrSImm";
}

// Just a different syntax for the c.nop hint: c.addi x0, simm6 vs c.nop simm6.
def C_ADDI_HINT_X0 : RVInst16CI<0b000, 0b01, (outs GPRX0:$rd_wb),
                                (ins GPRX0:$rd, simm6nonzero:$imm),
                                "c.addi", "$rd, $imm">,
                     Sched<[WriteIALU, ReadIALU]> {
  let Constraints = "$rd = $rd_wb";
  let Inst{6-2} = imm{4-0};
  let isAsmParserOnly = 1;
}

def C_ADDI_HINT_IMM_ZERO : RVInst16CI<0b000, 0b01, (outs GPRNoX0:$rd_wb),
                                      (ins GPRNoX0:$rd, immzero:$imm),
                                      "c.addi", "$rd, $imm">,
                           Sched<[WriteIALU, ReadIALU]> {
  let Constraints = "$rd = $rd_wb";
  let Inst{6-2} = 0;
  let isAsmParserOnly = 1;
}

def C_LI_HINT : RVInst16CI<0b010, 0b01, (outs GPRX0:$rd), (ins simm6:$imm),
                           "c.li", "$rd, $imm">,
                Sched<[WriteIALU]> {
  let Inst{6-2} = imm{4-0};
  let Inst{11-7} = 0;
  let DecoderMethod = "decodeRVCInstrRdSImm";
}

def C_LUI_HINT : RVInst16CI<0b011, 0b01, (outs GPRX0:$rd),
                            (ins c_lui_imm:$imm),
                            "c.lui", "$rd, $imm">,
                 Sched<[WriteIALU]> {
  let Inst{6-2} = imm{4-0};
  let Inst{11-7} = 0;
  let DecoderMethod = "decodeRVCInstrRdSImm";
}

def C_MV_HINT : RVInst16CR<0b1000, 0b10, (outs GPRX0:$rs1), (ins GPRNoX0:$rs2),
                           "c.mv", "$rs1, $rs2">, Sched<[WriteIALU, ReadIALU]>
{
  let Inst{11-7} = 0;
  let DecoderMethod = "decodeRVCInstrRdRs2";
}

def C_ADD_HINT : RVInst16CR<0b1001, 0b10, (outs GPRX0:$rs1_wb),
                            (ins GPRX0:$rs1, GPRNoX0:$rs2),
                            "c.add", "$rs1, $rs2">,
                 Sched<[WriteIALU, ReadIALU, ReadIALU]> {
  let Constraints = "$rs1 = $rs1_wb";
  let Inst{11-7} = 0;
  let DecoderMethod = "decodeRVCInstrRdRs1Rs2";
}

def C_SLLI_HINT : RVInst16CI<0b000, 0b10, (outs GPRX0:$rd_wb),
                             (ins GPRX0:$rd, uimmlog2xlennonzero:$imm),
                             "c.slli" ,"$rd, $imm">,
                  Sched<[WriteShift, ReadShift]> {
  let Constraints = "$rd = $rd_wb";
  let Inst{6-2} = imm{4-0};
  let Inst{11-7} = 0;
  let DecoderMethod = "decodeRVCInstrRdRs1UImm";
}
} // Predicates = [HasStdExtC, HasRVCHints], hasSideEffects = 0, mayLoad = 0,
  // mayStore = 0

let Predicates = [HasStdExtC] in {
  // The all zeros pattern isn't a valid RISC-V instruction. It's used by GNU
// binutils as 16-bit instruction known to be unimplemented (i.e., trapping).
let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in
def C_UNIMP : RVInst16<(outs), (ins), "c.unimp", "", [], InstFormatOther>,
              Sched<[]> {
  let Inst{15-0} = 0;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def C_ADDI_NOP : RVInst16CI<0b000, 0b01, (outs GPRX0:$rd_wb),
                            (ins GPRX0:$rd, immzero:$imm),
                            "c.addi", "$rd, $imm">,
                 Sched<[WriteIALU, ReadIALU]> {
  let Constraints = "$rd = $rd_wb";
  let Inst{6-2} = 0;
  let isAsmParserOnly = 1;
}

} // Predicates = [HasStdExtC]