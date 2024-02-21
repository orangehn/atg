//===----------------------------------------------------------------------===//
// Instruction Formats
//===----------------------------------------------------------------------===//
include "RISCVInstrFormats.td"
//include "gen/refer_addition_include.td"
include "gen/I_inst.td"


//===-----------------------------------------------------------------------===//
// not generate
//===-----------------------------------------------------------------------===//

let hasNoSchedulingInfo = 1,
    hasSideEffects = 1, mayLoad = 0, mayStore = 0 in
class CSR_ir<bits<3> funct3, string opcodestr>
    : RVInstI<funct3, OPC_SYSTEM, (outs GPR:$rd), (ins csr_sysreg:$imm12, GPR:$rs1),
              opcodestr, "$rd, $imm12, $rs1">, Sched<[WriteCSR, ReadCSR]>;

let hasNoSchedulingInfo = 1,
    hasSideEffects = 1, mayLoad = 0, mayStore = 0 in
class CSR_ii<bits<3> funct3, string opcodestr>
    : RVInstI<funct3, OPC_SYSTEM, (outs GPR:$rd),
              (ins csr_sysreg:$imm12, uimm5:$rs1),
              opcodestr, "$rd, $imm12, $rs1">, Sched<[WriteCSR]>;

def CSRRW : CSR_ir<0b001, "csrrw">;
def CSRRS : CSR_ir<0b010, "csrrs">;
def CSRRC : CSR_ir<0b011, "csrrc">;

def CSRRWI : CSR_ii<0b101, "csrrwi">;
def CSRRSI : CSR_ii<0b110, "csrrsi">;
def CSRRCI : CSR_ii<0b111, "csrrci">;

let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in {
// This is a de facto standard (as set by GNU binutils) 32-bit unimplemented
// instruction (i.e., it should always trap, if your implementation has invalid
// instruction traps).
def UNIMP : RVInstI<0b001, OPC_SYSTEM, (outs), (ins), "unimp", "">,
            Sched<[]> {
  let rs1 = 0;
  let rd = 0;
  let imm12 = 0b110000000000;
}
} // hasSideEffects = 1, mayLoad = 0, mayStore = 0


let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in
class Priv<string opcodestr, bits<7> funct7>
    : RVInstR<funct7, 0b000, OPC_SYSTEM, (outs), (ins GPR:$rs1, GPR:$rs2),
              opcodestr, "">;

//===----------------------------------------------------------------------===//
// Debug instructions
//===----------------------------------------------------------------------===//

let isBarrier = 1, isReturn = 1, isTerminator = 1 in {
def DRET : Priv<"dret", 0b0111101>, Sched<[]> {
  let rd = 0;
  let rs1 = 0;
  let rs2 = 0b10010;
}
} // isBarrier = 1, isReturn = 1, isTerminator = 1

