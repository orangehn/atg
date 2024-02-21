// llvm最简形式

class GeneralTemplate<bits<6> opcode, int Size=4>{
    int Size = Size;
    ...
}


def ADD { // MMRel StdMMR6Rel InstructionEncoding Instruction PredicateControl MipsInst InstSE ArithLogicR StdArch ADD_FM ISA_MIPS1
// static attr
  int Size = 4;
  string Namespace = "Mips";
  dag OutOperandList = (outs GPR32Opnd:$rd);
  dag InOperandList = (ins GPR32Opnd:$rs, GPR32Opnd:$rt);
  string AsmString = "add $rd, $rs, $rt";
  list<dag> Pattern = [(set GPR32Opnd:$rd, (null_frag GPR32Opnd:$rs, GPR32Opnd:$rt))];
  string TwoOperandAliasConstraint = "$rd = $rs";
  list<Predicate> Predicates = [HasStdEnc, NotInMicroMips];

  string DecoderNamespace = "Mips";
  string DecoderMethod = "";
  bit isCommutable = 1;
  bit isReMaterializable = 1;
  string Arch = "se";
  string BaseOpcode = "add";

  InstrItinClass Itinerary = II_ADD;
  list<Predicate> EncodingPredicates = [HasStdEnc];
  list<Predicate> GPRPredicates = [];
  list<Predicate> PTRPredicates = [];
  list<Predicate> SYMPredicates = [];
  list<Predicate> FGRPredicates = [];
  list<Predicate> InsnPredicates = [];
  list<Predicate> ASEPredicate = [];
  list<Predicate> HardFloatPredicate = [];
  list<Predicate> AdditionalPredicates = [NotInMicroMips];

// dynamic attr
  field bits<32> SoftFail = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

// dynamic attr
  field bits<32> Inst = { 0, 0, 0, 0, 0, 0, rs{4}, rs{3}, rs{2}, rs{1}, rs{0}, rt{4}, rt{3}, rt{2}, rt{1}, rt{0}, rd{4}, rd{3}, rd{2}, rd{1}, rd{0}, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 };
  bits<5> rd = { ?, ?, ?, ?, ? };
  bits<5> rs = { ?, ?, ?, ?, ? };
  bits<5> rt = { ?, ?, ?, ?, ? };
  bits<6> Opcode = { 0, 0, 0, 0, 0, 0 };

  bits<64> TSFlags = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };
  Format Form = FrmR;
  bits<4> FormBits = { 0, 0, 0, 1 };
  bit isCTI = 0;
  bit hasForbiddenSlot = 0;
  bit hasFCCRegOperand = 0;
}

class ADD_FM<bits<7> opcode, bits<7> funct7, bits<3> funct3> {
   bits<5> rs2;
   bits<5> rs1;
   bits<5> rd;
   // bits<7> opcode = 0b0110011;

   bits<32> Inst;

   let Inst{31-25} = funct7;
   let Inst{24-20} = rs2;
   let Inst{19-15} = rs1;
   let Inst{14-12} = funct3;
   let Inst{11-7} = rd;
   let Inst{6-0}= opcode;
}

class StaticTemplate{

}


class ADD: Mipsxxxx

class ADD: ADDTemplate
class ADD: Mipsxxx, StaticTemplate<xxx, xxx>
class ADD: Mipsxx, StaicTempalte<xxx, xxx>


class InstructionEncoding {
  // Size of encoded instruction.
  int Size;

  // The "namespace" in which this instruction exists, on targets like ARM
  // which multiple ISA namespaces exist.
  string DecoderNamespace = "";

  // List of predicates which will be turned into isel matching code.
  list<Predicate> Predicates = [];

  string DecoderMethod = "";

  // Is the instruction decoder method able to completely determine if the
  // given instruction is valid or not. If the TableGen definition of the
  // instruction specifies bitpattern A??B where A and B are static bits, the
  // hasCompleteDecoder flag says whether the decoder method fully handles the
  // ?? space, i.e. if it is a final arbiter for the instruction validity.
  // If not then the decoder attempts to continue decoding when the decoder
  // method fails.
  //
  // This allows to handle situations where the encoding is not fully
  // orthogonal. Example:
  // * InstA with bitpattern 0b0000????,
  // * InstB with bitpattern 0b000000?? but the associated decoder method
  //   DecodeInstB() returns Fail when ?? is 0b00 or 0b11.
  //
  // The decoder tries to decode a bitpattern that matches both InstA and
  // InstB bitpatterns first as InstB (because it is the most specific
  // encoding). In the default case (hasCompleteDecoder = 1), when
  // DecodeInstB() returns Fail the bitpattern gets rejected. By setting
  // hasCompleteDecoder = 0 in InstB, the decoder is informed that
  // DecodeInstB() is not able to determine if all possible values of ?? are
  // valid or not. If DecodeInstB() returns Fail the decoder will attempt to
  // decode the bitpattern as InstA too.
  bit hasCompleteDecoder = true;
}

class Instruction : InstructionEncoding {
  string Namespace = "";

  dag OutOperandList;       // An dag containing the MI def operand list.
  dag InOperandList;        // An dag containing the MI use operand list.
  string AsmString = "";    // The .s format to print the instruction with.

  // Allows specifying a canonical InstructionEncoding by HwMode. If non-empty,
  // the Inst member of this Instruction is ignored.
  EncodingByHwMode EncodingInfos;

  // Pattern - Set to the DAG pattern for this instruction, if we know of one,
  // otherwise, uninitialized.
  list<dag> Pattern;

  // The follow state will eventually be inferred automatically from the
  // instruction pattern.

  list<Register> Uses = []; // Default to using no non-operand registers
  list<Register> Defs = []; // Default to modifying no non-operand registers

  // Predicates - List of predicates which will be turned into isel matching
  // code.
  list<Predicate> Predicates = [];

  // Size - Size of encoded instruction, or zero if the size cannot be determined
  // from the opcode.
  int Size = 0;

  // Code size, for instruction selection.
  // FIXME: What does this actually mean?
  int CodeSize = 0;

  // Added complexity passed onto matching pattern.
  int AddedComplexity  = 0;

  // Indicates if this is a pre-isel opcode that should be
  // legalized/regbankselected/selected.
  bit isPreISelOpcode = false;

  // These bits capture information about the high-level semantics of the
  // instruction.
  bit isReturn     = false;     // Is this instruction a return instruction?
  bit isBranch     = false;     // Is this instruction a branch instruction?
  bit isEHScopeReturn = false;  // Does this instruction end an EH scope?
  bit isIndirectBranch = false; // Is this instruction an indirect branch?
  bit isCompare    = false;     // Is this instruction a comparison instruction?
  bit isMoveImm    = false;     // Is this instruction a move immediate instruction?
  bit isMoveReg    = false;     // Is this instruction a move register instruction?
  bit isBitcast    = false;     // Is this instruction a bitcast instruction?
  bit isSelect     = false;     // Is this instruction a select instruction?
  bit isBarrier    = false;     // Can control flow fall through this instruction?
  bit isCall       = false;     // Is this instruction a call instruction?
  bit isAdd        = false;     // Is this instruction an add instruction?
  bit isTrap       = false;     // Is this instruction a trap instruction?
  bit canFoldAsLoad = false;    // Can this be folded as a simple memory operand?
  bit mayLoad      = ?;         // Is it possible for this inst to read memory?
  bit mayStore     = ?;         // Is it possible for this inst to write memory?
  bit mayRaiseFPException = false; // Can this raise a floating-point exception?
  bit isConvertibleToThreeAddress = false;  // Can this 2-addr instruction promote?
  bit isCommutable = false;     // Is this 3 operand instruction commutable?
  bit isTerminator = false;     // Is this part of the terminator for a basic block?
  bit isReMaterializable = false; // Is this instruction re-materializable?
  bit isPredicable = false;     // 1 means this instruction is predicable
                                // even if it does not have any operand
                                // tablegen can identify as a predicate
  bit isUnpredicable = false;   // 1 means this instruction is not predicable
                                // even if it _does_ have a predicate operand
  bit hasDelaySlot = false;     // Does this instruction have an delay slot?
  bit usesCustomInserter = false; // Pseudo instr needing special help.
  bit hasPostISelHook = false;  // To be *adjusted* after isel by target hook.
  bit hasCtrlDep   = false;     // Does this instruction r/w ctrl-flow chains?
  bit isNotDuplicable = false;  // Is it unsafe to duplicate this instruction?
  bit isConvergent = false;     // Is this instruction convergent?
  bit isAuthenticated = false;  // Does this instruction authenticate a pointer?
  bit isAsCheapAsAMove = false; // As cheap (or cheaper) than a move instruction.
  bit hasExtraSrcRegAllocReq = false; // Sources have special regalloc requirement?
  bit hasExtraDefRegAllocReq = false; // Defs have special regalloc requirement?
  bit isRegSequence = false;    // Is this instruction a kind of reg sequence?
                                // If so, make sure to override
                                // TargetInstrInfo::getRegSequenceLikeInputs.
  bit isPseudo     = false;     // Is this instruction a pseudo-instruction?
                                // If so, won't have encoding information for
                                // the [MC]CodeEmitter stuff.
  bit isExtractSubreg = false;  // Is this instruction a kind of extract subreg?
                                // If so, make sure to override
                                // TargetInstrInfo::getExtractSubregLikeInputs.
  bit isInsertSubreg = false;   // Is this instruction a kind of insert subreg?
                                // If so, make sure to override
                                // TargetInstrInfo::getInsertSubregLikeInputs.
  bit variadicOpsAreDefs = false; // Are variadic operands definitions?

  // Does the instruction have side effects that are not captured by any
  // operands of the instruction or other flags?
  bit hasSideEffects = ?;

  // Is this instruction a "real" instruction (with a distinct machine
  // encoding), or is it a pseudo instruction used for codegen modeling
  // purposes.
  // FIXME: For now this is distinct from isPseudo, above, as code-gen-only
  // instructions can (and often do) still have encoding information
  // associated with them. Once we've migrated all of them over to true
  // pseudo-instructions that are lowered to real instructions prior to
  // the printer/emitter, we can remove this attribute and just use isPseudo.
  //
  // The intended use is:
  // isPseudo: Does not have encoding information and should be expanded,
  //   at the latest, during lowering to MCInst.
  //
  // isCodeGenOnly: Does have encoding information and can go through to the
  //   CodeEmitter unchanged, but duplicates a canonical instruction
  //   definition's encoding and should be ignored when constructing the
  //   assembler match tables.
  bit isCodeGenOnly = false;

  // Is this instruction a pseudo instruction for use by the assembler parser.
  bit isAsmParserOnly = false;

  // This instruction is not expected to be queried for scheduling latencies
  // and therefore needs no scheduling information even for a complete
  // scheduling model.
  bit hasNoSchedulingInfo = false;

  InstrItinClass Itinerary = NoItinerary;// Execution steps used for scheduling.

  // Scheduling information from TargetSchedule.td.
  list<SchedReadWrite> SchedRW;

  string Constraints = "";  // OperandConstraint, e.g. $src = $dst.

  /// DisableEncoding - List of operand names (e.g. "$op1,$op2") that should not
  /// be encoded into the output machineinstr.
  string DisableEncoding = "";

  string PostEncoderMethod = "";

  /// Target-specific flags. This becomes the TSFlags field in TargetInstrDesc.
  bits<64> TSFlags = 0;

  ///@name Assembler Parser Support
  ///@{

  string AsmMatchConverter = "";

  /// TwoOperandAliasConstraint - Enable TableGen to auto-generate a
  /// two-operand matcher inst-alias for a three operand instruction.
  /// For example, the arm instruction "add r3, r3, r5" can be written
  /// as "add r3, r5". The constraint is of the same form as a tied-operand
  /// constraint. For example, "$Rn = $Rd".
  string TwoOperandAliasConstraint = "";

  /// Assembler variant name to use for this instruction. If specified then
  /// instruction will be presented only in MatchTable for this variant. If
  /// not specified then assembler variants will be determined based on
  /// AsmString
  string AsmVariantName = "";

  ///@}

  /// UseNamedOperandTable - If set, the operand indices of this instruction
  /// can be queried via the getNamedOperandIdx() function which is generated
  /// by TableGen.
  bit UseNamedOperandTable = false;

  /// Should FastISel ignore this instruction. For certain ISAs, they have
  /// instructions which map to the same ISD Opcode, value type operands and
  /// instruction selection predicates. FastISel cannot handle such cases, but
  /// SelectionDAG can.
  bit FastISelShouldIgnore = false;
}
