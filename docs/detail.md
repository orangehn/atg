### 1. Discussion on Limitations

Despite the re-usability, ATG is still incomplete to
generate files in languages except TableGen. Efforts aiming at C or C++ code are now undergoing for
a more comprehensive target support. Moreover, further efforts about extending ATG to a wider scope such
as the instruction scheduling in the compiler are undergoing. Currently, the evaluation of ATG still depends
on the available LLVM infrastructure such as instruction selection, and instruction emission. Therefore, ATG does not support
customized instructions with complex mode or properties(such as pattern, operand def) due to lack of relevant knowledge from existing
ISAs. However, this more intellectual auto-designing approach is hopeful to further lower the threshold for
compiler development fundamentally. 

In this project, we simply use ATG to automatically generate the instruction definition part 
as a prototype display. We've done several version updates, so the line of generated code(LOC) 
(and some data associated with it) may be slightly different from previous versions, which is acceptable. 
