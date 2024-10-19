MLIR Playground
===============

This is a simple standalone program where you can manually construct some MLIR.
Then you can watch as it is converted to [LLVM-dialect][1] MLIR, then look at
the LLVM IR it becomes, and run the JIT'd code.

Getting Started
---------------

I have only tested this on a Linux machine. If you are on Windows, I would
suggest WSL. If you are on macOS, please refer to [this comic][2].

1. Install LLVM 19.1.2 and make sure `llvm-config` is in your `$PATH` and
   `$MLIR_DIR` is set to `$YOUR_LLVM_INSTALL_DIR/lib/cmake/mlir/`. If I sent
   this repository to you, you probably have already done this.
2. Inside a cloned copy of this repository, do the classic:
   ```
   $ mkdir build && cd build
   $ cmake -G Ninja ..
   $ ninja
   ```
3. Then, if you are still inside the `build/` directory you made above, you can
   run `./mlir-playground`. You should see some output like this:
   ```llvm
   Initial MLIR:
   =============
   module {
     func.func @calculate_sum(%arg0: i32, %arg1: i32) -> i32 {
       %0 = arith.addi %arg0, %arg1 : i32
       return %0 : i32
     }
   }

   LLVM-dialect MLIR:
   ==================
   module {
     llvm.func @calculate_sum(%arg0: i32, %arg1: i32) -> i32 {
       %0 = llvm.add %arg0, %arg1 : i32
       llvm.return %0 : i32
     }
   }

   LLVM IR:
   ========
   ; ModuleID = 'LLVMDialectModule'
   source_filename = "LLVMDialectModule"
   target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
   target triple = "x86_64-unknown-linux-gnu"

   define i32 @calculate_sum(i32 %0, i32 %1) {
     %3 = add i32 %0, %1
     ret i32 %3
   }

   define void @_mlir_calculate_sum(ptr %0) {
     %2 = getelementptr ptr, ptr %0, i64 0
     %3 = load ptr, ptr %2, align 8
     %4 = load i32, ptr %3, align 4
     %5 = getelementptr ptr, ptr %0, i64 1
     %6 = load ptr, ptr %5, align 8
     %7 = load i32, ptr %6, align 4
     %8 = call i32 @calculate_sum(i32 %4, i32 %7)
     %9 = getelementptr ptr, ptr %0, i64 2
     %10 = load ptr, ptr %9, align 8
     store i32 %8, ptr %10, align 4
     ret void
   }

   Execution Result:
   =================
   7
   ```
4. Go into `createExampleIR()` and generate some different MLIR and see what
   happens. For example, why not [an `scf.if` op?][3]

[1]: https://mlir.llvm.org/docs/Dialects/LLVM/
[2]: https://junk.ausb.in/memes/heres-a-nickel.png
[3]: https://mlir.llvm.org/docs/Dialects/SCFDialect/#scfif-scfifop
