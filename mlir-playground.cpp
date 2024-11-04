#include "llvm/Support/TargetSelect.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/DialectConversion.h"

namespace {

// This configures MLIR with the dialects we want to use. Not too exciting
void contextSetup(mlir::MLIRContext &ctx) {
    ctx.getOrLoadDialect<mlir::func::FuncDialect>();
    ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
    ctx.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    ctx.getOrLoadDialect<mlir::scf::SCFDialect>();
	ctx.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
    mlir::registerBuiltinDialectTranslation(ctx);
    mlir::registerLLVMDialectTranslation(ctx);
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
}

// You can modify this to create your own IR. Right now it is just a function
// that returns the sum of both its 32-bit integer operands.
void createExampleIR(mlir::ModuleOp module) {
    mlir::OpBuilder builder(module.getContext());
    builder.setInsertionPointToEnd(module.getBody());

    mlir::Location loc = builder.getUnknownLoc();
    mlir::FunctionType func_type = builder.getFunctionType(
        {builder.getI32Type(), builder.getI32Type()}, builder.getI32Type());
    mlir::func::FuncOp func = builder.create<mlir::func::FuncOp>(
        loc, "calculate_sum", func_type);
    // This may be useful if you want to pass a memref into the function. (It
    // also seems to be useful for ExecutionEngine::invoke(); see the comment
    // below.) But none of that is helpful at the moment. See also:
    // https://mlir.llvm.org/docs/TargetLLVMIR/#c-compatible-wrapper-emission
    //func->setDiscardableAttr(
    //    mlir::LLVM::LLVMDialect::getEmitCWrapperAttrName(),
    //    builder.getUnitAttr());

    llvm::SmallVector<mlir::Location> arg_locs(func_type.getNumInputs(), loc);
    // Sets insert point to end of this block
    mlir::Block *entryBlock = builder.createBlock(
        &func.getBody(), {}, func_type.getInputs(), arg_locs);
    mlir::Value sum =
        builder.create<mlir::arith::AddIOp>(
           loc, entryBlock->getArgument(0), entryBlock->getArgument(1));

    //Personal Implementation of a sample if statement
    mlir::Value comparison = builder.create<mlir::arith::CmpIOp>(
		loc, mlir::arith::CmpIPredicate::eq, entryBlock->getArgument(0), entryBlock->getArgument(1)
    );


	mlir::scf::IfOp ifStatement = builder.create<mlir::scf::IfOp>(loc, builder.getI32Type(), comparison, true);
	mlir::Block* thenBlockPtr = ifStatement.thenBlock();
	mlir::Block* elseBlockPtr = ifStatement.elseBlock();
	
	builder.setInsertionPointToStart(thenBlockPtr);
	mlir::Value ifVal = builder.create<mlir::arith::AddIOp>(loc, entryBlock->getArgument(0), entryBlock->getArgument(1));
	builder.create<mlir::scf::YieldOp>(loc, ifVal);

	builder.setInsertionPointToStart(elseBlockPtr);
    mlir::Value elseVal = builder.create<mlir::arith::SubIOp>(loc, entryBlock->getArgument(0), entryBlock->getArgument(1));
	builder.create<mlir::scf::YieldOp>(loc, elseVal);
	
	// sets insert point to end of entryBlock
	builder.setInsertionPointToEnd(entryBlock);

    builder.create<mlir::func::ReturnOp>(loc, ifStatement.getResults());
}

// Converts the module in-place to the LLVM dialect of MLIR. (The LLVM dialect
// of MLIR models LLVM IR inside MLIR, but isn't actually LLVM IR itself. For
// example, there are still block arguments, which LLVM IR does not have.)
// Normally I would write this as an MLIR pass, but for simplicity, I'm just
// doing this in a function here.
mlir::LogicalResult lowerToLLVMIR(mlir::ModuleOp module) {
    mlir::LLVMTypeConverter type_converter(module.getContext());
    mlir::LLVMConversionTarget target(*module.getContext());
    target.addLegalOp<mlir::ModuleOp>();

    mlir::RewritePatternSet patterns(module.getContext());
    mlir::arith::populateArithToLLVMConversionPatterns(type_converter,
                                                       patterns);
    mlir::populateFuncToLLVMConversionPatterns(type_converter, patterns);
    mlir::populateSCFToControlFlowConversionPatterns(patterns);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(type_converter,
                                                          patterns);
	

    return mlir::applyFullConversion(module, target, std::move(patterns));
}

// Transforms LLVM-dialect MLIR to LLVM IR (not explicitly shown below, but it
// does happen) and then JIT-compiles that LLVM IR to native code and puts it
// in memory. Then we can get a function pointer to that code and call the
// function pointer just like any old function pointer to any old C function.
// I tried to write this function such that you won't need to modify it (hence
// the C++ template usage).
template<typename RetType, typename... ArgTypes>
mlir::FailureOr<RetType> invoke(mlir::ModuleOp module, std::string funcop_name,
                                ArgTypes... args) {
    auto transformer = [](llvm::Module *llvm_module) -> llvm::Error {
        llvm::errs() << "\nLLVM IR:\n"
                     << "========\n";
        llvm_module->dump();
        return llvm::Error::success();
    };
    mlir::ExecutionEngineOptions engine_options;
    engine_options.transformer = transformer;
    // This might be useful instead of our transformer above if you want to run
    // the -O3 pipeline on the LLVM IR. For now, we just print it out and leave
    // it as is.
    //auto opt_pipeline = mlir::makeOptimizingTransformer(
    //        /*optLevel=*/3, /*sizeLevel=*/0, /*targetMachine=*/nullptr);
    //engine_options.transformer = opt_pipeline;

    llvm::Expected<std::unique_ptr<mlir::ExecutionEngine>> maybe_engine =
        mlir::ExecutionEngine::create(module, engine_options);

    if (!maybe_engine) {
        llvm::errs() << "Failed to construct ExecutionEngine. "
                     << "Hopefully there are details on stderr.";
        return mlir::failure();
    }
    std::unique_ptr<mlir::ExecutionEngine> engine =
        std::move(maybe_engine.get());

    llvm::Expected<void *> maybe_func_addr = engine->lookup(funcop_name);
    if (!maybe_func_addr) {
        llvm::errs() << "Could not find address of function\n";
        return mlir::failure();
    }

    void *func_addr = *maybe_func_addr;

    RetType (*func_ptr)(ArgTypes...) =
        (RetType (*)(ArgTypes...))func_addr;

    return func_ptr(args...);

    // Here's another way to do the above. However, it requires the commented
    // setDiscardableAttr() on the FuncOp above, which bloats the IR. So do the
    // more interesting function pointer strategy above for now.
    //RetType res;
    //if (engine->invoke(funcop_name, args...,
    //                   mlir::ExecutionEngine::Result(res))) {
    //    llvm::errs() << "Invoking kernel failed";
    //    return mlir::failure();
    //}
    //return res;

}

} // namespace

int main(int argc, char **argv) {
    mlir::MLIRContext ctx;
    contextSetup(ctx);

    mlir::OwningOpRef<mlir::ModuleOp> module_ref =
        mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mlir::ModuleOp module = *module_ref;

    createExampleIR(module);
    if (mlir::failed(mlir::verify(module))) {
        llvm::errs() << "failed to verify module\n";
        return 1;
    }

    llvm::errs() << "Initial MLIR:\n"
                 << "=============\n";
    module->dump();

    if (mlir::failed(lowerToLLVMIR(module))) {
        llvm::errs() << "failed to lower to LLVM IR\n";
        return 1;
    }

    llvm::errs() << "Canonicolize MLIR:\n"
                 << "=============\n";
    module->dump();

    if (mlir::failed(lowerToLLVMIR(module))) {
	
        llvm::errs() << "failed to lower to LLVM IR\n";
        return 1;
    }


    llvm::errs() << "\nLLVM-dialect MLIR:\n"
                 << "==================\n";
    module->dump();

    mlir::FailureOr<uint32_t> invoke_result =
        invoke</*result type */ uint32_t,
               /*arg types */ uint32_t, uint32_t>(
            module, "calculate_sum", 3, 4);
    if (mlir::failed(invoke_result)) {
        llvm::errs() << "invocation failed!\n";
    }
    uint32_t result = *invoke_result;
    llvm::errs() << "\nExecution Result:\n"
                 << "=================\n"
                 << result << "\n";

    return 0;
}
