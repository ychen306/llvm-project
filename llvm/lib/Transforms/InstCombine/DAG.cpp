#include "DAG.h"
#include "llvm/Support/ErrorHandling.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/EHPersonalities.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/TargetFolder.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Pass.h"
#include "llvm/Support/CBindingWrapping.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugCounter.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/InstCombine/InstCombineWorklist.h"
#include "llvm/Transforms/Utils/Local.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

using namespace llvm;

bool DAG::canAnalyze(User *U) {
  auto *I = dyn_cast<Instruction>(U);
  if (!I)
    return false;
  if (isa<InsertElementInst>(I) ||
      isa<ExtractElementInst>(I) ||
      isa<ShuffleVectorInst>(I))
    return true;

  switch (I->getOpcode()) {
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
  case Instruction::SDiv:
  case Instruction::SRem:
  case Instruction::UDiv:
  case Instruction::URem:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
  case Instruction::SExt:
  case Instruction::ZExt:
  case Instruction::Trunc:
  case Instruction::PtrToInt:
  case Instruction::BitCast:
  case Instruction::ICmp:
  case Instruction::Select:
  case Instruction::FAdd:
  case Instruction::FSub:
  case Instruction::FMul:
  case Instruction::FDiv:
  case Instruction::FRem:
  case Instruction::FCmp:
    return true;
  default:
    return false;
  }
}

static EquivalenceClasses<Value *> computeConnectedDAGs(Function &F) {
  EquivalenceClasses<Value *> ConnectedDAGs;

  std::vector<User *> Worklist;
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (!isInstructionTriviallyDead(&I))
        Worklist.push_back(&I);
    }
  }

  DenseSet<User *> Visited;
  while (!Worklist.empty()) {
    User *U = Worklist.back();
    Worklist.pop_back();

    if (!Visited.insert(U).second)
      continue;

    Value *UseLeader = nullptr;
    if (DAG::canAnalyze(U))
      UseLeader = ConnectedDAGs.getOrInsertLeaderValue(U);

    for (Value *Op : U->operand_values()) {
      User *Used = dyn_cast<User>(Op);
      if (!Used)
        continue;

      Worklist.push_back(Used);

      if (UseLeader && DAG::canAnalyze(Used)) {
        auto *UsedLeader = ConnectedDAGs.getOrInsertLeaderValue(Used);
        ConnectedDAGs.unionSets(UseLeader, UsedLeader);
      }
    }
  }

  return ConnectedDAGs;
}

static DAG::Opcode getOpcode(const Value *V) {
  auto *I = dyn_cast<Instruction>(V);
  if (!I)
    return DAG::Opaque;
  if (isa<InsertElementInst>(I))
    return DAG::InsertElement;
  if (isa<ExtractElementInst>(I))
    return DAG::ExtractElement;
  if (isa<ShuffleVectorInst>(I))
    return DAG::ShuffleVector;

  switch (I->getOpcode()) {
  case Instruction::Add:
    return DAG::Add;
  case Instruction::Sub:
    return DAG::Sub;
  case Instruction::Mul:
    return DAG::Mul;
  case Instruction::SDiv:
    return DAG::SDiv;
  case Instruction::SRem:
    return DAG::SRem;
  case Instruction::UDiv:
    return DAG::UDiv;
  case Instruction::URem:
    return DAG::URem;
  case Instruction::Shl:
    return DAG::Shl;
  case Instruction::LShr:
    return DAG::LShr;
  case Instruction::AShr:
    return DAG::AShr;
  case Instruction::And:
    return DAG::And;
  case Instruction::Or:
    return DAG::Or;
  case Instruction::Xor:
    return DAG::Xor;
  case Instruction::SExt:
    return DAG::SExt;
  case Instruction::ZExt:
    return DAG::ZExt;
  case Instruction::Trunc:
    return DAG::Trunc;
  case Instruction::PtrToInt:
    return DAG::PtrToInt;
  case Instruction::BitCast:
    return DAG::BitCast;
  case Instruction::ICmp:
    switch (cast<ICmpInst>(I)->getPredicate()) {
    default:
      llvm_unreachable("should not reach here");

    case CmpInst::ICMP_EQ:
      return DAG::Eq;
    case CmpInst::ICMP_NE:
      return DAG::Ne;
    case CmpInst::ICMP_UGT:
      return DAG::Ugt;
    case CmpInst::ICMP_UGE:
      return DAG::Uge;
    case CmpInst::ICMP_ULT:
      return DAG::Ult;
    case CmpInst::ICMP_ULE:
      return DAG::Ule;
    case CmpInst::ICMP_SGT:
      return DAG::Sgt;
    case CmpInst::ICMP_SGE:
      return DAG::Sge;
    case CmpInst::ICMP_SLT:
      return DAG::Slt;
    case CmpInst::ICMP_SLE:
      return DAG::Sle;
    }
  case Instruction::FAdd:
    return DAG::FAdd;
  case Instruction::FSub:
    return DAG::FSub;
  case Instruction::FMul:
    return DAG::FMul;
  case Instruction::FDiv:
    return DAG::FDiv;
  case Instruction::FRem:
    return DAG::FRem;
  case Instruction::FCmp:
    switch (cast<FCmpInst>(I)->getPredicate()) {
    default:
      return DAG::Opaque;
    case CmpInst::FCMP_OEQ:
      return DAG::Foeq;
    case CmpInst::FCMP_ONE:
      return DAG::Fone;
    case CmpInst::FCMP_OGT:
      return DAG::Fogt;
    case CmpInst::FCMP_OGE:
      return DAG::Foge;
    case CmpInst::FCMP_OLT:
      return DAG::Folt;
    case CmpInst::FCMP_OLE:
      return DAG::Fole;
    case CmpInst::FCMP_UEQ:
      return DAG::Fueq;
    case CmpInst::FCMP_UNE:
      return DAG::Fune;
    case CmpInst::FCMP_UGT:
      return DAG::Fugt;
    case CmpInst::FCMP_UGE:
      return DAG::Fuge;
    case CmpInst::FCMP_ULT:
      return DAG::Fult;
    case CmpInst::FCMP_ULE:
      return DAG::Fule;
    case CmpInst::FCMP_TRUE:
      return DAG::Ftrue;
    }
  case Instruction::Select:
    return DAG::Select;
  default:
    return DAG::Opaque;
  }
}

static unsigned getBitWidth(const Value *V, Function &F) {
  Module *M = F.getParent();
  assert(M && "Function doesn't have module");
  auto &Ctx = F.getContext();
  auto &DL = M->getDataLayout();

  Type *Ty = V->getType();
  if (auto *VecTy = dyn_cast<VectorType>(Ty)) {
    Ty = VecTy->getElementType();
  }

  if (!Ty->isIntegerTy() && !Ty->isPointerTy() && !Ty->isFloatTy() && !Ty->isDoubleTy())
    return 0;

  if (auto *PtrTy = dyn_cast<PointerType>(Ty))
    Ty = DL.getIntPtrType(Ctx, PtrTy->getAddressSpace());

  if (Ty->isFloatTy())
    return 32;

  if (Ty->isDoubleTy())
    return 64;

  return cast<IntegerType>(Ty)->getBitWidth();
}

static unsigned getVectorWidth(const Value *V) {
  Type *Ty = V->getType();
  auto *VecTy = dyn_cast<VectorType>(Ty);
  if (VecTy) {
    return VecTy->getVectorNumElements();;
  }
  return 1;
}

template <typename ElemTy>
static bool isConstantVector(Value *V) {
  Constant *C = dyn_cast<Constant>(V);
  if (!C)
    return false;

  if (!isa<VectorType>(C->getType()))
    return false;

  unsigned NumElts = V->getType()->getVectorNumElements();
  for (unsigned i = 0; i != NumElts; ++i) {
    Constant *CElt = C->getAggregateElement(i);
    if (!CElt || !isa<ElemTy>(CElt))
      return false;
  }
  return true;
}

DAGSet::DAGSet(Function &F) {
  ConnectedECs = computeConnectedDAGs(F);

  // create an empty DAG for each EC
  for (auto ECI = ConnectedECs.begin(), ECE = ConnectedECs.end(); ECI != ECE;
       ++ECI) {
    if (!ECI->isLeader())
      continue;

    Value *Leader = *ConnectedECs.findLeader(ECI);
    DAGs.emplace_back();
    EC2DAGIdMap[Leader] = DAGs.size()-1;
  }

  assert(EC2DAGIdMap.size() == ConnectedECs.getNumClasses());

  auto CreateNode = [&](Value *V, DAG *G) -> DAG::NodeRef {
    DAG::Opcode Op;
    std::vector<uint64_t> Values;
    if (auto *CI = dyn_cast<ConstantInt>(V)) {
      Op = DAG::Constant;
      Values.push_back(CI->getLimitedValue());
    } else if (auto *C = dyn_cast<ConstantFP>(V)) {
      Op = DAG::FConstant;
      if (C->isExactlyValue(0.0))
        Values.push_back(0);
      else if (C->isExactlyValue(1.0))
        Values.push_back(1);
      else
        Op = DAG::Opaque;
    } else if (isConstantVector<ConstantInt>(V)) {
      Op = DAG::Constant;
      auto *CV = cast<Constant>(V);
      unsigned VWidth = CV->getType()->getVectorNumElements();
      for (unsigned i = 0; i < VWidth; i++) {
        auto *ElemVal = cast<ConstantInt>(CV->getAggregateElement(i));
        Values.push_back(ElemVal->getLimitedValue());
      }
    } else if (isConstantVector<ConstantFP>(V)) {
      // we only recognize floating point literal of value 1.0 and 0.0
      Op = DAG::FConstant;
      auto *CV = cast<Constant>(V);
      unsigned VWidth = CV->getType()->getVectorNumElements();
      for (unsigned i = 0; i < VWidth; i++) {
        auto *ElemVal = cast<ConstantFP>(CV->getAggregateElement(i));
        if (ElemVal->isExactlyValue(1.0)) {
          Values.push_back(1);
        } else if (ElemVal->isExactlyValue(0.0)) {
          Values.push_back(0);
        } else {
          // bail... this is getting too complicated
          Op = DAG::Opaque;
          Values.clear();
          break;
        }
      }
    } else {
      Op = getOpcode(V);
    }

    DAG::NodeRef N = G->createNode(Op, getBitWidth(V, F), getVectorWidth(V), Values);
    Value2NodeMap[{V, G}] = N;
    return N;
  };

  auto GetOrCreateNode = [&](Value *V, DAG *G) -> DAG::NodeRef {
    if (!Value2NodeMap.count({V, G})) {
      auto N = CreateNode(V, G);

      Instruction *I = dyn_cast<Instruction>(V);
      if (I && !DAG::canAnalyze(I))
        I = nullptr;

      assert((int)G->Insts.size() == N.NodeId);
      G->Insts.emplace_back(I);
    }

    return Value2NodeMap.lookup({V, G});
  };

  for (auto &BB : F)
    for (auto &I : BB) {
      if (!DAG::canAnalyze(&I))
        continue;
      if (isInstructionTriviallyDead(&I))
        continue;

      DAG *G = getDAGForInst(&I);
      auto N = GetOrCreateNode(&I, G);

      assert(I.getNumOperands() <= 3);
      unsigned i = 0;
      for (Value *Op : I.operand_values()) {
        DAG::NodeRef Child = GetOrCreateNode(Op, G);

        if (i == 0)
          N->O1 = Child;
        else if (i == 1)
          N->O2 = Child;
        else // if (i == 2)
          N->O3 = Child;

        i++;
      }
    }
}

DAG *DAGSet::getDAGForInst(Instruction *I) {
  // find out which DAG this instruction belongs to
  auto *Leader = ConnectedECs.getLeaderValue(I);
  return &DAGs[EC2DAGIdMap.at(Leader)];
}

const DAG *DAGSet::getDAGForInst(Instruction *I) const {
  // find out which DAG this instruction belongs to
  auto *Leader = ConnectedECs.getLeaderValue(I);
  return &DAGs[EC2DAGIdMap.at(Leader)];
}

DAG::NodeRef DAGSet::getNodeForInst(llvm::Instruction *I) const {
  // must be a new instruction
  if (ConnectedECs.findValue(I) == ConnectedECs.end())
    return DAG::NodeRef();

  return Value2NodeMap.lookup({I, const_cast<DAG *>(getDAGForInst(I))});
}

StringRef getOpName(DAG::Opcode Op) {
  switch(Op) {
    case DAG::Opaque: return "Opaque";
    case DAG::Add: return "Add";
    case DAG::Sub: return "Sub";
    case DAG::Mul: return "Mul";
    case DAG::SDiv: return "SDiv";
    case DAG::SRem: return "SRem";
    case DAG::UDiv: return "UDiv";
    case DAG::URem: return "URem";
    case DAG::Shl: return "Shl";
    case DAG::LShr: return "LShr";
    case DAG::AShr: return "AShr";
    case DAG::And: return "And";
    case DAG::Or: return "Or";
    case DAG::Xor: return "Xor";
    case DAG::SExt: return "SExt";
    case DAG::ZExt: return "ZExt";
    case DAG::Trunc: return "Trunc";
    case DAG::PtrToInt: return "PtrToInt";
    case DAG::BitCast: return "BitCast";
    case DAG::Eq: return "Eq";
    case DAG::Ne: return "Ne";
    case DAG::Ugt: return "Ugt";
    case DAG::Uge: return "Uge";
    case DAG::Ult: return "Ult";
    case DAG::Ule: return "Ule";
    case DAG::Sgt: return "Sgt";
    case DAG::Sge: return "Sge";
    case DAG::Slt: return "Slt";
    case DAG::Sle: return "Sle";
    case DAG::Select: return "Select";

    case DAG::FAdd: return "FAdd";
    case DAG::FSub: return "FSub";
    case DAG::FMul: return "FMul";
    case DAG::FDiv: return "FDiv";
    case DAG::FRem: return "FRem";
    case DAG::Foeq: return "Foeq";
    case DAG::Fone: return "Fone";
    case DAG::Fogt: return "Fogt";
    case DAG::Foge: return "Foge";
    case DAG::Folt: return "Folt";
    case DAG::Fole: return "Fole";
    case DAG::Fueq: return "Fueq";
    case DAG::Fune: return "Fune";
    case DAG::Fugt: return "Fugt";
    case DAG::Fuge: return "Fuge";
    case DAG::Fult: return "Fult";
    case DAG::Fule: return "Fule";
    case DAG::Ftrue: return "Ftrue";

    case DAG::InsertElement: return "InsertElement";
    case DAG::ExtractElement: return "ExtractElement";
    case DAG::ShuffleVector: return "ShuffleVector";

    case DAG::Constant: return "Constant";
    case DAG::FConstant: return "FConstant";
  }
  llvm_unreachable("Unexhaustive handling of instructions");
}

void DAG::dump(raw_ostream &OS) const {
  unsigned NodeId = 0;
  for (const DAG::Node &N : Nodes) {
    OS << NodeId << ","
      << getOpName(N.Op) << ","
      << N.BitWidth << ","
      << N.VectorWidth << ","
      << N.O1.NodeId << ","
      << N.O2.NodeId << ","
      << N.O3.NodeId << ":";
    unsigned i = 0;
    for (uint64_t V : N.Values) {
      OS << V;
      if (++i < N.Values.size())
        OS << ",";
    }
    OS << "\n";

    NodeId++;
  }
}
