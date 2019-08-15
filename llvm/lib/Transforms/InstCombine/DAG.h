#ifndef DAG_H
#define DAG_H

#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>

//
//  Library to represent LLVM expression dag.
//  We can build dags from LLVM IR and serialize these dags.
//
//'add': 'Instruction::Add',
//'sub': 'Instruction::Sub',
//'mul': 'Instruction::Mul',
//'sdiv': 'Instruction::SDiv',
//'srem': 'Instruction::SRem',
//'udiv': 'Instruction::UDiv',
//'urem': 'Instruction::URem',
//'shl':  'Instruction::Shl',
//'lshr': 'Instruction::LShr',
//'ashr': 'Instruction::AShr',
//'and':  'Instruction::And',
//'or':   'Instruction::Or',
//'xor':  'Instruction::Xor',
//'sext': 'Instruction::SExt',
//'zext': 'Instruction::ZExt',
//'trunc': 'Instruction::Trunc',
//'ptrtoint': 'Instruction::PtrToInt',
//'inttoptr': 'Instruction::IntToPtr',
//'bitcast': 'Instruction::BitCast',
//'icmp': 'Instruction::ICmp',
//'select': 'Instruction::Select',
//
// floating point ops:
// fadd,
// fsub,
// fmul,
// fdiv,
// frem
// 
// vector ops:
// extractelement,
// insertelement,
// shufflevector

class DAGSet; // set of expression dags within a function

// A single *connected* expression dag
class DAG {
  friend class DAGSet;

public:
  enum Opcode {
    // use this to represent expressions that we don't analyze, such as phi
    Opaque,

    Add,
    Sub,
    Mul,
    SDiv,
    SRem,
    UDiv,
    URem,
    Shl,
    LShr,
    AShr,
    And,
    Or,
    Xor,
    SExt,
    ZExt,
    Trunc,
    PtrToInt,
    BitCast,

    // Floating point arith
    FAdd,
    FSub,
    FMul,
    FDiv,
    FRem,

    // Floating point icmp
    Foeq,
    Fone,
    Fogt,
    Foge,
    Folt,
    Fole,
    Fueq,
    Fune,
    Fugt,
    Fuge,
    Fult,
    Fule,
    Ftrue,

    // ICmp
    Eq,
    Ne,
    Ugt,
    Uge,
    Ult,
    Ule,
    Sgt,
    Sge,
    Slt,
    Sle,

    Select,

    // vector insts
    InsertElement,
    ExtractElement,
    ShuffleVector,

    Constant,
    FConstant
  };

  struct Node;
  std::vector<Node> Nodes;

public:
  struct NodeRef {
    DAG *G;
    int NodeId;

    operator bool() const { return NodeId >= 0; }
    Node *operator->() const { return &G->Nodes[NodeId]; }

    NodeRef(DAG *G = nullptr, int NodeId = -1) : G(G), NodeId(NodeId) {}
    NodeRef &operator=(const NodeRef &) = default;
    NodeRef(const NodeRef &) = default;
  };

  // TODO: model flags such as nsw or exact
  // TODO: print out cost of instruction here
  struct Node {
    Opcode Op;
    unsigned BitWidth; // bitwidth of the result
    unsigned VectorWidth; // 1 if scalar otherwise ... duh
    std::vector<uint64_t> Values;    // in case this is a node for constant
    // operands of an instr, null if applicable
    NodeRef O1, O2, O3;

    Node(Opcode Op, unsigned BitWidth, unsigned VectorWidth, const std::vector<uint64_t> &Values, NodeRef O1, NodeRef O2,
         NodeRef O3)
        : Op(Op), BitWidth(BitWidth), VectorWidth(VectorWidth), Values(Values), O1(O1), O2(O2), O3(O3) {}
  };

private:
  NodeRef createNode(Opcode Op, unsigned BitWidth, unsigned VectorWidth, std::vector<uint64_t> Values = std::vector<uint64_t>(),
                     NodeRef O1 = NodeRef(), NodeRef O2 = NodeRef(),
                     NodeRef O3 = NodeRef()) {
    Nodes.push_back(Node(Op, BitWidth, VectorWidth, Values, O1, O2, O3));
    return NodeRef(this, Nodes.size() - 1);
  }

  std::vector<llvm::Instruction *> Insts;

public:
  void dump(llvm::raw_ostream &) const;
  static bool canAnalyze(llvm::User *U);
};

class DAGSet {
  std::vector<DAG> DAGs;

  // break values into equivalence classes based on which they are in
  llvm::EquivalenceClasses<llvm::Value *> ConnectedECs;

  // mapping inst -> to the dag it's used in
  std::map<llvm::Value *, unsigned> EC2DAGIdMap;

  DAG *getDAGForInst(llvm::Instruction *I);
  const DAG *getDAGForInst(llvm::Instruction *I) const;

  // an value might show up in two *distinct*a DAGs
  // due to presence of function arguments and instructions
  // such as phi and loads
  llvm::DenseMap<std::pair<llvm::Value *, DAG *>, DAG::NodeRef> Value2NodeMap;

public:
  typedef decltype(DAGs)::const_iterator iterator;

  iterator begin() const { return DAGs.begin(); }
  iterator end() const { return DAGs.end(); }

  llvm::Instruction *getInstForNode(DAG::NodeRef N) const {
    return N.G->Insts[N.NodeId];
  }

  DAG::NodeRef getNodeForInst(llvm::Instruction *) const;

  DAGSet(llvm::Function &F);
  DAGSet() = default;
  DAGSet &operator=(DAGSet &&) = default;
};

#endif // DAG_H
