/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Simple tree data structure used by tree_rnn.cc.

#ifndef TENSORFLOW_FOLD_LLGTM_EXAMPLES_PARSETREE_H_
#define TENSORFLOW_FOLD_LLGTM_EXAMPLES_PARSETREE_H_

#include <initializer_list>
#include <memory>
#include <string>

#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "tensorflow_fold/llgtm/llgtm.h"

namespace llgtm {
namespace parse_trees {

class LeafNode;
class PhraseNode;

// Base class for nodes in the parse tree.
// Trees are an algebraic data type:
// data TreeNode = LeafNode string | ParseNode TreeNode TreeNode
// [https://en.wikipedia.org/wiki/Algebraic_data_type]
class TreeNode {
 public:
  TreeNode() {}
  TreeNode(const TreeNode&) = delete;
  virtual ~TreeNode() {}

  TreeNode& operator=(const TreeNode&) = delete;

  // Derived classes must override one of the following to return this.
  virtual LeafNode*   get_leaf()   { return nullptr; }
  virtual PhraseNode* get_phrase() { return nullptr; }

  // Pattern match on the type of the TreeNode.
  // Execute leaf_case(this) or phrase_case(this) depending on type.
  template<typename F1, typename F2>
  auto SwitchOnType(F1 leaf_case, F2 phrase_case)
      -> decltype(leaf_case(static_cast<LeafNode*>(nullptr))) {
    if (auto* leaf = get_leaf()) {
      return leaf_case(leaf);
    } else {
      auto* phrase = get_phrase();
      CHECK(phrase != nullptr);
      return phrase_case(phrase);
    }
  }
};

// A leaf (terminal) node in the parse tree, which contains a single word.
class LeafNode : public TreeNode {
 public:
  explicit LeafNode(string word) : word_(std::move(word)) {}
  ~LeafNode() override {}

  absl::string_view word() const { return word_; }

  LeafNode* get_leaf() override { return this; }

 private:
  const string word_;
};

// A phrase (non-terminal) in the parse tree, which contains sub-phrases.
class PhraseNode : public TreeNode {
 public:
  using NodeType = std::unique_ptr<TreeNode>;

  PhraseNode() = delete;
  ~PhraseNode() override {}

  explicit PhraseNode(NodeType a) {
    sub_nodes_.emplace_back(std::move(a));
  }

  PhraseNode(NodeType a, NodeType b) {
    sub_nodes_.emplace_back(std::move(a));
    sub_nodes_.emplace_back(std::move(b));
  }

  PhraseNode(NodeType a, NodeType b, NodeType c) {
    // std::initializer_list doesn't work with move-only types, so we have
    // to do this the hard way.
    sub_nodes_.emplace_back(std::move(a));
    sub_nodes_.emplace_back(std::move(b));
    sub_nodes_.emplace_back(std::move(c));
  }

  const std::vector<NodeType>& sub_nodes() const { return sub_nodes_; }

  PhraseNode* get_phrase() override { return this; }

 private:
  std::vector<NodeType> sub_nodes_;
};

// Creates a new LeafNode.
inline std::unique_ptr<TreeNode> Leaf(string str) {
  return absl::make_unique<LeafNode>(std::move(str));
}

// Creates a new PhraseNode.
template <class... Args>
inline std::unique_ptr<TreeNode> Phrase(Args... args) {
  return absl::make_unique<PhraseNode>(std::move(args)...);
}

}  // namespace parse_trees
}  // namespace llgtm

#endif  // TENSORFLOW_FOLD_LLGTM_EXAMPLES_PARSETREE_H_
