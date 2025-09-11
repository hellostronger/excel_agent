"""Feature tree structure for hierarchical table representation (inspired by ST-Raptor)."""

import json
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

from ..utils.config import *


def serial(level_list: List[int]) -> str:
    """Create serial representation of level list."""
    s = ""
    for i in level_list:
        s += str(i) + "."
    return s[:-1]


@dataclass
class TreeNode:
    """Base tree node class."""
    value: Optional[str] = None
    children: List['TreeNode'] = field(default_factory=list)
    
    def add_child(self, child_node: 'TreeNode'):
        """Add a child node."""
        self.children.append(child_node)
    
    def remove_child(self, child_node: 'TreeNode'):
        """Remove a child node."""
        if child_node in self.children:
            self.children.remove(child_node)


@dataclass
class IndexNode(TreeNode):
    """Index node with body and classification information."""
    body: List[TreeNode] = field(default_factory=list)
    father: Optional['IndexNode'] = None
    
    # Attributes for leaf nodes (need build_split_info() processing)
    group_type: Optional[int] = None  # Classification type: discrete, continuous, embedding
    group_name_list: Optional[List[str]] = None  # All group name classifications
    group_id_list: Optional[List[int]] = None  # All group id collections
    name2id: Optional[Dict[str, int]] = None  # dict, name to id mapping
    id2name: Optional[Dict[int, str]] = None  # dict, id to name mapping
    
    def add_body_node(self, node: TreeNode):
        """Add a node to the body."""
        self.body.append(node)


@dataclass
class FeatureTree:
    """Feature tree for hierarchical table representation."""
    
    def __init__(self):
        self.root: Optional[IndexNode] = None
        self.table_id: Optional[str] = None
        self.file_path: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
        self.schema_info: Dict[str, Any] = {}
        self._all_values: Optional[List[str]] = None
        
    def set_root(self, root: IndexNode):
        """Set the root node."""
        self.root = root
    
    def set_metadata(self, metadata: Dict[str, Any]):
        """Set table metadata."""
        self.metadata = metadata
    
    def set_schema_info(self, schema: Dict[str, Any]):
        """Set schema information."""
        self.schema_info = schema
    
    def all_value_list(self) -> List[str]:
        """Get all values in the tree for embedding."""
        if self._all_values is not None:
            return self._all_values
            
        values = []
        
        def collect_values(node: TreeNode):
            if node.value is not None:
                values.append(str(node.value))
            
            if isinstance(node, IndexNode):
                for body_node in node.body:
                    collect_values(body_node)
            
            for child in node.children:
                collect_values(child)
        
        if self.root:
            collect_values(self.root)
        
        self._all_values = list(set(values))  # Remove duplicates
        return self._all_values
    
    def __index__(self) -> str:
        """Get index representation of the tree (schema)."""
        if not self.root:
            return \"Empty tree\"\n        \n        def build_index(node: IndexNode, level: int = 0) -> str:\n            indent = \"  \" * level\n            result = f\"{indent}{node.value or 'root'}\\n\"\n            \n            # Add body information\n            if node.body:\n                result += f\"{indent}  [Body: {len(node.body)} items]\\n\"\n            \n            # Add children\n            for child in node.children:\n                if isinstance(child, IndexNode):\n                    result += build_index(child, level + 1)\n                else:\n                    result += f\"{indent}  - {child.value}\\n\"\n            \n            return result\n        \n        return build_index(self.root)\n    \n    def __str__(self, levels: Optional[List[int]] = None) -> str:\n        \"\"\"Get string representation of the tree.\"\"\"\n        if not self.root:\n            return \"Empty FeatureTree\"\n        \n        def build_string(node: IndexNode, current_level: int = 0, path: List[int] = None) -> str:\n            if path is None:\n                path = []\n            \n            # Check if we should include this level\n            if levels and current_level not in levels:\n                result = \"\"\n                for i, child in enumerate(node.children):\n                    if isinstance(child, IndexNode):\n                        result += build_string(child, current_level + 1, path + [i])\n                return result\n            \n            indent = \"  \" * current_level\n            result = f\"{indent}[{serial(path)}] {node.value or 'ROOT'}\\n\"\n            \n            # Add body information if exists\n            if node.body:\n                for i, body_item in enumerate(node.body[:3]):  # Limit to first 3 items\n                    result += f\"{indent}  * {body_item.value}\\n\"\n                if len(node.body) > 3:\n                    result += f\"{indent}  * ... ({len(node.body) - 3} more items)\\n\"\n            \n            # Add children\n            for i, child in enumerate(node.children):\n                if isinstance(child, IndexNode):\n                    result += build_string(child, current_level + 1, path + [i])\n                else:\n                    result += f\"{indent}  - {child.value}\\n\"\n            \n            return result\n        \n        return build_string(self.root)\n    \n    def __json__(self) -> Dict[str, Any]:\n        \"\"\"Get JSON representation of the tree.\"\"\"\n        def node_to_dict(node: TreeNode) -> Dict[str, Any]:\n            node_dict = {\n                \"value\": node.value,\n                \"type\": type(node).__name__\n            }\n            \n            if isinstance(node, IndexNode):\n                node_dict.update({\n                    \"body\": [node_to_dict(body_node) for body_node in node.body],\n                    \"group_type\": node.group_type,\n                    \"group_name_list\": node.group_name_list,\n                    \"group_id_list\": node.group_id_list,\n                    \"name2id\": node.name2id,\n                    \"id2name\": node.id2name\n                })\n            \n            if node.children:\n                node_dict[\"children\"] = [node_to_dict(child) for child in node.children]\n            \n            return node_dict\n        \n        return {\n            \"table_id\": self.table_id,\n            \"file_path\": self.file_path,\n            \"metadata\": self.metadata,\n            \"schema_info\": self.schema_info,\n            \"root\": node_to_dict(self.root) if self.root else None\n        }\n    \n    @classmethod\n    def from_json(cls, json_data: Dict[str, Any]) -> 'FeatureTree':\n        \"\"\"Create FeatureTree from JSON data.\"\"\"\n        tree = cls()\n        tree.table_id = json_data.get(\"table_id\")\n        tree.file_path = json_data.get(\"file_path\")\n        tree.metadata = json_data.get(\"metadata\", {})\n        tree.schema_info = json_data.get(\"schema_info\", {})\n        \n        def dict_to_node(node_data: Dict[str, Any]) -> TreeNode:\n            if node_data[\"type\"] == \"IndexNode\":\n                node = IndexNode(\n                    value=node_data.get(\"value\"),\n                    group_type=node_data.get(\"group_type\"),\n                    group_name_list=node_data.get(\"group_name_list\"),\n                    group_id_list=node_data.get(\"group_id_list\"),\n                    name2id=node_data.get(\"name2id\"),\n                    id2name=node_data.get(\"id2name\")\n                )\n                \n                # Add body nodes\n                for body_data in node_data.get(\"body\", []):\n                    node.add_body_node(dict_to_node(body_data))\n                    \n            else:\n                node = TreeNode(value=node_data.get(\"value\"))\n            \n            # Add children\n            for child_data in node_data.get(\"children\", []):\n                child_node = dict_to_node(child_data)\n                if isinstance(child_node, IndexNode):\n                    child_node.father = node if isinstance(node, IndexNode) else None\n                node.add_child(child_node)\n            \n            return node\n        \n        if json_data.get(\"root\"):\n            tree.root = dict_to_node(json_data[\"root\"])\n        \n        return tree\n    \n    def find_nodes_by_value(self, target_value: str, exact_match: bool = True) -> List[TreeNode]:\n        \"\"\"Find nodes by value.\"\"\"\n        results = []\n        \n        def search_node(node: TreeNode):\n            if node.value:\n                if exact_match:\n                    if node.value == target_value:\n                        results.append(node)\n                else:\n                    if target_value.lower() in node.value.lower():\n                        results.append(node)\n            \n            if isinstance(node, IndexNode):\n                for body_node in node.body:\n                    search_node(body_node)\n            \n            for child in node.children:\n                search_node(child)\n        \n        if self.root:\n            search_node(self.root)\n        \n        return results\n    \n    def get_path_to_node(self, target_node: TreeNode) -> Optional[List[int]]:\n        \"\"\"Get path from root to target node.\"\"\"\n        def find_path(node: TreeNode, path: List[int]) -> Optional[List[int]]:\n            if node == target_node:\n                return path\n            \n            for i, child in enumerate(node.children):\n                result = find_path(child, path + [i])\n                if result is not None:\n                    return result\n            \n            return None\n        \n        if self.root:\n            return find_path(self.root, [])\n        return None\n    \n    def get_subtree_data(self, node: TreeNode, max_depth: int = 3) -> Dict[str, Any]:\n        \"\"\"Get subtree data for context-aware processing.\"\"\"\n        def collect_subtree(current_node: TreeNode, depth: int = 0) -> Dict[str, Any]:\n            if depth > max_depth:\n                return {\"value\": current_node.value, \"truncated\": True}\n            \n            node_data = {\"value\": current_node.value}\n            \n            if isinstance(current_node, IndexNode) and current_node.body:\n                node_data[\"body\"] = [body_node.value for body_node in current_node.body[:5]]  # Limit body items\n                if len(current_node.body) > 5:\n                    node_data[\"body_truncated\"] = len(current_node.body) - 5\n            \n            if current_node.children:\n                node_data[\"children\"] = [\n                    collect_subtree(child, depth + 1) for child in current_node.children[:10]  # Limit children\n                ]\n                if len(current_node.children) > 10:\n                    node_data[\"children_truncated\"] = len(current_node.children) - 10\n            \n            return node_data\n        \n        return collect_subtree(node)\n    \n    def get_statistics(self) -> Dict[str, Any]:\n        \"\"\"Get tree statistics.\"\"\"\n        stats = {\n            \"total_nodes\": 0,\n            \"index_nodes\": 0,\n            \"leaf_nodes\": 0,\n            \"max_depth\": 0,\n            \"total_values\": 0,\n            \"body_items\": 0\n        }\n        \n        def analyze_node(node: TreeNode, depth: int = 0):\n            stats[\"total_nodes\"] += 1\n            stats[\"max_depth\"] = max(stats[\"max_depth\"], depth)\n            \n            if node.value:\n                stats[\"total_values\"] += 1\n            \n            if isinstance(node, IndexNode):\n                stats[\"index_nodes\"] += 1\n                stats[\"body_items\"] += len(node.body)\n            \n            if not node.children:\n                stats[\"leaf_nodes\"] += 1\n            \n            for child in node.children:\n                analyze_node(child, depth + 1)\n        \n        if self.root:\n            analyze_node(self.root)\n        \n        return stats