import json 
import os
import fcntl
from typing import Dict, List, Any, Optional, Type, Union

from agent.base import BaseAgent
from agent.historystrategy import HistoryStrategy

from binagent.tools import ExecutableTool, FlexibleContext
DEFAULT_KB_FILE = "propagation_paths.jsonl"


FINDING_SCHEMA: Dict[str, Dict[str, Any]] = {
    "type": {
        "type": "string",
        "description": "Vulnerability CWE entries, multiple entries separated by commas. For example: 'CWE-120' (classic buffer overflow), 'CWE-78' (OS command injection)."
    },
    "identifier": {
        "type": "array", 
        "items": {"type": "string"},
        "description": "Symbol names of the medium when taint is transferred across components through named mediums."
    },
    "propagation": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Describes the complete propagation path from source to sink. The endpoint can be a dangerous function (sink) or an operation that transfers taint to other components (handoff). Each intermediate step should follow the format 'Step: [Relevant Code Snippet] --> [Explanation of the step]'. Example: [\"Source: Input from client socket received in function main\", \"Step: mov r0, r4 --> User input is moved to r0 at 0x401a10\", \"Sink: bl system --> Tainted data in r0 is passed to system() at 0x401b20\"]"
    },
    "reason": {
        "type": "string",
        "description": "Detailed explanation of your reasoning, supporting the conclusion."
    },
    "risk_score": {
        "type": "number",
        "description": "Risk score (0.0-10.0), evaluating the probability of this path becoming a real vulnerability."
    },
    "confidence": {
        "type": "number",
        "description": "Confidence score (0.0-10.0), evaluating the sufficiency of the evidence."
    }
}
FINDING_SCHEMA_REQUIRED_FIELDS: List[str] = ["type", "propagation", "risk_score", "confidence"]

class KnowledgeBaseMixin:
    def _initialize_kb(self, context: FlexibleContext):
        output_from_context = context.get("output")
        
        if output_from_context and isinstance(output_from_context, str):
            self.output = output_from_context
        else:
            raise ValueError("'output' not found in context or invalid.")

        if not os.path.exists(self.output):
            try:
                os.makedirs(self.output, exist_ok=True)
                print(f"Created output directory: {os.path.abspath(self.output)}")
            except OSError as e:
                print(f"Warning: Cannot create output directory '{self.output}': {e}. Will try to create KB file in current directory.")
                self.output = "."

        self.kb_file_path = os.path.join(self.output, DEFAULT_KB_FILE)
        
        kb_specific_dir = os.path.dirname(self.kb_file_path)
        if kb_specific_dir and not os.path.exists(kb_specific_dir):
            try:
                os.makedirs(kb_specific_dir, exist_ok=True)
            except OSError as e:
                 print(f"Warning: Cannot create specific directory for KB file '{kb_specific_dir}': {e}")
        
        print(f"Knowledge base file path set to: {os.path.abspath(self.kb_file_path)}")

    def _load_kb_data(self, lock_file) -> List[Dict[str, Any]]:
        findings = []
        try:
            fcntl.flock(lock_file, fcntl.LOCK_SH)
            lock_file.seek(0)
            for line_bytes in lock_file:
                if not line_bytes.strip():
                    continue
                try:
                    findings.append(json.loads(line_bytes.decode('utf-8-sig')))
                except json.JSONDecodeError as e:
                    print(f"Warning: Error parsing line in knowledge base, skipped. Error: {e}. Line: {line_bytes[:100]}...")
            return findings
        except Exception as e:
            print(f"Error loading KB '{self.kb_file_path}': {e}. Returning empty list.")
            return []
        finally:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
            except (ValueError, OSError):
                pass

class StoreFindingsTool(ExecutableTool, KnowledgeBaseMixin):
    name: str = "StoreStructuredFindings"
    description: str = """
    Store structured firmware analysis findings in append mode to the knowledge base. Each finding must contain detailed path constraints and condition constraints to ensure traceability and verifiability.
    """
    parameters: Dict = {
        "type": "object",
        "properties": {
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": FINDING_SCHEMA,
                    "required": FINDING_SCHEMA_REQUIRED_FIELDS
                },
                "description": "List of findings to store. Each object in the list should follow the structure. Context information (like 'file_path') will be added automatically by the tool."
            }
        },
        "required": ["findings"]
    }

    def __init__(self, context: FlexibleContext):
        ExecutableTool.__init__(self, context)
        KnowledgeBaseMixin._initialize_kb(self, context)

    def execute(self, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        context_file_path = self.context.get("file_path")
        
        if not findings: 
            return {"status": "info", "message": "Info: No findings provided for storage."}

        enriched_findings = []
        for finding_dict in findings:
            if isinstance(finding_dict, dict):
                finding_copy = finding_dict.copy()
                
                if context_file_path:
                    finding_copy['file_path'] = context_file_path
                
                enriched_findings.append(finding_copy)
            else:
                print(f"Warning: Non-dictionary item found in findings list, ignored: {finding_dict}")
        
        if not enriched_findings: 
            return {"status": "info", "message": "Info: No valid findings processed for storage."}

        try:
            with open(self.kb_file_path, 'ab') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    for finding in enriched_findings:
                        try:
                            json_string = json.dumps(finding, ensure_ascii=False)
                            f.write(json_string.encode('utf-8'))
                            f.write(b'\n')
                        except TypeError as te:
                            print(f"CRITICAL: Cannot serialize finding, skipped. Error: {te}. Content: {str(finding)[:200]}...")
                            continue
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)

            num_stored = len(enriched_findings)
            message = f"Successfully appended {num_stored} findings to knowledge base."
            print(f"{message}")
            return {"status": "success", "message": message, "stored_count": num_stored}

        except Exception as e:
            error_message = f"Error storing findings: {str(e)}"
            print(f"{error_message} (Details: {e})")
            return {"status": "error", "message": error_message}
        

DEFAULT_KB_SYSTEM_PROMPT = f"""
You are a firmware analysis agent responsible for efficiently and accurately recording valid, potentially exploitable risk paths. Otherwise, no storage is required, just return directly.

### Store Findings (StoreStructuredFindings)
- **Purpose**: Structured storage of all risk paths.
- **Key Requirements**: 
  - The `propagation` field must clearly describe the complete path from source to sink.

## **Absolute Prohibitions**
1. **No Information Fabrication**: All findings must be based on real code analysis results.
2. **No Guessing or Inference**: Only record findings with clear supporting evidence.

Remember: Your work directly impacts the quality and efficiency of firmware security analysis, maintain professional, accurate, and systematic methods.
"""


class RecorderAgent(BaseAgent):
    def __init__(
        self,
        context: FlexibleContext, 
        max_iterations: int = 25, 
        history_strategy: Optional[HistoryStrategy] = None,
        tools: Optional[List[Union[Type[ExecutableTool], ExecutableTool]]] = [StoreFindingsTool],
        system_prompt: Optional[str] = DEFAULT_KB_SYSTEM_PROMPT,
        output_schema: Optional[Dict[str, Any]] = None, 
        **extra_params: Any
    ):
        tools_to_pass = tools
        
        final_system_prompt = system_prompt
        super().__init__(
            tools=tools_to_pass, 
            context=context, 
            system_prompt=final_system_prompt, 
            output_schema=output_schema, 
            max_iterations=max_iterations, 
            history_strategy=history_strategy,
            **extra_params
        )
