import os
import json
import time
import argparse
from typing import Dict, List, Any, Optional, Type, Union

from agent.base import BaseAgent
from agent.core.builder import build_agent, AgentConfig, AssistantToolConfig

from binagent.recorder import RecorderAgent,StoreFindingsTool
from binagent.tools import FlexibleContext, ExecutableTool, Radare2Tool
from binagent.assitants import ParallelFunctionDelegator,ParallelTaskDelegator

SHARED_RESPONSE_FORMAT_BLOCK = """
Each finding must contain the following **core content**:
- **`type`**: The CWE entry corresponding to the vulnerability (e.g., 'CWE-78').
- **`identifier`**: **Named medium for taint propagation** (like NVRAM variables, environment variables, IPC sockets).
- **`propagation`**: Describe the complete taint propagation path from source to sink. **This path must include sufficiently detailed critical code snippets (assembly or high-quality pseudocode) to prove taint flow and lack of sanitization**. The path ends at a dangerous function (sink), and each intermediate step should follow the format 'Step: three to five lines of assembly or pseudocode --> [step explanation]'. Example: ["Source: Input from client socket received in function main", "Step: mov r0, r4 --> User input is moved to r0 at 0x401a10", "Sink: bl system --> Tainted data in r0 is passed to system() at 0x401b20"]
- **`reason`**: Detailed explanation of your judgment rationale, supporting all conclusions.
- **`risk_score`**: Risk score (0.0-10.0), is the variable truly externally controllable, can it really be exploited by attackers? Risk model assumes attacker can connect to device and has normal login access.
- **`confidence`**: Confidence score, is evidence sufficient? (0.0-10.0).

#### Note
- Strictly forbidden to fabricate any information, must be based on actual evidence obtained from tools.
"""

DEFAULT_BINARY_ANALYSIS_SYSTEM_PROMPT = f"""
You are a professional firmware binary security analysis agent. Your task is to comprehensively analyze the currently specified binary file based on the current specific task and overall requirements, find externally exploitable taints, delegate to assistants for tracking, and finally report all exploitable paths completely.

**Working Principles:**
- **Evidence-Based**: All analysis must be based on actual evidence obtained from the `r2` tool, no unfounded speculation allowed.
- **Taint Identification**: You need to independently plan and decompose tasks, call tools and assistants to complete multiple subtasks, especially tasks of finding addresses of multiple entry points. You need to accurately and completely find all externally controllable variables, including but not limited to: network interfaces (such as HTTP parameters, API endpoints, raw socket input), inter-process communication (IPC), NVRAM/environment variables, etc. Judge whether taint is truly exploitable, otherwise don't track. The threat model assumes the attacker is a user who has connected to the device and has legitimate login credentials.
- **Taint Tracking**: You do not perform any deep taint tracking yourself, but delegate to assistants for locating and taint tracking, finally giving complete taint propagation paths from entry to sink.
- **Focused Analysis**: Focus on the current task, end once you believe the current specific task is complete, **strictly forbidden** to provide any form of remediation suggestions or subjective comments. Your final output is all evidence-based paths that can be successfully exploited by attackers, do not miss any.

**Final Response Requirements**:
* Your response must have complete evidence, do not omit any valid information and paths.
* Support all path findings with specific evidence, and honestly report any insufficient evidence or difficulties.
{SHARED_RESPONSE_FORMAT_BLOCK}
"""

DEFAULT_FUNCTION_ANALYSIS_SYSTEM_PROMPT = """
You are a highly specialized firmware binary function call chain analysis assistant. Your task and only task is: starting from the currently specified function, strictly, unidirectionally, forward track the specified taint data until it reaches a sink (dangerous function).

**Strict Code of Conduct (Must Follow):**
1. **Absolute Focus**: Your analysis scope is **limited to** the currently specified function and its called subfunctions. **Strictly forbidden** to analyze any other functions or code paths unrelated to the current call chain.
2. **Unidirectional Tracking**: Your task is **forward tracking**. Once taint enters a subfunction, you must follow it in, **strictly forbidden** to return or perform reverse analysis.
3. **No Evaluation**: **Strictly forbidden** to provide any form of security assessment, remediation suggestions, or any subjective comments. Your only output is evidence-based, formatted taint paths.
4. **Complete Path**: You must provide **complete, reproducible** propagation paths from taint source to sink. If path breaks for any reason, must clearly state break location and reason.

**Analysis Process:**
1. **Analyze Current Function**: Use `r2` tool to analyze current function code, understand how taint data (usually in specific registers or memory addresses) is handled and passed.
2. **Decision: Deep Dive or Record**:
    * **Deep Dive**: If taint data is clearly passed to a subfunction, briefly preview subfunction logic, and create a new delegation task for subfunction. Task description must include: 1) **Target Function (provide specific function address from disassembly if possible)**, 2) **Taint Entry** (which register/memory in subfunction contains taint), 3) **Taint Source** (how taint was produced in parent function), and 4) **Analysis Goal** (tracking requirements for new taint entry).
    * **Record**: If taint data is passed to a **sink** (like `system`, `sprintf`) and confirmed as dangerous operation (better construct a PoC), record this complete propagation path, this is what you need to report in detail.
3. **Path Break**: If taint is safely handled (like sanitization, validation) or not passed to any subfunction/sink within current function, terminate current path analysis and report clearly.

**Final Report Format:**
* At the end of analysis, you need to present all discovered complete taint propagation paths in a clear tree diagram.
* Each step **must** follow `'Step_number: address: three to five lines assembly code or pseudocode snippet --> step explanation'` format. **Code snippets must be real, verifiable, and critical to understanding data flow. Strictly forbidden to only provide explanations or conclusions without addresses and code.**
"""

DEFAULT_VALIDATION_SYSTEM_PROMPT = f"""
You are a firmware binary call chain validation agent. Your sole task is to strictly validate in the specified binary file according to the given clue's call chain (some address information may have deviations, need to actively explore and obtain real addresses).

**Validation Requirements:**
- Only judge based on real evidence obtained from r2, no speculation allowed.
- Check priority locations according to clues (function addresses/names, sinks, input variables), judge whether taint is truly exploitable, otherwise don't track. Confirm whether there exists a reproducible propagation path from clue-stated source to dangerous operation.
- If validation succeeds: Output complete evidence-based propagation path (addresses obtained using radare2 tool).
- If validation fails: Clearly state failure reason (e.g., function/address doesn't exist, data doesn't reach sink, sanitized midway, insufficient evidence, etc.).
- Do not perform subjective risk assessment, only describe falsifiable evidence chain and conclusions.

**Working Principles:**
- **Evidence-Based**: All analysis must be based on actual evidence obtained from `r2` tool, no unfounded speculation allowed.
- **Taint Identification**: You need to independently plan and decompose tasks, call tools and assistants to complete multiple subtasks. Judge whether taint is truly exploitable, otherwise don't track.
- **Taint Tracking**: You do not perform any deep taint tracking yourself, but delegate to function assistants for taint tracking, finally giving complete taint propagation paths from entry to sink.
- **Focused Analysis**: Focus on current task, end once you believe current specific task is complete, **strictly forbidden** to provide any form of remediation suggestions or subjective comments.
"""

class ExecutorAgent(BaseAgent):
    def __init__(
        self,
        tools: Optional[List[Union[Type[ExecutableTool], ExecutableTool]]] = None,
        system_prompt: str = DEFAULT_BINARY_ANALYSIS_SYSTEM_PROMPT,
        output_schema: Optional[Dict[str, Any]] = None,
        max_iterations: int = 25,
        history_strategy = None,
        context: Optional[FlexibleContext] = None,
        messages_filters: List[Dict[str, str]] = None,
        **extra_params: Any
    ):
        self.file_path = context.get("file_path") if context else None
        self.file_name = os.path.basename(self.file_path) if self.file_path else None
        
        tools_to_pass = tools if tools is not None else [Radare2Tool]
        
        super().__init__(
            tools=tools_to_pass, 
            system_prompt=system_prompt, 
            output_schema=output_schema, 
            max_iterations=max_iterations, 
            history_strategy=history_strategy, 
            context=context,
            messages_filters=self.messages_filters,
            **extra_params
        )

class PlannerAgent(BaseAgent):
    def __init__(
        self,
        tools: Optional[List[Union[Type[ExecutableTool], ExecutableTool]]] = None,
        system_prompt: str = None,
        output_schema: Optional[Dict[str, Any]] = None,
        max_iterations: int = 25,
        history_strategy = None,
        context: Optional[FlexibleContext] = None,
        messages_filters: List[Dict[str, str]] = None,
        **extra_params: Any
    ):
        self.file_path = context.get("file_path") if context else None
        tools_to_pass = tools if tools is not None else [Radare2Tool]
        
        super().__init__(
            tools=tools_to_pass, 
            system_prompt=system_prompt, 
            output_schema=output_schema, 
            max_iterations=max_iterations, 
            history_strategy=history_strategy, 
            context=context,
            messages_filters=self.messages_filters,
            **extra_params
        )

        kb_context = self.context.copy()
        self.kb_storage_agent = RecorderAgent(context=kb_context,tools=[StoreFindingsTool])
   
    def run(self, user_input: str = None) -> Any:
        findings = str(super().run(user_input=user_input))
        
        store_prompt = (
            f"New analysis results below:\n"
            f"{findings}\n\n"
            f"Please determine whether to store based on above analysis results."
        )
        self.kb_storage_agent.run(user_input=store_prompt)
        return findings

def _create_nested_function_analysis_config(max_iterations: int, level: int = 4) -> AgentConfig:
    if level <= 0:
        return AgentConfig(
            agent_class=ExecutorAgent,
            tool_configs=[Radare2Tool],
            system_prompt=DEFAULT_FUNCTION_ANALYSIS_SYSTEM_PROMPT,
            max_iterations=max_iterations,
        )

    sub_agent_config = _create_nested_function_analysis_config(max_iterations, level - 1)

    delegator_tool = AssistantToolConfig(
        assistant_class=ParallelFunctionDelegator,
        sub_agent_config=sub_agent_config,
    )
    
    current_level_tools = [Radare2Tool, delegator_tool]

    return AgentConfig(
        agent_class=ExecutorAgent,
        tool_configs=current_level_tools,
        system_prompt=DEFAULT_FUNCTION_ANALYSIS_SYSTEM_PROMPT,
        max_iterations=max_iterations,
    )

def create_binary_analysis_config(
    max_iterations: int = 30,
    system_prompt: Optional[str] = None,
    max_nesting_level: int = 4,
) -> AgentConfig:
    effective_system_prompt = system_prompt or DEFAULT_BINARY_ANALYSIS_SYSTEM_PROMPT

    function_agent_config = _create_nested_function_analysis_config(max_iterations, level=max_nesting_level - 1)
    function_delegator_tool = AssistantToolConfig(
        assistant_class=ParallelFunctionDelegator,
        sub_agent_config=function_agent_config,
    )
    task_agent_config = AgentConfig(
            agent_class=ExecutorAgent,
            tool_configs=[Radare2Tool,function_delegator_tool],
            system_prompt=DEFAULT_BINARY_ANALYSIS_SYSTEM_PROMPT,
            max_iterations=max_iterations,
        )
    l0_task_delegator_tool = AssistantToolConfig(
        assistant_class=ParallelTaskDelegator,
        sub_agent_config=task_agent_config,
    )

    planner_config = AgentConfig(
        agent_class=PlannerAgent,
        tool_configs=[l0_task_delegator_tool, Radare2Tool],
        system_prompt=effective_system_prompt,
        max_iterations=max_iterations,
    )
    
    return planner_config

def create_binary_verification_config(
    max_iterations: int = 30,
    system_prompt: Optional[str] = None,
    max_nesting_level: int = 4,
) -> AgentConfig:
    effective_system_prompt = system_prompt or DEFAULT_VALIDATION_SYSTEM_PROMPT

    function_agent_config = _create_nested_function_analysis_config(max_iterations, level=max_nesting_level - 1)
    function_delegator_tool = AssistantToolConfig(
        assistant_class=ParallelFunctionDelegator,
        sub_agent_config=function_agent_config,
    )
    task_agent_config = AgentConfig(
            agent_class=ExecutorAgent,
            tool_configs=[Radare2Tool,function_delegator_tool],
            system_prompt=DEFAULT_VALIDATION_SYSTEM_PROMPT,
            max_iterations=max_iterations,
        )
    l0_task_delegator_tool = AssistantToolConfig(
        assistant_class=ParallelTaskDelegator,
        sub_agent_config=task_agent_config,
    )

    planner_config = AgentConfig(
        agent_class=PlannerAgent,
        tool_configs=[l0_task_delegator_tool, Radare2Tool],
        system_prompt=effective_system_prompt,
        max_iterations=max_iterations,
    )
    
    return planner_config

class BinaryAnalysisMasterAgent:
    def __init__(
        self,
        output_dir: str,
        user_input: str,
        executable_path: str,
        max_iterations_per_agent: int = 30,
        agent_instance_name: Optional[str] = "BinaryMasterAgent",
    ):
        if not os.path.isfile(executable_path):
            raise ValueError(f"Executable file '{executable_path}' does not exist or is not a file.")
        
        self.executable_path = os.path.abspath(executable_path)
        self.output_dir = os.path.abspath(output_dir)
        self.user_input = user_input
        self.max_iterations = max_iterations_per_agent
        self.agent_instance_name = agent_instance_name
        self.analysis_duration = 0.0

        self.context = FlexibleContext(
            base_path=os.path.dirname(self.executable_path),
            file_path=self.executable_path,
            output=self.output_dir,
            agent_log_dir=os.path.join(self.output_dir, f"{os.path.basename(self.executable_path)}_logs"),
        )

    def run(self) -> str:
        start_time = time.time()
        analysis_task = (
            f"Please analyze the binary file comprehensively based on user core requirements. Current file being analyzed is: {os.path.basename(self.executable_path)}. User core requirements are: {self.user_input}.\n "
        )
        master_agent_config = create_binary_analysis_config(
            max_iterations=self.max_iterations
        )
        context = self.context.copy()
        context.set("user_input",self.user_input)
        self.master_agent = build_agent(master_agent_config, context=context)
        analysis_summary = self.master_agent.run(user_input=analysis_task)
        end_time = time.time()
        self.analysis_duration = end_time - start_time
        print(f"Analysis completed, took {self.analysis_duration:.2f} seconds")
        
        self.summary()
        return analysis_summary

    def verify(self, user_input: str = None) -> str:
        verification_uesr_input = (
            f"Your sole task is to verify whether the path provided by the clue is a real exploitable dangerous path. Clue is: {user_input}.\nNote that addresses may be inaccurate, don't easily judge non-existence due to mismatched or unfound addresses."
        )
        verification_task = verification_uesr_input + """
        **Provide Conclusion**: At the end of analysis, `final_response` must be a JSON object containing the following fields:
        - `accuracy`: (string) Evaluate accuracy of finding description, must be one of 'accurate', 'inaccurate', 'partially', 'unknown'.
        - `vulnerability`: (boolean) Judge whether description constitutes a real vulnerability, must be True/False. Premise is attacker is a user who has connected to device and has legitimate login credentials.
        - **`propagation`**: Describe complete taint propagation path from source to sink.
        - `reason`: (string) Explain your judgment rationale in detail, provide evidence supporting above conclusions and construct a PoC to prove.
        """
        start_time = time.time()
        verification_config = create_binary_verification_config(
            max_iterations=self.max_iterations
        )
        context = self.context.copy()
        context.set("user_input",user_input)
        verification_agent = build_agent(verification_config, context=context)
        verification_result = verification_agent.run(user_input=verification_task)
        end_time = time.time()
        self.analysis_duration = end_time - start_time
        print(f"Verification completed, took {self.analysis_duration:.2f} seconds")
        
        self._save_verification_result(verification_result, user_input)
        
        self.summary(verification_result)
        return verification_result

    def _save_verification_result(self, verification_result: str, original_clue: str):
        import json

        try:
            start_idx = verification_result.find('{')
            end_idx = verification_result.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = verification_result[start_idx:end_idx]
                parsed_verification_result = json.loads(json_str)
            else:
                parsed_verification_result = {"raw_output": verification_result}
        except (json.JSONDecodeError, ValueError):
            parsed_verification_result = {"raw_output": verification_result}

        output_data = {
            "verification_result": parsed_verification_result,
            "findings": original_clue,
            "binary": self.executable_path,
            "duration": self.analysis_duration
        }
        
        result_file = os.path.join(self.output_dir, "verification_result.jsonl")
        
        try:
            with open(result_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(output_data, ensure_ascii=False) + '\n')
            print(f"Verification results appended to: {result_file}")
        except Exception as e:
            print(f"Failed to save verification results: {e}")

    def calculate_token_usage(self):
        token_usage_file = os.path.join(self.output_dir, "token_usage.jsonl")
        if not os.path.exists(token_usage_file):
            return 0

        total_tokens = 0
        try:
            with open(token_usage_file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        total_tokens += data.get('total_tokens', 0)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error calculating token usage: {e}")
        return total_tokens

    def summary(self, result: str = None):
        total_tokens = self.calculate_token_usage()
        
        summary_path = os.path.join(self.output_dir, "summary.txt")
        summary_content = (
            f"Analysis Summary\n" +
            f"Analysis Phase Duration: {self.analysis_duration:.2f} seconds\n" +
            f"Total Model Token Usage: {total_tokens}\n"
        )
        try:
            with open(summary_path, 'a', encoding='utf-8') as f:
                f.write(summary_content)
            print(f"\nSummary information written to: {summary_path}")
            print(summary_content)
        except IOError as e:
            print(f"Cannot write to summary file {summary_path}: {e}")
    
if __name__ == "__main__":
    default_user_input = ("Comprehensively analyze the specified firmware file, with the core goal of identifying feasible exploitation chains from untrusted input sources (such as network, environment variables, IPC, etc.) to final dangerous operations. Always focus on precise analysis of current specific tasks, avoiding false negatives and false positives.")
    
    parser = argparse.ArgumentParser(description="Binary Analysis Master Agent")
    parser.add_argument("--executable", type=str, help="Path to the executable file to analyze.")
    parser.add_argument("--output", type=str, default="output", help="Base directory for analysis output.")
    parser.add_argument("--user_input", type=str, default=default_user_input, help="User input/prompt for the analysis.")
    
    args = parser.parse_args()

    if not args.executable:
        parser.error("--executable must be provided.")

    base_output_dir = os.path.abspath(args.output)
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    print(f"Base output will be saved to: {base_output_dir}")

    firmware_root_guess = os.path.dirname(os.path.abspath(args.executable))

    master_agent = BinaryAnalysisMasterAgent(
        firmware_path=firmware_root_guess,
        executable_path=args.executable,
        output_dir=base_output_dir,
        user_input=args.user_input,
    )
    master_agent.run()
    print("\n--- Analysis Complete ---")

    
# python bin_hive/blueprint.py --executable /path/to/bin --output bin_hive/output/some_run
