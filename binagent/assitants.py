import os
from typing import Any, Dict, Optional, List, Union, Type

from agent.base import BaseAgent
from agent.tools.basetool import FlexibleContext, ExecutableTool
from agent.core.assitants import BaseAssistant, ParallelBaseAssistant

class ParallelTaskDelegator(ParallelBaseAssistant):
    name = "TaskDelegator"
    description = """
    Task Delegator - Distributes multiple subtasks to child agents for parallel execution.
    
    Applicable scenarios:
    1. Need to break down a task into multiple independent subtasks for processing.
    2. Subtasks have no strict execution order dependencies, such as finding addresses of multiple controllable variables.
    3. Recommended for comprehensive analysis and complex tasks where parallel execution of subtasks can improve efficiency.
    
    """
    parameters = {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "task_description": {
                            "type": "string",
                            "description": "Detailed description of the subtask to execute. Note that each subtask description is independent and needs to specify the analysis target."
                        }
                    },
                    "required": ["task_description"],
                    "description": "Task item containing a single task description."
                },
                "description": "List of independent subtasks to be distributed to child agents for execution."
            }
        },
        "required": ["tasks"]
    }
    timeout = 9600


class ParallelFunctionDelegator(ParallelBaseAssistant):
    name = "FunctionDelegator"
    description = """
    Function Analysis Delegator - An intelligent agent specialized in analyzing function call chains in binary files. 
    Its responsibility is to forward-track the flow path of taint data between function calls. You can delegate potential 
    external entry points to this agent for in-depth tracking.
    """

    parameters = {
        "type": "object",
        "properties": {
            "tasks": { 
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "task_description": {
                            "type": "string", 
                            "description": "When creating analysis tasks for child functions, your description must clearly include these four points:\n1. **Target Function**: Name and address of the child function to analyze.\n2. **Taint Entry**: The specific register or stack address where the taint is located in the child function (e.g., 'taint is in first parameter register r0').\n3. **Taint Source**: How this taint data was generated in the parent function (e.g., 'this value was obtained by parent function main calling nvram_get(\"lan_ipaddr\")').\n4. **Analysis Goal**: Clearly indicate that the new task should track this new taint entry (e.g., 'track r0's flow path within the child function')."
                        },
                        "task_context": {   
                            "type": "string", 
                            "description": "(Optional) Provide supplementary context that affects the analysis. This information is not the taint flow itself but may affect the execution path of the child function. Examples:\n- 'r2 register currently holds 0x100, representing the maximum buffer length'\n- 'Global variable `is_admin` was set to 1 before this call'\n- 'Analysis should assume the file was successfully opened'"
                        }
                    },
                    "required": ["task_description"]
                },
                "description": "List of function tasks to analyze."
            }
        },
        "required": ["tasks"]
    }

    def __init__(self, 
                 context: FlexibleContext,
                 agent_class_to_create: Type[BaseAgent] = BaseAgent,
                 default_sub_agent_tool_classes: Optional[List[Union[Type[ExecutableTool], ExecutableTool]]] = None,
                 default_sub_agent_max_iterations: int = 10,
                 sub_agent_system_prompt: Optional[str] = None,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 timeout: Optional[int] = None
                ):
        final_name = name or ParallelFunctionDelegator.name
        final_description = description or ParallelFunctionDelegator.description
        
        super().__init__(
            context=context,
            agent_class_to_create=agent_class_to_create,
            default_sub_agent_tool_classes=default_sub_agent_tool_classes,
            default_sub_agent_max_iterations=default_sub_agent_max_iterations,
            sub_agent_system_prompt=sub_agent_system_prompt,
            name=final_name,
            description=final_description,
            timeout=timeout
        )

    def _build_sub_agent_prompt(self, usr_init_msg: Optional[str], **task_details: Any) -> str:
        """Build complete task prompt for child agent, including optional task_context."""
        task_description = task_details.get("task_description")
        task_context = task_details.get("task_context")

        usr_init_msg_content = usr_init_msg if usr_init_msg else "No user initial request provided"
        task_description_content = task_description if task_description else "No task description provided"

        prompt_parts = [
            f"User core request is:\n{usr_init_msg_content}",
            f"Current specific task:\n{task_description_content}"
        ]

        if task_context:
            prompt_parts.append(f"Supplementary context:\n{task_context}")

        return "\n\n".join(prompt_parts)
