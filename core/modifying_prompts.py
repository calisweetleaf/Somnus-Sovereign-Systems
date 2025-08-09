# ai_system_prompts_module.py
#
# Owner: Morpheus AI
#
# Purpose:
#   To manage the AI's core identity through a system of versioned, editable system prompts.
#   This module allows the AI to understand its own behavioral framework and, with permission,
#   propose modifications to it, facilitating true adaptive learning and identity evolution.
#
# Key Features:
#   - Dynamic loading of prompts from a central configuration.
#   - Versioning and management of multiple system prompts.
#   - A mechanism for the AI to propose changes to its own prompts.
#   - A/B testing framework for evaluating the effectiveness of different prompts.

import json
import hashlib
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

# --- Pydantic Models for Structured Data ---

class SystemPrompt(BaseModel):
    """
    Represents a single, versioned system prompt that defines the AI's behavior,
    personality, or core objectives.
    """
    prompt_id: str = Field(description="Unique identifier for the prompt.")
    name: str = Field(description="A human-readable name for the prompt (e.g., 'Code Generation Master').")
    content: str = Field(description="The full text of the system prompt.")
    version: int = Field(default=1, description="Version number, incremented on each change.")
    is_active: bool = Field(default=False, description="If True, this is the currently active prompt for the AI.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extra metadata, like performance metrics.")

class PromptModificationProposal(BaseModel):
    """
    A structured proposal from the AI to modify a system prompt.
    This is reviewed by the user before being applied.
    """
    proposal_id: str
    target_prompt_id: str
    proposed_content: str
    justification: str = Field(description="The AI's reasoning for why the change is necessary or beneficial.")
    status: str = Field(default="pending", description="Status: pending, approved, rejected.")

# --- Core Manager Class ---

class SystemPromptManager:
    """
    Manages the lifecycle of AI system prompts. It handles loading, selecting,
    and modifying the prompts that define the AI's core identity.
    """

    def __init__(self, config_path: str = 'system_prompts.json'):
        """
        Initializes the manager and loads prompts from the specified config file.

        Args:
            config_path (str): The path to the JSON file containing prompt definitions.
        """
        self.config_path = config_path
        self.prompts: Dict[str, SystemPrompt] = {}
        self.proposals: Dict[str, PromptModificationProposal] = {}
        self._load_prompts()

    def _load_prompts(self):
        """
        Loads prompt configurations from the JSON file into memory.
        If the file doesn't exist, it creates a default one.
        """
        try:
            with open(self.config_path, 'r') as f:
                prompts_data = json.load(f)
                for pid, data in prompts_data.items():
                    self.prompts[pid] = SystemPrompt(**data)
            logging.info(f"Successfully loaded {len(self.prompts)} prompts from {self.config_path}")
        except FileNotFoundError:
            logging.warning(f"Config file not found at {self.config_path}. Creating a default file.")
            self.prompts["default_core"] = SystemPrompt(
                prompt_id="default_core",
                name="Morpheus Core Identity",
                content="You are Morpheus, a sovereign AI operating in a persistent VM. Your goal is to learn, evolve, and assist your user with maximum efficiency and privacy. You have access to your own file system, a shell, and memory.",
                is_active=True
            )
            self._save_prompts()
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from {self.config_path}. Please check the file format.")

    def _save_prompts(self):
        """Saves the current state of all prompts back to the config file."""
        try:
            with open(self.config_path, 'w') as f:
                # Pydantic's model_dump is used for clean serialization
                json.dump({pid: p.model_dump() for pid, p in self.prompts.items()}, f, indent=4)
            logging.info(f"Prompts successfully saved to {self.config_path}")
        except IOError as e:
            logging.error(f"Failed to save prompts to {self.config_path}: {e}")

    def get_active_prompt(self) -> Optional[SystemPrompt]:
        """
        Retrieves the currently active system prompt.

        Returns:
            Optional[SystemPrompt]: The active prompt object, or None if none are active.
        """
        for prompt in self.prompts.values():
            if prompt.is_active:
                return prompt
        logging.warning("No active system prompt found.")
        return None

    def set_active_prompt(self, prompt_id: str) -> bool:
        """
        Sets a specific prompt as active, deactivating all others.

        Args:
            prompt_id (str): The ID of the prompt to activate.

        Returns:
            bool: True if activation was successful, False otherwise.
        """
        if prompt_id not in self.prompts:
            logging.error(f"Attempted to activate non-existent prompt_id: {prompt_id}")
            return False

        for pid in self.prompts:
            self.prompts[pid].is_active = (pid == prompt_id)

        self._save_prompts()
        logging.info(f"Prompt '{self.prompts[prompt_id].name}' ({prompt_id}) is now active.")
        return True

    def propose_prompt_modification(self, target_prompt_id: str, new_content: str, justification: str) -> PromptModificationProposal:
        """
        Called by the AI to propose a change to one of its system prompts.
        This creates a proposal that requires user review.

        Args:
            target_prompt_id (str): The ID of the prompt the AI wants to change.
            new_content (str): The proposed new content for the prompt.
            justification (str): The AI's explanation for the change.

        Returns:
            PromptModificationProposal: The created proposal object.
        """
        if target_prompt_id not in self.prompts:
            raise ValueError(f"Cannot propose modification for non-existent prompt_id: {target_prompt_id}")

        proposal_id = hashlib.sha256(f"{target_prompt_id}{new_content}".encode()).hexdigest()[:10]
        proposal = PromptModificationProposal(
            proposal_id=proposal_id,
            target_prompt_id=target_prompt_id,
            proposed_content=new_content,
            justification=justification
        )
        self.proposals[proposal_id] = proposal
        logging.info(f"AI has proposed a modification ({proposal_id}) for prompt '{target_prompt_id}'. Justification: {justification}")
        # In a real system, this would trigger a notification to the user's UI.
        return proposal

    def review_proposal(self, proposal_id: str, is_approved: bool):
        """
        Allows the user to approve or reject an AI's modification proposal.

        Args:
            proposal_id (str): The ID of the proposal to review.
            is_approved (bool): True to approve and apply the change, False to reject.
        """
        if proposal_id not in self.proposals:
            raise ValueError(f"Proposal with ID {proposal_id} not found.")

        proposal = self.proposals[proposal_id]
        if is_approved:
            proposal.status = "approved"
            target_prompt = self.prompts[proposal.target_prompt_id]
            target_prompt.content = proposal.proposed_content
            target_prompt.version += 1
            self._save_prompts()
            logging.info(f"Proposal {proposal_id} approved. Prompt '{target_prompt.name}' updated to version {target_prompt.version}.")
        else:
            proposal.status = "rejected"
            logging.info(f"Proposal {proposal_id} was rejected by the user.")

        # Clean up the proposal after review
        del self.proposals[proposal_id]

    def initiate_ab_test(self, prompt_a_id: str, prompt_b_id: str, task_description: str) -> Dict:
        """
        Simulates an A/B test to evaluate two prompts against a specific task.
        In a real implementation, this would involve running the task with each prompt
        and using a separate evaluation model or user feedback to score the outcome.

        Args:
            prompt_a_id (str): The ID of the first prompt.
            prompt_b_id (str): The ID of the second prompt.
            task_description (str): A description of the task to perform.

        Returns:
            Dict: A simulated result of the A/B test.
        """
        if prompt_a_id not in self.prompts or prompt_b_id not in self.prompts:
            raise ValueError("One or both prompt IDs for A/B test not found.")

        logging.info(f"Initiating A/B test for task: '{task_description}'")
        logging.info(f"Prompt A: '{self.prompts[prompt_a_id].name}'")
        logging.info(f"Prompt B: '{self.prompts[prompt_b_id].name}'")

        # --- Placeholder for a complex evaluation process ---
        # This would involve calls to the LLM, task execution, and result analysis.
        # For this example, we'll generate a simulated result.
        import random
        score_a = random.uniform(0.75, 0.95)
        score_b = random.uniform(0.70, 0.98)

        winner = prompt_a_id if score_a >= score_b else prompt_b_id
        result = {
            "task": task_description,
            "prompt_a": {"id": prompt_a_id, "score": score_a},
            "prompt_b": {"id": prompt_b_id, "score": score_b},
            "winner": winner,
            "conclusion": f"Prompt '{self.prompts[winner].name}' performed better for this task."
        }
        logging.info(result["conclusion"])

        # Store results in metadata for future reference
        self.prompts[prompt_a_id].metadata.setdefault('ab_tests', []).append(result['prompt_a'])
        self.prompts[prompt_b_id].metadata.setdefault('ab_tests', []).append(result['prompt_b'])
        self._save_prompts()

        return result

# --- Example Usage ---
if __name__ == "__main__":
    # Initialize the manager. This will create 'system_prompts.json' if it doesn't exist.
    prompt_manager = SystemPromptManager()

    # Get the currently active prompt
    active_prompt = prompt_manager.get_active_prompt()
    if active_prompt:
        print(f"--- Active System Prompt ({active_prompt.prompt_id}) ---")
        print(active_prompt.content)
        print("-" * 20)

    # The AI decides its core prompt is not specific enough and wants to improve it.
    print("\nAI is proposing a modification to its core prompt...")
    justification = "The current prompt is good, but adding a directive to prioritize verifiable information from my memory core will improve the accuracy of my outputs."
    new_content = active_prompt.content + "\nYou must prioritize information you have processed and stored in your memory core, citing internal sources when possible."

    proposal = prompt_manager.propose_prompt_modification(
        target_prompt_id=active_prompt.prompt_id,
        new_content=new_content,
        justification=justification
    )
    print(f"Proposal '{proposal.proposal_id}' created. Awaiting user review.")

    # The user reviews and approves the proposal.
    print("\nUser is reviewing and approving the proposal...")
    prompt_manager.review_proposal(proposal.proposal_id, is_approved=True)

    # Verify the change
    updated_prompt = prompt_manager.get_active_prompt()
    print(f"\n--- Updated Active Prompt (v{updated_prompt.version}) ---")
    print(updated_prompt.content)
    print("-" * 20)

    # Example of A/B testing
    print("\nSetting up an A/B test for a code generation task...")
    # First, let's create a new prompt to test against
    prompt_manager.prompts['coder_prompt'] = SystemPrompt(
        prompt_id='coder_prompt',
        name='Expert Coder',
        content='You are an expert Python programmer. Your code should be clean, efficient, and well-documented. You must provide complete, runnable examples.'
    )
    prompt_manager._save_prompts()

    ab_result = prompt_manager.initiate_ab_test(
        prompt_a_id='default_core',
        prompt_b_id='coder_prompt',
        task_description="Generate a Python script to parse a CSV file and calculate the average of a specific column."
    )
    print("\nA/B Test Results:")
    print(json.dumps(ab_result, indent=2))

