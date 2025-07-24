import os
import json
import torch
import textwrap
import re
import asyncio
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ==============================================================================
# Terminal Logger
# ==============================================================================
class Logger:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

    def print_header(self, text: str):
        print(f"\n{self.HEADER}{self.BOLD}--- {text} ---{self.ENDC}")

    def info(self, text: str):
        print(f"{self.BLUE}[INFO] {text}{self.ENDC}")

    def success(self, text: str):
        print(f"{self.GREEN}[SUCCESS] {text}{self.ENDC}")

    def warning(self, text: str):
        print(f"{self.YELLOW}[WARNING] {text}{self.ENDC}")

    def prompt(self, text: str):
        short_prompt = textwrap.shorten(text.replace('\n', ' '), width=200, placeholder='...')
        print(f"\n{self.CYAN}{self.BOLD}[PROMPT PREVIEW]{self.ENDC} {short_prompt}")

    def response(self, text: str):
        print(f"\n{self.YELLOW}{self.BOLD}[ASSISTANT]{self.ENDC}\n{text}")

    def ask_for_confirmation(self, question: str) -> bool:
        while True:
            response = input(f"{self.RED}{self.BOLD}{question} (y/n): {self.ENDC}").lower()
            if response in ['y', 'yes']: return True
            if response in ['n', 'no']: return False

# ==============================================================================
# Statistics Tracker 
# ==============================================================================
class StatsTracker:
    def __init__(self, logger: Logger):
        self.logger = logger
        self.full_raw_history: List[Tuple[str, str]] = []

    def _count_tokens(self, text: str) -> int:
        return len(text.split())

    def add_turn(self, user_query: str, bot_response: str):
        self.full_raw_history.append((user_query, bot_response))

    def calculate_and_print_stats(self, actual_prompt_text: str):
        self.logger.print_header("Token Efficiency Stats")
        actual_tokens_sent = self._count_tokens(actual_prompt_text)
        self.logger.info(f"Tokens Sent (Our Method): ~{actual_tokens_sent}")
        naive_history_text = "\n".join([f"user: {u}\nbot: {b}" for u, b in self.full_raw_history])
        naive_tokens = self._count_tokens(naive_history_text)
        self.logger.info(f"Tokens if Naive (Full History): ~{naive_tokens}")
        if naive_tokens > 0 and actual_tokens_sent < naive_tokens:
            reduction = 100 * (1 - (actual_tokens_sent / naive_tokens))
            self.logger.success(f"Context Reduction: {reduction:.2f}%")
        else:
            self.logger.info("Context Reduction: N/A")

# ==============================================================================
# (Config, QuantumEncoder)
# ==============================================================================
class Config:
    def __init__(self, logger: Logger, threshold: int):
        self.GEMINI_API_KEY = "<your-gemini-api-key>"
        if not self.GEMINI_API_KEY:
            logger.warning("No GEMINI_API_KEY environment variable set.")
            raise ValueError("Please set the GEMINI_API_KEY environment variable.")
        self.MODEL_NAME = "gemini-2.5-flash"
        self.CONTEXT_THRESHOLD = threshold
        self.SAFETY = []
        self.SYMBOLIC_MEMORY_FILE = "symbolic_memory.json"

class QuantumEncoder:
    def __init__(self, logger: Logger):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.dim = self.model.get_sentence_embedding_dimension()
        logger.success(f"QuantumEncoder on {self.device}, dim {self.dim}")

    def encode(self, text: str) -> torch.Tensor:
        v = self.model.encode(text, convert_to_tensor=True, device=self.device)
        return torch.outer(v, v)

    def compress(self, matrices: List[torch.Tensor]) -> torch.Tensor:
        return torch.mean(torch.stack(matrices), dim=0) if matrices else torch.zeros((self.dim, self.dim), device=self.device)

# ==============================================================================
# Symbolic Memory
# ==============================================================================
class SymbolicMemory:
    def __init__(self, logger: Logger, filepath: str):
        self.logger = logger
        self.filepath = filepath
        self.memory_data = self.load()

    def get_full_memory_json(self) -> str:
        return json.dumps(self.memory_data, indent=2)

    def update_memory(self, memory_update_json: str):
        try:
            # First, try to parse the entire string as JSON
            update_data = json.loads(memory_update_json)
        except json.JSONDecodeError:
            # If it fails, search for a JSON block within ```
            match = re.search(r'```json\s*(\{.*?\})\s*```', memory_update_json, re.DOTALL)
            if match:
                try:
                    update_data = json.loads(match.group(1))
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse JSON within ``` block: {e}")
                    return
            else:
                self.logger.warning("Received a non-JSON string and no JSON block was found.")
                return

        if "project_context" in update_data:
            self.memory_data["project_context"].update(update_data["project_context"])
            self.logger.info("Project context updated.")
        if "symbols" in update_data:
            self.memory_data["symbols"].update(update_data["symbols"])
            self.logger.info(f"Symbol map updated with {len(update_data['symbols'])} symbols.")
        self.save()


    def save(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.memory_data, f, indent=4)
        self.logger.info("Memory saved to file.")

    def load(self) -> Dict:
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r') as f:
                try:
                    data = json.load(f)
                    data.setdefault("project_context", {})
                    data.setdefault("symbols", {})
                    self.logger.success("Loaded existing memory from file.")
                    return data
                except json.JSONDecodeError:
                    self.logger.warning("Memory file is corrupted. Starting fresh.")
                    return {"project_context": {}, "symbols": {}}
        return {"project_context": {}, "symbols": {}}

# ==============================================================================
# Gemini Model
# ==============================================================================
class GeminiModel:
    def __init__(self, config: Config, logger: Logger):
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(config.MODEL_NAME)
        self.safety = config.SAFETY
        self.logger = logger

    async def generate_async(self, prompt: str) -> str:
        self.logger.prompt(prompt)
        if not self.logger.ask_for_confirmation("Send this prompt to Gemini?"):
            self.logger.warning("Cancelled by user.")
            return "ACTION_CANCELLED"
        self.logger.info("Sending request to Gemini...")
        try:
            response = await asyncio.to_thread(
                self.model.generate_content, prompt, safety_settings=self.safety
            )
            return response.text.strip()
        except Exception as e:
            self.logger.warning(f"Gemini API Error: {e}")
            return "⚠️ Error: Failed to generate content from API."

# ==============================================================================
# Context Manager 
# ==============================================================================
class ContextManager:
    def __init__(self, threshold: int):
        self.logger = Logger()
        self.config = Config(self.logger, threshold)
        self.encoder = QuantumEncoder(self.logger)
        self.memory = SymbolicMemory(self.logger, self.config.SYMBOLIC_MEMORY_FILE)
        self.gemini = GeminiModel(self.config, self.logger)
        self.stats_tracker = StatsTracker(self.logger)
        self.history: List[Tuple[str, str]] = []
        self.embeddings: List[torch.Tensor] = []

    async def initialize_project(self):
        project_context = self.memory.memory_data["project_context"]
        if not project_context.get('project_name'):
            self.logger.print_header("Project Initialization")
            name = input(f"{self.logger.CYAN}{self.logger.BOLD}Project Name: {self.logger.ENDC} ")
            lang = input(f"{self.logger.CYAN}{self.logger.BOLD}Main Language(s) (e.g., HTML, JavaScript): {self.logger.ENDC} ")
            project_context['project_name'] = name
            project_context['language'] = lang
            self.memory.save()
            self.logger.success(f"Initialized project '{name}' with language '{lang}'.")

    async def process_turn(self, user_input: str):
        # Build the new, stricter prompt
        prompt = self.build_prompt(user_input)
        
        # Generate the response
        response_text = await self.gemini.generate_async(prompt)
        
        # Separate user-facing text from memory update
        user_facing_response = re.sub(r'<memory_update>.*</memory_update>', '', response_text, flags=re.DOTALL).strip()
        memory_match = re.search(r'<memory_update>\s*(.*?)\s*</memory_update>', response_text, re.DOTALL)
        
        self.logger.response(user_facing_response)

        if "ACTION_CANCELLED" in response_text or "⚠️ Error" in response_text:
            return

        # Update symbolic memory immediately
        if memory_match:
            memory_json = memory_match.group(1)
            self.memory.update_memory(memory_json)
        else:
            self.logger.warning("No <memory_update> block found in the response. Memory not updated.")

        # Add to vector history for future compression
        self.history.append((user_input, user_facing_response))
        self.embeddings.extend([self.encoder.encode(user_input), self.encoder.encode(user_facing_response)])
        self.stats_tracker.add_turn(user_input, user_facing_response)
        
        # Compress vector history if needed
        await self.compress_vectors_if_needed()
        self.stats_tracker.calculate_and_print_stats(prompt)

    async def compress_vectors_if_needed(self):
        
        """
        -----------------------------------------------------------------------------
        🧠 Semantic Compression (Quantum Collapse)
        -----------------------------------------------------------------------------
        This compresses the semantic history of user + assistant turns into a single
        second-order tensor (outer-product embedding). While not used directly in the
        prompt yet, it serves as a "context fingerprint" that summarizes the meaning
        of the dialogue history compactly and consistently.
        
        This enables future capabilities such as:
        - Semantic recall and session similarity
        - Lightweight context anchoring without full history
        - Session clustering, deduplication, and meta-reasoning
        - Prompt routing or intent estimation based on latent context
        
        Think of it as a low-dimensional, compressed signature of the session's meaning.
        It's the semantic counterpart to version control: small, consistent, and reusable.
        """
        
        if len(self.history) >= self.config.CONTEXT_THRESHOLD:
            self.logger.print_header("Compressing Vector History")
            compressed_vector = self.encoder.compress(self.embeddings)
            # Replace history with a summary and the compressed vector
            self.history = [("summary", "Vector history was compressed. Symbolic memory remains complete.")]
            self.embeddings = [compressed_vector]
            self.logger.info("Vector history compressed.")

    def build_prompt(self, latest_query: str) -> str:
        # The new prompt with strict instructions
        return textwrap.dedent(f"""
        **ROLE: Expert Code Developer**
        Your task is to develop a software project step-by-step based on user requests.

        **CRITICAL INSTRUCTIONS:**
        1.  **MAINTAIN & MODIFY:** You MUST treat the `symbolic_map` as the current state of the code. Your primary goal is to MODIFY or ADD to this map. DO NOT start from scratch unless the map is empty.
        2.  **INTEGRATE NEW FEATURES:** When the user asks for a new feature, you must integrate it into the existing code defined in the `symbolic_map`.
        3.  **FULL SCRIPT ASSEMBLY:** If the user asks for the "full script" or "the whole code," you MUST assemble all the `implementation` parts from the `symbolic_map` into a single, runnable HTML file. Combine them logically.
        4.  **MEMORY UPDATE FORMAT:** Your response MUST contain two parts: the user-facing answer (with code snippets or the full code) AND a special block at the end: `<memory_update>JSON_HERE</memory_update>`.
        5.  **JSON CONTENT:** The JSON inside `<memory_update>` MUST reflect the NEW state of the project. If you modify a function, update its entry. If you add a function, create a new entry.

        ---
        [CURRENT PROJECT STATE (READ-ONLY)]
        {self.memory.get_full_memory_json()}
        ---
        [USER REQUEST]
        {latest_query}
        ---

        **RESPONSE STRUCTURE:**
        1.  (Your user-facing answer and code)
        2.  <memory_update>
            {{
              "project_context": {{ ...updated context... }},
              "symbols": {{
                "functionName": {{
                  "type": "function",
                  "description": "A brief description of the function.",
                  "implementation": "function functionName() {{ ...code... }}"
                }},
                "newFunctionName": {{ ... }}
              }}
            }}
            </memory_update>
        """)

# ==============================================================================
# Main Async Loop
# ==============================================================================
async def main():
    logger = Logger()
    if os.path.exists("symbolic_memory.json"):
        if logger.ask_for_confirmation("An old memory file was found. Clear it for a new session?"):
            os.remove("symbolic_memory.json")
            logger.success("Old memory file cleared.")

    manager = ContextManager(threshold=4) # Increased threshold slightly
    await manager.initialize_project()
    context = manager.memory.memory_data["project_context"]
    logger.print_header(f"Project '{context.get('project_name')}' ({context.get('language')}) Ready")

    while True:
        try:
            user_input = input(f"\n{logger.GREEN}{logger.BOLD}You: {logger.ENDC} ")
            if user_input.lower() in ['exit', 'quit']: break
            await manager.process_turn(user_input)
        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            logger.warning(f"\nAn unexpected error occurred: {e}")
    logger.success("Session ended.")

if __name__ == "__main__":
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass
    asyncio.run(main())
