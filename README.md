# Symbolic Memory Engine: An Advanced Framework for LLM-Powered Code Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-Gemini%20API-red)](https://ai.google.dev/)

This repository contains an advanced framework for stateful, iterative software development using Large Language Models (LLMs). It overcomes the inherent limitations of standard LLM chat interfaces by implementing a sophisticated memory architecture, enabling the construction of complex applications step-by-step, just like a human developer would.

This project is not just a chatbot; it's a proof-of-concept for a new paradigm of AI-assisted development, demonstrating remarkable gains in token efficiency and development consistency.

---

## The Problem: The "Amnesia" of Standard LLM Interfaces

Anyone who has tried to build software by having a continuous conversation with an LLM has encountered the same fundamental problems:

1.  **Context Window Limitations:** As the conversation and code grow, the context window fills up, forcing the model to "forget" the beginning of the project.
2.  **State Management Failure:** The model often loses track of the current state of the codebase. It might regenerate code from scratch, ignore previous instructions, or introduce conflicting logic.
3.  **Extreme Inefficiency:** To maintain context, standard chat interfaces repeatedly send the entire conversation history, leading to massive token consumption, higher costs, and slower response times.


## Our Solution: A Synergistic Memory Architecture

This framework solves these problems by treating memory not as a simple transcript, but as a structured, multi-layered system. Our approach is built on two core pillars:

### 1. The Symbolic Memory Map

The cornerstone of our solution. Instead of relying on a raw conversation log, we maintain a structured JSON object that represents the **current, canonical state of the codebase**.

-   **Code as Data:** Every function, variable block, and CSS style is abstracted into a "symbol" with properties like `type`, `description`, and `implementation`.
-   **Living Blueprint:** This map is updated after *every* successful development step. It acts as a single source of truth for the AI.
-   **Instructional Integrity:** When a new request is made, the entire Symbolic Map is passed to the LLM with the strict instruction to **MODIFY and EXTEND** this map, not to start over.

### 2. Semantic Context Compression

While the Symbolic Map holds the *what* (the code), we also need a way to remember the *why* (the conversational context) without sending the full transcript.

-   **Vector Encoding:** We use a sentence-transformer model (`all-MiniLM-L6-v2`) to encode the user-assistant dialogue into dense vector embeddings, capturing the semantic essence of each turn.
-   **Context Averaging:** To maintain efficiency, when the dialogue history reaches a certain threshold, these embeddings are mathematically compressed into a single, averaged vector. This vector represents the "gist" of the recent conversation.
-   **The "Collapse" Analogy:** A useful metaphor for this process is to think of it as a "context collapse." Much like a complex quantum state with many possibilities collapses to a single point upon measurement, the entire recent conversational history is "collapsed" into a single, dense vector that efficiently preserves its core meaning for the LLM.

These two components are combined into a **Synergistic Prompt** that gives the LLM everything it needs in the most efficient format possible:
1.  **Strict Instructions** on its role.
2.  The full **Symbolic Memory Map** as the current code state.
3.  The **compressed vector history** as conversational context.
4.  The new **user request**.

---

## Case Study: Building the "dot" Game

To validate this framework, we built a complete HTML5 game from the ground up through a series of natural language instructions.

**Process:** We started with "I want a dot game" and iteratively added features:
- Falling bricks to dodge.
- A 90-second win timer.
- A steady speed mechanic for the dot.
- A progressive speed increase every 5 seconds.
- A "star" power-up granting a temporary shield.
- A full login system (`username: kostas`, `password: 1234`).
- Progressive stages with increasing difficulty.
- On-screen transitional messages instead of browser alerts.

At every step, the framework correctly modified the existing code, integrating new features seamlessly.

### The Results: Tangible Token Efficiency

The true power of this framework is demonstrated by the dramatic reduction in token usage compared to a standard, "naive" chat session.

| Development Step | Naive Method Tokens (AI Studio Equivalent) | Our Method Tokens (Symbolic Memory) | **Context Reduction** |
| :--- | :--- | :--- | :--- |
| Add Timer (Turn 5) | ~1,528 | ~843 | **44.83%** |
| Add Steady Speed (Turn 7)| ~1,814 | ~977 | **46.14%** |
| Add Shield (Turn 11) | ~4,692 | ~1,187 | **74.70%** |
| Add Login (Turn 13) | ~7,748 | ~1,604 | **79.30%** |
| Add Stages (Turn 15) | ~9,821 | ~2,017 | **79.46%** |
| **Final Code (Turn 19)** | **~14,512** | **~2,336** | **🔥 83.90%** |

> **Conclusion:** By the end of the project, our Symbolic Memory Engine was **over 83% more efficient** than a standard chat interface, while ensuring 100% state consistency.

---

## How It Works: The Development Loop

The core logic operates in a continuous loop:

```
+----------------------+
|     User Input       |
+----------------------+
           |
           v
+----------------------+      +--------------------------+
|   Context Manager    |----->|   Symbolic Memory File   |
| (build_prompt)       |      |    (symbolic_map.json)   |
+----------------------+      +--------------------------+
           |
           v
+----------------------+
|    Gemini API Call   |
+----------------------+
           |
           v
+----------------------+
|  Parse LLM Response  |
| (Code + Memory Update) |
+----------------------+
           |
           v
+----------------------+
|  Update Symbolic Map |
|   (Save to File)     |
+----------------------+
           |
           v
+----------------------+
|  Compress Vector     |
|   History (if needed)|
+----------------------+
```


## Future Work

This framework is a powerful foundation. Future enhancements could include:
-   **Multi-file Support:** Extending the Symbolic Map to manage an entire directory of files.
-   **Refactoring Intelligence:** Giving the AI the ability to suggest and perform code refactoring based on the entire map.
-   **Automated Testing:** Requesting the AI to generate unit tests for the functions in its memory.
-   **UI Integration:** Building a simple web interface around this engine for a more user-friendly experience.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
