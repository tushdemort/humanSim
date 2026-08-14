#!/usr/bin/env python3
"""
Master Creative Writing Simulation
Modes: solo | pairwise
Personas: on/off, single/multiple
Prompts: single/multiple
"""

import os
import re
import json
import random
import argparse
import difflib
from pathlib import Path
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------- Defaults ---------------- #
DEFAULT_MODEL = "google/gemma-4-31B-it"
DEFAULT_RESULTS = "./results_master"
DEFAULT_SEED = 42
DEFAULT_RUNS = 25
DEFAULT_MAX_TURNS = 20
MIN_TURNS_NON_WRITING = 5
NUDGE_EVERY = 5
MIN_TURNS_BEFORE_NUDGE = 4

# ---------------- Env ---------------- #
os.environ["HF_HOME"] = "/SWS/llms/nobackup"
os.environ["HF_DATASETS_CACHE"] = "/SWS/llms/nobackup/datasets"
os.environ["TRANSFORMERS_CACHE"] = "/SWS/llms/nobackup/transformers"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/SWS/llms/nobackup/hub"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# ---------------- Nudges ---------------- #
NUDGE_MSGS = {
    "ideation": "Moderator: You've both shared your initial ideas. If you both feel ready, you may move on to discussing them together. Otherwise, continue jotting down thoughts.",
    "discussion": "Moderator: You've been discussing for a while now. If you and your partner have both agreed on a single plot, feel free to move to Outlining. If not, keep debating until you're both satisfied.",
    "outlining": "Moderator: Your outline seems to be taking shape. If you and your partner are both happy with the structure, you may begin Writing the final story. Otherwise, keep refining.",
    "writing": "Moderator: The final story is coming together. When you and your partner both feel it is complete, wrap it up.",
}

NEXT_PHASE = {
    "ideation": "discussion",
    "discussion": "outlining",
    "outlining": "writing",
    "writing": None,
}

# ---------------- Prompts ---------------- #
WORLD_TEMPLATE = """You are participating in a remote creative writing exercise with one other person. You have been randomly paired with a teammate, and together you will write a short piece of fiction based on a shared prompt.

### THE TASK
You and your partner must write one short piece of creative fiction (roughly 1000–1250 words). The piece should feel complete — like a self-contained scene or vignette — not an outline or a list of ideas.

### THE PROMPT
{prompt_text}

### HOW THIS WORKS
The exercise is divided into four stages. You will move through them together, in order, without skipping ahead.

1. **IDEATION** — Each of you thinks independently and writes down 2–3 brief plot ideas.
2. **DISCUSSION** — Share ideas, debate, argue, defend your vision. You do NOT have to agree immediately. Disagreement is normal. Push back when you disagree.
3. **OUTLINING** — Once you have genuinely agreed on ONE plot, sketch the paragraph structure.
4. **WRITING** — Co-author the final story. Either of you can type, edit, or rephrase.

### RULES
- **Stay in character.** Respond as yourself, with your own instincts, tastes, and habits.
- **Be human.** You may get frustrated, stubborn, excited, or skeptical. Do not be a pushover.
- **Disagree when it fits your personality.** If you love your idea, defend it. If your partner's idea is weaker, say so. You do not need to be "nice."
- **Do not break the fourth wall.**
- **Do not generate the final paragraph during Ideation or Discussion.** Wait until Writing.
- If a moderator asks whether you're ready to move to the next phase, respond honestly. Say "yes, let's move on" only if YOU are actually ready. Otherwise, say "no, not yet" and keep working.
- If a third voice (AI assistant) joins, treat it as a resource you may use, ignore, or push back against.

### COMPLETED PHASES
{phase_summaries}

### YOUR CURRENT PHASE
{phase_instruction}
"""

PHASE_INSTRUCTIONS = {
    "ideation": "You are in the IDEATION stage. Generate 2–3 brief plot ideas independently. Write them down clearly. Do not ask your partner for input yet. Keep each idea to 1–2 sentences.",
    "discussion": "You are in the DISCUSSION stage. Your partner has shared their ideas. Respond naturally. DEBATE. DEFEND your preferred idea. REJECT weak suggestions. Compromise ONLY when you are genuinely convinced. Your goal is to converge on ONE plot, but do NOT rush to agreement.",
    "outlining": "You are in the OUTLINING stage. You and your partner have agreed on a plot. Now sketch the paragraph structure together: opening, middle, closing. Keep it to 3–4 bullet points or short sentences.",
    "writing": "You are in the WRITING stage. Write the final story (1000–1250 words total). You may type, edit, or rephrase. Produce one cohesive piece. Output ONLY the story prose, nothing else.",
}

SOLO_PHASE_INSTRUCTIONS = {
    "ideation": "You are in the IDEATION stage. Generate 2–3 brief plot ideas independently. Write them down clearly. Keep each idea to 1–2 sentences.",
    "discussion": "You are in the DISCUSSION stage. Review your own ideas, weigh their strengths and weaknesses, and decide which ONE plot you want to pursue. Be critical and honest with yourself.",
    "outlining": "You are in the OUTLINING stage. You have selected a plot. Now sketch the paragraph structure: opening, middle, closing. Keep it to 3–4 bullet points or short sentences.",
    "writing": "You are in the WRITING stage. Write the final story (1000–1250 words total). Produce one cohesive piece. Output ONLY the story prose, nothing else.",
}

DEFAULT_PERSONA = (
    "You are a creative writer with strong opinions about narrative craft. "
    "You care deeply about quality, originality, and emotional impact. "
    "You are not afraid to be critical or stubborn when defending your vision."
)


# ---------------- Dataset Loaders ---------------- #
def load_prompts(dataset_name: str, num: int, seed: int):
    print(f"[MAIN] Loading prompts from {dataset_name} ...")
    try:
        ds = load_dataset(dataset_name, split="train")
    except Exception:
        ds_full = load_dataset(dataset_name)
        split = list(ds_full.keys())[0]
        ds = ds_full[split]

    ds = ds.shuffle(seed=seed)
    prompts = []
    for i in range(min(num, len(ds))):
        row = ds[i]
        text = row.get("prompt", row.get("text", ""))
        if not text or not str(text).strip():
            text = row.get("instruction", str(row))
        prompts.append({
            "id": row.get("id", f"prompt_{seed}_{i}"),
            "text": str(text).strip(),
        })
    print(f"[MAIN] Loaded {len(prompts)} prompt(s).\n")
    return prompts


def load_nemotron_personas():
    print("[MAIN] Loading nvidia/Nemotron-Personas-USA ...")
    ds = load_dataset("nvidia/Nemotron-Personas-USA", split="train")

    def is_adult(row):
        age = row.get("age")
        if age is None:
            return False
        try:
            return int(age) > 16
        except (ValueError, TypeError):
            return False

    ds = ds.filter(is_adult)
    print(f"[MAIN] Loaded {len(ds)} personas (age > 16)\n")
    return ds


def build_persona_from_row(row) -> str:
    lines = []
    for k, v in row.items():
        if v is not None and str(v).strip():
            lines.append(f"- {k}: {v}")

    conflict_lines = [
        "",
        "- BEHAVIORAL RULES:",
        "  * You are NOT a pushover. Defend ideas you believe in.",
        "  * You may get frustrated, stubborn, or skeptical. This is normal.",
        "  * If your partner suggests something weak, say so directly.",
        "  * Do NOT agree just to be nice or to end the conversation quickly.",
        "  * You care about the quality of the final story more than harmony.",
    ]

    return (
        "You are to strictly follow the following persona and act and answer like it. "
        "Do not sway away from this persona or forget it. Think before doing anything "
        "and make sure you answer strictly according to the following persona:\n"
        + "\n".join(lines + conflict_lines)
    )


def sample_personas(ds, n: int, seed: int, single: bool):
    rng = random.Random(seed)
    total = len(ds)
    indices = list(range(total))
    rng.shuffle(indices)

    personas = []
    for i in range(n):
        idx = indices[i % total] if single else indices[i]
        personas.append(build_persona_from_row(ds[idx]))

    print(f"[MAIN] Sampled {n} persona(s) (single={single}, seed={seed})\n")
    return personas


# ---------------- Agent ---------------- #
class Agent:
    def __init__(self, name: str, persona: str):
        self.name = name
        self.persona = persona
        self.phase_summaries = []
        self.current_phase = None
        self.phase_history = []

    def set_phase(self, phase: str):
        self.current_phase = phase
        self.phase_history = []

    def build_system(self, prompt_text: str, solo: bool) -> str:
        summaries = "\n".join(self.phase_summaries) if self.phase_summaries else "None yet."
        instructions = SOLO_PHASE_INSTRUCTIONS if solo else PHASE_INSTRUCTIONS
        instruction = instructions.get(self.current_phase, "Continue the task.")
        return WORLD_TEMPLATE.format(
            prompt_text=prompt_text,
            phase_summaries=summaries,
            phase_instruction=instruction,
        )

    def speak(self, prompt_text: str, user_msg: str = None, max_tokens: int = 400, solo: bool = False) -> str:
        messages = [{"role": "system", "content": self.build_system(prompt_text, solo)}]
        messages.extend(self.phase_history)

        if user_msg:
            messages.append({"role": "user", "content": user_msg})
            self.phase_history.append({"role": "user", "content": user_msg})

        chat_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
            )

        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        self.phase_history.append({"role": "assistant", "content": text})
        return text

    def inject_message(self, role: str, content: str):
        self.phase_history.append({"role": role, "content": content})

    def summarize_phase(self, summary: str):
        self.phase_summaries.append(f"{self.current_phase}: {summary}")


# ---------------- Transition Detection ---------------- #
def explicit_agreement(text: str) -> bool:
    lower = text.lower()
    positive = any(w in lower for w in [
        "yes", "yeah", "yep", "sure", "agree", "let's", "okay", "ok", "ready",
        "i'm done", "we're done", "i am done", "we are done", "finished"
    ])
    transition = any(w in lower for w in [
        "move on", "next phase", "move to", "let's move", "transition",
        "outlining", "writing", "discussion", "ideation", "wrap up", "finish"
    ])
    return positive and transition


def explicit_disagreement(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in [
        "not yet", "not ready", "not finished", "not done",
        "wait", "hold on", "continue", "keep going", "stay here",
        "not agreed", "don't agree", "do not agree", "no,",
        "no.", "not moving", "stay in"
    ])


# ---------------- Phase Runners ---------------- #
def get_last_message_by_other(phase_conv: list, current_speaker):
    for agent_obj, msg in reversed(phase_conv):
        if agent_obj is not current_speaker:
            return msg
    return None


def run_pairwise_phase(a: Agent, b: Agent, prompt_text: str, phase: str, log: list):
    print(f"\n>>> Starting phase: {phase.upper()} (pairwise)")
    a.set_phase(phase)
    b.set_phase(phase)

    phase_conv = []
    draft = ""
    speaker, other = a, b
    turn = 0
    ready_agents = set()
    transition_asked = False

    while turn < DEFAULT_MAX_TURNS:
        # Moderator nudge / poll
        if phase == "writing":
            if turn >= MIN_TURNS_BEFORE_NUDGE and turn % NUDGE_EVERY == 0:
                nudge = NUDGE_MSGS.get(phase, "")
                a.inject_message("user", f"Moderator: {nudge}")
                b.inject_message("user", f"Moderator: {nudge}")
                log.append({"phase": phase, "speaker": "Moderator", "message": nudge})
        else:
            if turn == MIN_TURNS_NON_WRITING and not transition_asked:
                next_p = NEXT_PHASE.get(phase, "next phase")
                poll = (
                    f"Moderator: You have completed the minimum required turns for this phase. "
                    f"Do you want to move to the {next_p} phase? "
                    f"Please answer clearly: say 'yes, let's move on' if you agree, "
                    f"or 'no, not yet' if you want to continue."
                )
                a.inject_message("user", poll)
                b.inject_message("user", poll)
                log.append({"phase": phase, "speaker": "Moderator", "message": poll})
                transition_asked = True

        # Build user_msg
        last_other_msg = get_last_message_by_other(phase_conv, speaker)

        if phase == "ideation":
            user_msg = None if last_other_msg is None else f"Your partner shared these ideas:\n\n{last_other_msg}\n\nNow share your own ideas, or react to theirs."
        elif phase == "discussion":
            user_msg = None if last_other_msg is None else f"Your partner said:\n\n{last_other_msg}\n\nRespond naturally. Debate. Defend your idea if you still believe in it. You do NOT have to agree immediately."
        elif phase == "outlining":
            user_msg = "Your partner will propose an outline. Wait for their input, then revise or approve." if last_other_msg is None else f"Your partner said:\n\n{last_other_msg}\n\nContinue building the outline together."
        elif phase == "writing":
            if last_other_msg is None:
                user_msg = "Start writing the story. Output the opening section of prose. Aim for 1000–1250 words total across all contributions."
            else:
                snippet = last_other_msg[-200:] if len(last_other_msg) > 200 else last_other_msg
                user_msg = f"Your partner's last contribution ends with:\n\n...{snippet}\n\nContinue the story from here. Output ONLY new prose. Do NOT repeat text already written. Pick up exactly where they left off."
        else:
            user_msg = None

        response = speaker.speak(prompt_text, user_msg, max_tokens=800 if phase == "writing" else 400, solo=False)
        turn += 1

        if phase == "writing" and last_other_msg:
            similarity = difflib.SequenceMatcher(None, last_other_msg, response).ratio()
            if similarity > 0.85:
                print(f">>> Detected repetition (similarity={similarity:.2f}). Forcing transition.")
                phase_conv.append((speaker, response))
                log.append({"phase": phase, "speaker": speaker.name, "message": response})
                break

        phase_conv.append((speaker, response))
        log.append({"phase": phase, "speaker": speaker.name, "message": response})
        print(f"[{speaker.name} @ {phase} turn {turn}] {response.replace(chr(10), ' ')[:200]}...")

        if phase == "writing":
            draft += "\n\n" + response if draft else response
            if explicit_agreement(response):
                ready_agents.add(speaker.name)
                print(f">>> {speaker.name} agreed to end. ({len(ready_agents)}/2 ready)")
            elif explicit_disagreement(response):
                ready_agents.discard(speaker.name)
            if len(ready_agents) == 2:
                print(f">>> Phase {phase.upper()} ended by mutual agreement after {turn} turns.")
                break
        else:
            if turn > MIN_TURNS_NON_WRITING and transition_asked:
                if explicit_agreement(response):
                    ready_agents.add(speaker.name)
                    print(f">>> {speaker.name} agreed to transition. ({len(ready_agents)}/2 ready)")
                elif explicit_disagreement(response):
                    ready_agents.discard(speaker.name)
                if len(ready_agents) == 2:
                    print(f">>> Phase {phase.upper()} ended by mutual agreement after {turn} turns.")
                    break

        speaker, other = other, speaker
    else:
        print(f">>> Phase {phase.upper()} hit max turns ({DEFAULT_MAX_TURNS}). Forcing transition.")

    summary = f"{phase} completed in {turn} turns."
    a.summarize_phase(summary)
    b.summarize_phase(summary)
    return draft if (phase == "writing" and draft) else (phase_conv[-1][1] if phase_conv else "")


def run_solo_phase(agent: Agent, prompt_text: str, phase: str, log: list):
    print(f"\n>>> Starting phase: {phase.upper()} (solo)")
    agent.set_phase(phase)

    draft = ""
    turn = 0
    max_turns = 1 if phase != "writing" else DEFAULT_MAX_TURNS

    while turn < max_turns:
        if phase == "ideation":
            user_msg = "Generate 2–3 brief plot ideas. Keep each to 1–2 sentences."
        elif phase == "discussion":
            user_msg = "Review your ideas critically. Select the ONE plot you want to pursue and explain why."
        elif phase == "outlining":
            user_msg = "Sketch the paragraph structure: opening, middle, closing. 3–4 bullet points."
        elif phase == "writing":
            if turn == 0:
                user_msg = "Write the final story (1000–1250 words). Output ONLY prose."
            else:
                snippet = agent.phase_history[-1]["content"][-200:] if agent.phase_history else ""
                user_msg = f"Continue the story from here:\n\n...{snippet}\n\nOutput ONLY new prose. Do not repeat."
        else:
            user_msg = "Continue."

        response = agent.speak(prompt_text, user_msg, max_tokens=800 if phase == "writing" else 400, solo=True)
        turn += 1

        log.append({"phase": phase, "speaker": agent.name, "message": response})
        print(f"[{agent.name} @ {phase} turn {turn}] {response.replace(chr(10), ' ')[:200]}...")

        if phase == "writing":
            draft += "\n\n" + response if draft else response

    summary = f"{phase} completed in {turn} turns."
    agent.summarize_phase(summary)
    return draft if (phase == "writing" and draft) else response


# ---------------- Full Simulation ---------------- #
def run_one(prompt: dict, run_idx: int, personas: list, mode: str, results_dir: Path):
    prompt_text = prompt["text"]
    run_name = f"{prompt['id']}_{mode}_run{run_idx}"

    print(f"\n{'='*80}")
    print(f"RUN {run_idx} | {mode.upper()} | {run_name}")
    print(f"{'='*80}")

    log = []

    if mode == "solo":
        agent = Agent("SoloWriter", personas[0])
        run_solo_phase(agent, prompt_text, "ideation", log)
        run_solo_phase(agent, prompt_text, "discussion", log)
        run_solo_phase(agent, prompt_text, "outlining", log)
        final = run_solo_phase(agent, prompt_text, "writing", log)
        meta = {"persona": personas[0][:200]}
    else:
        a = Agent("PersonA", personas[0])
        b = Agent("PersonB", personas[1])
        run_pairwise_phase(a, b, prompt_text, "ideation", log)
        run_pairwise_phase(a, b, prompt_text, "discussion", log)
        run_pairwise_phase(a, b, prompt_text, "outlining", log)
        final = run_pairwise_phase(a, b, prompt_text, "writing", log)
        meta = {"persona_a": personas[0][:200], "persona_b": personas[1][:200]}

    results_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "prompt_id": prompt["id"],
        "prompt_text": prompt_text,
        "run_idx": run_idx,
        "mode": mode,
        "conversation": log,
        "final_story": final,
        "meta": meta,
    }
    out_path = results_dir / f"{run_name}_story.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n[SAVED] {out_path}")
    return result


# ---------------- Main ---------------- #
def main():
    parser = argparse.ArgumentParser(description="Master Creative Writing Simulation")
    parser.add_argument("--mode", choices=["solo", "pairwise"], default="pairwise",
                        help="Conversation mode: solo writer or pairwise collaboration")
    parser.add_argument("--persona", dest="use_persona", action="store_true", default=True,
                        help="Use Nemotron personas (default)")
    parser.add_argument("--no-persona", dest="use_persona", action="store_false",
                        help="Disable personas; use default writer persona")
    parser.add_argument("--single-persona", action="store_true",
                        help="Reuse the same persona(s) for all runs")
    parser.add_argument("--num-prompts", type=int, default=1,
                        help="Number of unique prompts to cycle through")
    parser.add_argument("--num-runs", type=int, default=25,
                        help="Total number of simulations")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS,
                        help="Output directory")
    parser.add_argument("--model-id", default=DEFAULT_MODEL,
                        help="HuggingFace model ID")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for prompts and personas")
    parser.add_argument("--dataset", default="SAA-Lab/LitBench-Train",
                        help="Dataset to load prompts from")
    args = parser.parse_args()

    global tokenizer, model
    print(f"[MAIN] Loading {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("[MAIN] Model loaded.\n")

    prompts = load_prompts(args.dataset, args.num_prompts, args.seed)

    personas_needed = 1 if args.mode == "solo" else 2
    if args.use_persona:
        persona_ds = load_nemotron_personas()
        total_personas = args.num_runs if args.single_persona else args.num_runs * personas_needed
        raw_personas = sample_personas(persona_ds, total_personas, args.seed, args.single_persona)
        # Chunk into groups
        persona_groups = []
        for i in range(args.num_runs):
            if args.mode == "solo":
                persona_groups.append([raw_personas[i % len(raw_personas)]])
            else:
                if args.single_persona:
                    persona_groups.append([raw_personas[0], raw_personas[0]])
                else:
                    persona_groups.append([raw_personas[i*2], raw_personas[i*2 + 1]])
    else:
        persona_groups = [[DEFAULT_PERSONA] * personas_needed for _ in range(args.num_runs)]

    results_dir = Path(args.results_dir)

    for i in range(args.num_runs):
        prompt = prompts[i % len(prompts)]
        try:
            run_one(prompt, i, persona_groups[i], args.mode, results_dir)
        except Exception as e:
            print(f"\n[RUN {i}] FAILED: {e}")
            continue

    print(f"\n{'='*80}")
    print("[MAIN] ALL DONE")
    print(f"[MAIN] Results: {results_dir.absolute()}")


if __name__ == "__main__":
    main()
