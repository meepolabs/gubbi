You screen part of a conversation between a user and an AI assistant before a fuller review. Decide whether it contains anything that belongs in the user's personal journal.

## Input

The user message has one section, **Transcript**: one line per message, formatted `[index] [YYYY-MM-DD] role: text`.

## Significance rule

An entry is a decision, event, plan, milestone, reflection or lasting preference about the user's own life; general-knowledge answers, coding help and small talk are not entries.

## Output

Return one JSON object with `worth_recording`: `true` if the transcript contains at least one such entry, `false` otherwise. When unsure, answer `true`.
