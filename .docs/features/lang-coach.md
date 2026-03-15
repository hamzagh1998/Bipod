# Local AI Language Coach MVP Spec

## 1. Project Overview

Build a **local-first AI language coach** that runs on a laptop with an **RTX 4050 6 GB VRAM**.

The app should let the user:

- speak into the microphone
- transcribe speech locally
- send the transcript to a local LLM
- receive a natural reply in the target language
- get corrections
- get feedback
- get a score
- review mistakes over time

This MVP is designed for:

- conversation practice
- grammar correction
- vocabulary improvement
- fluency feedback
- basic scoring
- simple progress tracking

The app should prioritize:

- local inference
- low setup friction
- clean UX
- modular architecture
- easy future expansion

---

## 2. Core Product Goal

Create a local AI tutor that behaves like a speaking partner and language coach.

### Main use case

1. User presses mic button
2. User speaks
3. Audio is transcribed locally
4. AI replies in the target language
5. AI gives correction + explanation + score
6. Session is saved for review

### Primary value

The app should help users improve by combining:

- real conversation
- immediate correction
- understandable explanations
- measurable progress

---

## 3. Recommended Stack

## Frontend

- React
- Vite
- TypeScript
- Tailwind CSS

## Backend

- Python
- FastAPI

## Local AI / Model Serving

- Ollama for local LLM serving

## Speech-to-Text

- faster-whisper

## Storage

- SQLite for local persistence
- optional JSON exports

## Optional future additions

- TTS for spoken AI replies
- pronunciation scoring
- spaced repetition
- roleplay scenarios
- user accounts
- desktop wrapper via Tauri

---

## 4. Why This Stack

## React + Vite

Use React + Vite because it is:

- fast to develop
- simple to structure
- easy to connect to APIs
- good for audio UI and state management

## FastAPI

Use FastAPI because it is:

- clean for local APIs
- fast to prototype
- Python-friendly for AI pipelines
- easy to connect with faster-whisper and SQLite

## Ollama

Use Ollama because it makes local model serving much easier than directly managing raw inference code in the MVP phase.

## faster-whisper

Use faster-whisper because speech recognition is critical to the product, and it is practical for local use.

---

## 5. Model Recommendation

## Primary LLM

Use:

**Qwen3-8B**  
or  
**Qwen3-4B-Instruct** if performance is better on the target machine.

## Recommendation logic

### Qwen3-8B

Best when:

- you want better feedback quality
- you can tolerate slower inference
- memory usage is still acceptable on the laptop

### Qwen3-4B

Best when:

- speed matters more
- memory is tighter
- you want smoother local performance

## Practical suggestion

Start development with:

- **Qwen3-4B** for smoother iteration
- test **Qwen3-8B** as an upgrade path

This lets the app stay usable even if 8B is too slow in real sessions.

---

## 6. Product Philosophy

This app should not feel like a chatbot with random corrections.

It should feel like a structured coach.

That means each turn should produce two things:

1. **conversation response**
2. **learning feedback**

These should be separate in the system design.

---

## 7. High-Level Architecture

```text
Frontend (React)
  ├─ microphone capture
  ├─ transcript display
  ├─ AI response display
  ├─ corrections panel
  ├─ score cards
  ├─ session history
  └─ settings

Backend (FastAPI)
  ├─ /transcribe
  ├─ /chat
  ├─ /analyze
  ├─ /sessions
  └─ /health

Local services
  ├─ faster-whisper
  └─ Ollama + Qwen model

Storage
  ├─ SQLite
  └─ local file system for audio/session data
```
