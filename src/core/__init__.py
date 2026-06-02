"""core — shared Aura LLM + training infrastructure.

Imported by both the `st` (ASR/translation) and `tts` families. Owns the
transformer (llama3, model_factory, kvcache), the AuraLLM wrapper (aura),
the duration-bucket sampler, and the config/scheduler/DDP utilities.
Contains no speech-task-specific code.
"""
