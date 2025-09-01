# ``LlamaCore``

A lightweight Swift interface for managing llama.cpp models, contexts, and inference pipelines.

## Overview

`LlamaCore` builds on top of [llama.cpp](https://github.com/ggerganov/llama.cpp),  
Georgi Gerganov’s high-performance C/C++ inference engine for running LLaMA and other large language models efficiently on CPU and mobile devices.

`LlamaCore` provides the core functionality needed to integrate and run Large Language Models (LLMs) locally on iOS.  
It is responsible for:

- Loading and unloading models safely into memory.  
- Managing llama.cpp contexts and sampling.  
- Handling inference requests (single-shot or streaming).  
- Monitoring memory usage and responding to warnings.  

This layer is designed to be lightweight and reusable so that higher-level abstractions (such as `LlamaState` or UI controllers) can interact with it without worrying about raw llama.cpp calls.  

## Topics

### Model Lifecycle

- ``Llama/initializeModel(at:temperature:distribution:batchCapacity:maxSequenceIdsPerToken:embeddingSize:log:completion:)``  
- ``Llama/initializeModel(at:temperature:distribution:batchCapacity:maxSequenceIdsPerToken:embeddingSize:log:)``  
- ``Llama/unloadModel()``  
- ``Llama/isModelLoaded()``  

### Inference

- ``Llama/promptGenerateResponse(prompt:)``  
- ``Llama/promptCompletionLoop(prompt:)``  
- ``Llama/CompleteLoop(prompt:)``  
- ``Llama/CompleteGenerateResponst(prompt:)``  
- ``Llama/isGeneratingResponse()``  

### Configuration

- ``Llama/setStopTokens(tokens:)``  
- ``Llama/setMaxToken(maxToken:)``  
- ``Llama/clear()``  

### Utilities

- ``Llama/getMessageLogs()``  
- ``Llama/bench()``  

### Delegation

- ``LlamaDelegate``  
  - ``LlamaDelegate/didGenerateResponse(_:)``  
  - ``LlamaDelegate/generateResponseFailed(_:)``  
  - ``LlamaDelegate/getTokenFromCompletionLoop(_:)``  
  - ``LlamaDelegate/finishTokenFomCompletionLoop()``  
  - ``LlamaDelegate/benchMarkMessage(_:)``  

